import os
import copy
from pathlib import Path
import numpy as np
import argparse
import torch
from tqdm import tqdm
from MHFormer.model.mhformer import Model
from MHFormer.common.camera import normalize_screen_coordinates
from .share import IKeypoints3DInferencer
from progress import report_progress


class MHFormerInferencer(IKeypoints3DInferencer):
    def __init__(
        self, weight_path: Path, window: int, stride: int, device: str = "cuda"
    ):
        """
        初始化 MHFormer 模型
        :param weight_path: .pth 权重文件路径
        :param window: 时序窗口大小 (与预训练权重绑定的帧数，通常为 351, 81 或 27)
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.window = window
        self.stride = stride

        # 1. 模拟 argparse 配置构造 args
        args, _ = argparse.ArgumentParser().parse_known_args()
        args.layers = 3
        args.channel = 512
        args.d_hid = 1024
        args.frames = window  # 模型结构强依赖此参数
        args.n_joints = 17
        args.out_joints = 17

        # 2. 实例化并加载权重
        print(f"📦 正在初始化 MHFormer ({window} 帧版)...")
        self.model = Model(args).to(self.device)

        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"⚠️ 找不到 MHFormer 权重文件: {weight_path}")

        pre_dict = torch.load(weight_path, map_location=self.device)
        if "state_dict" in pre_dict:
            pre_dict = pre_dict["state_dict"]
        # Accept DataParallel exports, but never silently infer with random weights.
        state_dict = {k.removeprefix("module."): v for k, v in pre_dict.items()}
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

        # 定义骨骼左右翻转映射规则
        self.joints_left = [4, 5, 6, 11, 12, 13]
        self.joints_right = [1, 2, 3, 14, 15, 16]

    def _infer_chunk(
        self, chunk_2d: np.ndarray, video_width: int, video_height: int
    ) -> np.ndarray:
        """
        对形状为 [Window, 17, 2] 的单一块进行 3D 升维推理 (含 TTA 增强)
        """
        # 1. 归一化到屏幕坐标系 [-1, 1]
        input_2d = normalize_screen_coordinates(chunk_2d, w=video_width, h=video_height)

        # 2. TTA (Test-Time Augmentation): 构造翻转输入
        input_2d_aug = copy.deepcopy(input_2d)
        input_2d_aug[:, :, 0] *= -1  # X 轴反转
        input_2d_aug[:, self.joints_left + self.joints_right] = input_2d_aug[
            :, self.joints_right + self.joints_left
        ]

        # 将正向和翻转堆叠成 Batch [2, Window, 17, 2]
        input_batch = np.stack([input_2d, input_2d_aug], axis=0)
        input_tensor = torch.from_numpy(input_batch.astype(np.float32)).to(self.device)

        # 3. 前向传播
        with torch.no_grad():
            output_3d_tensor = self.model(input_tensor)  # 输出: [2, Window, 17, 3]

        # 4. 解析翻转后的结果并求平均
        out_non_flip = output_3d_tensor[0:1]  # [1, Window, 17, 3]
        out_flip = output_3d_tensor[1:2].clone()

        out_flip[:, :, :, 0] *= -1
        out_flip[:, :, self.joints_left + self.joints_right, :] = out_flip[
            :, :, self.joints_right + self.joints_left, :
        ]

        out_3d = (out_non_flip + out_flip) / 2.0
        out_3d = out_3d.squeeze(0).cpu().numpy()  # [Window, 17, 3]

        # 5. Root-Relative (骨盆归零，以此作为中心点就足够了)
        out_3d = out_3d - out_3d[:, 0:1, :]

        # 6. [删除] MHFormer 特有的相机视角对齐 (转到世界坐标系)
        # rot = np.array([0.1407..., -0.1500..., -0.755..., 0.622...], dtype=np.float32)
        # out_3d = camera_to_world(out_3d, R=rot, t=0)

        return out_3d

    def run_3d_keypoints_inference(
        self,
        keypoints_2d: np.ndarray,
        video_width: int,
        video_height: int,
    ) -> np.ndarray:
        total_frames = keypoints_2d.shape[0]
        if total_frames == 0:
            return np.zeros((0, 17, 3), dtype=np.float32)

        # 丢弃置信度，只保留 x, y 坐标进行推理
        kps_2d = keypoints_2d[:, :, :2].copy()

        # ==========================================
        # 边界情况：视频长度小于模型窗口
        # ==========================================
        if total_frames <= self.window:
            report_progress("3d", 0, 1, "窗口", "重建 3D 骨架（短视频补齐至 351 帧）", force=True)
            pad_length = self.window - total_frames
            # 使用边缘填充策略补齐帧数
            padded_2d = np.pad(kps_2d, ((0, pad_length), (0, 0), (0, 0)), mode="edge")
            pred_3d = self._infer_chunk(padded_2d, video_width, video_height)
            report_progress("3d", 1, 1, "窗口", "3D 重建完成", force=True)

            # Keep the same pelvis-relative camera coordinates as long videos.
            return pred_3d[:total_frames]

        # ==========================================
        # 滑动窗口分块推理 (核心提速逻辑)
        # ==========================================
        final_3d = np.zeros((total_frames, 17, 3), dtype=np.float32)
        weight_counts = np.zeros((total_frames, 1, 1), dtype=np.float32)

        starts = list(range(0, total_frames - self.window + 1, self.stride))
        # 防坑：确保最后几帧绝对被覆盖到
        if starts[-1] + self.window < total_frames:
            starts.append(total_frames - self.window)

        print(
            f"🔄 启动 MHFormer 块级滑动推理: 总帧数 {total_frames}, 切分 {len(starts)} 块..."
        )

        report_progress("3d", 0, len(starts), "窗口", "滑动窗口重建 3D 骨架", force=True)
        for index, start in enumerate(tqdm(starts, desc="MHFormer 3D 升维"), 1):
            end = start + self.window
            chunk_2d = kps_2d[start:end]

            # 执行块预测
            chunk_3d = self._infer_chunk(chunk_2d, video_width, video_height)

            # 累加预测结果
            final_3d[start:end] += chunk_3d
            weight_counts[start:end] += 1
            report_progress("3d", index, len(starts), "窗口", "滑动窗口重建 3D 骨架")

        # 对重叠区域求平均值 (时序平滑)
        final_3d = final_3d / weight_counts

        # 把整个视频序列的地面拉平 (按帧校准最低点，复刻官方 Demo 的视觉效果)
        # final_3d[:, :, 2] -= np.min(final_3d[:, :, 2], axis=1, keepdims=True)

        print("✅ 全序列 3D 推理完毕！")
        return final_3d
