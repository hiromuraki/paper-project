import cv2
import torch
from tqdm import tqdm
from .share import IKeypoints2DInferencer
from mmpose.apis import MMPoseInferencer
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from progress import report_progress


class RTMPoseInferencer(IKeypoints2DInferencer):
    """
    使用 MMPose 的 RTMPose-l 模型进行 2D 关键点检测，并输出统一格式的张量。
    输出形状为 [Persons, Frames, 17, 3]，格式为 COCO 的 17 点关键点，其中第三维为 (x, y, confidence)。
    """

    def __init__(self, batch_size: int = 1, device: str = "cuda:0") -> None:
        super().__init__()
        self.batch_size = batch_size

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("⚠️ CUDA 不可用，已自动切换到 CPU")
            self.device = "cpu"
        else:
            self.device = device

    def run_2d_keypoints_inference(
        self,
        input_video: Path,
    ) -> np.ndarray:
        # ================================================
        # 阶段一：使用 MMPose 的 RTMPose-l 模型提取 2D 关键点
        # ================================================
        input_video = input_video.expanduser().resolve()
        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")

        # 🌟 优化：利用 OpenCV 提前获取视频总帧数，喂给 tqdm 产生完美进度条
        cap = cv2.VideoCapture(str(input_video))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        print("正在初始化 RTMPose-l 和 RTMDet-m（首次运行可能需要下载权重）...")
        report_progress("init_2d", message="加载 RTMPose / RTMDet，首次运行会下载权重", force=True)
        inferencer = MMPoseInferencer(
            det_model="rtmdet-m",
            pose2d="rtmpose-l",
            device=self.device,
        )

        print(f"开始提取 2D 关键点: {input_video} (共 {total_frames} 帧)")
        frame_predictions: List[List[Dict[str, np.ndarray]]] = []
        report_progress("2d", 0, total_frames, "帧", "逐帧检测 2D 骨架", force=True)
        result_generator = inferencer(
            str(input_video),
            show=False,
            return_vis=False,
            save_predictions=False,
            batch_size=self.batch_size,
        )

        # 🌟 替换：使用 tqdm 包装生成器，并传入 total_frames
        for result in tqdm(result_generator, total=total_frames, desc="RTMPose 提取中"):
            frame_predictions.append(RTMPoseInferencer.__extract_instances(result))
            report_progress("2d", len(frame_predictions), total_frames, "帧", "逐帧检测 2D 骨架")

        report_progress("2d", len(frame_predictions), len(frame_predictions), "帧", "2D 检测完成", force=True)

        print(f"关键点提取完成，实际提取帧数: {len(frame_predictions)}")

        # ===================================================
        # 阶段2：转化为 [Persons, Frames, 17, 3] 的统一张量格式
        # ===================================================
        print("正在转化为统一的 [Persons, Frames, 17, 3] 张量格式...")
        frames_count = len(frame_predictions)
        if frames_count == 0:
            return np.zeros((0, 0, 17, 3), dtype=np.float32)

        # 1. 找到整个视频中，单帧出现的“最多人数”，以此作为固定的 Person 维度
        max_persons = max(len(persons) for persons in frame_predictions)

        if max_persons == 0:
            print("⚠️ 视频中未检测到任何人！")
            return np.zeros((0, frames_count, 17, 3), dtype=np.float32)

        # 2. 初始化全 0 的多维张量
        # (某帧如果人数少于 max_persons，后面的人天然就是 0，起到了 Padding 的作用)
        output_tensor = np.zeros((max_persons, frames_count, 17, 3), dtype=np.float32)

        # 3. 填装数据
        for frame_idx, persons in enumerate(frame_predictions):
            for person_idx, instance in enumerate(persons):
                # 取出 17x2 的坐标
                kps = np.array(instance["keypoints"], dtype=np.float32)
                # 取出 17 维的置信度，若由于某种原因缺失，默认给 1.0
                scores = np.array(
                    instance.get("keypoint_scores", np.ones(17)), dtype=np.float32
                )

                # 合并到输出张量的对应切片中
                output_tensor[person_idx, frame_idx, :, :2] = kps
                output_tensor[person_idx, frame_idx, :, 2] = scores

        print(
            f"✅ 张量重构完成！最终形状为: {output_tensor.shape} (Persons, Frames, Joints, Channels)"
        )

        return output_tensor

    @classmethod
    def __build_instances(
        cls, keypoints: Any, keypoint_scores: Any = None
    ) -> List[Dict[str, np.ndarray]]:
        keypoints_array = np.asarray(keypoints, dtype=np.float32)
        if keypoints_array.ndim == 2:
            keypoints_array = keypoints_array[None, ...]
        if keypoints_array.ndim != 3:
            return []

        scores_array = None
        if keypoint_scores is not None:
            scores_array = np.asarray(keypoint_scores, dtype=np.float32)
            if scores_array.ndim == 1:
                scores_array = scores_array[None, ...]

        instances: List[Dict] = []
        for index, instance_keypoints in enumerate(keypoints_array):
            instance: Dict[str, np.ndarray] = {"keypoints": instance_keypoints}
            if scores_array is not None and index < len(scores_array):
                instance["keypoint_scores"] = scores_array[index]
            instances.append(instance)
        return instances

    @classmethod
    def __extract_instances(cls, prediction: Any) -> List[Dict[str, np.ndarray]]:
        if prediction is None:
            return []

        if isinstance(prediction, dict):
            if "keypoints" in prediction:
                return cls.__build_instances(
                    prediction["keypoints"], prediction.get("keypoint_scores")
                )

            if "predictions" in prediction:
                return cls.__extract_instances(prediction["predictions"])

            instances: List[Dict[str, np.ndarray]] = []
            for value in prediction.values():
                instances.extend(cls.__extract_instances(value))
            return instances

        if isinstance(prediction, (list, tuple)):
            instances: List[Dict[str, np.ndarray]] = []
            for item in prediction:
                instances.extend(cls.__extract_instances(item))
            return instances

        return []
