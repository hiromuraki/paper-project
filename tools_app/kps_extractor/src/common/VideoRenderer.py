from tqdm import tqdm
from pathlib import Path
from typing import Dict, Literal, Tuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')


class VideoRenderer:
    COCO17_POSE_CONNECTIONS = [
        # --- 下半身 (双腿) ---
        (15, 13),  # 左小腿: 左脚踝(15) -> 左膝(13)
        (13, 11),  # 左大腿: 左膝(13)   -> 左髋(11)
        (16, 14),  # 右小腿: 右脚踝(16) -> 右膝(14)
        (14, 12),  # 右大腿: 右膝(14)   -> 右髋(12)

        # --- 躯干 (连接四肢的核心) ---
        (11, 12),  # 骨盆底: 左髋(11)   -> 右髋(12)
        (5, 11),   # 左侧腰: 左肩(5)    -> 左髋(11)
        (6, 12),   # 右侧腰: 右肩(6)    -> 右髋(12)
        (5, 6),    # 肩膀线: 左肩(5)    -> 右肩(6)

        # --- 上半身 (双臂) ---
        (5, 7),    # 左大臂: 左肩(5)    -> 左肘(7)
        (7, 9),    # 左小臂: 左肘(7)    -> 左腕(9)
        (6, 8),    # 右大臂: 右肩(6)    -> 右肘(8)
        (8, 10),   # 右小臂: 右肘(8)    -> 右腕(10)

        # --- 头部 (五官及脖颈连接) ---
        (1, 2),    # 双眼连线: 左眼(1) -> 右眼(2)
        (0, 1),    # 左侧鼻梁: 鼻子(0) -> 左眼(1)
        (0, 2),    # 右侧鼻梁: 鼻子(0) -> 右眼(2)
        (1, 3),    # 左侧脸颊: 左眼(1) -> 左耳(3)
        (2, 4),    # 右侧脸颊: 右眼(2) -> 右耳(4)

        # --- 颈部 (连接头与躯干) ---
        (3, 5),    # 左侧脖子: 左耳(3) -> 左肩(5)
        (4, 6),    # 右侧脖子: 右耳(4) -> 右肩(6)
    ]

    H36M_POSE_CONNECTIONS = [
        # 下半身
        (0, 1), (1, 2), (2, 3),       # 右腿: 骨盆(0) -> 右髋(1) -> 右膝(2) -> 右脚(3)
        (0, 4), (4, 5), (5, 6),       # 左腿: 骨盆(0) -> 左髋(4) -> 左膝(5) -> 左脚(6)

        # 躯干与头部
        (0, 7), (7, 8),               # 脊椎: 骨盆(0) -> 脊椎中心(7) -> 胸腔/脖子底部(8)
        (8, 9), (9, 10),              # 头部: 胸腔(8) -> 鼻子/面部(9) -> 头顶(10)

        # 上半身 (手臂从胸腔节点辐射出去)
        (8, 14), (14, 15), (15, 16),  # 右臂: 胸腔(8) -> 右肩(14) -> 右肘(15) -> 右腕(16)
        (8, 11), (11, 12), (12, 13)   # 左臂: 胸腔(8) -> 左肩(11) -> 左肘(12) -> 左腕(13)
    ]

    PERSON_COLORS = [
        # 1号人: 纯红色的点, 纯蓝色的线 (极其醒目)
        ((0, 0, 255), (255, 0, 0)),
        # 2号人: 纯蓝色的点, 纯红色的线 (反转一下以示区分)
        ((255, 0, 0), (0, 0, 255)),
        # 3号人: 紫色的点, 绿色的线 (备用高对比度)
        ((255, 0, 255), (0, 255, 0)),
        # 4号人: 纯黑色的点, 深橙色的线
        ((0, 0, 0), (0, 140, 255)),
    ]

    @classmethod
    def render_3d_keypoints(
        cls,
        keypoints_3d: np.ndarray,
        output_video_path: Path,
        flip_x: bool = False,
        flip_y: bool = False,
        fps: int = 30,
    ):
        """
        渲染 3D 骨骼的关键点，并输出为演示视频。
        输入: keypoints_3d 形状为 [Frames, 17, 3]
        """
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        frames = keypoints_3d.shape[0]

        if frames == 0:
            print("⚠️ 输入的 3D 帧数为 0，跳过渲染。")
            return

        # ==========================================
        # 步骤 1: 确定统一的三维边界 (防止画面在视频里忽大忽小)
        # ==========================================
        # 3D 升维模型输出的坐标系通常是：X(左右), Y(上下/向下为正), Z(深度/前后)
        # 我们计算整个动作序列的活动半径，来锁定摄像机视角
        radius = np.max(np.max(keypoints_3d, axis=(0, 1)) - np.min(keypoints_3d, axis=(0, 1))) / 2.0

        # ==========================================
        # 步骤 2: 初始化 Matplotlib 画布
        # ==========================================
        # 创建 800x800 的正方形画布
        fig = plt.figure(figsize=(8, 8), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        fig.tight_layout()

        # 预先绘制一帧来获取实际的像素分辨率，初始化 OpenCV 写入器
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        print(f"🎬 开始渲染 3D 骨骼演示视频: {output_video_path}")

        # ==========================================
        # 步骤 3: 逐帧渲染与合并
        # ==========================================
        for i in tqdm(range(frames), desc="3D 渲染中"):
            ax.clear()

            # 设置观察视角 (Elevation仰角, Azimuth方位角)
            # elev=15, azim=70 是经典的 MHFormer 侧后方观察视角
            ax.view_init(elev=15, azim=70)
            ax.set_box_aspect([1, 1, 1])  # 保持长宽高比例为 1:1:1

            kps = keypoints_3d[i]

            # 🌟 核心：坐标系映射
            # 神经网络输出：X 是左右，Y 是向下（图像坐标系），Z 是深度
            # Matplotlib 3D 坐标系：要让人立起来，Z 轴必须是向上
            # 所以我们将输出的 Y 轴取负值映射给 Matplotlib 的 Z 轴
            x = kps[:, 0] if not flip_x else -kps[:, 0]
            y = kps[:, 2] if not flip_y else -kps[:, 2]  # 把深度映射到 Matplotlib 的水平纵深 Y 轴
            z = -kps[:, 1]  # 把向下正值的图像 Y 轴反转，变成向上的高度 Z 轴

            # 获取当前帧的中心点，用于摄像机追踪
            center_x, center_y, center_z = np.mean(x), np.mean(y), np.mean(z)

            # 1. 绘制 17 个黑色关节点
            ax.scatter(x, y, z, c='black', s=20)

            # 2. 绘制彩色骨骼连线
            for start_idx, end_idx in cls.H36M_POSE_CONNECTIONS:
                # 按照国际学术界惯例区分左右半身
                if start_idx in [1, 2, 3, 14, 15, 16] or end_idx in [1, 2, 3, 14, 15, 16]:
                    color = '#0000FF'  # 右侧：纯蓝色
                elif start_idx in [4, 5, 6, 11, 12, 13] or end_idx in [4, 5, 6, 11, 12, 13]:
                    color = '#FF0000'  # 左侧：纯红色
                else:
                    color = '#800080'  # 躯干中心线：紫色

                ax.plot([x[start_idx], x[end_idx]],
                        [y[start_idx], y[end_idx]],
                        [z[start_idx], z[end_idx]],
                        color=color, linewidth=2.5)

            # 3. 锁定摄像机边界 (追踪人物但不改变变焦倍率)
            ax.set_xlim3d([center_x - radius, center_x + radius])
            ax.set_ylim3d([center_y - radius, center_y + radius])
            ax.set_zlim3d([center_z - radius, center_z + radius])

            # 4. 美化：隐藏烦人的坐标轴刻度数字，只留网格线体现空间感
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_zticklabels([])

            # 如果你想画面完全干净，连网格线都不要，可以解除下面这句的注释
            # ax.axis('off')

            # ==========================================
            # 步骤 4: 提取画布像素并写入 OpenCV
            # ==========================================
            fig.canvas.draw()
            # 获取 RGB 图像缓冲
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

            # 转换成 OpenCV 需要的 BGR 格式
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            writer.write(img_bgr)

        # 释放资源
        writer.release()
        plt.close(fig)
        print(f"✅ 3D 渲染成功！已导出: {output_video_path}")

    @classmethod
    def render_2d_keypoints(
        cls,
        video_path: Path,
        keypoints_2d: np.ndarray,  # 接收形状为 [Frames, 17, 2] 或 [Frames, 17, 3] 的张量
        bones_mode: Literal["coco17", "h36m"],
        output_video_path: Path,
    ) -> Path:
        input_video = Path(video_path).expanduser().resolve()
        output_video = Path(output_video_path).expanduser().resolve()

        if not input_video.exists():
            raise FileNotFoundError(f"Input video not found: {input_video}")

        output_video.parent.mkdir(parents=True, exist_ok=True)

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open input video: {input_video}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 尝试获取视频总帧数用于 tqdm 的进度计算
        video_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = keypoints_2d.shape[0]
        # 如果获取不到真实的视频帧数，就兜底使用关键点的长度
        if video_total_frames <= 0:
            video_total_frames = total_frames

        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),  # 修正了老版本 opencv 的调用方式警告
            fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open output video for writing: {output_video}")

        print(f"🎬 开始渲染 2D 骨骼视频: {output_video}")
        frame_index = 0

        # 🌟 使用 with 语句自动管理 tqdm 进度条的生命周期
        with tqdm(total=video_total_frames, desc="2D 渲染中") as pbar:
            while True:
                success, frame = capture.read()
                if not success:
                    break

                rendered_frame = frame.copy()

                # 如果当前视频帧在关键点数组长度范围内，且该帧不是全 0
                if frame_index < total_frames:
                    frame_kps = keypoints_2d[frame_index]
                    # 传入 person_index=0 因为现在的张量每帧只代表 1 个人（主体）
                    cls.__draw_pose_instance(rendered_frame, frame_kps, 0, bones_mode)

                writer.write(rendered_frame)

                # 更新循环变量和进度条
                frame_index += 1
                pbar.update(1)

        capture.release()
        writer.release()

        print(f"✅ 2D 骨骼视频已导出: {output_video}")
        return output_video

    @classmethod
    def __draw_pose_instance(
        cls,
        frame: np.ndarray,
        keypoints: np.ndarray,  # 形状为 [17, 2] 或 [17, 3]
        person_index: int,
        bones_mode: Literal["coco17", "h36m"]
    ) -> None:
        point_color, line_color = cls.PERSON_COLORS[person_index % len(cls.PERSON_COLORS)]
        visible_points: Dict[int, Tuple[int, int]] = {}

        for keypoint_index, point in enumerate(keypoints):
            x_coord = int(round(float(point[0])))
            y_coord = int(round(float(point[1])))

            # 过滤掉填充为 0 的无效点（防止在画面左上角原点画出一团线）
            if x_coord == 0 and y_coord == 0:
                continue

            visible_points[keypoint_index] = (x_coord, y_coord)
            cv2.circle(frame, (x_coord, y_coord), 4, point_color, -1)

        pose_connections = cls.COCO17_POSE_CONNECTIONS if bones_mode == "coco17" else cls.H36M_POSE_CONNECTIONS
        for start_index, end_index in pose_connections:
            if start_index not in visible_points or end_index not in visible_points:
                continue
            cv2.line(frame, visible_points[start_index], visible_points[end_index], line_color, 2)
