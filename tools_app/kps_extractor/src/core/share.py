from pathlib import Path
import numpy as np


class IKeypoints3DInferencer:
    def run_3d_keypoints_inference(
        self,
        keypoints_2d: np.ndarray,  # shape: [Frames, 17, 3] (x, y, confidence)
        video_width: int,
        video_height: int,
        stride: int,
    ) -> np.ndarray:  # shape: [Frames, 17, 3]
        """
        使用重叠滑动窗口策略，对任意长度的视频进行平滑 3D 升维。
        :param keypoints_2d: 形状为 [Frames, 17, 3] 的全量 2D 数据
        :param video_width: 视频的原始宽度（像素），用于归一化计算
        :param video_height: 视频的原始高度（像素），用于归一化计算
        :param window: 模型的固定时序长度
        :param stride: 滑动步长 (推荐为 window 的一半)
        """
        raise NotImplementedError


class IKeypoints2DInferencer:
    def run_2d_keypoints_inference(
        self,
        input_video: Path,
    ) -> np.ndarray:  # shape: [Persons, Frames, 17, 3]
        """
        对输入视频进行 2D 关键点检测。
        形状为 [Persons, Frames, 17, 3]，其中第三维为 (x, y, confidence)。
        """
        raise NotImplementedError
