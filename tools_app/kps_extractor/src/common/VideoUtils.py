from pathlib import Path
from typing import Tuple
import cv2


class VideoUtils:
    @classmethod
    def get_video_size(cls, video_path: Path) -> Tuple[int, int]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        capture.release()
        return width, height
