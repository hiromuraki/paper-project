"""Video -> RTMPose COCO/H36M 2D -> MHFormer 3D; models load lazily."""
from pathlib import Path
import argparse
import os
import sys

SOURCE_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHT = SOURCE_DIR / "MHFormer/checkpoint/pretrained/351/model_4294.pth"
sys.path.insert(0, str(SOURCE_DIR / "MHFormer"))
sys.path.insert(0, str(SOURCE_DIR))
from progress import report_progress


def extract_3d_keypoints(input_video_filepath, output_dir=None, device="auto", weight_path=None):
    # Keep downloaded model weights in this tool; Windows profile caches can be read-only.
    os.environ.setdefault("TORCH_HOME", str(SOURCE_DIR.parent / ".model-cache"))
    report_progress("initializing", message="加载提取依赖并检查视频", force=True)
    import cv2
    import numpy as np
    from core import RTMPoseInferencer, MHFormerInferencer
    from common import DataConverter, VideoUtils, Serializer

    video = Path(input_video_filepath).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve() if output_dir is not None else video.parent
    weight = Path(weight_path).expanduser().resolve() if weight_path else DEFAULT_WEIGHT
    if not video.is_file():
        raise FileNotFoundError("视频不存在: {}".format(video))
    if not weight.is_file():
        raise FileNotFoundError("MHFormer 权重不存在: {}".format(weight))
    width, height = VideoUtils.get_video_size(video)
    cap = cv2.VideoCapture(str(video))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("视频尺寸或 FPS 无效，请转码后重试")
    output.mkdir(parents=True, exist_ok=True)
    selected_device = "cuda:0" if device == "auto" else device
    print("STAGE:2d", flush=True)
    coco = RTMPoseInferencer(device=selected_device).run_2d_keypoints_inference(video)
    if coco.shape[0] == 0 or coco.shape[1] == 0:
        raise ValueError("视频中未检测到人物或没有可解码帧")
    # Same person policy as the original pipeline: first detection, no tracking.
    h36m = DataConverter.coco_to_h36m_with_confidence(coco[0])
    report_progress("save_2d", message="转换 H36M 并保存 2D 骨架", force=True)
    metadata = dict(fps=fps, width=width, height=height)
    paths = {}
    for name, data in (("2d_coco17", coco), ("2d_h36m", h36m)):
        paths[name] = output / (video.stem + "_" + name + "_kps.npz")
        Serializer.save_keypoints_to_npz(data, paths[name], **metadata)
    print("STAGE:3d", flush=True)
    report_progress("init_3d", message="加载 MHFormer 权重", force=True)
    model = MHFormerInferencer(weight_path=weight, window=351, stride=60, device=selected_device)
    points = model.run_3d_keypoints_inference(h36m, video_width=width, video_height=height)
    paths["3d"] = output / (video.stem + "_3d_kps.npz")
    report_progress("save_3d", message="保存 3D 骨架", force=True)
    Serializer.save_keypoints_to_npz(points, paths["3d"], **metadata)
    print("STAGE:done", flush=True)
    report_progress("done", message="2D / 3D 骨架均已生成", force=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description="从视频提取 2D / 3D 骨架")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda:0"), default="auto")
    parser.add_argument("--weight", help="MHFormer 351 帧权重路径")
    args = parser.parse_args()
    extract_3d_keypoints(args.input, args.output, args.device, args.weight)


if __name__ == "__main__":
    main()
