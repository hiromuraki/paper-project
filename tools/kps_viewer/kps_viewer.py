"""读取项目 dataset 数据包的骨骼序列查看器。运行 tools/kps_viewer/run.ps1 或 run.sh。"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

# H36M 17 关节，与 src/skeleton_demo/data.py 的 H36M_PARENTS 保持一致。
# 这里保留轻量副本而不是 import，是为了让查看器不必连带导入 torch。
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]
BONES = [[child, parent] for child, parent in enumerate(H36M_PARENTS) if parent >= 0]

# 优先尝试的数据键：本项目 kp2d.npz / kp3d.npz 用 keypoints，
# reconstruction 兼容 keypoints_3d_viewer，kp / x 兼容旧数据格式。
KEY_CANDIDATES = ("keypoints", "reconstruction", "kp", "x")

HTML_PATH = Path(__file__).with_suffix(".html")
VENDOR_DIR = Path(__file__).parent / "vendor"
DATASET_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FPS = 30

app = FastAPI(title="KPS Viewer")


def pick_array(npz: np.lib.npyio.NpzFile) -> tuple[np.ndarray, str]:
    """从 npz 里挑出骨架数组，返回 (数组, 键名)。"""
    for key in KEY_CANDIDATES:
        if key in npz.files and npz[key].ndim >= 2:
            return npz[key], key

    for key in npz.files:
        if npz[key].ndim >= 2:
            return npz[key], key

    raise ValueError(
        f"npz 里没有找到骨架数组（可用键: {', '.join(npz.files)}）；"
        "期望 [T, 17, 2] 或 [T, 17, 3]"
    )


def normalize_frames(array: np.ndarray, slot: str = "auto") -> dict:
    """把骨架数组整理成 [T, J, 3]，返回解析结果字典。

    slot 决定通道语义：
      - "2d"：3 通道按 (x, y, 置信度) 解读，置信度会被丢弃（叠加只用 x/y）；
      - "3d"：3 通道按 (x, y, z) 解读，2 通道补 z=0；
      - "auto"：按通道数判断（2 → 2D，3 → 3D）。
    """
    if slot not in ("auto", "2d", "3d"):
        raise ValueError(f"slot 只能是 auto / 2d / 3d，实际是 {slot!r}")

    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 4:  # [N, T, J, D] 带 batch 维，取第一条
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"期望 [T, J, 2] 或 [T, J, 3]，实际是 {tuple(array.shape)}")

    num_frames, num_joints, channels = array.shape
    if channels not in (2, 3):
        raise ValueError(
            f"最后一维应为 2 或 3（坐标，2D 时第 3 维可为置信度），"
            f"实际是 {channels}，shape={tuple(array.shape)}"
        )
    if num_frames == 0 or num_joints == 0:
        raise ValueError(f"骨架序列为空，shape={tuple(array.shape)}")

    # NaN / inf 无法进入 JSON，统一置零并计数上报。
    # 注意不能用 nan_to_num 的默认值：inf 会被换成 float32 最大值，
    # 后面 np.round 再乘 1e6 会溢出回 inf，且巨大坐标会把相机推到天文距离。
    finite = np.isfinite(array)
    num_nonfinite = int(finite.size - finite.sum())
    if num_nonfinite:
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=True)

    has_confidence = False
    confidence_range: list[float] | None = None
    confidence_suspect = False

    if slot == "2d":
        if channels == 3:
            has_confidence = True
            third = array[..., 2]
            low, high = float(third.min()), float(third.max())
            confidence_range = [round(low, 4), round(high, 4)]
            # 置信度应落在 0–1；越界说明第 3 通道是 z，文件放错栏位了
            confidence_suspect = low < -0.05 or high > 1.2
        array = array[..., :2]
        dim = 2
    elif slot == "3d":
        dim = 3
    else:
        dim = channels

    if array.shape[-1] == 2:  # 补 z=0，两栏统一按三维坐标下发
        array = np.concatenate([array, np.zeros_like(array[..., :1])], axis=-1)

    return {
        "frames": array,
        "dim": dim,
        "nonfinite": num_nonfinite,
        "has_confidence": has_confidence,
        "confidence_range": confidence_range,
        "confidence_suspect": confidence_suspect,
    }


def pick_fps(npz: np.lib.npyio.NpzFile) -> int:
    if "fps" in npz.files:
        value = npz["fps"]
        if value.size == 1:
            fps = int(np.asarray(value).reshape(-1)[0])
            if 1 <= fps <= 1000:
                return fps
    return DEFAULT_FPS


def parse_npz(source, slot: str) -> dict:
    with np.load(source, allow_pickle=False) as npz:
        array, source_key = pick_array(npz)
        parsed = normalize_frames(array, slot)
        fps = pick_fps(npz)
    frames = parsed.pop("frames")
    return {
        "status": "success", "source_key": source_key,
        "dim": parsed["dim"], "has_confidence": parsed["has_confidence"],
        "confidence_range": parsed["confidence_range"],
        "confidence_suspect": parsed["confidence_suspect"],
        "nonfinite_replaced": parsed["nonfinite"],
        "num_frames": int(frames.shape[0]), "num_joints": int(frames.shape[1]),
        "fps": fps, "bones_topology": BONES,
        "frames": np.round(frames.astype(np.float64), 6).tolist(),
    }


@app.post("/api/upload_npz")
async def upload_npz(file: UploadFile = File(...), slot: str = "auto") -> JSONResponse:
    try:
        return JSONResponse(parse_npz(io.BytesIO(await file.read()), slot))
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)}, status_code=400)


def dataset_root(source: str) -> Path:
    root = DATASET_ROOT.resolve()
    if Path(source).name != source or not (source == "dataset" or source.startswith("dataset-")):
        raise HTTPException(400, "数据源必须是根目录下的 dataset 或 dataset-* 目录")
    path = (root / source).resolve()
    if path.parent != root:
        raise HTTPException(403, "数据源必须位于指定根目录下")
    if not path.is_dir():
        raise HTTPException(404, "数据源目录不存在")
    return path


@app.get("/api/sources")
def sources() -> dict:
    root = DATASET_ROOT.resolve()
    names = sorted(path.name for path in root.iterdir()
                   if path.is_dir() and not path.is_symlink()
                   and (path.name == "dataset" or path.name.startswith("dataset-"))) if root.is_dir() else []
    return {"root": str(root), "sources": names}


def dataset_path(relative: str, source: str = "dataset") -> Path:
    root = dataset_root(source)
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise HTTPException(403, f"路径必须位于 {source} 内")
    return path


def package_folder(relative: str, source: str = "dataset") -> Path:
    path = dataset_path(relative, source)
    if len(path.relative_to(dataset_root(source)).parts) != 2:
        raise HTTPException(400, f"资源包必须位于 {source}/<分类>/<包名>")
    if not path.is_dir():
        raise HTTPException(404, "数据包不存在")
    return path


@app.get("/api/packages")
def packages(source: str = "dataset") -> dict:
    root = dataset_root(source)
    # dataset 是第一级，分类第二级，资源包第三级；不深入包内发现子包。
    items = []
    groups = []
    if root.is_dir():
        for group in sorted(root.iterdir()):
            if not group.is_dir() or group.is_symlink():
                continue
            groups.append(group.name)
            for folder in sorted(group.iterdir()):
                if not folder.is_dir() or folder.is_symlink():
                    continue
                items.append({"path": folder.relative_to(root).as_posix(),
                              "files": [name for name in ("kp2d.npz", "kp3d.npz", "video.mp4", "remark.txt", "remakr.txt", "meta.json") if (folder / name).is_file()]})
    return {"root": f"./{source}", "source": source, "groups": groups, "packages": items}


@app.get("/api/package")
def package(path: str, source: str = "dataset") -> dict:
    folder = package_folder(path, source)
    result = {"path": path, "tracks": {}, "errors": [], "texts": {}, "video": None}
    for slot in ("2d", "3d"):
        name = f"kp{slot}.npz"
        file = dataset_path(f"{path}/{name}", source)
        if file.is_file():
            try:
                result["tracks"][slot] = parse_npz(file, slot)
            except Exception as exc:
                result["errors"].append(f"{name}: {exc}")
    for name in ("remark.txt", "remakr.txt", "meta.json"):
        file = dataset_path(f"{path}/{name}", source)
        if file.is_file():
            try:
                with file.open(encoding="utf-8-sig", newline="") as stream:
                    result["texts"][name] = stream.read()
            except Exception as exc:
                result["errors"].append(f"{name}: {exc}")
    if dataset_path(f"{path}/video.mp4", source).is_file():
        from urllib.parse import quote
        result["video"] = "/api/video?source=" + quote(source, safe="") + "&path=" + quote(path, safe="")
    return result


@app.get("/api/video")
def package_video(path: str, source: str = "dataset") -> FileResponse:
    package_folder(path, source)
    file = dataset_path(f"{path}/video.mp4", source)
    if not file.is_file():
        raise HTTPException(404, "video.mp4 不存在")
    return FileResponse(file, media_type="video/mp4")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(HTML_PATH)


@app.get("/vendor/{name}")
def vendor(name: str) -> FileResponse:
    """提供内置的 three.js / OrbitControls（不依赖外网 CDN）。"""
    path = (VENDOR_DIR / name).resolve()
    if path.parent != VENDOR_DIR.resolve() or not path.is_file():
        raise HTTPException(status_code=404, detail=f"vendor 里没有 {name}")
    return FileResponse(path, media_type="application/javascript")


@app.get("/favicon.ico")
def favicon() -> Response:
    return Response(status_code=204)


def main() -> None:
    global DATASET_ROOT
    parser = argparse.ArgumentParser(description="骨骼序列动画查看器")
    parser.add_argument("-dataset-root", "--dataset-root", type=Path, default=DATASET_ROOT,
                        help="包含 dataset 和 dataset-* 的根目录，默认项目根目录")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认仅本机")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    DATASET_ROOT = args.dataset_root.expanduser().resolve()
    if not DATASET_ROOT.is_dir():
        parser.error(f"根目录不存在: {DATASET_ROOT}")

    local = args.host.startswith("127.") or args.host in ("::1", "localhost", "")
    notes = [] if local else [f"监听 {args.host}，局域网可用本机 IP 访问"]
    notes.append("Ctrl+C 退出")
    shown = "127.0.0.1" if local else args.host
    print(f"KPS Viewer: http://{shown}:{args.port}  ({'；'.join(notes)})", flush=True)
    started = time.monotonic()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    print(f"已退出，运行 {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
