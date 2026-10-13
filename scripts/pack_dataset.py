from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_keypoints(path: Path) -> torch.Tensor | None:
    if not path.exists():
        return None
    if path.stat().st_size == 0:
        return None
    with np.load(path, allow_pickle=True) as data:
        if "keypoints" in data:
            array = data["keypoints"]
        elif "kp" in data:
            array = data["kp"]
        elif "x" in data:
            array = data["x"]
        else:
            raise ValueError(f"{path} must contain one of: keypoints, kp, x")
    return torch.as_tensor(array, dtype=torch.float32)


def load_meta(clip_dir: Path) -> dict:
    meta_path = clip_dir / "meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))

    name = clip_dir.name.lower()
    if "good" in name:
        label = 1
    elif "bad" in name or "error" in name:
        label = 0
    else:
        raise ValueError(f"{clip_dir} has no meta.json and label cannot be inferred from name")

    return {
        "clip_id": clip_dir.name,
        "action_id": "unknown",
        "label": label,
        "subject_id": "unknown",
        "view": "unknown",
        "error_type": None,
    }


def collect_split(split_dir: Path) -> dict:
    x2d: list[torch.Tensor] = []
    x3d: list[torch.Tensor] = []
    y: list[float] = []
    clip_id: list[str] = []
    subject: list[str] = []
    view: list[str] = []
    error_type: list[str] = []
    has2d: list[bool] = []
    has3d: list[bool] = []

    if not split_dir.exists():
        raise FileNotFoundError(f"Missing split directory: {split_dir}")

    clip_dirs = sorted([path for path in split_dir.iterdir() if path.is_dir()])
    if not clip_dirs:
        raise ValueError(f"No clip folders found in {split_dir}")

    for clip_dir in clip_dirs:
        meta = load_meta(clip_dir)
        kp2d = load_keypoints(clip_dir / "kp2d.npz")
        kp3d = load_keypoints(clip_dir / "kp3d.npz")
        if kp2d is None and kp3d is None:
            raise ValueError(f"{clip_dir} has neither kp2d.npz nor kp3d.npz")

        if kp2d is not None:
            x2d.append(kp2d)
        if kp3d is not None:
            x3d.append(kp3d)

        y.append(float(meta["label"]))
        clip_id.append(str(meta.get("clip_id", clip_dir.name)))
        subject.append(str(meta.get("subject_id", "unknown")))
        view.append(str(meta.get("view", "unknown")))
        error_type.append(str(meta.get("error_type", "")))
        has2d.append(kp2d is not None)
        has3d.append(kp3d is not None)

    return {
        "x2d": x2d if x2d else None,
        "x3d": x3d if x3d else None,
        "y": torch.tensor(y, dtype=torch.float32),
        "clip_id": clip_id,
        "subject": subject,
        "view": view,
        "error_type": error_type,
        "has2d": has2d,
        "has3d": has3d,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("../dataset/mock"))
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--action", type=str, default="squat")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        payload = collect_split(args.dataset / split)
        path = args.out / f"{args.action}_{split}.pt"
        torch.save(payload, path)
        print(f"saved {path} ({len(payload['y'])} clips)")


if __name__ == "__main__":
    main()

