from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def make_sequence(num_frames: int, dim: int, good: bool, seed: int) -> np.ndarray:
    """Create a valid but synthetic skeleton clip for format testing only."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)
    sequence = np.zeros((num_frames, 17, dim), dtype=np.float32)

    for joint in range(17):
        frequency = 1.0 + 0.2 * (joint % 3)
        phase = 0.0 if good else float(rng.uniform(-np.pi, np.pi))
        amplitude = 0.05 if good else 0.14

        sequence[:, joint, 0] = 0.5 + 0.02 * (joint % 5) + amplitude * np.sin(
            2.0 * np.pi * frequency * t + phase
        )
        sequence[:, joint, 1] = 0.4 + 0.01 * joint + amplitude * np.cos(
            2.0 * np.pi * frequency * t + phase
        )
        if dim == 3:
            sequence[:, joint, 2] = 0.03 * np.sin(2.0 * np.pi * t + 0.3 * joint)

    noise = 0.002 if good else 0.01
    sequence += noise * rng.normal(size=sequence.shape).astype(np.float32)
    return sequence


def write_clip(
    root: Path,
    clip_id: str,
    label: int,
    subject_id: str,
    view: str,
    error_type: str | None,
    seed: int,
) -> None:
    clip_dir = root / clip_id
    clip_dir.mkdir(parents=True, exist_ok=True)

    x2d = make_sequence(num_frames=96, dim=2, good=label == 1, seed=seed)
    x3d = make_sequence(num_frames=96, dim=3, good=label == 1, seed=seed + 1000)

    np.savez_compressed(clip_dir / "kp2d.npz", keypoints=x2d, fps=np.array(30, dtype=np.int32))
    np.savez_compressed(clip_dir / "kp3d.npz", keypoints=x3d, fps=np.array(30, dtype=np.int32))

    meta = {
        "clip_id": clip_id,
        "action_id": "squat",
        "label": label,
        "label_name": "good" if label == 1 else "bad",
        "subject_id": subject_id,
        "view": view,
        "fps": 30,
        "num_frames": int(x3d.shape[0]),
        "error_type": error_type,
        "severity": 0 if label == 1 else 1,
        "start_frame": 0,
        "end_frame": int(x3d.shape[0] - 1),
        "has_2d": True,
        "has_3d": True,
        "joint_format": "h36m_17",
    }
    (clip_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (clip_dir / "remark.txt").write_text(
        "mock clip, generated for format testing only\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("../dataset/mock"))
    args = parser.parse_args()

    root = args.out
    root.mkdir(parents=True, exist_ok=True)

    clips = [
        ("train", "clip_good_0001", 1, "s01", "front", None, 1),
        ("train", "clip_good_0002", 1, "s02", "side", None, 2),
        ("train", "clip_bad_0001", 0, "s01", "front", "insufficient_depth", 3),
        ("train", "clip_bad_0002", 0, "s02", "side", "trunk_lean", 4),
        ("val", "clip_good_1001", 1, "s03", "front", None, 5),
        ("val", "clip_bad_1001", 0, "s03", "front", "asymmetry", 6),
        ("test", "clip_good_2001", 1, "s04", "front", None, 7),
        ("test", "clip_bad_2001", 0, "s04", "front", "insufficient_depth", 8),
    ]

    for split, clip_id, label, subject_id, view, error_type, seed in clips:
        write_clip(
            root / split,
            clip_id=clip_id,
            label=label,
            subject_id=subject_id,
            view=view,
            error_type=error_type,
            seed=seed,
        )

    print(f"mock dataset written to: {root.resolve()}")


if __name__ == "__main__":
    main()

