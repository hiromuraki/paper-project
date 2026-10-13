from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch


def make_sequence(num_frames: int, dim: int, positive: bool) -> torch.Tensor:
    """Generate a smooth skeleton-like sequence with a binary label.

    This is only for smoke-testing the training pipeline. It is not a real
    motion model and is not expected to generalize to real skeleton data.
    """
    t = torch.linspace(0.0, 1.0, num_frames)
    joints = []

    for j in range(17):
        freq = 1.0 + 0.35 * (j % 4)
        phase = 0.0 if positive else float(torch.empty(1).uniform_(-math.pi, math.pi))
        amp = 0.45 + 0.08 * (j % 5) if positive else float(torch.empty(1).uniform_(0.15, 1.4))

        wave = amp * torch.sin(2.0 * math.pi * freq * t + phase)
        wave = wave + 0.15 * torch.cos(4.0 * math.pi * t + 0.5 * j)

        joint = torch.stack([wave, 0.35 * wave, 0.25 * wave][:dim], dim=-1)
        if dim == 2:
            joint = torch.stack([wave, 0.35 * wave], dim=-1)
        else:
            joint = torch.stack([wave, 0.35 * wave, 0.25 * wave], dim=-1)

        joints.append(joint)

    seq = torch.stack(joints, dim=1)
    noise_scale = 0.01 if positive else 0.04
    seq = seq + noise_scale * torch.randn_like(seq)
    return seq


def make_split(
    num_samples: int,
    target_dim: int,
    positive_ratio: float,
    seed: int,
) -> dict:
    generator = torch.Generator().manual_seed(seed)
    x_list = []
    y_list = []
    subjects = []
    views = []
    error_types = []

    for i in range(num_samples):
        positive = bool(torch.rand(1, generator=generator).item() < positive_ratio)
        num_frames = int(torch.randint(48, 120, (1,), generator=generator).item())
        x_list.append(make_sequence(num_frames, target_dim, positive))
        y_list.append(1.0 if positive else 0.0)
        subjects.append(i % max(4, num_samples // 8))
        views.append(i % 3)
        error_types.append("good" if positive else f"error_{i % 6}")

    return {
        "x2d": None if target_dim != 2 else x_list,
        "x3d": None if target_dim != 3 else x_list,
        "y": torch.tensor(y_list, dtype=torch.float32),
        "subject": torch.tensor(subjects, dtype=torch.long),
        "view": torch.tensor(views, dtype=torch.long),
        "error_type": error_types,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--dim", type=int, choices=[2, 3], default=3)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--positive-ratio", type=float, default=0.5)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    train = make_split(args.train_size, args.dim, args.positive_ratio, seed=42)
    val = make_split(args.val_size, args.dim, args.positive_ratio, seed=123)

    torch.save(train, args.out / "squat_train.pt")
    torch.save(val, args.out / "squat_val.pt")
    print(f"saved: {args.out / 'squat_train.pt'}")
    print(f"saved: {args.out / 'squat_val.pt'}")


if __name__ == "__main__":
    main()
