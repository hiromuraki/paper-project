from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .data import build_features, load_skeleton_table, normalize_skeleton, resample_time
from .model import BinarySkeletonModel


def first_sequence(value) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0]
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim == 4:
        tensor = tensor[0]
    if tensor.ndim != 3:
        raise ValueError(f"Expected [T, 17, D] or [N, T, 17, D], got {tuple(tensor.shape)}")
    return tensor


def prepare(x: torch.Tensor | None, target_len: int, dim: int) -> torch.Tensor:
    if x is None:
        return torch.zeros(target_len, 17, dim * 3)
    x = normalize_skeleton(x.unsqueeze(0))
    x = resample_time(x, target_len)
    return build_features(x)[0]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = BinarySkeletonModel(**cfg["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    table = load_skeleton_table(args.input)
    x2d = first_sequence(table.get("x2d"))
    x3d = first_sequence(table.get("x3d"))

    if x2d is None and x3d is None:
        raise ValueError("Input must contain x2d or x3d.")

    target_len = int(cfg["data"].get("target_len", 64))
    f2d = prepare(x2d, target_len, 2).unsqueeze(0)
    f3d = prepare(x3d, target_len, 3).unsqueeze(0)

    logits = model(
        f2d,
        f3d,
        torch.tensor([float(x2d is not None)]),
        torch.tensor([float(x3d is not None)]),
    )
    prob = float(torch.sigmoid(logits).item())
    decision = prob >= args.threshold
    print(f"prob: {prob:.4f}")
    print(f"decision: {decision}")


if __name__ == "__main__":
    main()
