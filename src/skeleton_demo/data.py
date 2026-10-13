from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset


# H36M 17 joints:
# 0 Hip, 1 RHip, 2 RKnee, 3 RFoot, 4 LHip, 5 LKnee, 6 LFoot,
# 7 Spine, 8 Thorax, 9 Neck/Nose, 10 Head,
# 11 LShoulder, 12 LElbow, 13 LWrist,
# 14 RShoulder, 15 RElbow, 16 RWrist
H36M_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]

# Pairs used for left/right mirror augmentation.
LEFT_RIGHT_PAIRS = [(1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16)]


def load_skeleton_table(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)

    if path.suffix == ".npz":
        import numpy as np

        with np.load(path, allow_pickle=True) as data:
            table: dict[str, Any] = {}
            for key in data.files:
                value = data[key]
                if key in {"x2d", "x3d", "y"}:
                    table[key] = torch.as_tensor(value, dtype=torch.float32)
                else:
                    table[key] = value.tolist() if getattr(value, "ndim", 0) > 0 else value.item()
            return table

    raise ValueError(f"Unsupported data format: {path.suffix}. Use .pt or .npz.")


def normalize_skeleton(x: Tensor) -> Tensor:
    """Root-center and scale-normalize a skeleton sequence.

    Args:
        x: [B, T, J, D]
    """
    root = x[:, :, 0:1, :]
    x = x - root
    scale = x.norm(dim=-1).mean(dim=(1, 2), keepdim=True).clamp_min(1e-6).unsqueeze(-1)
    return x / scale


def resample_time(x: Tensor, target_len: int) -> Tensor:
    """Linearly resample [B, T, J, D] along the time dimension."""
    if x.shape[1] == target_len:
        return x
    if x.shape[1] < 2:
        return x.expand(-1, target_len, -1, -1)

    batch, _, joints, dim = x.shape
    # [B, T, J, D] -> [B, J * D, T] -> interpolate -> [B, T, J, D]
    x = x.permute(0, 2, 3, 1).reshape(batch, joints * dim, -1)
    x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    x = x.reshape(batch, joints, dim, target_len)
    return x.permute(0, 3, 1, 2)


def build_features(x: Tensor) -> Tensor:
    """Build coordinate, bone-vector and velocity features.

    Args:
        x: [B, T, J, D]

    Returns:
        [B, T, J, 3 * D]
    """
    parents = torch.tensor(H36M_PARENTS, dtype=torch.long, device=x.device).clamp_min(0)
    parent_coords = x[:, :, parents, :]
    bone = x - parent_coords
    bone[:, :, 0, :] = 0.0

    velocity = torch.zeros_like(x)
    velocity[:, 1:] = x[:, 1:] - x[:, :-1]

    return torch.cat([x, bone, velocity], dim=-1)


def mirror_skeleton(x: Tensor) -> Tensor:
    """Mirror skeleton left/right. x: [B, T, J, D]."""
    x = x.clone()
    x[..., 0] = -x[..., 0]
    for left, right in LEFT_RIGHT_PAIRS:
        x[:, :, [left, right], :] = x[:, :, [right, left], :]
    return x


def rotate_y(x: Tensor, angle: float) -> Tensor:
    """Rotate 3D skeleton around the Y axis. x: [B, T, J, 3]."""
    cos_a = torch.cos(torch.tensor(angle, device=x.device))
    sin_a = torch.sin(torch.tensor(angle, device=x.device))
    rotation = torch.tensor(
        [
            [cos_a, 0.0, sin_a],
            [0.0, 1.0, 0.0],
            [-sin_a, 0.0, cos_a],
        ],
        device=x.device,
        dtype=x.dtype,
    )
    return torch.einsum("b t j d, d e -> b t j e", x, rotation)


class SkeletonDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        target_len: int = 64,
        augment: bool = False,
        modality_dropout: float = 0.0,
    ) -> None:
        table = load_skeleton_table(path)
        self.x2d = self._coerce_sequences(table.get("x2d"))
        self.x3d = self._coerce_sequences(table.get("x3d"))
        self.y = torch.as_tensor(table["y"], dtype=torch.float32).view(-1)
        self.target_len = target_len
        self.augment = augment
        self.modality_dropout = modality_dropout

        if self.x2d is None and self.x3d is None:
            raise ValueError("At least one of x2d or x3d must be provided.")
        for name, sequences in (("x2d", self.x2d), ("x3d", self.x3d)):
            if sequences is not None and len(sequences) != len(self.y):
                raise ValueError(f"{name} length must match y; repack misaligned data.")
        for i in range(len(self.y)):
            if all(s is None or s[i] is None for s in (self.x2d, self.x3d)):
                raise ValueError(f"Sample {i} has neither modality.")

    @staticmethod
    def _coerce_sequences(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            return value.float().unsqueeze(0) if value.ndim == 3 else value.float()
        return [None if item is None else torch.as_tensor(item, dtype=torch.float32) for item in value]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def _prepare_one(self, x: Tensor | None, idx: int) -> Tensor | None:
        if x is None:
            return None

        if torch.is_tensor(x):
            if x.ndim == 4:
                seq = x[idx]
            elif x.ndim == 3:
                seq = x
            else:
                raise ValueError(f"Expected [N, T, J, D] or [T, J, D], got {tuple(x.shape)}")
        else:
            seq = x[idx]

        if seq is None:
            return None

        seq = seq.unsqueeze(0)  # [1, T, J, D]
        seq = normalize_skeleton(seq)
        seq = resample_time(seq, self.target_len)
        return seq[0]

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x2d = self._prepare_one(self.x2d, idx)
        x3d = self._prepare_one(self.x3d, idx)

        if self.augment:
            scale = float(torch.empty(1).uniform_(0.9, 1.1).item())
            if x2d is not None:
                x2d = x2d * scale
            if x3d is not None:
                x3d = x3d * scale

            if torch.rand(1).item() < 0.5:
                if x2d is not None:
                    x2d = mirror_skeleton(x2d.unsqueeze(0))[0]
                if x3d is not None:
                    x3d = mirror_skeleton(x3d.unsqueeze(0))[0]

            if x3d is not None:
                angle = float(torch.empty(1).uniform_(-0.25, 0.25).item())
                x3d = rotate_y(x3d.unsqueeze(0), angle)[0]

            noise = 0.01
            if x2d is not None:
                x2d = x2d + noise * torch.randn_like(x2d)
            if x3d is not None:
                x3d = x3d + noise * torch.randn_like(x3d)

        has2d = x2d is not None
        has3d = x3d is not None

        if self.augment and self.modality_dropout > 0.0 and has2d and has3d:
            if torch.rand(1).item() < self.modality_dropout:
                if torch.rand(1).item() < 0.5:
                    has2d = False
                else:
                    has3d = False

        f2d = build_features(x2d.unsqueeze(0))[0] if x2d is not None else torch.zeros(
            self.target_len, 17, 6
        )
        f3d = build_features(x3d.unsqueeze(0))[0] if x3d is not None else torch.zeros(
            self.target_len, 17, 9
        )

        return {
            "x2d": f2d,
            "x3d": f3d,
            "has2d": torch.tensor(float(has2d)),
            "has3d": torch.tensor(float(has3d)),
            "y": self.y[idx],
        }


def collate_fn(batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
    output: dict[str, Tensor] = {}
    for key in batch[0]:
        values = [item[key] for item in batch]
        output[key] = torch.stack(values, dim=0)
    return output
