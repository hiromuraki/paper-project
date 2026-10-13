from __future__ import annotations

import torch
from torch import Tensor, nn


class SkeletonEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_joints: int = 17,
    ) -> None:
        super().__init__()
        layer = lambda: nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.proj = nn.Linear(in_dim, d_model)
        self.joint_embed = nn.Parameter(torch.zeros(1, 1, num_joints, d_model))
        self.spatial = nn.TransformerEncoder(layer(), num_layers=1)
        self.temporal = nn.TransformerEncoder(layer(), num_layers=1)
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, J, in_dim]
        batch, time, joints, _ = x.shape
        hidden = self.proj(x) + self.joint_embed

        for _ in range(self.num_layers):
            hidden = hidden.reshape(batch * time, joints, -1)
            hidden = self.spatial(hidden)
            hidden = hidden.reshape(batch, time, joints, -1)

            hidden = hidden.permute(0, 2, 1, 3).reshape(batch * joints, time, -1)
            hidden = self.temporal(hidden)
            hidden = hidden.reshape(batch, joints, time, -1).permute(0, 2, 1, 3)

        return self.norm(hidden)


def pool_sequence(hidden: Tensor) -> Tensor:
    """Pool [B, T, J, D] to [B, 2D]."""
    mean = hidden.mean(dim=(1, 2))
    max_value = hidden.amax(dim=(1, 2))
    return torch.cat([mean, max_value], dim=-1)


class BinarySkeletonModel(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        num_joints: int = 17,
    ) -> None:
        super().__init__()
        self.encoder_2d = SkeletonEncoder(
            in_dim=2 * 3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            num_joints=num_joints,
        )
        self.encoder_3d = SkeletonEncoder(
            in_dim=3 * 3,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            num_joints=num_joints,
        )
        self.gate = nn.Sequential(
            nn.Linear(4 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x2d: Tensor, x3d: Tensor, has2d: Tensor, has3d: Tensor) -> Tensor:
        # x2d: [B, T, J, 6], x3d: [B, T, J, 9]
        hidden_2d = self.encoder_2d(x2d)
        hidden_3d = self.encoder_3d(x3d)

        pooled_2d = pool_sequence(hidden_2d)
        pooled_3d = pool_sequence(hidden_3d)

        gate = torch.sigmoid(self.gate(torch.cat([pooled_2d, pooled_3d], dim=-1)))
        has2d = has2d.view(-1, 1)
        has3d = has3d.view(-1, 1)
        both = has2d * has3d
        gate = torch.where(both > 0.0, gate, has2d)

        fused = gate * pooled_2d + (1.0 - gate) * pooled_3d
        logits = self.head(fused).squeeze(-1)
        return logits

