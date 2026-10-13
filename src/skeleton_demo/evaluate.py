from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from .data import SkeletonDataset, collate_fn
from .metrics import binary_metrics
from .model import BinarySkeletonModel


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/squat.yaml"))
    parser.add_argument("--ckpt", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SkeletonDataset(
        cfg["data"]["val"],
        target_len=int(cfg["data"].get("target_len", 64)),
        augment=False,
        modality_dropout=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    model = BinarySkeletonModel(**cfg["model"]).to(device)
    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        out = model(
            batch["x2d"].to(device),
            batch["x3d"].to(device),
            batch["has2d"].to(device),
            batch["has3d"].to(device),
        )
        logits.append(out.detach().cpu())
        labels.append(batch["y"])

    metrics = binary_metrics(torch.cat(labels), torch.sigmoid(torch.cat(logits)))
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()

