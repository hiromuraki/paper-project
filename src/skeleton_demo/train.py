from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .data import SkeletonDataset, collate_fn
from .metrics import binary_metrics
from .model import BinarySkeletonModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    for batch in loader:
        x2d = batch["x2d"].to(device)
        x3d = batch["x3d"].to(device)
        has2d = batch["has2d"].to(device)
        has3d = batch["has3d"].to(device)
        y = batch["y"].to(device)

        logits = model(x2d, x3d, has2d, has3d)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    probs = torch.sigmoid(logits)
    return binary_metrics(labels, probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/squat.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["train"].get("seed", 42)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = SkeletonDataset(
        cfg["data"]["train"],
        target_len=int(cfg["data"].get("target_len", 64)),
        augment=bool(cfg["data"].get("augment", True)),
        modality_dropout=float(cfg["data"].get("modality_dropout", 0.0)),
    )
    val_ds = SkeletonDataset(
        cfg["data"]["val"],
        target_len=int(cfg["data"].get("target_len", 64)),
        augment=False,
        modality_dropout=0.0,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"].get("batch_size", 32)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_fn,
    )

    model = BinarySkeletonModel(**cfg["model"]).to(device)
    num_pos = float(train_ds.y.sum().item())
    num_neg = float(len(train_ds) - num_pos)
    pos_weight = torch.tensor([num_neg / max(num_pos, 1.0)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"].get("epochs", 50)),
    )

    output_dir = Path(cfg["train"].get("output_dir", "runs/squat"))
    output_dir.mkdir(parents=True, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, int(cfg["train"].get("epochs", 50)) + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            x2d = batch["x2d"].to(device)
            x3d = batch["x3d"].to(device)
            has2d = batch["has2d"].to(device)
            has3d = batch["has3d"].to(device)
            y = batch["y"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x2d, x3d, has2d, has3d)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            running_loss += float(loss.item()) * y.shape[0]

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device)
        train_loss = running_loss / max(len(train_ds), 1)
        print(
            f"epoch={epoch:03d} "
            f"loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_auroc={val_metrics['auroc']:.4f}"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "metrics": val_metrics,
                },
                output_dir / "best.pt",
            )

    print(f"best val f1={best_f1:.4f}")
    print(f"checkpoint: {output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()

