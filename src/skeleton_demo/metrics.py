from __future__ import annotations

import torch
from torch import Tensor


def binary_metrics(y_true: Tensor, y_prob: Tensor, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.float().view(-1)
    y_prob = y_prob.float().view(-1)
    pred = (y_prob >= threshold).float()

    tp = ((pred == 1) & (y_true == 1)).sum().item()
    tn = ((pred == 0) & (y_true == 0)).sum().item()
    fp = ((pred == 1) & (y_true == 0)).sum().item()
    fn = ((pred == 0) & (y_true == 1)).sum().item()

    eps = 1e-9
    accuracy = (tp + tn) / max(tp + tn + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    f1 = 2.0 * precision * recall / max(precision + recall, eps)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": binary_auroc(y_true, y_prob),
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def binary_auroc(y_true: Tensor, y_score: Tensor) -> float:
    y_true = y_true.float().view(-1)
    y_score = y_score.float().view(-1)

    pos = y_score[y_true == 1].view(-1, 1)
    neg = y_score[y_true == 0].view(1, -1)
    if pos.numel() == 0 or neg.numel() == 0:
        return float("nan")

    greater = (pos > neg).float()
    equal = (pos == neg).float()
    return float((greater + 0.5 * equal).mean().item())

