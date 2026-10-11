"""Segmentation losses. All take logits and are computed only on pixels inside the FOV."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """Soft Dice over the whole batch (stable when a patch contains no vessel)."""
    p = torch.sigmoid(logits) * fov
    t = target * fov
    return 1 - (2 * (p * t).sum() + eps) / (p.sum() + t.sum() + eps)


def bce_loss(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (loss * fov).sum() / fov.sum().clamp(min=1)


def focal_loss(logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor, gamma: float = 2.0,
               alpha: float = 0.75) -> torch.Tensor:
    """Focal loss: down-weights easy pixels by (1 - p_t)^gamma; alpha up-weights the vessel class."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = torch.exp(-bce)
    a_t = alpha * target + (1 - alpha) * (1 - target)
    return (a_t * (1 - p_t) ** gamma * bce * fov).sum() / fov.sum().clamp(min=1)


def seg_loss(name: str, logits: torch.Tensor, target: torch.Tensor, fov: torch.Tensor) -> torch.Tensor:
    logits = logits.float()
    if name == "dice_bce":
        return dice_loss(logits, target, fov) + bce_loss(logits, target, fov)
    if name == "bce":
        return bce_loss(logits, target, fov)
    if name == "focal":
        return focal_loss(logits, target, fov)
    if name == "dice_focal":
        return dice_loss(logits, target, fov) + focal_loss(logits, target, fov)
    raise ValueError(f"unknown loss: {name}")
