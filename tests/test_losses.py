from __future__ import annotations

import pytest
import torch

from src.losses import bce_loss, dice_loss, focal_loss, seg_loss

LOSSES = ["dice_bce", "bce", "focal", "dice_focal"]


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    target = (torch.rand(2, 1, 16, 16, generator=g) < 0.2).float()
    logits = torch.randn(2, 1, 16, 16, generator=g)
    fov = torch.zeros_like(target)
    fov[..., 2:-2, 2:-2] = 1
    return logits, target, fov


def test_dice_loss_extremes(batch):
    _, target, fov = batch
    assert dice_loss((target * 2 - 1) * 50, target, fov).item() < 1e-2   # confident and correct
    assert dice_loss((1 - target * 2) * 50, target, fov).item() > 0.95  # confident and wrong


def test_dice_loss_stable_without_vessels():
    zeros = torch.zeros(1, 1, 8, 8)
    loss = dice_loss(torch.full_like(zeros, -20.0), zeros, torch.ones_like(zeros))
    assert torch.isfinite(loss) and loss.item() < 1e-3


@pytest.mark.parametrize("name", LOSSES)
def test_losses_ignore_pixels_outside_fov(batch, name):
    logits, target, fov = batch
    moved = torch.where(fov.bool(), logits, logits + 100.0)
    assert seg_loss(name, logits, target, fov).item() == pytest.approx(seg_loss(name, moved, target, fov).item(), rel=1e-5)


@pytest.mark.parametrize("name", LOSSES)
def test_losses_backpropagate(batch, name):
    logits, target, fov = batch
    logits = logits.clone().requires_grad_(True)
    loss = seg_loss(name, logits, target, fov)
    loss.backward()
    assert torch.isfinite(loss) and logits.grad is not None
    assert logits.grad[fov == 0].abs().max().item() == 0.0


def test_combined_losses_are_sums(batch):
    logits, target, fov = batch
    assert seg_loss("dice_bce", logits, target, fov).item() == pytest.approx(
        (dice_loss(logits, target, fov) + bce_loss(logits, target, fov)).item())
    assert seg_loss("dice_focal", logits, target, fov).item() == pytest.approx(
        (dice_loss(logits, target, fov) + focal_loss(logits, target, fov)).item())


def test_focal_downweights_easy_pixels(batch):
    logits, target, fov = batch
    easy = (target * 2 - 1) * 5
    assert focal_loss(easy, target, fov).item() < bce_loss(easy, target, fov).item()


def test_unknown_loss_raises(batch):
    with pytest.raises(ValueError, match="unknown loss"):
        seg_loss("nope", *batch)
