from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.seg_data import (PatchDataset, Sample, _positions, frame_shape, full_frame_predict, preprocess,
                               sliding_window_predict, to_native)


def fundus(h: int = 80, w: int = 72, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Random RGB image with a circular field of view."""
    rgb = np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    fov = (yy - h / 2) ** 2 + (xx - w / 2) ** 2 < (min(h, w) / 2.2) ** 2
    return rgb, fov


def pointwise_model() -> torch.nn.Module:
    """A 1x1 conv: its output at a pixel depends on that pixel only, so tiling must not change the result."""
    conv = torch.nn.Conv2d(1, 1, 1)
    with torch.no_grad():
        conv.weight.fill_(1.5)
        conv.bias.fill_(-0.3)
    return conv


@pytest.mark.parametrize("mode,channels", [("green", 1), ("rgb", 3)])
def test_preprocess_normalises_inside_fov(mode, channels):
    rgb, fov = fundus()
    x = preprocess(rgb, fov, mode)
    assert x.shape == (channels, *fov.shape) and x.dtype == np.float32
    assert np.all(x[:, ~fov] == 0)
    np.testing.assert_allclose(x[:, fov].mean(axis=1), 0, atol=1e-4)
    np.testing.assert_allclose(x[:, fov].std(axis=1), 1, atol=1e-3)


def test_preprocess_unknown_mode():
    with pytest.raises(ValueError, match="unknown input mode"):
        preprocess(*fundus(), "hsv")


@pytest.mark.parametrize("size,patch,stride", [(100, 32, 16), (64, 32, 32), (32, 32, 16), (70, 64, 32)])
def test_positions_cover_whole_axis(size, patch, stride):
    pos = _positions(size, patch, stride)
    assert pos[0] == 0 and pos[-1] == size - patch
    covered = np.zeros(size, bool)
    for p in pos:
        covered[p:p + patch] = True
    assert covered.all()


def test_frame_shape():
    assert frame_shape(584, 565) == (608, 576)
    assert frame_shape(64, 64) == (64, 64)


@pytest.mark.parametrize("shape", [(1, 80, 72), (1, 20, 24)])  # second case is smaller than the patch -> padded
def test_sliding_window_matches_pointwise_model(shape):
    image = np.random.default_rng(0).normal(size=shape).astype(np.float32)
    model = pointwise_model()
    prob = sliding_window_predict(model, image, patch=32, stride=16, device=torch.device("cpu"), batch_size=5)
    expected = torch.sigmoid(1.5 * torch.from_numpy(image[0]) - 0.3).numpy()
    assert prob.shape == shape[1:]
    np.testing.assert_allclose(prob, expected, atol=1e-6)


def test_full_frame_matches_sliding_window():
    image = np.random.default_rng(1).normal(size=(1, 50, 45)).astype(np.float32)
    model = pointwise_model()
    full = full_frame_predict(model, image, torch.device("cpu"))
    slid = sliding_window_predict(model, image, 32, 16, torch.device("cpu"))
    assert full.shape == (50, 45)
    np.testing.assert_allclose(full, slid, atol=1e-6)


def test_to_native():
    prob = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert to_native(prob, (32, 32)) is prob
    up = to_native(prob, (64, 48))
    assert up.shape == (64, 48) and up.min() >= 0 and up.max() <= 1


def make_sample(name: str = "a", h: int = 80, w: int = 72) -> Sample:
    rgb, fov = fundus(h, w)
    mask = (np.random.default_rng(1).random((h, w)) < 0.15).astype(np.uint8) * fov
    return Sample(name, preprocess(rgb, fov, "green"), mask, fov, rgb, None, None)


def test_patch_dataset_shapes_and_centre_inside_fov():
    np.random.seed(0)
    s = make_sample()
    ds = PatchDataset([s], patch=16, patches_per_epoch=50, augment=False)
    assert len(ds) == 50
    for i in range(len(ds)):
        img, msk, fov = ds[i]
        assert img.shape == (1, 16, 16) and msk.shape == fov.shape == (1, 16, 16)
        assert img.dtype == msk.dtype == fov.dtype == torch.float32
        assert fov[0, 8, 8] == 1  # the sampler keeps patch centres inside the FOV


def test_patch_dataset_augmentation_keeps_alignment():
    """Image, mask and FOV must be flipped / rotated together."""
    np.random.seed(0)
    s = make_sample()
    s.image = np.stack([s.mask.astype(np.float32)])  # image == mask, so any misalignment shows up
    ds = PatchDataset([s], patch=16, patches_per_epoch=30, augment=True)
    for i in range(len(ds)):
        img, msk, _ = ds[i]
        # the intensity jitter is affine (a * x + b), so pixels with mask 1 are all larger than those with mask 0
        on, off = img[msk == 1], img[msk == 0]
        if len(on) and len(off):
            assert on.min() > off.max()
