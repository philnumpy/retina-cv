from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.cls_data import MEAN, STD, ben_graham, build_transforms, crop_fundus, denormalise, pad_to_square


def bordered(h: int = 60, w: int = 90, top: int = 5, left: int = 12) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    img[top:h - top, left:w - left] = 150
    return img


def test_crop_fundus_removes_black_border():
    out = crop_fundus(bordered())
    assert out.shape == (50, 66, 3) and out.min() == 150


def test_crop_fundus_leaves_dark_image_untouched():
    dark = np.full((40, 40, 3), 3, np.uint8)
    assert crop_fundus(dark) is dark


@pytest.mark.parametrize("h,w", [(30, 50), (50, 30), (40, 40)])
def test_pad_to_square_centres_image(h, w):
    img = np.full((h, w, 3), 200, np.uint8)
    out = pad_to_square(img)
    s = max(h, w)
    assert out.shape == (s, s, 3)
    assert (out == 200).all(axis=2).sum() == h * w
    assert out[s // 2, s // 2, 0] == 200


def test_ben_graham_outside_circle_is_grey():
    rng = np.random.default_rng(0)
    out = ben_graham(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    assert out.shape == (64, 64, 3)
    assert (out[0, 0] == 128).all() and (out[-1, -1] == 128).all()


def test_ben_graham_flattens_global_colour_cast():
    flat = np.full((64, 64, 3), (180, 90, 40), np.uint8)
    out = ben_graham(flat)
    assert np.abs(out[32, 32].astype(int) - 128).max() <= 1  # a uniform image has no local detail left


@pytest.mark.parametrize("train", [True, False])
def test_transforms_output_size(train):
    img = np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    out = build_transforms(48, train)(image=img)["image"]
    assert out.shape == (48, 48, 3) and out.dtype == np.float32


def test_denormalise_inverts_eval_transform():
    img = np.random.default_rng(0).integers(0, 256, (32, 32, 3), dtype=np.uint8)
    x = torch.from_numpy(build_transforms(32, False)(image=img)["image"].transpose(2, 0, 1))
    np.testing.assert_allclose(denormalise(x), img / 255.0, atol=1e-5)
    assert len(MEAN) == len(STD) == 3
