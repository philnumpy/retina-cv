"""Vessel-segmentation data: preprocessing, random-patch training set, sliding-window inference."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.common import ROOT

PROC = ROOT / "data" / "processed"


@dataclass
class Sample:
    """One fundus image, already preprocessed and held in RAM."""
    name: str
    image: np.ndarray   # float32 (C, H, W), normalised network input
    mask: np.ndarray    # uint8 (H, W) in {0, 1} at working resolution
    fov: np.ndarray     # bool (H, W) field-of-view mask at working resolution
    rgb: np.ndarray     # uint8 (H, W, 3) for visualisation
    mask_path: Path     # native-resolution mask (evaluation is done at native resolution)
    fov_path: Path


def clahe(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)


def preprocess(rgb: np.ndarray, fov: np.ndarray, mode: str) -> np.ndarray:
    """Return a (C, H, W) float32 input.

    green: CLAHE on the green channel (highest vessel contrast) -> 1 channel.
    rgb:   CLAHE on the lightness channel of LAB, colour kept    -> 3 channels.
    Both are z-scored per image using FOV pixels only, and zeroed outside the FOV.
    """
    if mode == "green":
        x = clahe(rgb[..., 1])[..., None].astype(np.float32)
    elif mode == "rgb":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe(lab[..., 0])
        x = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float32)
    else:
        raise ValueError(f"unknown input mode: {mode}")
    inside = x[fov]
    x = (x - inside.mean(axis=0)) / (inside.std(axis=0) + 1e-6)
    x[~fov] = 0.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def read_mask(path: Path) -> np.ndarray:
    return (cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)


def list_split(dataset: str, split: str) -> list[str]:
    """Image stems for 'train' / 'val' / 'test'. train/val come from the saved split file."""
    if split == "test":
        return sorted(p.stem for p in (PROC / dataset / "test" / "images").glob("*.png"))
    lines = (ROOT / "data" / "splits" / f"{dataset}_{split}.txt").read_text().split()
    return sorted(lines)


def load_samples(dataset: str, split: str, mode: str, work_size: int | None, limit: int | None = None) -> list[Sample]:
    """Load and preprocess a split. `work_size` (FIVES) resizes the 2048 px images for the network."""
    folder = PROC / dataset / ("test" if split == "test" else "train")
    samples = []
    for name in list_split(dataset, split)[:limit]:
        rgb = cv2.cvtColor(cv2.imread(str(folder / "images" / f"{name}.png")), cv2.COLOR_BGR2RGB)
        mask_path, fov_path = folder / "masks" / f"{name}.png", folder / "fov" / f"{name}.png"
        mask, fov = read_mask(mask_path), read_mask(fov_path)
        if work_size and rgb.shape[0] != work_size:
            rgb = cv2.resize(rgb, (work_size, work_size), interpolation=cv2.INTER_AREA)
            mask = (cv2.resize(mask * 255, (work_size, work_size), interpolation=cv2.INTER_AREA) > 127).astype(np.uint8)
            fov = cv2.resize(fov, (work_size, work_size), interpolation=cv2.INTER_NEAREST)
        fov = fov.astype(bool)
        samples.append(Sample(name, preprocess(rgb, fov, mode), mask, fov, rgb, mask_path, fov_path))
    return samples


class PatchDataset(Dataset):
    """Random training patches, sampled on the fly.

    Every item draws a random image and a random patch whose centre lies inside the FOV,
    then applies flips, 90-degree rotations and a mild intensity jitter. One "epoch" is
    simply `patches_per_epoch` draws.
    """

    def __init__(self, samples: list[Sample], patch: int, patches_per_epoch: int, augment: bool = True) -> None:
        self.samples, self.patch, self.n, self.augment = samples, patch, patches_per_epoch, augment

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[np.random.randint(len(self.samples))]
        h, w = s.mask.shape
        p = self.patch
        for _try in range(20):
            y, x = np.random.randint(0, h - p + 1), np.random.randint(0, w - p + 1)
            if s.fov[y + p // 2, x + p // 2]:
                break
        img = s.image[:, y:y + p, x:x + p]
        msk = s.mask[y:y + p, x:x + p]
        fov = s.fov[y:y + p, x:x + p]
        if self.augment:
            k = np.random.randint(4)
            img, msk, fov = np.rot90(img, k, (1, 2)), np.rot90(msk, k), np.rot90(fov, k)
            if np.random.rand() < 0.5:
                img, msk, fov = img[:, :, ::-1], msk[:, ::-1], fov[:, ::-1]
            # contrast / brightness jitter in z-score space
            img = img * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.2, 0.2)
        return (torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(msk, dtype=np.float32))[None],
                torch.from_numpy(np.ascontiguousarray(fov, dtype=np.float32))[None])


def _positions(size: int, patch: int, stride: int) -> list[int]:
    pos = list(range(0, max(size - patch, 0) + 1, stride))
    if pos[-1] != max(size - patch, 0):
        pos.append(size - patch)  # last window is flush with the border
    return pos


@torch.no_grad()
def sliding_window_predict(model: torch.nn.Module, image: np.ndarray, patch: int, stride: int, device: torch.device,
                           batch_size: int = 64, amp: bool = True) -> np.ndarray:
    """Full-image vessel probability: overlapping windows, probabilities averaged where they overlap."""
    model.eval()
    _, h, w = image.shape
    ph, pw = max(patch - h, 0), max(patch - w, 0)
    x = torch.from_numpy(np.pad(image, ((0, 0), (0, ph), (0, pw)), mode="reflect")).to(device)
    H, W = x.shape[1:]
    coords = [(y, xx) for y in _positions(H, patch, stride) for xx in _positions(W, patch, stride)]
    prob_sum = torch.zeros((H, W), device=device)
    count = torch.zeros((H, W), device=device)
    for i in range(0, len(coords), batch_size):
        chunk = coords[i:i + batch_size]
        batch = torch.stack([x[:, y:y + patch, xx:xx + patch] for y, xx in chunk])
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            prob = torch.sigmoid(model(batch).float())[:, 0]
        for (y, xx), p in zip(chunk, prob):
            prob_sum[y:y + patch, xx:xx + patch] += p
            count[y:y + patch, xx:xx + patch] += 1
    return (prob_sum / count)[:h, :w].cpu().numpy()


def frame_shape(h: int, w: int, multiple: int = 32) -> tuple[int, int]:
    """Smallest size >= (h, w) that the encoder-decoder can process (divisible by 32)."""
    return (-(-h // multiple) * multiple, -(-w // multiple) * multiple)


@torch.no_grad()
def full_frame_predict(model: torch.nn.Module, image: np.ndarray, device: torch.device) -> np.ndarray:
    """Single forward pass over the whole (reflect-padded) image: the fast deployment path.

    Runs in FP32: half precision overflowed to NaN on whole frames for one ResNet34 checkpoint.
    """
    model.eval()
    _, h, w = image.shape
    H, W = frame_shape(h, w)
    x = torch.from_numpy(np.pad(image, ((0, 0), (0, H - h), (0, W - w)), mode="reflect"))[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0]
    return prob[:h, :w].cpu().numpy()


def to_native(prob: np.ndarray, native_shape: tuple[int, int]) -> np.ndarray:
    """Upsample a working-resolution probability map back to the native image size."""
    if prob.shape == native_shape:
        return prob
    return cv2.resize(prob, (native_shape[1], native_shape[0]), interpolation=cv2.INTER_LINEAR)


def describe(samples: list[Sample]) -> dict[str, Any]:
    return {"n": len(samples), "shape": list(samples[0].image.shape),
            "vessel_fraction_in_fov": float(np.mean([s.mask[s.fov].mean() for s in samples]))}
