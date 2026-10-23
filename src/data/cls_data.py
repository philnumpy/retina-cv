"""APTOS 2019 data: fundus cropping, Ben Graham normalisation, dataset and augmentations."""
from __future__ import annotations


import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.common import ROOT

APTOS = ROOT / "data" / "processed" / "aptos"
SPLIT_FILE = ROOT / "data" / "splits" / "aptos_split.csv"
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # ImageNet statistics (pretrained backbones)


def crop_fundus(rgb: np.ndarray, tol: int = 7) -> np.ndarray:
    """Crop away the black border around the retina (bounding box of pixels brighter than `tol`)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    rows, cols = np.where((gray > tol).any(axis=1))[0], np.where((gray > tol).any(axis=0))[0]
    if len(rows) < 10 or len(cols) < 10:  # almost entirely dark image: leave untouched
        return rgb
    return rgb[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(rgb: np.ndarray) -> np.ndarray:
    """Zero-pad the shorter side so the retina keeps its aspect ratio when resized."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    out = np.zeros((s, s, 3), dtype=rgb.dtype)
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = rgb
    return out


def square_fundus(rgb: np.ndarray, size: int = 512) -> np.ndarray:
    """Crop the black border, pad to a square and resize: the 'plain' cache that prepare.py writes."""
    return cv2.resize(pad_to_square(crop_fundus(rgb)), (size, size), interpolation=cv2.INTER_AREA)


def ben_graham(rgb: np.ndarray) -> np.ndarray:
    """Ben Graham's local-colour normalisation (Kaggle DR 2015 winner).

    Subtract a heavily blurred copy (the local average colour), amplify what remains and
    re-centre on grey: out = 4*img - 4*blur + 128. This removes lighting / camera colour
    differences and makes small lesions stand out. A circular mask hides the boundary halo.
    """
    size = rgb.shape[0]
    blur = cv2.GaussianBlur(rgb, (0, 0), sigmaX=size / 30)
    out = cv2.addWeighted(rgb, 4, blur, -4, 128)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), int(size * 0.48), 1, -1)
    return out * mask[..., None] + 128 * (1 - mask[..., None])


def build_transforms(size: int, train: bool) -> A.Compose:
    """Geometric + mild photometric augmentation. No hue/saturation shifts: lesion colour is diagnostic."""
    if train:
        return A.Compose([
            A.RandomResizedCrop(size=(size, size), scale=(0.8, 1.0), ratio=(0.95, 1.05)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.8),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.Normalize(MEAN, STD),
        ])
    return A.Compose([A.Resize(size, size), A.Normalize(MEAN, STD)])


def load_split() -> pd.DataFrame:
    return pd.read_csv(SPLIT_FILE)


def load_image(image_id: str, variant: str) -> np.ndarray:
    """Cached, cropped 512x512 RGB image. variant: 'plain' or 'ben'."""
    return cv2.cvtColor(cv2.imread(str(APTOS / variant / f"{image_id}.jpg")), cv2.COLOR_BGR2RGB)


class AptosDataset(Dataset):
    def __init__(self, df: pd.DataFrame, variant: str, size: int, train: bool) -> None:
        self.ids, self.labels = df["id_code"].tolist(), df["diagnosis"].to_numpy()
        self.variant, self.tf = variant, build_transforms(size, train)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img = self.tf(image=load_image(self.ids[i], self.variant))["image"]
        return torch.from_numpy(img.transpose(2, 0, 1)), int(self.labels[i])


def denormalise(x: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor -> float RGB image in [0, 1] for plotting."""
    img = x.detach().cpu().numpy().transpose(1, 2, 0) * np.array(STD) + np.array(MEAN)
    return np.clip(img, 0, 1)
