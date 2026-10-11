"""Verify the raw downloads and build the processed datasets + split files.

    python -m src.data.prepare --dataset drive|fives|aptos|all

Writes data/processed/<dataset>/..., data/splits/* and results/data_summary_<dataset>.json.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from src.data.cls_data import ben_graham, crop_fundus, pad_to_square
from src.utils.common import ROOT, save_json

RAW, PROC, SPLITS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "splits"
SPLIT_SEED = 42


def _write_split(dataset: str, names: list[str], n_val: int) -> dict[str, int]:
    """Hold out `n_val` training images for validation (fixed seed) and save both lists."""
    rng = np.random.default_rng(SPLIT_SEED)
    names = sorted(names)
    val = sorted(rng.choice(names, size=n_val, replace=False).tolist())
    train = [n for n in names if n not in set(val)]
    SPLITS.mkdir(parents=True, exist_ok=True)
    (SPLITS / f"{dataset}_train.txt").write_text("\n".join(train) + "\n")
    (SPLITS / f"{dataset}_val.txt").write_text("\n".join(val) + "\n")
    return {"train": len(train), "val": len(val)}


# ----------------------------------------------------------------------------- DRIVE
def prepare_drive() -> None:
    """DRIVE: 20 training + 20 test images (565x584) with manual vessel masks and FOV masks."""
    src = next(RAW.glob("drive/**/training")).parent
    summary: dict[str, object] = {"dataset": "DRIVE", "source": str(src.relative_to(ROOT))}
    for split_src, split_dst in (("training", "train"), ("test", "test")):
        out = PROC / "drive" / split_dst
        for sub in ("images", "masks", "fov", "masks_observer2"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        shapes, fractions = Counter(), []
        images = sorted((src / split_src / "images").glob("*.tif"))
        for path in images:
            idx = path.stem.split("_")[0]
            img = np.array(Image.open(path).convert("RGB"))
            mask = np.array(Image.open(src / split_src / "1st_manual" / f"{idx}_manual1.gif").convert("L"))
            fov = np.array(Image.open(src / split_src / "mask" / f"{idx}_{split_src}_mask.gif").convert("L"))
            assert img.shape[:2] == mask.shape == fov.shape, f"shape mismatch for {path.name}"
            assert set(np.unique(mask)) <= {0, 255}, "vessel mask is not binary"
            cv2.imwrite(str(out / "images" / f"{idx}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out / "masks" / f"{idx}.png"), mask)
            cv2.imwrite(str(out / "fov" / f"{idx}.png"), ((fov > 127) * 255).astype(np.uint8))
            second = src / split_src / "2nd_manual" / f"{idx}_manual2.gif"
            if second.exists():  # second human observer, test set only: used as a human reference
                cv2.imwrite(str(out / "masks_observer2" / f"{idx}.png"), np.array(Image.open(second).convert("L")))
            shapes[img.shape[:2]] += 1
            fractions.append(float((mask > 127)[fov > 127].mean()))
        summary[split_dst] = {"images": len(images), "masks": len(list((out / "masks").glob("*.png"))),
                              "fov_masks": len(list((out / "fov").glob("*.png"))),
                              "resolutions_hw": {str(k): v for k, v in shapes.items()},
                              "vessel_fraction_in_fov_mean": float(np.mean(fractions))}
    summary["split"] = _write_split("drive", [p.stem for p in (PROC / "drive" / "train" / "images").glob("*.png")], 4)
    save_json(ROOT / "results" / "data_summary_drive.json", summary)
    print(pd.Series(summary).to_string(), "\n")


# ----------------------------------------------------------------------------- FIVES
def _fives_one(args: tuple[Path, Path, Path, str]) -> tuple[tuple[int, int], float, str]:
    img_path, mask_path, out, name = args
    img = cv2.imread(str(img_path))
    mask = (cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    # FIVES ships no FOV masks: the FOV is every pixel that is not near-black, cleaned up morphologically
    fov = (img.max(axis=2) > 10).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8)) * 255
    cv2.imwrite(str(out / "images" / f"{name}.png"), img)
    cv2.imwrite(str(out / "masks" / f"{name}.png"), mask)
    cv2.imwrite(str(out / "fov" / f"{name}.png"), fov)
    return img.shape[:2], float((mask > 0)[fov > 0].mean()), name.split("_")[-1]


def prepare_fives() -> None:
    """FIVES: 600 train / 200 test images, 2048x2048, categories A(MD), D(R), G(laucoma), N(ormal)."""
    summary: dict[str, object] = {"dataset": "FIVES"}
    for split in ("train", "test"):
        base = next(p for p in RAW.glob("fives/**/" + split) if p.is_dir())
        orig = next(p for p in base.iterdir() if p.name.lower().startswith("orig"))
        gt = next(p for p in base.iterdir() if p.name.lower().startswith("ground"))
        out = PROC / "fives" / split
        for sub in ("images", "masks", "fov"):
            (out / sub).mkdir(parents=True, exist_ok=True)
        jobs = [(p, gt / p.name, out, p.stem) for p in sorted(orig.glob("*.png")) if (gt / p.name).exists()]
        with ProcessPoolExecutor(16) as ex:
            res = list(ex.map(_fives_one, jobs, chunksize=8))
        summary[split] = {"images": len(list(orig.glob("*.png"))), "image_mask_pairs": len(jobs),
                          "resolutions_hw": {str(k): v for k, v in Counter(r[0] for r in res).items()},
                          "category_counts": dict(sorted(Counter(r[2] for r in res).items())),
                          "vessel_fraction_in_fov_mean": float(np.mean([r[1] for r in res]))}
    summary["split"] = _write_split("fives", [p.stem for p in (PROC / "fives" / "train" / "images").glob("*.png")], 60)
    save_json(ROOT / "results" / "data_summary_fives.json", summary)
    print(pd.Series(summary).to_string(), "\n")


# ----------------------------------------------------------------------------- APTOS
def _aptos_one(args: tuple[str, int]) -> tuple[str, tuple[int, int], str]:
    image_id, size = args
    bgr = cv2.imread(str(next(RAW.glob(f"aptos/**/train_images/{image_id}.png"))))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    sq = cv2.resize(pad_to_square(crop_fundus(rgb)), (size, size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(PROC / "aptos" / "plain" / f"{image_id}.jpg"), cv2.cvtColor(sq, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(str(PROC / "aptos" / "ben" / f"{image_id}.jpg"), cv2.cvtColor(ben_graham(sq).astype(np.uint8), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 95])
    # content hash of a small grayscale thumbnail: identical photos saved twice get the same hash
    thumb = cv2.resize(cv2.cvtColor(sq, cv2.COLOR_RGB2GRAY), (16, 16), interpolation=cv2.INTER_AREA) // 16
    return image_id, rgb.shape[:2], hashlib.md5(thumb.tobytes()).hexdigest()


def prepare_aptos(size: int = 512) -> None:
    """APTOS 2019: crop + resize every image once, detect duplicates, write a stratified 70/15/15 split."""
    df = pd.read_csv(next(RAW.glob("aptos/**/train.csv")))
    for v in ("plain", "ben"):
        (PROC / "aptos" / v).mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(24) as ex:
        res = list(ex.map(_aptos_one, [(i, size) for i in df.id_code], chunksize=16))
    df["hash"] = [r[2] for r in res]
    shapes = Counter(r[1] for r in res)

    # Duplicates: APTOS contains the same photograph under several ids, sometimes with different grades.
    # Keep one copy per duplicate group when all copies agree on the grade; drop groups that disagree.
    groups = df.groupby("hash")
    n_dup_groups = int((groups.size() > 1).sum())
    conflicting = groups.diagnosis.nunique() > 1
    n_conflict_images = int(df.hash.isin(conflicting[conflicting].index).sum())
    clean = df[~df.hash.isin(conflicting[conflicting].index)].drop_duplicates("hash").reset_index(drop=True)

    train, rest = train_test_split(clean, test_size=0.30, stratify=clean.diagnosis, random_state=SPLIT_SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.diagnosis, random_state=SPLIT_SEED)
    clean["split"] = "train"
    clean.loc[val.index, "split"] = "val"
    clean.loc[test.index, "split"] = "test"
    SPLITS.mkdir(parents=True, exist_ok=True)
    clean[["id_code", "diagnosis", "split"]].to_csv(SPLITS / "aptos_split.csv", index=False)

    summary = {
        "dataset": "APTOS 2019", "labelled_images": len(df), "images_found": len(res),
        "class_counts_raw": np.bincount(df.diagnosis, minlength=5).tolist(),
        "distinct_resolutions": len(shapes), "most_common_resolutions_hw": {str(k): v for k, v in shapes.most_common(5)},
        "duplicate_groups": n_dup_groups, "images_in_duplicate_groups": int((df.hash.map(groups.size()) > 1).sum()),
        "images_dropped_conflicting_labels": n_conflict_images, "images_after_dedup": len(clean),
        "split_class_counts": {s: np.bincount(clean[clean.split == s].diagnosis, minlength=5).tolist() for s in ("train", "val", "test")},
        "split_sizes": clean.split.value_counts().to_dict(), "cached_size": size,
    }
    save_json(ROOT / "results" / "data_summary_aptos.json", summary)
    print(pd.Series(summary).to_string(), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", choices=["drive", "fives", "aptos", "all"])
    which = ap.parse_args().dataset
    for name, fn in (("drive", prepare_drive), ("fives", prepare_fives), ("aptos", prepare_aptos)):
        if which in (name, "all"):
            fn()
