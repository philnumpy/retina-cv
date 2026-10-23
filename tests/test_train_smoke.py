"""End-to-end `--smoke` runs of both training scripts on tiny synthetic data, on CPU, with workers=0.

Nothing is downloaded: dataset loaders and output folders are redirected to pytest's tmp_path.
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pandas as pd
import yaml

import src.data.cls_data as cls_data
import src.train_cls as train_cls
import src.train_seg as train_seg
from src.models.classifier import build_classifier
from tests.test_seg_data import make_sample


def run_main(monkeypatch, module, tmp_path, cfg: dict) -> tuple:
    config = tmp_path / "exp.yaml"
    config.write_text(yaml.safe_dump(cfg))
    res, ckpt = tmp_path / "results", tmp_path / "runs"
    res.mkdir(), ckpt.mkdir()
    monkeypatch.setattr(module, "run_dirs", lambda task, name: (res, ckpt))
    monkeypatch.setattr(sys, "argv", ["prog", "--config", str(config), "--smoke", "--set", "workers=0"])
    module.main()
    return res, ckpt


def test_train_seg_smoke(monkeypatch, tmp_path):
    samples = [make_sample(f"img{i}") for i in range(3)]
    monkeypatch.setattr(train_seg, "load_samples", lambda *a, **k: samples)
    cfg = {"dataset": "synthetic", "arch": "unet", "base": 8, "input": "green", "loss": "dice_bce", "patch": 32,
           "patches_per_epoch": 64, "batch_size": 4, "eval_batch": 16, "epochs": 5, "lr": 1e-3, "workers": 8}
    res, ckpt = run_main(monkeypatch, train_seg, tmp_path, cfg)
    history = json.loads((res / "history.json").read_text())
    assert len(history["history"]) == 1  # --smoke forces a single epoch
    assert (ckpt / "best.pt").exists() and (res / "training_curves.png").exists()
    assert yaml.safe_load((res / "config.yaml").read_text())["workers"] == 0


def test_train_cls_smoke(monkeypatch, tmp_path):
    rng = np.random.default_rng(0)
    images = tmp_path / "aptos"
    (images / "plain").mkdir(parents=True)
    rows = []
    for i in range(30):
        cv2.imwrite(str(images / "plain" / f"id{i}.jpg"), rng.integers(0, 256, (48, 48, 3), dtype=np.uint8))
        rows.append({"id_code": f"id{i}", "diagnosis": i % 5, "split": "train" if i < 20 else "val"})
    monkeypatch.setattr(cls_data, "APTOS", images)
    monkeypatch.setattr(train_cls, "load_split", lambda: pd.DataFrame(rows))
    monkeypatch.setattr(train_cls, "build_classifier", lambda cfg: build_classifier(cfg, pretrained=False))
    cfg = {"arch": "resnet18", "variant": "plain", "head": "classification", "image_size": 32, "batch_size": 4,
           "epochs": 5, "lr": 1e-4, "workers": 8}
    res, ckpt = run_main(monkeypatch, train_cls, tmp_path, cfg)
    history = json.loads((res / "history.json").read_text())
    assert len(history["history"]) == 1
    assert (ckpt / "best.pt").exists()
