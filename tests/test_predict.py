"""src.predict end to end on synthetic fundus photos, with tiny untrained checkpoints saved to tmp_path."""
from __future__ import annotations

import sys

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

import src.predict as predict
from src.data.seg_data import estimate_fov
from src.models.classifier import build_classifier
from src.models.unet import build_seg_model


def photo(h: int, w: int, radius: int, seed: int = 0) -> np.ndarray:
    """A bright noisy disc on black: the shape of a fundus photograph."""
    rgb = np.random.default_rng(seed).integers(60, 200, (h, w, 3), dtype=np.uint8)
    yy, xx = np.mgrid[:h, :w]
    rgb[(yy - h / 2) ** 2 + (xx - w / 2) ** 2 > radius ** 2] = 0
    return rgb


@pytest.fixture
def photos(tmp_path):
    folder = tmp_path / "photos"
    folder.mkdir()
    cv2.imwrite(str(folder / "a.png"), photo(90, 120, 40))
    cv2.imwrite(str(folder / "b.jpg"), photo(150, 150, 70, seed=1))
    (folder / "notes.txt").write_text("not an image")
    return folder


def save_run(tmp_path, task: str, cfg: dict) -> tuple:
    config = tmp_path / f"{cfg['name']}.yaml"
    config.write_text(yaml.safe_dump(cfg))
    model = build_seg_model({**cfg, "encoder_weights": None}) if task == "seg" else build_classifier(cfg, pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    return config, ckpt


def run(monkeypatch, *argv: str) -> None:
    monkeypatch.setattr(sys, "argv", ["predict", *argv])
    predict.main()


SEG_CFG = {"name": "pytest_seg_run", "dataset": "drive", "arch": "unet", "base": 8, "input": "green", "patch": 32,
           "eval_batch": 16}


def test_estimate_fov_finds_disc():
    fov = estimate_fov(photo(100, 100, 40))
    assert fov[50, 50] and not fov[2, 2]
    assert fov.sum() == pytest.approx(np.pi * 40 ** 2, rel=0.05)


def test_segment_rescales_fov_and_keeps_input_size():
    model = build_seg_model({**SEG_CFG, "encoder_weights": None}).eval()
    rgb = photo(120, 160, 50)
    prob, fov, scale = predict.segment(model, rgb, SEG_CFG, fov_diameter=50, device=torch.device("cpu"))
    assert prob.shape == fov.shape == (120, 160)
    assert scale == pytest.approx(0.5, abs=0.02)  # FOV ~100 px across -> rescaled to 50
    assert np.all(prob[~fov] == 0) and 0 <= prob.min() and prob.max() <= 1


def test_predict_seg(monkeypatch, tmp_path, photos):
    config, ckpt = save_run(tmp_path, "seg", SEG_CFG)
    out = tmp_path / "out"
    run(monkeypatch, "--task", "seg", "--config", str(config), "--checkpoint", str(ckpt), "--input", str(photos),
        "--output", str(out), "--fov-diameter", "64", "--device", "cpu")
    df = pd.read_csv(out / "predictions.csv")
    assert list(df.image) == ["a.png", "b.jpg"]
    assert (df.threshold == 0.5).all()  # no metrics.json for this run -> default threshold
    for stem, shape in (("a", (90, 120)), ("b", (150, 150))):
        mask = cv2.imread(str(out / f"{stem}_mask.png"), cv2.IMREAD_GRAYSCALE)
        assert mask.shape == shape and set(np.unique(mask)) <= {0, 255}
        assert (out / f"{stem}_prob.png").exists() and (out / f"{stem}_overlay.png").exists()


def test_predict_seg_uses_validation_threshold(monkeypatch, tmp_path, photos):
    config, ckpt = save_run(tmp_path, "seg", SEG_CFG)
    results = tmp_path / "results" / "segmentation" / SEG_CFG["name"]
    results.mkdir(parents=True)
    (results / "metrics.json").write_text('{"threshold": 0.37}')
    monkeypatch.setattr(predict, "ROOT", tmp_path)
    out = tmp_path / "out"
    run(monkeypatch, "--task", "seg", "--config", str(config), "--checkpoint", str(ckpt), "--input", str(photos / "a.png"),
        "--output", str(out), "--fov-diameter", "64", "--device", "cpu")
    assert pd.read_csv(out / "predictions.csv").threshold.tolist() == [0.37]


@pytest.mark.parametrize("variant", ["plain", "ben"])
def test_predict_cls_classification_with_gradcam(monkeypatch, tmp_path, photos, variant):
    cfg = {"name": "pytest_cls_run", "arch": "resnet18", "variant": variant, "head": "classification", "image_size": 32}
    config, ckpt = save_run(tmp_path, "cls", cfg)
    out = tmp_path / "out"
    run(monkeypatch, "--task", "cls", "--config", str(config), "--checkpoint", str(ckpt), "--input", str(photos),
        "--output", str(out), "--gradcam", "--device", "cpu")
    df = pd.read_csv(out / "predictions.csv")
    probs = df[[f"p_grade{k}" for k in range(5)]].to_numpy()
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-3)
    assert (df.grade == probs.argmax(1)).all()
    assert (df.referable == (df.grade >= 2)).all()
    assert (out / "a_gradcam.png").exists() and (out / "b_gradcam.png").exists()


def test_predict_cls_regression(monkeypatch, tmp_path, photos):
    cfg = {"name": "pytest_reg_run", "arch": "resnet18", "variant": "plain", "head": "regression", "image_size": 32}
    config, ckpt = save_run(tmp_path, "cls", cfg)
    out = tmp_path / "out"
    run(monkeypatch, "--task", "cls", "--config", str(config), "--checkpoint", str(ckpt), "--input", str(photos),
        "--output", str(out), "--device", "cpu")
    df = pd.read_csv(out / "predictions.csv")
    expected = np.digitize(df.score, [0.5, 1.5, 2.5, 3.5])
    assert (df.grade == expected).all() and df.grade.between(0, 4).all()


def test_gradcam_rejected_for_regression(monkeypatch, tmp_path, photos):
    cfg = {"name": "pytest_reg_run", "arch": "resnet18", "variant": "plain", "head": "regression", "image_size": 32}
    config, ckpt = save_run(tmp_path, "cls", cfg)
    with pytest.raises(SystemExit, match="classification head only"):
        run(monkeypatch, "--task", "cls", "--config", str(config), "--checkpoint", str(ckpt), "--input", str(photos),
            "--output", str(tmp_path / "out"), "--gradcam", "--device", "cpu")


def test_no_images_is_a_clear_error(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="no images"):
        predict.list_images([str(tmp_path / "empty")])
