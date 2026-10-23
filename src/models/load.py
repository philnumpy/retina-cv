"""Rebuild a trained model from its config and checkpoint (used by export, benchmark and predict)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from src.data.seg_data import frame_shape
from src.models.classifier import build_classifier
from src.models.unet import build_seg_model
from src.utils.common import ROOT


def load_model(task: str, config: str, seed: int | None = None, checkpoint: str | Path | None = None
               ) -> tuple[torch.nn.Module, dict[str, Any], tuple[int, ...], str]:
    """Rebuild a trained model from its config + checkpoint. Returns (model, cfg, example input shape, run name).

    The checkpoint defaults to runs/<task>/<run>/best.pt; pass `checkpoint` to load one from elsewhere.
    """
    with open(config) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("name", Path(config).stem)
    run = cfg["name"] if seed in (None, cfg.get("seed", 0)) else f"{cfg['name']}__seed{seed}"
    folder = "segmentation" if task == "seg" else "classification"
    ckpt = torch.load(checkpoint or ROOT / "runs" / folder / run / "best.pt", map_location="cpu", weights_only=False)
    if task == "seg":
        model = build_seg_model({**cfg, "encoder_weights": None})
        # one whole frame per forward pass: DRIVE 584x565 padded to 608x576, FIVES at its working size
        h, w = (cfg["work_size"],) * 2 if cfg.get("work_size") else frame_shape(584, 565)
        shape = (1 if cfg["input"] == "green" else 3, h, w)
    else:
        model = build_classifier(cfg, pretrained=False)
        shape = (3, cfg["image_size"], cfg["image_size"])
    model.load_state_dict(ckpt["model"])
    return model.eval(), cfg, shape, run
