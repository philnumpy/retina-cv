"""DR grading models: any timm backbone with a 5-way classification or 1-output regression head."""
from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn


def build_classifier(cfg: dict[str, Any], pretrained: bool = True) -> nn.Module:
    """head='classification' -> 5 logits; head='regression' -> one continuous grade score."""
    num_out = 1 if cfg.get("head", "classification") == "regression" else 5
    return timm.create_model(cfg["arch"], pretrained=pretrained, num_classes=num_out, drop_rate=cfg.get("drop_rate", 0.2),
                             drop_path_rate=cfg.get("drop_path_rate", 0.1))


def gradcam_target_layer(model: nn.Module, arch: str) -> nn.Module:
    """Last convolutional feature map of each supported CNN family."""
    if arch.startswith("convnext"):
        return model.stages[-1]
    if arch.startswith("efficientnet") or arch.startswith("tf_efficientnet"):
        return model.conv_head
    if arch.startswith("resnet"):
        return model.layer4
    raise ValueError(f"no Grad-CAM layer registered for {arch}")
