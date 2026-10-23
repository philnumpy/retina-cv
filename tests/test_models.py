from __future__ import annotations

import pytest
import torch

from src.models.classifier import build_classifier, gradcam_target_layer
from src.models.unet import UNet, build_seg_model, count_parameters


@pytest.mark.parametrize("arch", ["unet", "attention_unet", "unet_resnet34"])
@pytest.mark.parametrize("input_mode,channels", [("green", 1), ("rgb", 3)])
def test_seg_model_output_shape(arch, input_mode, channels):
    model = build_seg_model({"arch": arch, "input": input_mode, "base": 8, "encoder_weights": None}).eval()
    with torch.no_grad():
        out = model(torch.randn(2, channels, 64, 64))
    assert out.shape == (2, 1, 64, 64)


def test_unet_handles_non_square_frames():
    model = UNet(1, base=8).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 96, 64)).shape == (1, 1, 96, 64)


def test_attention_adds_parameters():
    assert count_parameters(UNet(1, base=8, attention=True)) > count_parameters(UNet(1, base=8))


def test_unknown_seg_arch_raises():
    with pytest.raises(ValueError, match="unknown arch"):
        build_seg_model({"arch": "segformer", "input": "green"})


@pytest.mark.parametrize("head,n_out", [("classification", 5), ("regression", 1)])
def test_classifier_heads(head, n_out):
    model = build_classifier({"arch": "resnet18", "head": head}, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, n_out)


@pytest.mark.parametrize("arch", ["resnet18", "efficientnet_b0", "convnext_atto"])
def test_gradcam_target_layer_registered(arch):
    model = build_classifier({"arch": arch}, pretrained=False)
    layer = gradcam_target_layer(model, arch)
    assert any(m is layer for m in model.modules())


def test_gradcam_target_layer_unknown_arch():
    model = build_classifier({"arch": "vit_tiny_patch16_224"}, pretrained=False)
    with pytest.raises(ValueError, match="no Grad-CAM layer"):
        gradcam_target_layer(model, "vit_tiny_patch16_224")
