"""U-Net and Attention U-Net written out in full, plus a factory that also offers
a U-Net with an ImageNet-pretrained ResNet34 encoder (segmentation_models_pytorch)."""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """(3x3 conv -> BatchNorm -> ReLU) x 2, with optional dropout."""

    def __init__(self, c_in: int, c_out: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018).

    The decoder feature g (coarse, semantically strong) decides which positions of the
    skip feature x to pass through: alpha = sigmoid(psi(relu(Wg*g + Wx*x))), output = alpha * x.
    """

    def __init__(self, c_gate: int, c_skip: int, c_mid: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(nn.Conv2d(c_gate, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid))
        self.w_x = nn.Sequential(nn.Conv2d(c_skip, c_mid, 1, bias=False), nn.BatchNorm2d(c_mid))
        self.psi = nn.Sequential(nn.Conv2d(c_mid, 1, 1), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x * self.psi(F.relu(self.w_g(g) + self.w_x(x)))


class UNet(nn.Module):
    """Classic encoder-decoder with skip connections. `attention=True` gates every skip."""

    def __init__(self, in_channels: int = 1, base: int = 32, depth: int = 4, attention: bool = False,
                 dropout: float = 0.1) -> None:
        super().__init__()
        chans = [base * 2**i for i in range(depth + 1)]  # e.g. 32, 64, 128, 256, 512
        self.encoders = nn.ModuleList()
        c = in_channels
        for ch in chans[:-1]:
            self.encoders.append(ConvBlock(c, ch))
            c = ch
        self.bottleneck = ConvBlock(chans[-2], chans[-1], dropout)
        self.ups, self.decoders, self.gates = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for ch in reversed(chans[:-1]):
            self.ups.append(nn.ConvTranspose2d(ch * 2, ch, 2, stride=2))
            self.gates.append(AttentionGate(ch, ch, ch // 2) if attention else nn.Identity())
            self.decoders.append(ConvBlock(ch * 2, ch))
        self.head = nn.Conv2d(chans[0], 1, 1)
        self.attention = attention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)
            x = F.max_pool2d(x, 2)
        x = self.bottleneck(x)
        for up, gate, dec, skip in zip(self.ups, self.gates, self.decoders, reversed(skips)):
            x = up(x)
            skip = gate(x, skip) if self.attention else skip
            x = dec(torch.cat([skip, x], dim=1))
        return self.head(x)  # logits


def build_seg_model(cfg: dict[str, Any]) -> nn.Module:
    in_ch = 1 if cfg["input"] == "green" else 3
    arch = cfg["arch"]
    if arch == "unet":
        return UNet(in_ch, cfg.get("base", 32), cfg.get("depth", 4), attention=False)
    if arch == "attention_unet":
        return UNet(in_ch, cfg.get("base", 32), cfg.get("depth", 4), attention=True)
    if arch == "unet_resnet34":
        import segmentation_models_pytorch as smp
        return smp.Unet("resnet34", encoder_weights=cfg.get("encoder_weights", "imagenet"), in_channels=in_ch, classes=1)
    raise ValueError(f"unknown arch: {arch}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
