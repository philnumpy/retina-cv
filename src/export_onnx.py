"""Export a trained model to ONNX and verify it against PyTorch.

    python -m src.export_onnx --task seg --config configs/seg/drive_unet_green.yaml
    python -m src.export_onnx --task cls --config configs/cls/convnext_tiny_plain.yaml

Also reports parameter count and FLOPs. Output: results/efficiency/<run>/export.json + model.onnx (in runs/).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import torch
import yaml
from thop import profile

from src.data.seg_data import frame_shape
from src.models.classifier import build_classifier
from src.models.unet import build_seg_model
from src.utils.common import ROOT, save_json


def load_model(task: str, config: str, seed: int | None = None) -> tuple[torch.nn.Module, dict[str, Any], tuple[int, ...], str]:
    """Rebuild a trained model from its config + checkpoint. Returns (model, cfg, example input shape, run name)."""
    cfg = yaml.safe_load(open(config))
    cfg.setdefault("name", Path(config).stem)
    run = cfg["name"] if seed in (None, cfg.get("seed", 0)) else f"{cfg['name']}__seed{seed}"
    folder = "segmentation" if task == "seg" else "classification"
    ckpt = torch.load(ROOT / "runs" / folder / run / "best.pt", map_location="cpu", weights_only=False)
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


def real_inputs(task: str, cfg: dict[str, Any], shape: tuple[int, ...], n: int = 4) -> torch.Tensor:
    """A small batch of real, preprocessed test inputs with the export shape."""
    if task == "cls":
        from src.data.cls_data import AptosDataset, load_split
        split = load_split()
        ds = AptosDataset(split[split.split == "test"].head(n), cfg["variant"], cfg["image_size"], train=False)
        return torch.stack([ds[i][0] for i in range(n)])
    from src.data.seg_data import load_samples
    frames = []
    for s in load_samples(cfg["dataset"], "test", cfg["input"], cfg.get("work_size"), limit=n):
        _, h, w = s.image.shape
        frames.append(np.pad(s.image, ((0, 0), (0, shape[1] - h), (0, shape[2] - w)), mode="reflect"))
    return torch.from_numpy(np.stack(frames))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["seg", "cls"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    model, cfg, shape, run = load_model(args.task, args.config, args.seed)
    out_dir = ROOT / "results" / "efficiency" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = ROOT / "runs" / "onnx" / f"{run}.onnx"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    x = torch.randn(1, *shape)
    params = sum(p.numel() for p in model.parameters())
    macs, _ = profile(model, inputs=(x,), verbose=False)  # multiply-accumulates for one input
    for m in model.modules():  # thop registers counter buffers; remove them before export
        for name in ("total_ops", "total_params"):
            if name in m._buffers:
                del m._buffers[name]

    torch.onnx.export(model, x, onnx_path, input_names=["input"], output_names=["output"], opset_version=17,
                      dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    # Verify on real preprocessed test images: random noise is far outside the training distribution and
    # produces huge activations, which says little about agreement on actual inputs.
    xb = real_inputs(args.task, cfg, shape)
    with torch.no_grad():
        ref = model(xb).numpy()
    out = session.run(None, {"input": xb.numpy()})[0]
    max_diff = float(np.abs(ref - out).max())
    agree = float(((ref > 0) == (out > 0)).mean()) if args.task == "seg" or ref.shape[1] == 1 else \
        float((ref.argmax(1) == out.argmax(1)).mean())
    result = {"run": run, "task": args.task, "arch": cfg["arch"], "input_shape": list(shape), "parameters": params,
              "parameters_M": params / 1e6, "gmacs": macs / 1e9, "onnx_file_mb": onnx_path.stat().st_size / 1e6,
              "onnx_max_abs_diff": max_diff, "onnx_output_range": [float(ref.min()), float(ref.max())],
              "onnx_decision_agreement": agree, "onnx_verified_on": f"{len(xb)} real test inputs", "onnx_path": str(onnx_path.relative_to(ROOT))}
    save_json(out_dir / "export.json", result)
    for k, v in result.items():
        print(f"{k:<20}{v}")


if __name__ == "__main__":
    main()
