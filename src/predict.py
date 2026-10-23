"""Run a trained model on your own fundus photographs (a folder or single files).

    python -m src.predict --task seg --config configs/seg/drive_attunet_green.yaml --input photos/ --output out/
    python -m src.predict --task cls --config configs/cls/convnext_tiny_ben.yaml --input photos/ --output out/ [--gradcam]

The checkpoint defaults to runs/<task>/<run>/best.pt (use --checkpoint for one stored elsewhere).
Anything that was fitted on validation is reused from results/<task>/<run>/metrics.json when present:
the segmentation threshold, the softmax temperature and the regression cut points.

Segmentation: the field of view is estimated from the image (non-black pixels) and the image is rescaled
so that its FOV has about the same diameter, in pixels, as the training images. Vessel width in pixels is
what the network learned, so this matters more than the image size. Outputs per image:
<name>_prob.png (vessel probability x 255), <name>_mask.png (binary), <name>_overlay.png, plus predictions.csv.

Grading: the image is cropped, padded and resized exactly like the APTOS cache (Ben Graham normalisation
for 'ben' configs). Outputs predictions.csv and, with --gradcam (classification head), <name>_gradcam.png.

Research code: the outputs are not a diagnosis.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch

from src.data.cls_data import ben_graham, build_transforms, square_fundus
from src.data.seg_data import estimate_fov, preprocess, sliding_window_predict
from src.models.load import load_model
from src.utils.common import ROOT, load_json
from src.utils.metrics import scores_to_grades, softmax

EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
# FOV diameter (px) the networks were trained at. DRIVE's FOV is about 540 px across (dataset description);
# FIVES fills the frame, so its FOV is about as wide as the working size (2048 px at native resolution).
TRAIN_FOV_DIAMETER = {"drive": 540, "fives": 2048}


def list_images(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTENSIONS) if p.is_dir() else [p]
    if not files:
        raise SystemExit(f"no images ({', '.join(sorted(EXTENSIONS))}) found in {paths}")
    return files


def read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def fitted(task: str, run: str) -> dict[str, Any]:
    """Values fitted on validation by the eval scripts, if this run has been evaluated."""
    path = ROOT / "results" / ("segmentation" if task == "seg" else "classification") / run / "metrics.json"
    return load_json(path) if path.exists() else {}


# ---------------------------------------------------------------------------- segmentation
def train_fov_diameter(cfg: dict[str, Any]) -> int:
    if cfg["dataset"] == "fives" and cfg.get("work_size"):
        return cfg["work_size"]
    return TRAIN_FOV_DIAMETER[cfg["dataset"]]


def segment(model: torch.nn.Module, rgb: np.ndarray, cfg: dict[str, Any], fov_diameter: int,
            device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Vessel probability at the input resolution, the FOV mask, and the scale the network saw."""
    h, w = rgb.shape[:2]
    fov = estimate_fov(rgb)
    if fov.sum() < 0.05 * h * w:  # no usable dark border: treat the whole image as FOV
        fov = np.ones((h, w), bool)
    ys, xs = np.where(fov)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    scale = fov_diameter / max(y1 - y0, x1 - x0)
    size = (max(round((x1 - x0) * scale), 1), max(round((y1 - y0) * scale), 1))  # cv2 order: (w, h)
    crop = cv2.resize(rgb[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    crop_fov = cv2.resize(fov[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
    prob = sliding_window_predict(model, preprocess(crop, crop_fov, cfg["input"]), cfg["patch"], cfg["patch"] // 2,
                                  device, cfg.get("eval_batch", 64))
    full = np.zeros((h, w), np.float32)
    full[y0:y1, x0:x1] = cv2.resize(prob, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    return full * fov, fov, float(scale)


def overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    out[mask] = (0.35 * out[mask] + 0.65 * np.array([255, 220, 0])).astype(np.uint8)
    return out


def run_seg(args: argparse.Namespace, model: torch.nn.Module, cfg: dict[str, Any], run: str, device: torch.device,
            files: list[Path], out: Path) -> pd.DataFrame:
    threshold = args.threshold if args.threshold is not None else fitted("seg", run).get("threshold", 0.5)
    diameter = args.fov_diameter or train_fov_diameter(cfg)
    rows = []
    for f in files:
        rgb = read_rgb(f)
        prob, fov, scale = segment(model, rgb, cfg, diameter, device)
        mask = (prob >= threshold) & fov
        cv2.imwrite(str(out / f"{f.stem}_prob.png"), np.round(prob * 255).astype(np.uint8))
        cv2.imwrite(str(out / f"{f.stem}_mask.png"), mask.astype(np.uint8) * 255)
        cv2.imwrite(str(out / f"{f.stem}_overlay.png"), cv2.cvtColor(overlay(rgb, mask), cv2.COLOR_RGB2BGR))
        rows.append({"image": f.name, "height": rgb.shape[0], "width": rgb.shape[1], "scale": round(scale, 4),
                     "threshold": threshold, "vessel_density": float(mask[fov].mean())})
        print(f"{f.name}: vessel density {rows[-1]['vessel_density']:.4f} (scale {scale:.3f})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- grading
def grading_input(rgb: np.ndarray, cfg: dict[str, Any]) -> torch.Tensor:
    """Same steps as prepare.py (512 px cache, optional Ben Graham) followed by the eval transform."""
    sq = square_fundus(rgb)
    if cfg["variant"] == "ben":
        sq = ben_graham(sq).astype(np.uint8)
    x = build_transforms(cfg["image_size"], train=False)(image=sq)["image"]
    return torch.from_numpy(x.transpose(2, 0, 1))


def run_cls(args: argparse.Namespace, model: torch.nn.Module, cfg: dict[str, Any], run: str, device: torch.device,
            files: list[Path], out: Path) -> pd.DataFrame:
    head = cfg.get("head", "classification")
    saved = fitted("cls", run)
    temperature = saved.get("temperature", 1.0)
    thresholds = np.array(saved.get("thresholds", [0.5, 1.5, 2.5, 3.5]))
    cam = None
    if args.gradcam:
        if head != "classification":
            raise SystemExit("--gradcam is implemented for the classification head only")
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        from src.data.cls_data import denormalise
        from src.models.classifier import gradcam_target_layer
        cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model, cfg["arch"])])

    rows = []
    for f in files:
        x = grading_input(read_rgb(f), cfg)[None].to(device)
        with torch.no_grad():
            output = model(x).float().cpu().numpy()
        row: dict[str, Any] = {"image": f.name}
        if head == "regression":
            score = float(output[0, 0])
            grade = int(scores_to_grades(np.array([score]), thresholds)[0])
            row["score"] = round(score, 4)
        else:
            probs = softmax(output, temperature)[0]
            grade = int(probs.argmax())
            row.update({f"p_grade{k}": round(float(p), 4) for k, p in enumerate(probs)})
            row["p_referable"] = round(float(probs[2:].sum()), 4)
            if cam is not None:
                heat = cam(input_tensor=x, targets=[ClassifierOutputTarget(grade)])[0]
                vis = show_cam_on_image(denormalise(x[0]).astype(np.float32), heat, use_rgb=True)
                cv2.imwrite(str(out / f"{f.stem}_gradcam.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        row.update({"grade": grade, "grade_name": GRADE_NAMES[grade], "referable": grade >= 2})
        rows.append(row)
        print(f"{f.name}: grade {grade} ({GRADE_NAMES[grade]})")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["seg", "cls"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True, nargs="+", help="image files and/or folders")
    ap.add_argument("--output", required=True, help="folder for predictions.csv and output images")
    ap.add_argument("--seed", type=int, default=None, help="pick the run trained with this seed")
    ap.add_argument("--checkpoint", default=None, help="path to best.pt (default: runs/<task>/<run>/best.pt)")
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:1 ...")
    ap.add_argument("--threshold", type=float, default=None, help="seg: vessel threshold (default: chosen on validation)")
    ap.add_argument("--fov-diameter", type=int, default=None, help="seg: rescale so the FOV is this many px across")
    ap.add_argument("--gradcam", action="store_true", help="cls: also save a Grad-CAM heatmap per image")
    args = ap.parse_args()

    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    model, cfg, _, run = load_model(args.task, args.config, args.seed, args.checkpoint)
    model.to(device).eval()
    files = list_images(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    df = (run_seg if args.task == "seg" else run_cls)(args, model, cfg, run, device, files, out)
    df.to_csv(out / "predictions.csv", index=False)
    print(f"{len(df)} images -> {out / 'predictions.csv'}")


if __name__ == "__main__":
    main()
