"""Evaluate a trained segmentation model on the official test set.

    python -m src.eval_seg --config configs/seg/drive_unet_green.yaml [--seed 1] [--smoke]

The decision threshold is chosen on the validation images (max pooled Dice) and then
applied unchanged to the test set. All metrics are computed inside the FOV at native resolution.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import pearsonr

from src.data.seg_data import full_frame_predict, load_samples, read_mask, sliding_window_predict, to_native
from src.models.unet import build_seg_model
from src.utils import plots
from src.utils.common import get_logger, load_config, parse_args, print_table, run_dirs, save_json, seed_everything
from src.utils.metrics import PixelHistogram, dice_score

SCALARS = ["threshold", "dice", "iou", "sensitivity", "specificity", "precision", "accuracy", "auc_roc", "auc_pr"]


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    smoke = cfg.get("smoke", False)
    res_dir, ckpt_dir = run_dirs("segmentation", cfg["run_name"])
    log = get_logger(res_dir, "eval")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model = build_seg_model({**cfg, "encoder_weights": None}).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    limit = 2 if smoke else None

    def predict(split: str) -> tuple[list[Any], list[np.ndarray]]:
        samples = load_samples(cfg["dataset"], split, cfg["input"], cfg.get("work_size"), limit)
        return samples, [sliding_window_predict(model, s.image, cfg["patch"], cfg["patch"] // 2, device, cfg["eval_batch"])
                         for s in samples]

    # 1) threshold from validation only
    val_hist = PixelHistogram()
    val_samples, val_probs = predict("val")
    for s, prob in zip(val_samples, val_probs):
        mask, fov = read_mask(s.mask_path), read_mask(s.fov_path).astype(bool)
        val_hist.update(to_native(prob, mask.shape), mask, fov)
    threshold, val_dice = val_hist.best_threshold()
    log.info(f"threshold chosen on validation: {threshold:.4f} (val pooled Dice {val_dice:.4f})")

    # 2) test set
    hist = PixelHistogram()
    hist_full = PixelHistogram()  # same test images, one forward pass per image instead of sliding windows
    per_image: list[dict[str, Any]] = []
    vis: list[dict[str, Any]] = []
    test_samples, test_probs = predict("test")
    for s, prob in zip(test_samples, test_probs):
        mask, fov = read_mask(s.mask_path), read_mask(s.fov_path).astype(bool)
        prob = to_native(prob, mask.shape)
        pred = (prob >= threshold) & fov
        hist.update(prob, mask, fov)
        hist_full.update(to_native(full_frame_predict(model, s.image, device), mask.shape), mask, fov)
        per_image.append({"name": s.name, "dice": dice_score(pred, mask, fov),
                          "density_true": float(mask[fov].mean()), "density_pred": float(pred[fov].mean())})
        vis.append({"name": s.name, "dice": per_image[-1]["dice"], "prob": prob * fov, "pred": pred, "mask": mask, "fov": fov})

    summary = hist.summary(threshold)
    curves = summary.pop("curves")
    full = hist_full.summary(threshold)
    dice_list = [p["dice"] for p in per_image]
    d_true, d_pred = np.array([p["density_true"] for p in per_image]), np.array([p["density_pred"] for p in per_image])
    r = float(pearsonr(d_true, d_pred)[0]) if len(per_image) > 2 else float("nan")
    metrics = {
        "run": cfg["run_name"], "dataset": cfg["dataset"], "arch": cfg["arch"], "input": cfg["input"], "loss": cfg["loss"],
        "seed": cfg["seed"], "n_test_images": len(per_image), "best_epoch": ckpt["epoch"], "val_dice_at_0.5": ckpt["val_dice"],
        "val_dice_at_threshold": val_dice, **summary,
        "dice_per_image_mean": float(np.mean(dice_list)), "dice_per_image_std": float(np.std(dice_list)),
        "dice_per_image_min": float(np.min(dice_list)),
        "density_pearson_r": r, "density_mae": float(np.abs(d_true - d_pred).mean()),
        "density_true_mean": float(d_true.mean()), "density_pred_mean": float(d_pred.mean()),
        "full_frame_dice": full["dice"], "full_frame_auc_roc": full["auc_roc"],
        "per_image": per_image,
    }
    save_json(res_dir / "metrics.json", metrics)
    text = print_table(f"[{cfg['run_name']}] test metrics (inside FOV, {len(per_image)} images)",
                       {**{k: metrics[k] for k in SCALARS}, "dice_per_image_mean": metrics["dice_per_image_mean"],
                        "density_pearson_r": r, "full_frame_dice": full["dice"]})
    (res_dir / "metrics.txt").write_text(text + "\n")

    # 3) figures
    plots.roc_pr(curves, summary["auc_roc"], summary["auc_pr"], summary["vessel_fraction"], res_dir / "roc_pr.png")
    plots.dice_vs_threshold(curves["thr"], curves["dice"], val_hist.summary(threshold)["curves"]["dice"], threshold,
                            res_dir / "dice_vs_threshold.png")
    plots.per_image_dice([p["name"] for p in per_image], dice_list, res_dir / "per_image_dice.png")
    if len(per_image) > 2:
        plots.density_scatter(d_true, d_pred, r, res_dir / "vessel_density.png")
    # best, two around the median and the worst image -> an honest qualitative sample
    order = np.argsort(dice_list)
    picks = [order[-1], order[len(order) * 2 // 3], order[len(order) // 3], order[0]] if len(order) >= 4 else list(order)
    import cv2
    shown = []
    for i in picks:
        v, s = vis[i], test_samples[i]
        rgb = cv2.resize(s.rgb, (v["mask"].shape[1], v["mask"].shape[0])) if s.rgb.shape[:2] != v["mask"].shape else s.rgb
        shown.append({**v, "image": rgb, "pred": v["pred"].astype(np.float32), "mask": v["mask"].astype(np.float32)})
    plots.qualitative_grid(shown, res_dir / "qualitative_grid.png")
    plots.error_maps(shown, res_dir / "error_maps.png")
    log.info(f"saved metrics and figures to {res_dir}")


if __name__ == "__main__":
    main()
