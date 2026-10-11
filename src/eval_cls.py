"""Evaluate a DR grader on the held-out test split.

    python -m src.eval_cls --config configs/cls/convnext_tiny_plain.yaml [--seed 1] [--smoke]

Anything fitted after training (temperature, regression thresholds) is fitted on validation only.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.cls_data import load_image, load_split
from src.models.classifier import build_classifier
from src.train_cls import make_loader, predict
from src.utils import plots
from src.utils.common import get_logger, load_config, parse_args, print_table, run_dirs, save_json, seed_everything
from src.utils.metrics import (classification_metrics, expected_calibration_error, fit_temperature, optimise_thresholds,
                               scores_to_grades, softmax)


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    smoke = cfg.get("smoke", False)
    res_dir, ckpt_dir = run_dirs("classification", cfg["run_name"])
    log = get_logger(res_dir, "eval")
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = cfg.get("head", "classification")

    ckpt = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)
    model = build_classifier(cfg, pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])

    split = load_split()
    val_df, test_df = split[split.split == "val"], split[split.split == "test"]
    if smoke:
        val_df, test_df = val_df.groupby("diagnosis").head(8), test_df.groupby("diagnosis").head(8)
    val_out, val_true = predict(model, make_loader(val_df, cfg, False), device)
    test_out, y_true = predict(model, make_loader(test_df, cfg, False), device)

    extra: dict[str, object] = {}
    ovr: list[dict[str, object]] = []
    if head == "regression":
        thresholds = optimise_thresholds(val_out[:, 0], val_true)
        y_pred = scores_to_grades(test_out[:, 0], thresholds)
        y_pred_naive = scores_to_grades(test_out[:, 0], np.array([0.5, 1.5, 2.5, 3.5]))
        referable_score = test_out[:, 0]  # the continuous grade itself ranks referable vs. not
        m = classification_metrics(y_true, y_pred, referable_score)
        extra = {"thresholds": thresholds, "qwk_default_thresholds": classification_metrics(y_true, y_pred_naive, referable_score)["qwk"]}
        probs = None
    else:
        probs = softmax(test_out)
        y_pred = probs.argmax(1)
        referable_score = probs[:, 2:].sum(1)
        m = classification_metrics(y_true, y_pred, referable_score)
        # calibration: ECE before and after temperature scaling (T fitted on validation)
        temperature = fit_temperature(val_out, val_true)
        ece, bins = expected_calibration_error(probs.max(1), (y_pred == y_true).astype(float))
        probs_ts = softmax(test_out, temperature)
        ece_ts, bins_ts = expected_calibration_error(probs_ts.max(1), (probs_ts.argmax(1) == y_true).astype(float))
        extra = {"ece": ece, "ece_temperature_scaled": ece_ts, "temperature": temperature}
        plots.reliability_diagram(bins, ece, bins_ts, ece_ts, res_dir / "reliability.png")
        for k in range(5):
            if 0 < (y_true == k).sum() < len(y_true):
                fpr, tpr, _ = roc_curve(y_true == k, probs[:, k])
                ovr.append({"fpr": fpr, "tpr": tpr, "auc": float(roc_auc_score(y_true == k, probs[:, k]))})
        extra["ovr_auc"] = [r["auc"] for r in ovr]
        extra["macro_ovr_auc"] = float(np.mean(extra["ovr_auc"]))

    ref = {**m.pop("referable_roc"), "auc": m["referable_auc"], "sensitivity": m["referable_sensitivity"],
           "specificity": m["referable_specificity"]}
    plots.roc_curves_cls(ovr, ref, res_dir / "roc_curves.png")
    plots.confusion_matrices(m["confusion_matrix"], res_dir / "confusion_matrix.png")
    plots.class_distribution({s: np.bincount(split[split.split == s].diagnosis, minlength=5) for s in ("train", "val", "test")},
                             res_dir / "class_distribution.png")

    # most confident errors (classification head only: confidence = max softmax probability)
    if probs is not None:
        wrong = np.where(y_pred != y_true)[0]
        top = wrong[np.argsort(-probs.max(1)[wrong])][:12]
        ids = test_df["id_code"].tolist()
        if len(top):
            plots.image_grid([load_image(ids[i], cfg["variant"]) for i in top],
                             [f"true {y_true[i]} | pred {y_pred[i]} (p = {probs[i].max():.2f})" for i in top], 6,
                             res_dir / "confident_errors.png", "Most confident errors on the test set")

    report_text = m.pop("report_text")
    metrics = {"run": cfg["run_name"], "arch": cfg["arch"], "variant": cfg["variant"], "head": head, "seed": cfg["seed"],
               "image_size": cfg["image_size"], "n_test": int(len(y_true)), "best_epoch": ckpt["epoch"],
               "val_qwk": ckpt["val_qwk"], **m, **extra}
    save_json(res_dir / "metrics.json", metrics)
    np.savez(res_dir / "test_predictions.npz", outputs=test_out, y_true=y_true, y_pred=y_pred,
             ids=np.array(test_df["id_code"].tolist()))
    rows = {k: metrics[k] for k in ("qwk", "accuracy", "macro_f1", "referable_auc", "referable_sensitivity", "referable_specificity")}
    rows.update({k: v for k, v in extra.items() if isinstance(v, float)})
    text = print_table(f"[{cfg['run_name']}] test metrics (n = {len(y_true)})", rows)
    print(report_text)
    (res_dir / "metrics.txt").write_text(text + "\n\n" + report_text)
    log.info(f"saved metrics and figures to {res_dir}")


if __name__ == "__main__":
    main()
