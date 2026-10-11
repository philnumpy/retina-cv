"""All figures. One shared style so plots across experiments read as one set."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# categorical palette (fixed order) and neutrals
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Proliferative"]

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight", "figure.facecolor": "white",
    "axes.facecolor": "white", "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 2, "font.family": "DejaVu Sans",
    "axes.prop_cycle": plt.cycler(color=C),
})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path)
    plt.close(fig)


def training_curves(history: list[dict[str, float]], panels: Sequence[tuple[str, Sequence[str]]], path: Path) -> None:
    """One panel per quantity; each panel overlays the listed history keys (e.g. train/val)."""
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.6))
    epochs = [h["epoch"] for h in history]
    for ax, (title, keys) in zip(np.atleast_1d(axes), panels):
        for i, k in enumerate(keys):
            ax.plot(epochs, [h.get(k, np.nan) for h in history], color=C[i], label=k.replace("_", " "))
        ax.set(title=title, xlabel="epoch")
        if len(keys) > 1:
            ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------- segmentation
def roc_pr(curves: dict[str, np.ndarray], auc_roc: float, auc_pr: float, prevalence: float, path: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.2, 4))
    a.plot(curves["fpr"], curves["tpr"], color=C[0], label=f"model (AUC = {auc_roc:.4f})")
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="chance")
    a.set(title="ROC curve (test, inside FOV)", xlabel="false positive rate", ylabel="true positive rate")
    a.legend(loc="lower right")
    b.plot(curves["recall"], curves["precision"], color=C[0], label=f"model (AP = {auc_pr:.4f})")
    b.axhline(prevalence, color=MUTED, lw=1, ls="--", label=f"prevalence ({prevalence:.3f})")
    b.set(title="Precision-recall curve (test, inside FOV)", xlabel="recall", ylabel="precision", ylim=(0, 1.02))
    b.legend(loc="lower left")
    _save(fig, path)


def dice_vs_threshold(thr: np.ndarray, dice_test: np.ndarray, dice_val: np.ndarray, chosen: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(thr, dice_val, color=C[1], label="validation")
    ax.plot(thr, dice_test, color=C[0], label="test")
    ax.axvline(chosen, color=MUTED, lw=1, ls="--", label=f"threshold chosen on validation ({chosen:.3f})")
    ax.set(title="Dice vs. threshold", xlabel="probability threshold", ylabel="pooled Dice", xlim=(0, 1), ylim=(0, 1))
    ax.legend(loc="center")
    _save(fig, path)


def qualitative_grid(samples: list[dict[str, np.ndarray]], path: Path) -> None:
    """Rows = test images; columns = input | ground truth | prediction | probability map."""
    titles = ["input", "ground truth", "prediction", "probability"]
    fig, axes = plt.subplots(len(samples), 4, figsize=(12, 3 * len(samples)))
    for r, s in enumerate(samples):
        panels = [s["image"], s["mask"], s["pred"], s["prob"]]
        for c, (ax, im) in enumerate(zip(axes[r], panels)):
            ax.imshow(im, cmap=None if im.ndim == 3 else ("magma" if c == 3 else "gray"), vmin=0, vmax=None if im.ndim == 3 else 1)
            ax.axis("off")
            if r == 0:
                ax.set_title(titles[c])
        axes[r, 0].text(0.02, 0.97, f"{s['name']}  Dice {s['dice']:.3f}", transform=axes[r, 0].transAxes, color="white",
                        fontsize=8, va="top")
    fig.subplots_adjust(wspace=0.02, hspace=0.02)
    _save(fig, path)


def error_overlay(mask: np.ndarray, pred: np.ndarray, fov: np.ndarray) -> np.ndarray:
    """RGB map: true positive white, false positive orange, false negative blue."""
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    m, p = mask.astype(bool), pred.astype(bool) & fov
    out[m & p] = (255, 255, 255)
    out[~m & p] = (235, 104, 52)
    out[m & ~p & fov] = (42, 120, 214)
    return out


def error_maps(samples: list[dict[str, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(1, len(samples), figsize=(4 * len(samples), 4.3))
    for ax, s in zip(np.atleast_1d(axes), samples):
        ax.imshow(error_overlay(s["mask"], s["pred"], s["fov"]))
        ax.set_title(f"{s['name']}  (Dice {s['dice']:.3f})", fontsize=9)
        ax.axis("off")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=MUTED) for c in ("white", C[1], C[0])]
    fig.legend(handles, ["true positive", "false positive", "false negative"], loc="lower center", ncol=3)
    _save(fig, path)


def per_image_dice(names: list[str], dice: list[float], path: Path) -> None:
    order = np.argsort(dice)
    n = len(dice)
    if n <= 40:
        fig, ax = plt.subplots(figsize=(max(6, 0.36 * n), 3.6))
        ax.bar(range(n), np.array(dice)[order], color=C[0], width=0.7)
        ax.set_xticks(range(n), [names[i] for i in order], rotation=60, ha="right", fontsize=7)
        ax.set_ylim(min(dice) - 0.05, max(dice) + 0.02)
    else:
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.hist(dice, bins=30, color=C[0], edgecolor="white")
        ax.set_xlabel("Dice")
        ax.set_ylabel("test images")
    ax.axhline(np.mean(dice), color=MUTED, lw=1, ls="--") if n <= 40 else ax.axvline(np.mean(dice), color=MUTED, lw=1, ls="--")
    ax.set_title(f"Per-image Dice on the test set (mean {np.mean(dice):.4f}, min {np.min(dice):.4f}, n = {n})")
    _save(fig, path)


def density_scatter(true_d: np.ndarray, pred_d: np.ndarray, r: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    lo, hi = min(true_d.min(), pred_d.min()) * 0.95, max(true_d.max(), pred_d.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", label="identity")
    ax.scatter(true_d, pred_d, s=28, color=C[0], edgecolor="white", linewidth=0.8, label=f"test images (Pearson r = {r:.3f})")
    ax.set(title="Vessel density: predicted vs. manual", xlabel="manual annotation (vessel px / FOV px)",
           ylabel="prediction (vessel px / FOV px)", xlim=(lo, hi), ylim=(lo, hi))
    ax.legend(loc="upper left")
    _save(fig, path)


# -------------------------------------------------------------- classification
def confusion_matrices(cm: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    for ax, mat, title, fmt in ((axes[0], cm, "Confusion matrix (counts)", "d"), (axes[1], norm, "Row-normalised (recall)", ".2f")):
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set(title=title, xlabel="predicted grade", ylabel="true grade", xticks=range(5), yticks=range(5))
        ax.set_yticklabels(GRADES)
        ax.grid(False)
    fig.tight_layout()
    _save(fig, path)


def roc_curves_cls(ovr: list[dict[str, Any]], referable: dict[str, Any], path: Path) -> None:
    """Left: one-vs-rest ROC per grade (classification heads only). Right: referable-DR ROC."""
    fig, axes = plt.subplots(1, 2 if ovr else 1, figsize=(9.6 if ovr else 4.8, 4.1), squeeze=False)
    if ovr:
        a = axes[0, 0]
        for k, r in enumerate(ovr):
            a.plot(r["fpr"], r["tpr"], color=C[k], label=f"{GRADES[k]} (AUC = {r['auc']:.3f})")
        a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
        a.set(title="One-vs-rest ROC per grade (test)", xlabel="false positive rate", ylabel="true positive rate")
        a.legend(loc="lower right")
    b = axes[0, -1]
    b.plot(referable["fpr"], referable["tpr"], color=C[0], label=f"referable DR (AUC = {referable['auc']:.4f})")
    b.scatter([1 - referable["specificity"]], [referable["sensitivity"]], s=50, color=C[1], zorder=3, edgecolor="white",
              label=f"operating point (sens {referable['sensitivity']:.3f}, spec {referable['specificity']:.3f})")
    b.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    b.set(title="Referable DR (grade >= 2) ROC (test)", xlabel="false positive rate", ylabel="true positive rate")
    b.legend(loc="lower right")
    _save(fig, path)


def reliability_diagram(bins_raw: dict[str, np.ndarray], ece_raw: float, bins_ts: dict[str, np.ndarray] | None,
                        ece_ts: float | None, path: Path) -> None:
    items = [("uncalibrated", bins_raw, ece_raw)] + ([("temperature-scaled", bins_ts, ece_ts)] if bins_ts else [])
    fig, axes = plt.subplots(1, len(items), figsize=(4.6 * len(items), 4.2), squeeze=False)
    for ax, (name, b, ece) in zip(axes[0], items):
        centers, width = (b["edges"][:-1] + b["edges"][1:]) / 2, b["edges"][1] - b["edges"][0]
        m = b["count"] > 0
        ax.bar(centers[m], b["acc"][m], width=width * 0.9, color=C[0], label="accuracy per confidence bin")
        ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect calibration")
        ax.set(title=f"Reliability, {name} (ECE = {ece:.4f})", xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
        ax.legend(loc="upper left")
    _save(fig, path)


def class_distribution(counts: dict[str, np.ndarray], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    w = 0.26
    for i, (split, c) in enumerate(counts.items()):
        bars = ax.bar(np.arange(5) + (i - 1) * w, c, width=w - 0.02, color=C[i], label=f"{split} (n = {int(c.sum())})")
        ax.bar_label(bars, fontsize=7, color=MUTED, padding=1)
    ax.set_xticks(range(5), GRADES)
    ax.set(title="APTOS 2019 class distribution per split", ylabel="images")
    ax.legend()
    _save(fig, path)


def image_grid(images: list[np.ndarray], titles: list[str], ncols: int, path: Path, suptitle: str = "") -> None:
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t in zip(axes.ravel(), images, titles):
        ax.imshow(im)
        ax.set_title(t, fontsize=8, fontweight="normal")
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def bar_compare(labels: list[str], groups: dict[str, list[float]], ylabel: str, title: str, path: Path, log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(labels)), 3.9))
    w = 0.8 / len(groups)
    for i, (name, vals) in enumerate(groups.items()):
        bars = ax.bar(np.arange(len(labels)) + (i - (len(groups) - 1) / 2) * w, vals, width=w - 0.03, color=C[i], label=name)
        ax.bar_label(bars, fmt="%.1f", fontsize=7, color=MUTED, padding=1)
    ax.set_xticks(range(len(labels)), labels)
    ax.set(title=title, ylabel=ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend()
    _save(fig, path)
