"""Metrics for both tasks. Everything here is plain numpy so it is easy to audit."""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import (accuracy_score, classification_report, cohen_kappa_score, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)

# --------------------------------------------------------------------------------------
# Segmentation: pixel statistics are accumulated in probability histograms so that
# ROC / PR / Dice-vs-threshold can be computed over hundreds of millions of pixels
# without keeping them in memory. With 4096 bins the binning error is < 1e-3 in threshold.
# --------------------------------------------------------------------------------------
N_BINS = 4096


class PixelHistogram:
    """Histograms of predicted vessel probability for vessel and background pixels (FOV only)."""

    def __init__(self) -> None:
        self.pos = np.zeros(N_BINS, dtype=np.int64)
        self.neg = np.zeros(N_BINS, dtype=np.int64)

    def update(self, prob: np.ndarray, target: np.ndarray, fov: np.ndarray) -> None:
        idx = np.minimum((prob[fov] * N_BINS).astype(np.int64), N_BINS - 1)
        t = target[fov].astype(bool)
        self.pos += np.bincount(idx[t], minlength=N_BINS)
        self.neg += np.bincount(idx[~t], minlength=N_BINS)

    def curves(self) -> dict[str, np.ndarray]:
        """Confusion counts for every threshold t_k = k / N_BINS (predict vessel if prob >= t_k)."""
        tp = np.cumsum(self.pos[::-1])[::-1].astype(np.float64)
        fp = np.cumsum(self.neg[::-1])[::-1].astype(np.float64)
        fn = self.pos.sum() - tp
        tn = self.neg.sum() - fp
        return {"thr": np.arange(N_BINS) / N_BINS, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def summary(self, threshold: float) -> dict[str, Any]:
        c = self.curves()
        eps = 1e-12
        tpr = c["tp"] / (c["tp"] + c["fn"] + eps)
        fpr = c["fp"] / (c["fp"] + c["tn"] + eps)
        prec = c["tp"] / (c["tp"] + c["fp"] + eps)
        dice = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"] + eps)
        # append the "predict nothing" end point so the curves span the full range
        fpr_c, tpr_c = np.append(fpr, 0.0)[::-1], np.append(tpr, 0.0)[::-1]
        rec_c, prec_c = np.append(tpr, 0.0)[::-1], np.append(prec, 1.0)[::-1]
        auc_roc = float(np.trapezoid(tpr_c, fpr_c))
        # average precision: sum of precision * recall increment (step-wise, like sklearn)
        auc_pr = float(np.sum(np.diff(rec_c) * prec_c[1:]))
        k = min(int(round(threshold * N_BINS)), N_BINS - 1)
        tp, fp, fn, tn = c["tp"][k], c["fp"][k], c["fn"][k], c["tn"][k]
        return {
            "threshold": threshold,
            "dice": float(2 * tp / (2 * tp + fp + fn + eps)),
            "iou": float(tp / (tp + fp + fn + eps)),
            "sensitivity": float(tp / (tp + fn + eps)),
            "specificity": float(tn / (tn + fp + eps)),
            "precision": float(tp / (tp + fp + eps)),
            "accuracy": float((tp + tn) / (tp + tn + fp + fn + eps)),
            "auc_roc": auc_roc,
            "auc_pr": auc_pr,
            "n_pixels": int(tp + tn + fp + fn),
            "vessel_fraction": float((tp + fn) / (tp + tn + fp + fn + eps)),
            "curves": {"fpr": fpr_c, "tpr": tpr_c, "recall": rec_c, "precision": prec_c, "thr": c["thr"], "dice": dice},
        }

    def best_threshold(self) -> tuple[float, float]:
        """Threshold maximising pooled Dice (used on the validation set only)."""
        c = self.curves()
        dice = 2 * c["tp"] / (2 * c["tp"] + c["fp"] + c["fn"] + 1e-12)
        k = int(np.argmax(dice))
        return float(c["thr"][k]), float(dice[k])


def dice_score(pred: np.ndarray, target: np.ndarray, fov: np.ndarray) -> float:
    p, t = pred[fov].astype(bool), target[fov].astype(bool)
    return float(2 * (p & t).sum() / (p.sum() + t.sum() + 1e-12))


# --------------------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------------------
def qwk(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(5))))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> tuple[float, dict[str, np.ndarray]]:
    """ECE = sum_b (n_b / N) * |accuracy_b - confidence_b| over equal-width confidence bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    acc, avg_conf, count = np.zeros(n_bins), np.zeros(n_bins), np.zeros(n_bins)
    for b in range(n_bins):
        m = idx == b
        count[b] = m.sum()
        if m.any():
            acc[b], avg_conf[b] = correct[m].mean(), conf[m].mean()
    ece = float(np.sum(count / max(count.sum(), 1) * np.abs(acc - avg_conf)))
    return ece, {"edges": edges, "acc": acc, "conf": avg_conf, "count": count}


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling: one scalar T minimising validation NLL."""
    def nll(t: np.ndarray) -> float:
        p = softmax(logits, float(np.exp(t[0])))
        return float(-np.log(p[np.arange(len(labels)), labels] + 1e-12).mean())

    return float(np.exp(minimize(nll, x0=[0.0], method="Nelder-Mead").x[0]))


def scores_to_grades(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Regression head: grade = number of thresholds the score exceeds."""
    return np.digitize(scores, np.sort(thresholds)).astype(int)


def optimise_thresholds(scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Find the 4 cut points that maximise validation QWK (Nelder-Mead from 0.5/1.5/2.5/3.5)."""
    res = minimize(lambda t: -qwk(labels, scores_to_grades(scores, t)), x0=[0.5, 1.5, 2.5, 3.5], method="Nelder-Mead")
    return np.sort(res.x)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, referable_score: np.ndarray) -> dict[str, Any]:
    """Grading metrics plus the binary referable-DR task (grade >= 2)."""
    ref_true, ref_pred = (y_true >= 2).astype(int), (y_pred >= 2).astype(int)
    tn, fp, fn, tp = confusion_matrix(ref_true, ref_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(ref_true, referable_score)
    return {
        "qwk": qwk(y_true, y_pred),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(5)), zero_division=0)),
        "referable_auc": float(roc_auc_score(ref_true, referable_score)),
        "referable_sensitivity": float(tp / max(tp + fn, 1)),
        "referable_specificity": float(tn / max(tn + fp, 1)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))),
        "report": classification_report(y_true, y_pred, labels=list(range(5)), output_dict=True, zero_division=0),
        "report_text": classification_report(y_true, y_pred, labels=list(range(5)), digits=4, zero_division=0),
        "referable_roc": {"fpr": fpr, "tpr": tpr},
    }
