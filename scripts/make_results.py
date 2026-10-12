"""Build RESULTS.md from the metrics.json files written by the evaluation scripts.

    python scripts/make_results.py

Every number in the tables is read from results/**; nothing is typed in by hand.
The hand-written discussion lives in results/discussion.md and is appended verbatim.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"


def load_runs(task: str) -> dict[str, list[dict]]:
    """Group runs by experiment name (seeds are folders named <exp>__seed<k>)."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for f in sorted((RES / task).glob("*/metrics.json")):
        if f.parent.name.startswith("smoke_"):
            continue
        groups[f.parent.name.split("__seed")[0]].append(json.loads(f.read_text()))
    return groups


def cell(runs: list[dict], key: str, digits: int = 4) -> tuple[float, str]:
    vals = [r[key] for r in runs if r.get(key) is not None]
    if not vals:
        return float("nan"), "n/a"
    mean = float(np.mean(vals))
    return mean, f"{mean:.{digits}f}" + (f" ± {np.std(vals):.{digits}f}" if len(vals) > 1 else "")


def table(groups: dict[str, list[dict]], columns: list[tuple[str, str]], describe, higher_better: set[str],
          lower_better: set[str] = frozenset()) -> str:
    """Markdown table: one row per experiment, mean ± std over seeds, best mean per column in bold."""
    if not groups:
        return "_no runs yet_\n"
    cells = {name: {k: cell(runs, k) for k, _ in columns} for name, runs in groups.items()}
    best = {}
    for k, _ in columns:
        vals = {n: c[k][0] for n, c in cells.items() if not np.isnan(c[k][0])}
        if vals and k in higher_better:
            best[k] = max(vals, key=vals.get)
        elif vals and k in lower_better:
            best[k] = min(vals, key=vals.get)
    head = "| Experiment | Model | Seeds | " + " | ".join(h for _, h in columns) + " |"
    lines = [head, "|" + "---|" * (len(columns) + 3)]
    for name, runs in groups.items():
        row = [f"**{cells[name][k][1]}**" if best.get(k) == name else cells[name][k][1] for k, _ in columns]
        lines.append(f"| [`{name}`](results/{TASK_DIR[0]}/{name}/) | {describe(runs[0])} | {len(runs)} | " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


TASK_DIR = ["segmentation"]


def observer2_drive() -> str:
    """Agreement of the second human observer with the reference annotator on the DRIVE test set."""
    base = ROOT / "data" / "processed" / "drive" / "test"
    if not (base / "masks_observer2").exists():
        return ""
    tp = fp = fn = tn = 0
    for p in sorted((base / "masks_observer2").glob("*.png")):
        obs = cv2.imread(str(p), 0) > 127
        ref = cv2.imread(str(base / "masks" / p.name), 0) > 127
        fov = cv2.imread(str(base / "fov" / p.name), 0) > 127
        tp += (obs & ref & fov).sum(); fp += (obs & ~ref & fov).sum()
        fn += (~obs & ref & fov).sum(); tn += (~obs & ~ref & fov).sum()
    if tp == 0:
        return ""
    return (f"Human reference: the second observer's annotations scored against the first observer on the same 20 test "
            f"images give Dice **{2 * tp / (2 * tp + fp + fn):.4f}**, sensitivity {tp / (tp + fn):.4f}, "
            f"specificity {tn / (tn + fp):.4f}, accuracy {(tp + tn) / (tp + tn + fp + fn):.4f} (computed by this script).\n")


def efficiency_table() -> str:
    rows = []
    for f in sorted((RES / "efficiency").glob("*/benchmark.json")):
        b, e = json.loads(f.read_text()), json.loads((f.parent / "export.json").read_text())
        get = lambda backend, dev, k: next((f"{r[k]:.1f}" for r in b["results"] if r["backend"] == backend and r["device"] == dev), "n/a")
        shape = "×".join(str(s) for s in e["input_shape"])
        rows.append(f"| [`{b['run']}`](results/efficiency/{b['run']}/) | {b['arch']} | {shape} | {e['parameters_M']:.2f} | {e['gmacs']:.2f} | "
                    f"{e['onnx_max_abs_diff']:.1e} | {get('PyTorch', 'cpu', 'p50_ms')} / {get('PyTorch', 'cpu', 'p95_ms')} | "
                    f"{get('ONNX Runtime', 'cpu', 'p50_ms')} / {get('ONNX Runtime', 'cpu', 'p95_ms')} | "
                    f"{get('PyTorch', 'cuda', 'p50_ms')} / {get('PyTorch', 'cuda', 'p95_ms')} | "
                    f"{get('ONNX Runtime', 'cuda', 'p50_ms')} / {get('ONNX Runtime', 'cuda', 'p95_ms')} | "
                    f"{get('PyTorch', 'cpu', 'throughput_img_s')} / {get('ONNX Runtime', 'cpu', 'throughput_img_s')} | "
                    f"{get('PyTorch', 'cuda', 'throughput_img_s')} / {get('ONNX Runtime', 'cuda', 'throughput_img_s')} |")
    if not rows:
        return "_no benchmarks yet_\n"
    head = ("| Run | Model | Input (C×H×W) | Params (M) | GMACs | ONNX max abs logit diff (real inputs) | PyTorch CPU p50 / p95 ms | ORT CPU p50 / p95 ms | "
            "PyTorch GPU p50 / p95 ms | ORT GPU p50 / p95 ms | CPU img/s (PyTorch / ORT) | GPU img/s (PyTorch / ORT) |")
    return "\n".join([head, "|" + "---|" * 12] + rows) + "\n"


def best_by_val(groups: dict[str, list[dict]], key: str) -> tuple[str, list[dict]]:
    """Pick the experiment with the highest mean validation score (never the test score)."""
    name = max(groups, key=lambda n: np.mean([r[key] for r in groups[n]]))
    return name, groups[name]


def readme_summary(seg: dict[str, list[dict]], cls: dict[str, list[dict]]) -> str:
    """Headline table for the README: per benchmark, the model selected on validation."""
    lines = ["| Task | Selected model (by validation) | Seeds | Test result |", "|---|---|---|---|"]
    for ds, label in (("drive", "Vessel segmentation, DRIVE (20 test images)"), ("fives", "Vessel segmentation, FIVES (200 test images)")):
        sub = {k: v for k, v in seg.items() if v[0]["dataset"] == ds}
        if sub:
            name, runs = best_by_val(sub, "val_dice_at_threshold")
            c = lambda k: cell(runs, k)[1]
            lines.append(f"| {label} | `{name}` | {len(runs)} | Dice {c('dice')}, IoU {c('iou')}, sensitivity {c('sensitivity')}, "
                         f"specificity {c('specificity')}, AUC-ROC {c('auc_roc')}, AUC-PR {c('auc_pr')} |")
    if cls:
        name, runs = best_by_val(cls, "val_qwk")
        c = lambda k: cell(runs, k)[1]
        lines.append(f"| DR grading, APTOS 2019 ({runs[0]['n_test']} test images) | `{name}` | {len(runs)} | QWK {c('qwk')}, accuracy {c('accuracy')}, "
                     f"macro F1 {c('macro_f1')}; referable DR: AUC {c('referable_auc')}, sensitivity {c('referable_sensitivity')}, "
                     f"specificity {c('referable_specificity')} |")
        heads = {k: v for k, v in cls.items() if v[0]["head"] == "classification"}
        name, runs = best_by_val(heads, "val_qwk")
        c = lambda k: cell(runs, k)[1]
        lines.append(f"| Same, best 5-class softmax model | `{name}` | {len(runs)} | QWK {c('qwk')}, accuracy {c('accuracy')}, macro F1 {c('macro_f1')}, "
                     f"referable AUC {c('referable_auc')}, ECE {c('ece')} → {c('ece_temperature_scaled')} after temperature scaling |")
    return "\n".join(lines) + "\n"


def main() -> None:
    seg, cls = load_runs("segmentation"), load_runs("classification")
    seg_cols = [("val_dice_at_threshold", "Val Dice"), ("dice", "Dice"), ("iou", "IoU"), ("sensitivity", "Sens."), ("specificity", "Spec."), ("accuracy", "Acc."),
                ("auc_roc", "AUC-ROC"), ("auc_pr", "AUC-PR"), ("threshold", "Thr."), ("density_pearson_r", "Density r"),
                ("full_frame_dice", "Dice, single pass")]
    seg_hi = {"dice", "iou", "sensitivity", "specificity", "accuracy", "auc_roc", "auc_pr", "density_pearson_r", "full_frame_dice",
              "val_dice_at_threshold"}
    seg_desc = lambda r: f"{r['arch']}, {r['input']}, {r['loss']}"
    cls_cols = [("val_qwk", "Val QWK"), ("qwk", "QWK"), ("accuracy", "Acc."), ("macro_f1", "Macro F1"), ("referable_auc", "Ref. AUC"),
                ("referable_sensitivity", "Ref. sens."), ("referable_specificity", "Ref. spec."), ("ece", "ECE"),
                ("ece_temperature_scaled", "ECE (temp. scaled)")]
    cls_hi = {"qwk", "accuracy", "macro_f1", "referable_auc", "referable_sensitivity", "referable_specificity", "val_qwk"}
    cls_desc = lambda r: f"{r['arch']}, {r['variant']}, {r['head']}"

    out = ["# Results", "",
           "All numbers below are read from the `metrics.json` files produced by `src/eval_seg.py`, `src/eval_cls.py`, "
           "`src/export_onnx.py` and `src/benchmark.py`. Where an experiment was run with several seeds the table shows "
           "mean ± standard deviation over seeds; the best mean per column is in bold. "
           "Regenerate with `python scripts/make_results.py`.", ""]
    for ds, title, note in (
        ("drive", "Part A1: vessel segmentation on DRIVE (official 20-image test set)",
         "16 training / 4 validation images (fixed split in `data/splits/`). Metrics are pooled over all FOV pixels of the 20 "
         "test images. The threshold is the Dice-optimal threshold on the 4 validation images."),
        ("fives", "Part A2: vessel segmentation on FIVES (official 200-image test set)",
         "540 training / 60 validation images. The network runs at 1024×1024; probabilities are upsampled to the native "
         "2048×2048 and scored against the original masks. FIVES has no FOV masks, so the FOV is derived by thresholding.")):
        sub = {k: v for k, v in seg.items() if v[0]["dataset"] == ds}
        TASK_DIR[0] = "segmentation"
        out += [f"## {title}", "", note, "", table(sub, seg_cols, seg_desc, seg_hi)]
        if ds == "drive":
            out += [observer2_drive()]
    TASK_DIR[0] = "classification"
    n_test = next(iter(cls.values()))[0]["n_test"] if cls else "?"
    out += [f"## Part B: diabetic retinopathy grading on APTOS 2019 (held-out test split, n = {n_test})", "",
            "Stratified 70/15/15 split after duplicate removal (`data/splits/aptos_split.csv`). Referable DR = grade ≥ 2. "
            "ECE uses 15 confidence bins; it is not defined for the regression head.", "",
            table(cls, cls_cols, cls_desc, cls_hi, {"ece", "ece_temperature_scaled"})]
    out += ["## Part C: efficiency", "",
            "Latency at batch size 1 (p50 / p95), throughput at batch 8, FP32. Segmentation models process one whole frame "
            "per forward pass. Hardware and library versions are recorded in each `benchmark.json`.", "", efficiency_table()]
    disc = RES / "discussion.md"
    if disc.exists():
        out += [disc.read_text()]
    (ROOT / "RESULTS.md").write_text("\n".join(out))
    readme = ROOT / "README.md"
    if readme.exists() and "<!-- RESULTS:START -->" in readme.read_text():
        text = readme.read_text()
        a, b = text.index("<!-- RESULTS:START -->") + len("<!-- RESULTS:START -->"), text.index("<!-- RESULTS:END -->")
        readme.write_text(text[:a] + "\n" + readme_summary(seg, cls) + text[b:])
    print("\n".join(out))


if __name__ == "__main__":
    main()
