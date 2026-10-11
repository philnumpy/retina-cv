# Results

All numbers below are read from the `metrics.json` files produced by `src/eval_seg.py`, `src/eval_cls.py`, `src/export_onnx.py` and `src/benchmark.py`. Where an experiment was run with several seeds the table shows mean ± standard deviation over seeds; the best mean per column is in bold. Regenerate with `python scripts/make_results.py`.

## Part A1: vessel segmentation on DRIVE (official 20-image test set)

16 training / 4 validation images (fixed split in `data/splits/`). Metrics are pooled over all FOV pixels of the 20 test images. The threshold is the Dice-optimal threshold on the 4 validation images.

| Experiment | Model | Seeds | Dice | IoU | Sens. | Spec. | Acc. | AUC-ROC | AUC-PR | Thr. | Density r | Dice, single pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`drive_unet_green`](results/segmentation/drive_unet_green/) | unet, green, dice_bce | 5 | **0.8197 ± 0.0054** | **0.6945 ± 0.0077** | **0.8468 ± 0.0045** | **0.9680 ± 0.0023** | **0.9526 ± 0.0018** | **0.9774 ± 0.0019** | **0.9058 ± 0.0050** | 0.3612 ± 0.0299 | **0.4446 ± 0.0290** | n/a |

Human reference: the second observer's annotations scored against the first observer on the same 20 test images give Dice **0.7891**, sensitivity 0.7746, specificity 0.9725, accuracy 0.9473 (computed by this script).

## Part A2: vessel segmentation on FIVES (official 200-image test set)

540 training / 60 validation images. The network runs at 1024×1024; probabilities are upsampled to the native 2048×2048 and scored against the original masks. FIVES has no FOV masks, so the FOV is derived by thresholding.

_no runs yet_

## Part B: diabetic retinopathy grading on APTOS 2019 (held-out test split, n = ?)

Stratified 70/15/15 split after duplicate removal (`data/splits/aptos_split.csv`). Referable DR = grade ≥ 2. ECE uses 15 confidence bins; it is not defined for the regression head.

_no runs yet_

## Part C: efficiency

Latency at batch size 1 (p50 / p95), throughput at batch 8, FP32. Segmentation models process one whole frame per forward pass. Hardware and library versions are recorded in each `benchmark.json`.

| Run | Model | Input (C×H×W) | Params (M) | GMACs | ONNX max abs diff | PyTorch CPU p50 / p95 ms | ORT CPU p50 / p95 ms | PyTorch GPU p50 / p95 ms | ORT GPU p50 / p95 ms | CPU img/s (PyTorch / ORT) | GPU img/s (PyTorch / ORT) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [`drive_unet_green`](results/efficiency/drive_unet_green/) | unet | 1×64×64 | 7.76 | 0.86 | 5.2e-06 | 9.3 / 14.3 | 3.8 / 5.9 | 2.9 / 3.5 | n/a / n/a | 263.2 / 399.7 | 2939.6 / n/a |
