# Results

All numbers below are read from the `metrics.json` files produced by `src/eval_seg.py`, `src/eval_cls.py`, `src/export_onnx.py` and `src/benchmark.py`. Where an experiment was run with several seeds the table shows mean ± standard deviation over seeds; the best mean per column is in bold. Regenerate with `python scripts/make_results.py`.

## Part A1: vessel segmentation on DRIVE (official 20-image test set)

16 training / 4 validation images (fixed split in `data/splits/`). Metrics are pooled over all FOV pixels of the 20 test images. The threshold is the Dice-optimal threshold on the 4 validation images.

| Experiment | Model | Seeds | Dice | IoU | Sens. | Spec. | Acc. | AUC-ROC | AUC-PR | Thr. | Density r | Dice, single pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`drive_attunet_green`](results/segmentation/drive_attunet_green/) | attention_unet, green, dice_bce | 5 | 0.8218 ± 0.0010 | 0.6975 ± 0.0014 | 0.8464 ± 0.0041 | **0.9689 ± 0.0012** | **0.9533 ± 0.0005** | 0.9784 ± 0.0005 | 0.9083 ± 0.0007 | 0.3953 ± 0.0640 | 0.4239 ± 0.0609 | 0.8220 ± 0.0017 |
| [`drive_r34unet_green`](results/segmentation/drive_r34unet_green/) | unet_resnet34, green, dice_bce | 3 | 0.8162 ± 0.0051 | 0.6895 ± 0.0072 | 0.8405 ± 0.0096 | 0.9680 ± 0.0002 | 0.9518 ± 0.0011 | 0.9755 ± 0.0024 | 0.9011 ± 0.0057 | 0.3984 ± 0.0565 | **0.4453 ± 0.0257** | 0.6883 ± 0.1169 |
| [`drive_r34unet_rgb`](results/segmentation/drive_r34unet_rgb/) | unet_resnet34, rgb, dice_bce | 3 | 0.8217 ± 0.0012 | 0.6974 ± 0.0018 | 0.8527 ± 0.0008 | 0.9675 ± 0.0006 | 0.9529 ± 0.0004 | 0.9779 ± 0.0004 | 0.9067 ± 0.0014 | 0.3623 ± 0.0267 | 0.4383 ± 0.0319 | 0.7009 ± 0.0457 |
| [`drive_unet_green`](results/segmentation/drive_unet_green/) | unet, green, dice_bce | 5 | 0.8197 ± 0.0054 | 0.6945 ± 0.0077 | 0.8468 ± 0.0045 | 0.9680 ± 0.0023 | 0.9526 ± 0.0018 | 0.9774 ± 0.0019 | 0.9058 ± 0.0050 | 0.3612 ± 0.0299 | 0.4446 ± 0.0290 | n/a |
| [`drive_unet_green_focal`](results/segmentation/drive_unet_green_focal/) | unet, green, dice_focal | 5 | 0.8167 ± 0.0045 | 0.6903 ± 0.0064 | 0.8556 ± 0.0049 | 0.9650 ± 0.0016 | 0.9511 ± 0.0014 | 0.9770 ± 0.0015 | 0.9036 ± 0.0043 | 0.4088 ± 0.0448 | 0.4346 ± 0.0277 | 0.8191 ± 0.0036 |
| [`drive_unet_rgb`](results/segmentation/drive_unet_rgb/) | unet, rgb, dice_bce | 5 | **0.8220 ± 0.0040** | **0.6978 ± 0.0057** | **0.8570 ± 0.0042** | 0.9667 ± 0.0013 | 0.9528 ± 0.0012 | **0.9787 ± 0.0012** | **0.9092 ± 0.0038** | 0.3788 ± 0.0171 | 0.4044 ± 0.0217 | **0.8234 ± 0.0036** |

Human reference: the second observer's annotations scored against the first observer on the same 20 test images give Dice **0.7891**, sensitivity 0.7746, specificity 0.9725, accuracy 0.9473 (computed by this script).

## Part A2: vessel segmentation on FIVES (official 200-image test set)

540 training / 60 validation images. The network runs at 1024×1024; probabilities are upsampled to the native 2048×2048 and scored against the original masks. FIVES has no FOV masks, so the FOV is derived by thresholding.

| Experiment | Model | Seeds | Dice | IoU | Sens. | Spec. | Acc. | AUC-ROC | AUC-PR | Thr. | Density r | Dice, single pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`fives_attunet_green`](results/segmentation/fives_attunet_green/) | attention_unet, green, dice_bce | 3 | 0.9131 ± 0.0001 | 0.8401 ± 0.0001 | **0.9041 ± 0.0002** | 0.9921 ± 0.0000 | 0.9838 ± 0.0000 | 0.9938 ± 0.0000 | 0.9674 ± 0.0000 | 0.5699 ± 0.0046 | 0.9846 ± 0.0004 | 0.9167 ± 0.0001 |
| [`fives_r34unet_rgb`](results/segmentation/fives_r34unet_rgb/) | unet_resnet34, rgb, dice_bce | 3 | **0.9168 ± 0.0002** | **0.8463 ± 0.0004** | 0.9040 ± 0.0004 | **0.9929 ± 0.0000** | **0.9846 ± 0.0000** | **0.9942 ± 0.0000** | **0.9690 ± 0.0001** | 0.5821 ± 0.0072 | **0.9851 ± 0.0006** | **0.9196 ± 0.0001** |
| [`fives_unet_green`](results/segmentation/fives_unet_green/) | unet, green, dice_bce | 3 | 0.9131 ± 0.0001 | 0.8401 ± 0.0002 | 0.9040 ± 0.0004 | 0.9921 ± 0.0000 | 0.9838 ± 0.0000 | 0.9938 ± 0.0000 | 0.9673 ± 0.0001 | 0.5688 ± 0.0019 | 0.9842 ± 0.0001 | 0.9167 ± 0.0001 |

## Part B: diabetic retinopathy grading on APTOS 2019 (held-out test split, n = 524)

Stratified 70/15/15 split after duplicate removal (`data/splits/aptos_split.csv`). Referable DR = grade ≥ 2. ECE uses 15 confidence bins; it is not defined for the regression head.

| Experiment | Model | Seeds | QWK | Acc. | Macro F1 | Ref. AUC | Ref. sens. | Ref. spec. | ECE | ECE (temp. scaled) |
|---|---|---|---|---|---|---|---|---|---|---|
| [`convnext_tiny_ben`](results/classification/convnext_tiny_ben/) | convnext_tiny, ben, classification | 3 | 0.9311 ± 0.0023 | **0.8550 ± 0.0071** | **0.7114 ± 0.0124** | **0.9879 ± 0.0023** | 0.9245 ± 0.0023 | 0.9616 ± 0.0103 | 0.0737 ± 0.0240 | 0.0409 ± 0.0122 |
| [`convnext_tiny_plain`](results/classification/convnext_tiny_plain/) | convnext_tiny, plain, classification | 3 | 0.9331 ± 0.0057 | 0.8454 ± 0.0124 | 0.7091 ± 0.0158 | 0.9876 ± 0.0011 | 0.9097 ± 0.0093 | 0.9595 ± 0.0044 | 0.0575 ± 0.0161 | 0.0451 ± 0.0046 |
| [`convnext_tiny_plain_reg`](results/classification/convnext_tiny_plain_reg/) | convnext_tiny, plain, regression | 3 | **0.9337 ± 0.0040** | 0.8181 ± 0.0106 | 0.6765 ± 0.0120 | 0.9873 ± 0.0019 | **0.9360 ± 0.0106** | 0.9470 ± 0.0127 | n/a | n/a |
| [`effnet_b0_ben`](results/classification/effnet_b0_ben/) | efficientnet_b0, ben, classification | 3 | 0.9174 ± 0.0044 | 0.8461 ± 0.0024 | 0.6969 ± 0.0108 | 0.9821 ± 0.0015 | 0.9048 ± 0.0046 | 0.9533 ± 0.0067 | 0.1224 ± 0.0043 | 0.0386 ± 0.0008 |
| [`effnet_b0_plain`](results/classification/effnet_b0_plain/) | efficientnet_b0, plain, classification | 3 | 0.9175 ± 0.0085 | 0.8473 ± 0.0081 | 0.7005 ± 0.0156 | 0.9848 ± 0.0013 | 0.8933 ± 0.0061 | 0.9574 ± 0.0015 | 0.1108 ± 0.0026 | **0.0351 ± 0.0075** |
| [`effnet_b0_plain_reg`](results/classification/effnet_b0_plain_reg/) | efficientnet_b0, plain, regression | 3 | 0.9229 ± 0.0062 | 0.7812 ± 0.0197 | 0.6268 ± 0.0287 | 0.9831 ± 0.0022 | 0.9261 ± 0.0040 | 0.9346 ± 0.0159 | n/a | n/a |
| [`resnet50_plain`](results/classification/resnet50_plain/) | resnet50, plain, classification | 3 | 0.9322 ± 0.0022 | 0.8416 ± 0.0062 | 0.7028 ± 0.0107 | 0.9862 ± 0.0005 | 0.8949 ± 0.0236 | **0.9616 ± 0.0089** | **0.0508 ± 0.0169** | 0.0425 ± 0.0055 |

## Part C: efficiency

Latency at batch size 1 (p50 / p95), throughput at batch 8, FP32. Segmentation models process one whole frame per forward pass. Hardware and library versions are recorded in each `benchmark.json`.

| Run | Model | Input (C×H×W) | Params (M) | GMACs | ONNX max abs diff | PyTorch CPU p50 / p95 ms | ORT CPU p50 / p95 ms | PyTorch GPU p50 / p95 ms | ORT GPU p50 / p95 ms | CPU img/s (PyTorch / ORT) | GPU img/s (PyTorch / ORT) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [`drive_unet_green`](results/efficiency/drive_unet_green/) | unet | 1×64×64 | 7.76 | 0.86 | 5.2e-06 | 9.3 / 14.3 | 3.8 / 5.9 | 2.9 / 3.5 | n/a / n/a | 263.2 / 399.7 | 2939.6 / n/a |
