# Results

All numbers below are read from the `metrics.json` files produced by `src/eval_seg.py`, `src/eval_cls.py`, `src/export_onnx.py` and `src/benchmark.py`. Where an experiment was run with several seeds the table shows mean ± standard deviation over seeds; the best mean per column is in bold. Regenerate with `python scripts/make_results.py`.

## Part A1: vessel segmentation on DRIVE (official 20-image test set)

16 training / 4 validation images (fixed split in `data/splits/`). Metrics are pooled over all FOV pixels of the 20 test images. The threshold is the Dice-optimal threshold on the 4 validation images.

| Experiment | Model | Seeds | Val Dice | Dice | IoU | Sens. | Spec. | Acc. | AUC-ROC | AUC-PR | Thr. | Density r | Dice, single pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`drive_attunet_green`](results/segmentation/drive_attunet_green/) | attention_unet, green, dice_bce | 5 | **0.7822 ± 0.0043** | 0.8218 ± 0.0010 | 0.6975 ± 0.0014 | 0.8464 ± 0.0041 | **0.9689 ± 0.0012** | **0.9533 ± 0.0005** | 0.9784 ± 0.0005 | 0.9083 ± 0.0007 | 0.3953 ± 0.0640 | 0.4239 ± 0.0609 | 0.8220 ± 0.0017 |
| [`drive_r34unet_green`](results/segmentation/drive_r34unet_green/) | unet_resnet34, green, dice_bce | 5 | 0.7761 ± 0.0007 | 0.8177 ± 0.0045 | 0.6917 ± 0.0063 | 0.8409 ± 0.0091 | 0.9685 ± 0.0010 | 0.9523 ± 0.0010 | 0.9758 ± 0.0021 | 0.9022 ± 0.0051 | 0.3888 ± 0.0505 | 0.4368 ± 0.0323 | 0.6080 ± 0.1356 |
| [`drive_r34unet_rgb`](results/segmentation/drive_r34unet_rgb/) | unet_resnet34, rgb, dice_bce | 5 | 0.7761 ± 0.0011 | 0.8217 ± 0.0011 | 0.6973 ± 0.0016 | 0.8515 ± 0.0018 | 0.9677 ± 0.0007 | 0.9529 ± 0.0004 | 0.9778 ± 0.0006 | 0.9068 ± 0.0015 | 0.3742 ± 0.0253 | 0.4446 ± 0.0267 | 0.7129 ± 0.0667 |
| [`drive_unet_green`](results/segmentation/drive_unet_green/) | unet, green, dice_bce | 5 | 0.7808 ± 0.0039 | 0.8197 ± 0.0054 | 0.6945 ± 0.0077 | 0.8468 ± 0.0045 | 0.9680 ± 0.0023 | 0.9526 ± 0.0018 | 0.9774 ± 0.0019 | 0.9058 ± 0.0050 | 0.3612 ± 0.0299 | **0.4446 ± 0.0290** | 0.8206 ± 0.0051 |
| [`drive_unet_green_focal`](results/segmentation/drive_unet_green_focal/) | unet, green, dice_focal | 5 | 0.7783 ± 0.0035 | 0.8167 ± 0.0045 | 0.6903 ± 0.0064 | 0.8556 ± 0.0049 | 0.9650 ± 0.0016 | 0.9511 ± 0.0014 | 0.9770 ± 0.0015 | 0.9036 ± 0.0043 | 0.4088 ± 0.0448 | 0.4346 ± 0.0277 | 0.8191 ± 0.0036 |
| [`drive_unet_rgb`](results/segmentation/drive_unet_rgb/) | unet, rgb, dice_bce | 5 | 0.7754 ± 0.0021 | **0.8220 ± 0.0040** | **0.6978 ± 0.0057** | **0.8570 ± 0.0042** | 0.9667 ± 0.0013 | 0.9528 ± 0.0012 | **0.9787 ± 0.0012** | **0.9092 ± 0.0038** | 0.3788 ± 0.0171 | 0.4044 ± 0.0217 | **0.8234 ± 0.0036** |

Human reference: the second observer's annotations scored against the first observer on the same 20 test images give Dice **0.7891**, sensitivity 0.7746, specificity 0.9725, accuracy 0.9473 (computed by this script).

## Part A2: vessel segmentation on FIVES (official 200-image test set)

540 training / 60 validation images. The network runs at 1024×1024; probabilities are upsampled to the native 2048×2048 and scored against the original masks. FIVES has no FOV masks, so the FOV is derived by thresholding.

| Experiment | Model | Seeds | Val Dice | Dice | IoU | Sens. | Spec. | Acc. | AUC-ROC | AUC-PR | Thr. | Density r | Dice, single pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| [`fives_attunet_green`](results/segmentation/fives_attunet_green/) | attention_unet, green, dice_bce | 3 | 0.9342 ± 0.0001 | 0.9131 ± 0.0001 | 0.8401 ± 0.0001 | 0.9041 ± 0.0002 | 0.9921 ± 0.0000 | 0.9838 ± 0.0000 | 0.9938 ± 0.0000 | 0.9674 ± 0.0000 | 0.5699 ± 0.0046 | 0.9846 ± 0.0004 | 0.9167 ± 0.0001 |
| [`fives_r34unet_rgb`](results/segmentation/fives_r34unet_rgb/) | unet_resnet34, rgb, dice_bce | 3 | 0.9360 ± 0.0002 | 0.9168 ± 0.0002 | 0.8463 ± 0.0004 | 0.9040 ± 0.0004 | 0.9929 ± 0.0000 | 0.9846 ± 0.0000 | 0.9942 ± 0.0000 | 0.9690 ± 0.0001 | 0.5821 ± 0.0072 | 0.9851 ± 0.0006 | 0.9196 ± 0.0001 |
| [`fives_r34unet_rgb_native`](results/segmentation/fives_r34unet_rgb_native/) | unet_resnet34, rgb, dice_bce | 1 | **0.9459** | **0.9265** | **0.8631** | **0.9171** | **0.9935** | **0.9863** | **0.9947** | **0.9732** | 0.4666 | **0.9864** | **0.9278** |
| [`fives_unet_green`](results/segmentation/fives_unet_green/) | unet, green, dice_bce | 3 | 0.9340 ± 0.0001 | 0.9131 ± 0.0001 | 0.8401 ± 0.0002 | 0.9040 ± 0.0004 | 0.9921 ± 0.0000 | 0.9838 ± 0.0000 | 0.9938 ± 0.0000 | 0.9673 ± 0.0001 | 0.5688 ± 0.0019 | 0.9842 ± 0.0001 | 0.9167 ± 0.0001 |
| [`fives_unet_green_native`](results/segmentation/fives_unet_green_native/) | unet, green, dice_bce | 1 | 0.9359 | 0.9129 | 0.8398 | 0.9054 | 0.9919 | 0.9837 | 0.9932 | 0.9658 | 0.4539 | 0.9804 | 0.9145 |

## Part B: diabetic retinopathy grading on APTOS 2019 (held-out test split, n = 524)

Stratified 70/15/15 split after duplicate removal (`data/splits/aptos_split.csv`). Referable DR = grade ≥ 2. ECE uses 15 confidence bins; it is not defined for the regression head.

| Experiment | Model | Seeds | Val QWK | QWK | Acc. | Macro F1 | Ref. AUC | Ref. sens. | Ref. spec. | ECE | ECE (temp. scaled) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [`convnext_tiny_ben`](results/classification/convnext_tiny_ben/) | convnext_tiny, ben, classification | 3 | 0.9162 ± 0.0117 | 0.9311 ± 0.0023 | **0.8550 ± 0.0071** | **0.7114 ± 0.0124** | **0.9879 ± 0.0023** | 0.9245 ± 0.0023 | 0.9616 ± 0.0103 | 0.0737 ± 0.0240 | 0.0409 ± 0.0122 |
| [`convnext_tiny_plain`](results/classification/convnext_tiny_plain/) | convnext_tiny, plain, classification | 3 | 0.9148 ± 0.0081 | 0.9331 ± 0.0057 | 0.8454 ± 0.0124 | 0.7091 ± 0.0158 | 0.9876 ± 0.0011 | 0.9097 ± 0.0093 | 0.9595 ± 0.0044 | 0.0575 ± 0.0161 | 0.0451 ± 0.0046 |
| [`convnext_tiny_plain_reg`](results/classification/convnext_tiny_plain_reg/) | convnext_tiny, plain, regression | 3 | **0.9221 ± 0.0033** | **0.9337 ± 0.0040** | 0.8181 ± 0.0106 | 0.6765 ± 0.0120 | 0.9873 ± 0.0019 | **0.9360 ± 0.0106** | 0.9470 ± 0.0127 | n/a | n/a |
| [`effnet_b0_ben`](results/classification/effnet_b0_ben/) | efficientnet_b0, ben, classification | 3 | 0.9157 ± 0.0035 | 0.9174 ± 0.0044 | 0.8461 ± 0.0024 | 0.6969 ± 0.0108 | 0.9821 ± 0.0015 | 0.9048 ± 0.0046 | 0.9533 ± 0.0067 | 0.1224 ± 0.0043 | 0.0386 ± 0.0008 |
| [`effnet_b0_plain`](results/classification/effnet_b0_plain/) | efficientnet_b0, plain, classification | 3 | 0.9003 ± 0.0029 | 0.9175 ± 0.0085 | 0.8473 ± 0.0081 | 0.7005 ± 0.0156 | 0.9848 ± 0.0013 | 0.8933 ± 0.0061 | 0.9574 ± 0.0015 | 0.1108 ± 0.0026 | **0.0351 ± 0.0075** |
| [`effnet_b0_plain_reg`](results/classification/effnet_b0_plain_reg/) | efficientnet_b0, plain, regression | 3 | 0.9071 ± 0.0113 | 0.9229 ± 0.0062 | 0.7812 ± 0.0197 | 0.6268 ± 0.0287 | 0.9831 ± 0.0022 | 0.9261 ± 0.0040 | 0.9346 ± 0.0159 | n/a | n/a |
| [`resnet50_plain`](results/classification/resnet50_plain/) | resnet50, plain, classification | 3 | 0.8991 ± 0.0035 | 0.9322 ± 0.0022 | 0.8416 ± 0.0062 | 0.7028 ± 0.0107 | 0.9862 ± 0.0005 | 0.8949 ± 0.0236 | **0.9616 ± 0.0089** | **0.0508 ± 0.0169** | 0.0425 ± 0.0055 |

## Part C: efficiency

Latency at batch size 1 (p50 / p95), throughput at batch 8, FP32. Segmentation models process one whole frame per forward pass. Hardware and library versions are recorded in each `benchmark.json`.

| Run | Model | Input (C×H×W) | Params (M) | GMACs | ONNX max abs logit diff (real inputs) | PyTorch CPU p50 / p95 ms | ORT CPU p50 / p95 ms | PyTorch GPU p50 / p95 ms | ORT GPU p50 / p95 ms | CPU img/s (PyTorch / ORT) | GPU img/s (PyTorch / ORT) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| [`convnext_tiny_plain_reg`](results/efficiency/convnext_tiny_plain_reg/) | convnext_tiny | 3×512×512 | 27.82 | 23.27 | 1.1e-05 | 116.8 / 123.6 | 104.8 / 113.7 | 6.9 / 6.9 | 5.0 / 5.0 | 6.3 / 8.6 | 163.0 / 229.0 |
| [`drive_attunet_green`](results/efficiency/drive_attunet_green/) | attention_unet | 1×608×576 | 7.85 | 74.78 | 1.6e-05 | 454.2 / 478.0 | 237.6 / 338.1 | 11.4 / 11.4 | 10.2 / 10.4 | 1.6 / 2.7 | 105.3 / 106.0 |
| [`drive_unet_rgb`](results/efficiency/drive_unet_rgb/) | unet | 3×608×576 | 7.76 | 73.45 | 1.3e-05 | 394.0 / 414.6 | 199.5 / 205.2 | 9.5 / 9.5 | 8.8 / 9.0 | 2.0 / 3.1 | 121.8 / 122.7 |
| [`effnet_b0_plain`](results/efficiency/effnet_b0_plain/) | efficientnet_b0 | 3×512×512 | 4.01 | 2.01 | 8.6e-05 | 42.1 / 48.5 | 25.6 / 25.9 | 4.4 / 4.6 | 2.2 / 2.2 | 12.3 / 34.3 | 545.6 / 558.2 |
| [`fives_r34unet_rgb`](results/efficiency/fives_r34unet_rgb/) | unet_resnet34 | 3×1024×1024 | 24.44 | 125.60 | 4.1e-05 | 610.2 / 645.1 | 349.7 / 361.7 | 17.2 / 17.9 | 16.1 / 16.3 | 1.3 / 2.0 | 64.5 / 64.4 |
| [`resnet50_plain`](results/efficiency/resnet50_plain/) | resnet50 | 3×512×512 | 23.52 | 21.59 | 2.4e-05 | 112.6 / 120.0 | 58.6 / 62.7 | 4.0 / 4.1 | 3.9 / 4.0 | 5.2 / 16.1 | 290.3 / 291.9 |

## What worked and what didn't

All statements below refer to the tables above. Models are ranked on validation; test numbers are quoted for reporting only.

### Segmentation

**Worked**

- **A plain U-Net is already at the level of the human second observer and above it.** On DRIVE every variant reaches test Dice 0.817–0.822 and AUC-ROC 0.975–0.979, against Dice 0.7891 for the second human observer scored on the same images.
- **Attention gates gave the best validation Dice and the most stable results.** Attention U-Net (green) has the highest validation Dice (0.7822) and the smallest seed-to-seed spread on test Dice (± 0.0010 vs. ± 0.0054 for the plain U-Net). The mean test Dice gain over the plain U-Net (0.8218 vs. 0.8197) is smaller than the plain U-Net's own standard deviation, so it is a stability gain rather than a clear accuracy gain.
- **Native resolution matters for the pretrained encoder on FIVES.** Training the ResNet34 U-Net on 512 px patches of the full 2048 px images raised test Dice from 0.9168 (1024 px working size, 3 seeds) to 0.9265 and sensitivity from 0.9040 to 0.9171 (1 seed). The native run is a single seed; the three 1024 px seeds differ by only ± 0.0002, so the gap is far outside seed noise.
- **Vessel density on FIVES is recovered almost exactly** (Pearson r 0.984–0.986 between predicted and manual density over 200 images).

**Did not work, or made no difference**

- **Focal loss did not beat Dice + BCE** (DRIVE Dice 0.8167 vs. 0.8197). It shifted the operating point towards higher sensitivity (0.8556 vs. 0.8468) at lower specificity, without improving AUC.
- **RGB vs. green input is a wash on DRIVE.** RGB has the best test Dice (0.8220) but the lowest validation Dice of the U-Net variants (0.7754), so it would not have been selected. With 4 validation images this choice is noisy.
- **The ImageNet-pretrained ResNet34 encoder did not help on DRIVE** (Dice 0.8177 green, 0.8217 RGB) despite 3× the parameters. It did help on FIVES (0.9168 vs. 0.9131 at 1024 px).
- **The ResNet34 U-Net trained on 64 px patches cannot be run on a whole frame in one pass.** Its single-pass Dice collapses to 0.61–0.71, while the plain and Attention U-Nets keep their accuracy (0.8206–0.8234). With 32× downsampling a 64 px patch has a 2×2 bottleneck, so the encoder never sees realistic context during training. For this model sliding-window inference is required; when it is trained on 256 px or 512 px patches (FIVES) the single pass works (0.9196–0.9278). One such checkpoint also overflowed to NaN in half precision, so single-pass inference runs in FP32.
- **Native resolution did not help the plain U-Net on FIVES** (0.9129 native vs. 0.9131 at 1024 px).
- **Vessel density is only weakly reproduced on DRIVE** (r 0.40–0.44). All 20 test images have similar density (roughly 0.10–0.15), so small per-image threshold effects dominate; per-image density from these models should not be used as a biomarker on that evidence.
- **Validation Dice on DRIVE (≈ 0.78) is lower than test Dice (≈ 0.82).** The four validation images come from the training half of DRIVE; the gap is consistent across all models and reflects the image subset, not tuning on test.

### DR grading

**Worked**

- **ConvNeXt-Tiny and ResNet50 are clearly better than EfficientNet-B0 on QWK** (0.931–0.934 vs. 0.917–0.923), with ConvNeXt-Tiny best on validation.
- **Referable-DR screening performance is strong for every model** (AUC 0.982–0.988). The regression head gives the highest referable sensitivity (0.9360) at specificity 0.9470.
- **Temperature scaling fixes most of the miscalibration.** EfficientNet-B0 is over-confident (ECE 0.11–0.12) and drops to 0.035–0.039 after scaling with one parameter fitted on validation; ConvNeXt-Tiny goes from 0.057–0.074 to 0.041–0.045.
- **De-duplicating APTOS before splitting** removed 266 duplicate images (77 with conflicting grades) that would otherwise have leaked across the split.

**Did not work, or made no difference**

- **Ben Graham preprocessing did not improve QWK** (ConvNeXt-Tiny 0.9311 with vs. 0.9331 without; EfficientNet-B0 0.9174 vs. 0.9175). It gave slightly higher accuracy for ConvNeXt-Tiny (0.8550 vs. 0.8454), within about one standard deviation.
- **The regression head trades accuracy for ordinal consistency.** QWK is marginally higher (0.9337 vs. 0.9331) but accuracy (0.8181 vs. 0.8454) and macro F1 (0.6765 vs. 0.7091) are lower. Optimising the four cut points on validation gave no measurable gain over fixed 0.5 / 1.5 / 2.5 / 3.5 cut points (per-seed QWK with default cut points is stored as `qwk_default_thresholds`).
- **Severe DR (grade 3) remains the weak class**: recall is 0.37–0.47 on only 25 test images, with most errors going to the neighbouring grades 2 and 4. Macro F1 (≈ 0.70) is far below accuracy (≈ 0.85) for that reason.
- **All models overfit**: validation loss rises after roughly epoch 10–15 while validation QWK plateaus, which is why checkpoints are selected on QWK with early stopping.
- **Grad-CAM shows some attention on image borders and corners**, not only on lesions. APTOS images come from different cameras with different crops, so part of the measured performance is likely an acquisition shortcut.

### Efficiency

- ONNX exports reproduce PyTorch logits to within 1e-5 – 9e-5 on real test images, with identical decisions on all checked inputs except 3 in 4.2 million pixels for one segmentation model.
- On CPU, ONNX Runtime was faster than PyTorch for every model at batch size 1 (1.1× for ConvNeXt-Tiny up to 2.0× for the U-Net and 1.9× for ResNet50). On GPU the two are close for the U-Nets and ResNet50, while ONNX Runtime is about 1.4× faster for ConvNeXt-Tiny and 2.0× for EfficientNet-B0.
- EfficientNet-B0 needs 12× fewer GMACs than ConvNeXt-Tiny (2.0 vs. 23.3) and is 4× faster on CPU with ONNX Runtime (25.6 ms vs. 104.8 ms), for 1.6 points less QWK.
- **Measurement conditions:** GPU timings were taken on an otherwise idle RTX A4000. CPU timings used 8 threads on a 32-core machine while two training jobs were running, so they are conservative and noisier than the GPU numbers. The native-resolution FIVES model was not benchmarked; the FIVES row is the 1024 px model.
