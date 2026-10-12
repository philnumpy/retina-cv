## What worked and what didn't

All statements below refer to the tables above. Models are ranked on validation; test numbers are quoted for reporting only.

### Segmentation

**Worked**

- **A plain U-Net is already at the level of the human second observer and above it.** On DRIVE every variant reaches test Dice 0.817–0.822 and AUC-ROC 0.975–0.979, against Dice 0.7891 for the second human observer scored on the same images.
- **Attention gates gave the best validation Dice and the most stable results.** Attention U-Net (green) has the highest validation Dice (0.7822) and the smallest seed-to-seed spread on test Dice (± 0.0010 vs. ± 0.0054 for the plain U-Net). The mean test Dice gain over the plain U-Net (0.8218 vs. 0.8197) is smaller than the plain U-Net's own standard deviation, so it is a stability gain rather than a clear accuracy gain.
- **Native resolution matters for the pretrained encoder on FIVES.** Training the ResNet34 U-Net on 512 px patches of the full 2048 px images raised test Dice from 0.9168 (1024 px working size, 3 seeds) to 0.9265 and sensitivity from 0.9040 to 0.9171 (1 seed). The native run is a single seed; the three 1024 px seeds differ by only ± 0.0002, so the gap is far outside seed noise.
- **Vessel density on FIVES is recovered almost exactly** (Pearson r 0.980–0.986 between predicted and manual density over 200 images).

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
- **Severe DR (grade 3) remains the weak class**: mean recall is 0.37–0.53 across the seven configurations on only 25 test images, with most errors going to the neighbouring grades 2 and 4. Macro F1 (≈ 0.70) is far below accuracy (≈ 0.85) for that reason.
- **All models overfit**: validation loss rises after roughly epoch 10–15 while validation QWK plateaus, which is why checkpoints are selected on QWK with early stopping.
- **Grad-CAM shows some attention on image borders and corners**, not only on lesions. APTOS images come from different cameras with different crops, so part of the measured performance is likely an acquisition shortcut.

### Efficiency

- ONNX exports reproduce PyTorch logits to within 1e-5 – 9e-5 on real test images, with identical decisions on all checked inputs except 1 pixel in 1.4 million for one segmentation model.
- On CPU, ONNX Runtime was faster than PyTorch for every model at batch size 1 (1.1× for ConvNeXt-Tiny up to 2.0× for the U-Net and 1.9× for ResNet50). On GPU the two are close for the U-Nets and ResNet50, while ONNX Runtime is about 1.4× faster for ConvNeXt-Tiny and 2.0× for EfficientNet-B0.
- EfficientNet-B0 needs 12× fewer GMACs than ConvNeXt-Tiny (2.0 vs. 23.3) and is 4× faster on CPU with ONNX Runtime (25.6 ms vs. 104.8 ms), for 1.6 points less QWK.
- **Measurement conditions:** GPU timings were taken on an otherwise idle RTX A4000. CPU timings used 8 threads on a 32-core machine while two training jobs were running, so they are conservative and noisier than the GPU numbers. The native-resolution FIVES model was not benchmarked; the FIVES row is the 1024 px model.
