#!/usr/bin/env bash
# Train + evaluate the listed DR-grading configs for seeds 0 1 2; Grad-CAM for seed 0 of classification heads.
#   CUDA_VISIBLE_DEVICES=0 scripts/run_cls_queue.sh convnext_tiny_plain convnext_tiny_ben
cd "$(dirname "$0")/.."
for c in "$@"; do
  scripts/run.sh cls configs/cls/$c.yaml 0 1 2
  grep -q "head: regression" configs/cls/$c.yaml || python -m src.explain --config configs/cls/$c.yaml
done
