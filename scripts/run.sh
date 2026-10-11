#!/usr/bin/env bash
# Train + evaluate one experiment for one or more seeds.
#   scripts/run.sh seg configs/seg/drive_unet_green.yaml 0 1 2
#   scripts/run.sh cls configs/cls/convnext_tiny_plain.yaml 0
set -eu
cd "$(dirname "$0")/.."
task=$1; config=$2; shift 2
for seed in "${@:-0}"; do
  python -m src.train_$task --config "$config" --seed "$seed"
  python -m src.eval_$task --config "$config" --seed "$seed"
done
