#!/usr/bin/env bash
# All DRIVE experiments, 5 seeds each (about 1.5 min per run on an RTX A4000).
cd "$(dirname "$0")/.."
for c in drive_unet_green drive_unet_rgb drive_unet_green_focal drive_attunet_green drive_r34unet_green drive_r34unet_rgb; do
  scripts/run.sh seg configs/seg/$c.yaml 0 1 2 3 4
done
