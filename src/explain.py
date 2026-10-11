"""Grad-CAM heatmaps for a trained DR grader.

    python -m src.explain --config configs/cls/convnext_tiny_plain.yaml [--seed 0]

Saves a grid of test images (half correct, half incorrect) with the class-activation map of
the predicted grade overlaid. Grad-CAM weights the last conv feature maps by the gradient of
the predicted-class score, showing which retinal regions drove the decision.
"""
from __future__ import annotations

import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.cls_data import AptosDataset, denormalise, load_split
from src.models.classifier import build_classifier, gradcam_target_layer
from src.utils import plots
from src.utils.common import load_config, parse_args, run_dirs, seed_everything


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    res_dir, ckpt_dir = run_dirs("classification", cfg["run_name"])
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert cfg.get("head", "classification") == "classification", "Grad-CAM grid is implemented for the classification head"

    model = build_classifier(cfg, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=False)["model"])
    model.eval()

    split = load_split()
    test_df = split[split.split == "test"].reset_index(drop=True)
    saved = np.load(res_dir / "test_predictions.npz", allow_pickle=True)  # written by eval_cls
    y_true, y_pred = saved["y_true"], saved["y_pred"]
    assert list(saved["ids"]) == test_df["id_code"].tolist()

    # 8 correct (covering every grade where possible) + 8 incorrect, picked with a fixed seed
    rng = np.random.default_rng(0)
    correct = [i for g in range(5) for i in rng.permutation(np.where((y_true == g) & (y_pred == g))[0])[:2]][:8]
    wrong = rng.permutation(np.where(y_true != y_pred)[0])[:8].tolist()
    picks = correct + wrong

    ds = AptosDataset(test_df, cfg["variant"], cfg["image_size"], train=False)
    cam = GradCAM(model=model, target_layers=[gradcam_target_layer(model, cfg["arch"])])
    images, titles = [], []
    for i in picks:
        x, _ = ds[i]
        heat = cam(input_tensor=x[None].to(device), targets=[ClassifierOutputTarget(int(y_pred[i]))])[0]
        images.append(show_cam_on_image(denormalise(x).astype(np.float32), heat, use_rgb=True))
        titles.append(f"{'correct' if y_true[i] == y_pred[i] else 'WRONG'}: true {y_true[i]} | pred {y_pred[i]}")
    plots.image_grid(images, titles, 8, res_dir / "gradcam.png",
                     "Grad-CAM for the predicted grade (top row: correct, bottom row: incorrect)")
    print(f"saved {res_dir / 'gradcam.png'}")


if __name__ == "__main__":
    main()
