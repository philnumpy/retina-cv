"""Train a vessel-segmentation model on random patches; validate on full images.

    python -m src.train_seg --config configs/seg/drive_unet_green.yaml [--seed 1] [--smoke]

Model selection (best checkpoint, early stopping) uses the validation images only.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.seg_data import PatchDataset, Sample, describe, load_samples, sliding_window_predict
from src.losses import seg_loss
from src.models.unet import build_seg_model, count_parameters
from src.utils import plots
from src.utils.common import (get_logger, load_config, log_run, parse_args, run_dirs, save_json, seed_everything,
                              seed_worker)
from src.utils.metrics import PixelHistogram


@torch.no_grad()
def validate(model: torch.nn.Module, samples: list[Sample], cfg: dict[str, Any], device: torch.device) -> dict[str, float]:
    """Full-image sliding-window inference on validation images: loss + pooled Dice at 0.5."""
    hist, losses = PixelHistogram(), []
    for s in samples:
        prob = sliding_window_predict(model, s.image, cfg["patch"], cfg["patch"] // 2, device, cfg["eval_batch"])
        hist.update(prob, s.mask, s.fov)
        p = torch.from_numpy(prob).clamp(1e-6, 1 - 1e-6)
        logits = torch.log(p / (1 - p))[None, None]
        t, f = torch.from_numpy(s.mask).float()[None, None], torch.from_numpy(s.fov).float()[None, None]
        losses.append(seg_loss(cfg["loss"], logits, t, f).item())
    return {"val_loss": float(np.mean(losses)), "val_dice": hist.summary(0.5)["dice"]}


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    smoke = cfg.get("smoke", False)
    res_dir, ckpt_dir = run_dirs("segmentation", cfg["run_name"])
    log = get_logger(res_dir)
    log_run(res_dir, cfg)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    limit = 2 if smoke else None
    train = load_samples(cfg["dataset"], "train", cfg["input"], cfg.get("work_size"), limit)
    val = load_samples(cfg["dataset"], "val", cfg["input"], cfg.get("work_size"), limit)
    log.info(f"train {describe(train)} | val {describe(val)}")

    n_patches = 3 * cfg["batch_size"] if smoke else cfg["patches_per_epoch"]
    workers = cfg.get("workers", 8)
    loader = DataLoader(PatchDataset(train, cfg["patch"], n_patches), batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=workers, worker_init_fn=seed_worker, drop_last=True,
                        persistent_workers=workers > 0, pin_memory=True)
    model = build_seg_model(cfg).to(device)
    log.info(f"{cfg['arch']} | {count_parameters(model) / 1e6:.2f} M parameters | device {device}")

    epochs = 1 if smoke else cfg["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=epochs * len(loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best, bad_epochs, t0 = -1.0, 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, inter, denom = 0.0, 0.0, 0.0
        for img, msk, fov in loader:
            img, msk, fov = img.to(device, non_blocking=True), msk.to(device), fov.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(img)
            loss = seg_loss(cfg["loss"], logits, msk, fov)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item()
            pred = (logits.detach() > 0).float() * fov
            inter += (pred * msk).sum().item()
            denom += pred.sum().item() + (msk * fov).sum().item()
        row = {"epoch": epoch, "train_loss": loss_sum / len(loader), "train_dice": 2 * inter / max(denom, 1.0),
               **validate(model, val, cfg, device), "lr": sched.get_last_lr()[0]}
        history.append(row)
        improved = row["val_dice"] > best
        if improved:
            best, bad_epochs = row["val_dice"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_dice": best}, ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
        log.info(f"epoch {epoch:3d}/{epochs} | train loss {row['train_loss']:.4f} dice {row['train_dice']:.4f} | "
                 f"val loss {row['val_loss']:.4f} dice {row['val_dice']:.4f}{' *' if improved else ''}")
        if bad_epochs >= cfg.get("patience", 15):
            log.info(f"early stopping: no validation improvement for {bad_epochs} epochs")
            break

    best_epoch = int(np.argmax([h["val_dice"] for h in history])) + 1
    save_json(res_dir / "history.json", {"history": history, "best_epoch": best_epoch, "best_val_dice": best,
                                         "train_minutes": (time.time() - t0) / 60,
                                         "parameters": count_parameters(model)})
    plots.training_curves(history, [("Loss", ["train_loss", "val_loss"]), ("Dice (threshold 0.5)", ["train_dice", "val_dice"])],
                          res_dir / "training_curves.png")
    log.info(f"done: best val Dice {best:.4f} at epoch {best_epoch} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
