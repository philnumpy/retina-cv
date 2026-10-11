"""Train a diabetic-retinopathy grader on APTOS 2019.

    python -m src.train_cls --config configs/cls/convnext_tiny_plain.yaml [--seed 1] [--smoke]

Best checkpoint and early stopping use validation QWK only.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.cls_data import AptosDataset, load_split
from src.models.classifier import build_classifier
from src.utils import plots
from src.utils.common import (get_logger, load_config, log_run, parse_args, run_dirs, save_json, seed_everything,
                              seed_worker)
from src.utils.metrics import qwk


def class_weights(labels: np.ndarray) -> torch.Tensor:
    """Inverse-frequency weights, normalised to mean 1 (rare grades count more in the loss)."""
    counts = np.bincount(labels, minlength=5).astype(np.float64)
    w = counts.sum() / (5 * counts)
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def outputs_to_grades(out: torch.Tensor, head: str) -> np.ndarray:
    """Classification: argmax. Regression: round the score (thresholds are optimised later, in eval)."""
    if head == "regression":
        return out[:, 0].round().clamp(0, 4).long().cpu().numpy()
    return out.argmax(1).cpu().numpy()


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device, max_batches: int | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
    """Raw model outputs (logits or scores) and labels for a whole loader."""
    model.eval()
    outs, labels = [], []
    for i, (x, y) in enumerate(loader):
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            outs.append(model(x.to(device, non_blocking=True)).float().cpu())
        labels.append(y)
        if max_batches and i + 1 >= max_batches:
            break
    return torch.cat(outs).numpy(), torch.cat(labels).numpy()


def make_loader(df: Any, cfg: dict[str, Any], train: bool) -> DataLoader:
    ds = AptosDataset(df, cfg["variant"], cfg["image_size"], train)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train, drop_last=train, num_workers=cfg.get("workers", 8),
                      worker_init_fn=seed_worker, pin_memory=True, persistent_workers=train,
                      generator=torch.Generator().manual_seed(cfg["seed"]))


def main() -> None:
    args = parse_args(__doc__)
    cfg = load_config(args)
    smoke = cfg.get("smoke", False)
    res_dir, ckpt_dir = run_dirs("classification", cfg["run_name"])
    log = get_logger(res_dir)
    log_run(res_dir, cfg)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = cfg.get("head", "classification")

    split = load_split()
    train_df, val_df = split[split.split == "train"], split[split.split == "val"]
    for name, df in (("train", train_df), ("val", val_df)):
        log.info(f"{name}: n = {len(df)} | class counts {np.bincount(df.diagnosis, minlength=5).tolist()}")
    train_loader, val_loader = make_loader(train_df, cfg, True), make_loader(val_df, cfg, False)
    max_batches = 3 if smoke else None

    model = build_classifier(cfg).to(device)
    weights = class_weights(train_df.diagnosis.to_numpy()).to(device)
    log.info(f"{cfg['arch']} | head {head} | {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters | "
             f"class weights {[round(w, 3) for w in weights.tolist()]}")

    def criterion(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if head == "regression":  # weight each sample by its grade's class weight
            return (F.smooth_l1_loss(out[:, 0], y.float(), reduction="none") * weights[y]).mean()
        return F.cross_entropy(out, y, weight=weights, label_smoothing=cfg.get("label_smoothing", 0.0))

    epochs = 1 if smoke else cfg["epochs"]
    steps = (3 if smoke else len(train_loader)) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history: list[dict[str, float]] = []
    best, bad_epochs, t0 = -1.0, 0, time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, n, preds, labels = 0.0, 0, [], []
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device, non_blocking=True), y.to(device)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                out = model(x)
            loss = criterion(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.item() * len(y)
            n += len(y)
            preds.append(outputs_to_grades(out.detach().float(), head))
            labels.append(y.cpu().numpy())
            if max_batches and i + 1 >= max_batches:
                break
        tr_pred, tr_true = np.concatenate(preds), np.concatenate(labels)
        val_out, val_true = predict(model, val_loader, device, max_batches)
        vt = torch.from_numpy(val_out)
        val_pred = outputs_to_grades(vt, head)
        with torch.no_grad():
            val_loss = criterion(vt.to(device), torch.from_numpy(val_true).to(device)).item()
        row = {"epoch": epoch, "train_loss": loss_sum / n, "val_loss": val_loss,
               "train_acc": float((tr_pred == tr_true).mean()), "val_acc": float((val_pred == val_true).mean()),
               "train_qwk": qwk(tr_true, tr_pred), "val_qwk": qwk(val_true, val_pred)}
        history.append(row)
        improved = row["val_qwk"] > best
        if improved:
            best, bad_epochs = row["val_qwk"], 0
            torch.save({"model": model.state_dict(), "cfg": cfg, "epoch": epoch, "val_qwk": best}, ckpt_dir / "best.pt")
        else:
            bad_epochs += 1
        log.info(f"epoch {epoch:3d}/{epochs} | train loss {row['train_loss']:.4f} acc {row['train_acc']:.4f} qwk {row['train_qwk']:.4f}"
                 f" | val loss {row['val_loss']:.4f} acc {row['val_acc']:.4f} qwk {row['val_qwk']:.4f}{' *' if improved else ''}")
        if bad_epochs >= cfg.get("patience", 8):
            log.info(f"early stopping: no validation QWK improvement for {bad_epochs} epochs")
            break

    best_epoch = int(np.argmax([h["val_qwk"] for h in history])) + 1
    save_json(res_dir / "history.json", {"history": history, "best_epoch": best_epoch, "best_val_qwk": best,
                                         "train_minutes": (time.time() - t0) / 60})
    plots.training_curves(history, [("Loss", ["train_loss", "val_loss"]), ("Accuracy", ["train_acc", "val_acc"]),
                                    ("Quadratic weighted kappa", ["train_qwk", "val_qwk"])], res_dir / "training_curves.png")
    log.info(f"done: best val QWK {best:.4f} at epoch {best_epoch} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
