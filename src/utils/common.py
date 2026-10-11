"""Shared run plumbing: config loading, seeding, run directories, environment logging."""
from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int) -> None:
    """Seed python, numpy and torch (CPU + CUDA) and make cuDNN deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: give every worker its own reproducible numpy/random stream."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _coerce(value: str) -> Any:
    return yaml.safe_load(value)


def parse_args(description: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="path to a YAML config")
    p.add_argument("--seed", type=int, default=None, help="override the config seed")
    p.add_argument("--smoke", action="store_true", help="1 epoch, 2-3 batches, tiny eval")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override config entries")
    return p.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load YAML, apply --set / --seed overrides and derive the run name."""
    with open(args.config) as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    for item in args.set:
        key, value = item.split("=", 1)
        cfg[key] = _coerce(value)
    cfg.setdefault("name", Path(args.config).stem)
    cfg.setdefault("seed", 0)
    base_seed = cfg["seed"]
    if args.seed is not None:
        cfg["seed"] = args.seed
    # the config's own seed keeps the plain name; extra seeds get a suffix
    cfg["run_name"] = cfg["name"] if cfg["seed"] == base_seed else f"{cfg['name']}__seed{cfg['seed']}"
    if getattr(args, "smoke", False):
        cfg["smoke"] = True
        cfg["run_name"] = "smoke_" + cfg["run_name"]
    return cfg


def run_dirs(task: str, run_name: str) -> tuple[Path, Path]:
    """results/<task>/<run> holds plots + metrics (committed); runs/<task>/<run> holds checkpoints."""
    res = ROOT / "results" / task / run_name
    ckpt = ROOT / "runs" / task / run_name
    res.mkdir(parents=True, exist_ok=True)
    ckpt.mkdir(parents=True, exist_ok=True)
    return res, ckpt


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True).stdout
        return out.stdout.strip() + ("-dirty" if dirty.strip() else "")
    except Exception:
        return "unknown"


def environment() -> dict[str, Any]:
    return {
        "git_commit": git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "argv": " ".join(sys.argv),
    }


def log_run(res_dir: Path, cfg: dict[str, Any]) -> None:
    """Persist the exact config and environment next to the results."""
    with open(res_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    save_json(res_dir / "env.json", environment())


def get_logger(res_dir: Path, name: str = "train") -> logging.Logger:
    logger = logging.getLogger(f"{name}:{res_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(res_dir / f"{name}.log", mode="w")):
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger


def save_json(path: Path, obj: Any) -> None:
    def default(o: Any) -> Any:
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)

    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def print_table(title: str, rows: dict[str, Any]) -> str:
    """Print a two-column metric table and return it as text."""
    width = max(len(k) for k in rows) + 2
    lines = [title, "-" * (width + 12)]
    for k, v in rows.items():
        lines.append(f"{k:<{width}}{v:>12.4f}" if isinstance(v, float) else f"{k:<{width}}{v!s:>12}")
    text = "\n".join(lines)
    print(text, flush=True)
    return text
