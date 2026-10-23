"""Latency / throughput: PyTorch vs. ONNX Runtime, on CPU and GPU.

    python -m src.benchmark --task seg --config configs/seg/drive_unet_green.yaml
    python -m src.benchmark --task cls --config configs/cls/convnext_tiny_plain.yaml

Latency is measured at batch size 1 (p50 / p95 over repeated runs after warm-up).
Throughput is measured at a larger batch. GPU timings synchronise CUDA before reading the clock.
Run export_onnx first. Output: results/efficiency/<run>/benchmark.json + benchmark.png
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np
import onnxruntime as ort
import torch

from src.models.load import load_model
from src.utils import plots
from src.utils.common import ROOT, environment, load_json, save_json


def time_fn(fn: Callable[[], None], warmup: int, runs: int) -> np.ndarray:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1000)
    return np.array(times)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", required=True, choices=["seg", "cls"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--batch", type=int, default=8, help="batch size for the throughput measurement")
    ap.add_argument("--cpu-threads", type=int, default=8)
    args = ap.parse_args()

    model, cfg, shape, run = load_model(args.task, args.config, args.seed)
    out_dir = ROOT / "results" / "efficiency" / run
    export = load_json(out_dir / "export.json")
    onnx_path = str(ROOT / export["onnx_path"])
    torch.set_num_threads(args.cpu_threads)
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()  # reuse the CUDA / cuDNN libraries that ship with PyTorch

    rows = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        warmup, runs = (5, 30) if dev == "cpu" else (20, 200)
        x1, xb = np.random.randn(1, *shape).astype(np.float32), np.random.randn(args.batch, *shape).astype(np.float32)

        # ---- PyTorch (FP32, eval mode, no autograd)
        m = model.to(dev)
        t1, tb = torch.from_numpy(x1).to(dev), torch.from_numpy(xb).to(dev)

        def torch_run(t: torch.Tensor) -> None:
            with torch.inference_mode():
                m(t)
            if dev == "cuda":
                torch.cuda.synchronize()

        lat = time_fn(lambda: torch_run(t1), warmup, runs)
        thr = time_fn(lambda: torch_run(tb), max(warmup // 2, 2), max(runs // 5, 5))
        rows.append({"backend": "PyTorch", "device": dev, "p50_ms": float(np.percentile(lat, 50)),
                     "p95_ms": float(np.percentile(lat, 95)), "throughput_img_s": float(args.batch / (np.median(thr) / 1000))})

        # ---- ONNX Runtime
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.cpu_threads
        providers = ["CPUExecutionProvider"] if dev == "cpu" else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        sess = ort.InferenceSession(onnx_path, sess_options=so, providers=providers)
        if dev == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
            print("ONNX Runtime CUDA provider unavailable; skipping ORT GPU benchmark")
            continue
        if dev == "cuda":  # keep input and output on the GPU so we time inference, not host<->device copies
            g1, gb = ort.OrtValue.ortvalue_from_numpy(x1, "cuda", 0), ort.OrtValue.ortvalue_from_numpy(xb, "cuda", 0)

            def ort_run_bound(v: ort.OrtValue) -> None:
                io = sess.io_binding()
                io.bind_ortvalue_input("input", v)
                io.bind_output("output", "cuda")
                sess.run_with_iobinding(io)

            lat = time_fn(lambda: ort_run_bound(g1), warmup, runs)
            thr = time_fn(lambda: ort_run_bound(gb), max(warmup // 2, 2), max(runs // 5, 5))
        else:
            lat = time_fn(lambda: sess.run(None, {"input": x1}), warmup, runs)
            thr = time_fn(lambda: sess.run(None, {"input": xb}), max(warmup // 2, 2), max(runs // 5, 5))
        rows.append({"backend": "ONNX Runtime", "device": dev, "p50_ms": float(np.percentile(lat, 50)),
                     "p95_ms": float(np.percentile(lat, 95)), "throughput_img_s": float(args.batch / (np.median(thr) / 1000))})

    result = {"run": run, "task": args.task, "arch": cfg["arch"], "input_shape": list(shape), "throughput_batch": args.batch,
              "cpu_threads": args.cpu_threads, "onnxruntime": ort.__version__, "env": environment(), "results": rows}
    save_json(out_dir / "benchmark.json", result)
    print(f"\n[{run}] input {shape} | latency at batch 1, throughput at batch {args.batch}")
    print(f"{'backend':<14}{'device':<7}{'p50 ms':>10}{'p95 ms':>10}{'img/s':>10}")
    for r in rows:
        print(f"{r['backend']:<14}{r['device']:<7}{r['p50_ms']:>10.2f}{r['p95_ms']:>10.2f}{r['throughput_img_s']:>10.1f}")
    labels = [d.upper() if d == "cpu" else "GPU" for d in devices]
    by = lambda b, k: [next((r[k] for r in rows if r["backend"] == b and r["device"] == d), 0.0) for d in devices]
    plots.bar_compare(labels, {"PyTorch": by("PyTorch", "p50_ms"), "ONNX Runtime": by("ONNX Runtime", "p50_ms")},
                      "p50 latency (ms, batch 1, log scale)", f"Inference latency: {cfg['arch']} at {shape[1]}x{shape[2]}",
                      out_dir / "benchmark.png", log=True)


if __name__ == "__main__":
    main()
