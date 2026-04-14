"""v0.2 Benchmark: memory + speed across batch sizes, LoRA vs full FT, checkpointing.

Benchmarks throughput and peak memory for ViT-B and ViT-L across every
combination of LoRA / full-FT × gradient checkpointing on/off × several
batch sizes. Writes a JSON sweep to ``benchmark_results/benchmark_m3pro.json``.

**Image pool**
By default the benchmark feeds a cached pool of synthetic (random-normal)
images preloaded on the GPU. To use real images instead, set the environment
variable ``BENCH_IMAGE_DIR`` to any directory containing ``.jpeg/.jpg/.png``
files — the pool is normalized with ImageNet stats and cached on-device, so
disk I/O does not affect the throughput measurement.

The pool being on-device means benchmark numbers are *pure forward/backward
compute*, regardless of whether the source is random noise or real patches.
"""

import sys
from pathlib import Path

# Make the repo importable regardless of where the script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import glob
import json
import os
import random
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from mlx_vit.vit import VisionTransformer, ViTConfig
from mlx_vit.lora import inject_lora
from mlx_vit.trainer import cross_entropy_loss


# --- Image pool: real jpegs if BENCH_IMAGE_DIR is set, otherwise synthetic ---
REAL_IMAGE_DIR = os.environ.get("BENCH_IMAGE_DIR", "").strip() or None
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_REAL_POOL_CACHE: dict = {}


def _load_pool_from_disk(directory: str, n: int, image_size: int) -> mx.array:
    patterns = ("*.jpeg", "*.jpg", "*.png")
    paths: list[str] = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(directory, pat)))
    paths.sort()
    if not paths:
        raise RuntimeError(
            f"BENCH_IMAGE_DIR={directory!r} contains no .jpeg/.jpg/.png files."
        )
    # If we have fewer than `n`, cycle with replacement rather than erroring.
    rng = random.Random(42)
    chosen = [paths[rng.randrange(len(paths))] for _ in range(n)]

    buf = np.empty((n, image_size, image_size, 3), dtype=np.float32)
    for i, p in enumerate(chosen):
        im = Image.open(p).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
        buf[i] = arr
    return mx.array(buf)


def _load_pool_synthetic(n: int, image_size: int) -> mx.array:
    rng = np.random.default_rng(42)
    buf = rng.standard_normal((n, image_size, image_size, 3)).astype(np.float32)
    return mx.array(buf)


def load_image_pool(n: int = 64, image_size: int = 224) -> mx.array:
    """Load (or synthesize) ``n`` images, cache them on the GPU, return as an
    ``mx.array`` of shape ``(n, H, W, 3)``. Source is controlled by the
    ``BENCH_IMAGE_DIR`` env var — unset means synthetic."""
    key = (n, image_size)
    if key in _REAL_POOL_CACHE:
        return _REAL_POOL_CACHE[key]

    if REAL_IMAGE_DIR:
        pool = _load_pool_from_disk(REAL_IMAGE_DIR, n, image_size)
    else:
        pool = _load_pool_synthetic(n, image_size)

    mx.eval(pool)
    _REAL_POOL_CACHE[key] = pool
    return pool


def get_real_batch(batch_size: int, image_size: int = 224) -> mx.array:
    """Return a batch of ``batch_size`` images as an ``mx.array``, drawn from
    the cached on-device pool — no disk I/O after the first call."""
    pool = load_image_pool(n=max(64, batch_size), image_size=image_size)
    idx = mx.array(np.arange(batch_size) % pool.shape[0])
    return pool[idx]


@dataclass
class BenchResult:
    arch: str
    mode: str  # "full_ft", "lora"
    checkpoint: bool
    batch_size: int
    peak_memory_mb: float
    train_img_s: float
    inf_img_s: float
    oom: bool = False


def clear_memory():
    gc.collect()
    mx.clear_cache()
    mx.reset_peak_memory()


def build_model(arch, num_classes, checkpoint):
    if arch == "vit_b":
        config = ViTConfig.vit_base_patch16(
            num_classes=num_classes, image_size=224,
            gradient_checkpointing=checkpoint,
        )
    elif arch == "vit_l":
        config = ViTConfig.vit_large_patch16(
            num_classes=num_classes, image_size=224,
            gradient_checkpointing=checkpoint,
        )
    return VisionTransformer(config)


def bench_inference(model, batch_size, image_size=224, warmup=5, iters=25):
    x = get_real_batch(batch_size, image_size)
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        out = model(x)
        mx.eval(out)

    # Timed
    times = []
    for _ in range(iters):
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        out = model(x)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    peak = mx.get_peak_memory() / (1024 ** 2)
    return batch_size / avg_time, peak


def bench_train_step(model, batch_size, num_classes, image_size=224, warmup=5, iters=25):
    loss_fn = nn.value_and_grad(model, cross_entropy_loss)

    x = get_real_batch(batch_size, image_size)
    labels = mx.random.randint(0, num_classes, (batch_size,))
    mx.eval(x, labels)

    optimizer = optim.AdamW(learning_rate=1e-4)

    # Warmup
    for _ in range(warmup):
        (loss, acc), grads = loss_fn(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # Timed
    mx.reset_peak_memory()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        (loss, acc), grads = loss_fn(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    peak = mx.get_peak_memory() / (1024 ** 2)
    return batch_size / avg_time, peak


def run_config(arch, mode, checkpoint, batch_size, num_classes=10):
    clear_memory()

    model = build_model(arch, num_classes, checkpoint)

    if mode == "lora":
        model, trainable = inject_lora(model, rank=8, alpha=8.0)

    mx.eval(model.parameters())
    model_mem = mx.get_active_memory() / (1024 ** 2)

    label = f"{arch} | {mode:7s} | ckpt={'Y' if checkpoint else 'N'} | bs={batch_size:2d}"

    try:
        inf_ips, inf_peak = bench_inference(model, batch_size)
    except Exception as e:
        print(f"  {label} | INF OOM: {e}")
        return BenchResult(arch, mode, checkpoint, batch_size, 0, 0, 0, oom=True)

    try:
        train_ips, train_peak = bench_train_step(model, batch_size, num_classes)
    except Exception as e:
        print(f"  {label} | TRAIN OOM: {e}")
        return BenchResult(arch, mode, checkpoint, batch_size, inf_peak, 0, inf_ips, oom=True)

    print(
        f"  {label} | "
        f"inf {inf_ips:5.1f} img/s | "
        f"train {train_ips:5.1f} img/s | "
        f"model {model_mem:7.1f} MB | "
        f"peak {train_peak:7.1f} MB"
    )

    return BenchResult(
        arch=arch, mode=mode, checkpoint=checkpoint,
        batch_size=batch_size,
        peak_memory_mb=train_peak,
        train_img_s=train_ips,
        inf_img_s=inf_ips,
    )


def main():
    print("=" * 90)
    print("v0.2 Benchmark: Memory + Speed")
    print("=" * 90)

    device = mx.device_info()
    print(f"Device: {device.get('architecture', '?')}")
    total_mem = device.get("memory_size", 0)
    if total_mem:
        print(f"Total memory: {total_mem / (1024**3):.0f} GB")

    # Preload real image pool once, on GPU, at max batch size.
    print(f"Loading real image pool from {REAL_IMAGE_DIR} ...")
    pool = load_real_image_pool(n=64, image_size=224)
    print(f"Real image pool: {pool.shape} {pool.dtype} — preloaded on device")
    print()

    results = []

    # ---- ViT-B benchmarks ----
    print("--- ViT-B/16 (224x224) ---")
    for batch_size in [1, 4, 8, 16, 32]:
        for mode in ["lora", "full_ft"]:
            for ckpt in [False, True]:
                r = run_config("vit_b", mode, ckpt, batch_size)
                results.append(r)
    print()

    # ---- ViT-L benchmarks ----
    print("--- ViT-L/16 (224x224) ---")
    for batch_size in [1, 2, 4, 8]:
        for mode in ["lora", "full_ft"]:
            for ckpt in [False, True]:
                r = run_config("vit_l", mode, ckpt, batch_size)
                results.append(r)
    print()

    # ---- Summary tables ----
    print("=" * 90)
    print("SUMMARY: ViT-B LoRA (v0.1 baseline vs v0.2)")
    print("=" * 90)
    print(f"{'BS':>4} | {'Mode':>8} | {'Ckpt':>4} | {'Train img/s':>12} | {'Peak MB':>10} | {'Notes'}")
    print("-" * 75)

    # v0.1 baselines from CLAUDE.md
    v01_baselines = {1: 5.6, 4: 12.0, 8: 14.0, 16: 16.0}

    for r in results:
        if r.oom:
            notes = "OOM"
        elif r.arch == "vit_b" and r.mode == "lora" and not r.checkpoint:
            baseline = v01_baselines.get(r.batch_size, 0)
            if baseline > 0:
                ratio = r.train_img_s / baseline
                notes = f"v0.1: {baseline:.1f} img/s ({ratio:.2f}x)"
            else:
                notes = ""
        else:
            notes = ""
        status = "OOM" if r.oom else f"{r.train_img_s:5.1f} img/s"
        print(
            f"{r.batch_size:4d} | {r.mode:>8s} | {'Y' if r.checkpoint else 'N':>4s} | "
            f"{status:>12s} | {r.peak_memory_mb:8.1f} MB | {notes}"
        )

    # Save raw results
    raw = [
        {
            "arch": r.arch, "mode": r.mode, "checkpoint": r.checkpoint,
            "batch_size": r.batch_size, "peak_memory_mb": round(r.peak_memory_mb, 1),
            "train_img_s": round(r.train_img_s, 1), "inf_img_s": round(r.inf_img_s, 1),
            "oom": r.oom,
        }
        for r in results
    ]
    out_path = "benchmark_results/benchmark_m3pro.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
