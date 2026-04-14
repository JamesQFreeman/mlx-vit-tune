"""v0.2 Benchmark: memory + speed across batch sizes, LoRA vs full FT, checkpointing."""

import sys
sys.path.insert(0, "/Volumes/ExternalHD/mlx-path-foundation-tune")

import gc
import json
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from mlx_vit.vit import VisionTransformer, ViTConfig
from mlx_vit.lora import inject_lora
from mlx_vit.trainer import cross_entropy_loss


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
    mx.metal.clear_cache()
    mx.metal.reset_peak_memory()


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


def bench_inference(model, batch_size, image_size=224, warmup=3, iters=10):
    x = mx.random.normal((batch_size, image_size, image_size, 3))
    mx.eval(x)

    # Warmup
    for _ in range(warmup):
        out = model(x)
        mx.eval(out)

    # Timed
    times = []
    for _ in range(iters):
        mx.metal.reset_peak_memory()
        t0 = time.perf_counter()
        out = model(x)
        mx.eval(out)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    peak = mx.metal.get_peak_memory() / (1024 ** 2)
    return batch_size / avg_time, peak


def bench_train_step(model, batch_size, num_classes, image_size=224, warmup=3, iters=10):
    loss_fn = nn.value_and_grad(model, cross_entropy_loss)

    x = mx.random.normal((batch_size, image_size, image_size, 3))
    labels = mx.random.randint(0, num_classes, (batch_size,))
    mx.eval(x, labels)

    optimizer = optim.AdamW(learning_rate=1e-4)

    # Warmup
    for _ in range(warmup):
        (loss, acc), grads = loss_fn(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    # Timed
    mx.metal.reset_peak_memory()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        (loss, acc), grads = loss_fn(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    peak = mx.metal.get_peak_memory() / (1024 ** 2)
    return batch_size / avg_time, peak


def run_config(arch, mode, checkpoint, batch_size, num_classes=10):
    clear_memory()

    model = build_model(arch, num_classes, checkpoint)

    if mode == "lora":
        model, trainable = inject_lora(model, rank=8, alpha=8.0)

    mx.eval(model.parameters())
    model_mem = mx.metal.get_active_memory() / (1024 ** 2)

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

    device = mx.metal.device_info()
    print(f"Device: {device.get('architecture', '?')}")
    total_mem = device.get("memory_size", 0)
    if total_mem:
        print(f"Total memory: {total_mem / (1024**3):.0f} GB")
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
    out_path = "/Volumes/ExternalHD/mlx-path-foundation-tune/benchmark_results/benchmark_v02.json"
    with open(out_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
