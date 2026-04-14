"""Tight interleaved A/B test: fp32 (v0.3 baseline) vs bf16 (v0.4 default).

Mirrors scripts/ab_sdpa.py — alternates variants in the same thermal window
so absolute thermal drift cancels out. Writes results to
``benchmark_results/ab_bf16_m3pro.json`` for plotting by plot_v04.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_vit.vit import VisionTransformer, ViTConfig
from mlx_vit.lora import inject_lora
from mlx_vit.trainer import cross_entropy_loss


def one_step_timer(model, x, labels, opt, loss_fn, iters):
    """Return list of iteration times after a single warmup."""
    (l, _), g = loss_fn(model, x, labels)
    opt.update(model, g)
    mx.eval(l, model.parameters(), opt.state)

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        (l, _), g = loss_fn(model, x, labels)
        opt.update(model, g)
        mx.eval(l, model.parameters(), opt.state)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def build(arch, dtype, ckpt):
    if arch == "vit_b":
        cfg = ViTConfig.vit_base_patch16(
            num_classes=10, image_size=224,
            gradient_checkpointing=ckpt, dtype=dtype,
        )
    else:
        cfg = ViTConfig.vit_large_patch16(
            num_classes=10, image_size=224,
            gradient_checkpointing=ckpt, dtype=dtype,
        )
    mx.random.seed(0)
    model = VisionTransformer(cfg)
    model, _ = inject_lora(model, rank=8, alpha=8.0)
    mx.eval(model.parameters())
    return model


def run(arch, bs, ckpt, rounds=3, iters_per_round=20):
    print()
    print(f"=== {arch}  bs={bs}  ckpt={ckpt} ===")
    mx.random.seed(0)
    x = mx.random.normal((bs, 224, 224, 3)).astype(mx.float32)  # arrives fp32
    labels = mx.random.randint(0, 10, (bs,))
    mx.eval(x, labels)

    model_fp32 = build(arch, mx.float32, ckpt)
    model_bf16 = build(arch, mx.bfloat16, ckpt)
    loss_fn_fp32 = nn.value_and_grad(model_fp32, cross_entropy_loss)
    loss_fn_bf16 = nn.value_and_grad(model_bf16, cross_entropy_loss)
    opt_fp32 = optim.AdamW(learning_rate=1e-4)
    opt_bf16 = optim.AdamW(learning_rate=1e-4)

    all_fp32, all_bf16 = [], []
    # Interleave to wash out thermal drift
    for r in range(rounds):
        all_fp32.extend(one_step_timer(model_fp32, x, labels, opt_fp32, loss_fn_fp32, iters_per_round))
        all_bf16.extend(one_step_timer(model_bf16, x, labels, opt_bf16, loss_fn_bf16, iters_per_round))

    mean_fp32 = statistics.mean(all_fp32) * 1000
    mean_bf16 = statistics.mean(all_bf16) * 1000
    std_fp32 = statistics.stdev(all_fp32) * 1000
    std_bf16 = statistics.stdev(all_bf16) * 1000
    speedup = mean_fp32 / mean_bf16

    print(f"  fp32: {mean_fp32:7.2f} ± {std_fp32:5.2f} ms   ({bs/(mean_fp32/1000):5.1f} img/s)   n={len(all_fp32)}")
    print(f"  bf16: {mean_bf16:7.2f} ± {std_bf16:5.2f} ms   ({bs/(mean_bf16/1000):5.1f} img/s)   n={len(all_bf16)}")
    print(f"  speedup: {speedup:.3f}×")
    return {
        "arch": arch, "bs": bs, "ckpt": ckpt,
        "fp32_ms": mean_fp32, "bf16_ms": mean_bf16,
        "fp32_std": std_fp32, "bf16_std": std_bf16,
        "speedup": speedup,
    }


def main():
    print("Interleaved A/B test: fp32 (v0.3) vs bf16 (v0.4 default)")
    print(f"Device: {mx.device_info().get('architecture', '?')}")

    configs = [
        ("vit_b",  1, True),
        ("vit_b",  4, True),
        ("vit_b",  8, True),
        ("vit_b", 16, True),
        ("vit_b", 32, True),
        ("vit_l",  1, True),
        ("vit_l",  2, True),
        ("vit_l",  4, True),
        ("vit_l",  8, True),
    ]

    results = []
    for arch, bs, ckpt in configs:
        results.append(run(arch, bs, ckpt))

    print()
    print("SUMMARY")
    print(f"{'config':25s}   {'fp32':>11s}   {'bf16':>11s}   {'speedup':>9s}")
    for r in results:
        tag = f"{r['arch']} bs={r['bs']} ckpt={r['ckpt']}"
        print(f"{tag:25s}   {r['fp32_ms']:7.2f} ms   {r['bf16_ms']:7.2f} ms   {r['speedup']:7.3f}×")

    import math
    geo = math.exp(sum(math.log(r["speedup"]) for r in results) / len(results))
    print(f"\nGeometric mean speedup: {geo:.3f}×")

    out = "benchmark_results/ab_bf16_m3pro.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
