"""Tight A/B test: use_fast_sdpa=True vs use_fast_sdpa=False on the same model
configuration, high statistics, interleaved to wash out thermal/cache effects."""

import sys
sys.path.insert(0, ".")

import statistics
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
from mlx_vit.vit import VisionTransformer, ViTConfig
from mlx_vit.lora import inject_lora
from mlx_vit.trainer import cross_entropy_loss


def one_step_timer(model, x, labels, opt, loss_fn, iters):
    """Return list of iteration times after a single warmup."""
    # warmup
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


def build(arch, use_fast_sdpa, ckpt):
    if arch == "vit_b":
        cfg = ViTConfig.vit_base_patch16(
            num_classes=10, image_size=224,
            gradient_checkpointing=ckpt, use_fast_sdpa=use_fast_sdpa,
        )
    else:
        cfg = ViTConfig.vit_large_patch16(
            num_classes=10, image_size=224,
            gradient_checkpointing=ckpt, use_fast_sdpa=use_fast_sdpa,
        )
    mx.random.seed(0)
    model = VisionTransformer(cfg)
    model, _ = inject_lora(model, rank=8, alpha=8.0)
    mx.eval(model.parameters())
    return model


def run(arch, bs, ckpt, rounds=5, iters_per_round=20):
    print()
    print(f"=== {arch}  bs={bs}  ckpt={ckpt} ===")
    mx.random.seed(0)
    x = mx.random.normal((bs, 224, 224, 3))
    labels = mx.random.randint(0, 10, (bs,))
    mx.eval(x, labels)

    # Build once for each flag. We won't share weights — the same seed is set
    # before build, and the inject_lora RNG is also deterministic, so both
    # models have identical weights at construction.
    model_manual = build(arch, False, ckpt)
    model_fast   = build(arch, True,  ckpt)
    loss_fn_m = nn.value_and_grad(model_manual, cross_entropy_loss)
    loss_fn_f = nn.value_and_grad(model_fast,   cross_entropy_loss)
    opt_m = optim.AdamW(learning_rate=1e-4)
    opt_f = optim.AdamW(learning_rate=1e-4)

    all_m, all_f = [], []
    # Interleave rounds to wash out thermal drift.
    for r in range(rounds):
        all_m.extend(one_step_timer(model_manual, x, labels, opt_m, loss_fn_m, iters_per_round))
        all_f.extend(one_step_timer(model_fast,   x, labels, opt_f, loss_fn_f, iters_per_round))

    mean_m = statistics.mean(all_m) * 1000
    mean_f = statistics.mean(all_f) * 1000
    std_m = statistics.stdev(all_m) * 1000
    std_f = statistics.stdev(all_f) * 1000
    speedup = mean_m / mean_f

    print(f"  manual  : {mean_m:7.2f} ± {std_m:5.2f} ms   ({bs/(mean_m/1000):5.1f} img/s)   n={len(all_m)}")
    print(f"  fast SDP: {mean_f:7.2f} ± {std_f:5.2f} ms   ({bs/(mean_f/1000):5.1f} img/s)   n={len(all_f)}")
    print(f"  speedup : {speedup:.3f}×   (positive = fast is faster)")
    return {"arch": arch, "bs": bs, "ckpt": ckpt,
            "manual_ms": mean_m, "fast_ms": mean_f,
            "manual_std": std_m, "fast_std": std_f,
            "speedup": speedup}


def main():
    import json

    print("Interleaved A/B test: manual attention vs mx.fast.scaled_dot_product_attention")
    print(f"Device: {mx.metal.device_info().get('architecture', '?')}")

    results = []
    # Headline configurations. All with gradient checkpointing (the default
    # high-throughput path for LoRA training on Apple Silicon).
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
    for arch, bs, ckpt in configs:
        results.append(run(arch, bs, ckpt, rounds=3, iters_per_round=20))

    print()
    print("SUMMARY")
    print(f"{'config':25s}   {'manual':>11s}   {'fast':>11s}   {'speedup':>9s}")
    for r in results:
        tag = f"{r['arch']} bs={r['bs']} ckpt={r['ckpt']}"
        print(f"{tag:25s}   {r['manual_ms']:7.2f} ms   {r['fast_ms']:7.2f} ms   {r['speedup']:7.3f}×")

    # Persist for plotting.
    out = "benchmark_results/ab_sdpa_m3pro.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
