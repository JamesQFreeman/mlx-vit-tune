"""Three-way training bench: PyTorch CPU vs PyTorch MPS vs mlx-vit-tune v0.4.

All three backends train a ViT with LoRA (rank 8, all linear layers) + gradient
checkpointing on randomly-initialized weights. The point is to show the
training-step wall time and peak memory you'd actually get if you fine-tuned
a ViT on a MacBook today.

* PyTorch CPU   = timm ViT + peft LoRA + torch.utils.checkpoint, fp32, CPU
* PyTorch MPS   = same, device='mps'                                  (fp32)
* mlx-vit-tune  = ViTConfig defaults (bf16 + mx.fast SDPA + grad ckpt)

Writes benchmark_results/bench_3way.json and prints a summary table.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import json
import os
import statistics
import time


# --------------------------------------------------------------------------
#  PyTorch (timm + peft)
# --------------------------------------------------------------------------

TIMM_NAMES = {
    "vit_b": "vit_base_patch16_224",
    "vit_l": "vit_large_patch16_224",
}


def bench_pytorch(
    arch: str,
    batch_size: int,
    device: str,          # "cpu" or "mps"
    warmup: int = 3,
    iters: int = 15,
) -> dict:
    import psutil
    import torch
    import torch.nn as nn
    import timm
    from peft import LoraConfig, get_peft_model

    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # timm ViT with grad checkpointing; peft for LoRA
    model = timm.create_model(TIMM_NAMES[arch], pretrained=False, num_classes=10)
    try:
        model.set_grad_checkpointing(enable=True)
    except Exception:
        pass

    # timm ViT uses fused qkv (one Linear), plus proj, fc1, fc2.
    # (enable_input_require_grads is a HF transformers method timm doesn't
    # have; not needed here since the input images aren't parameters.)
    peft_cfg = LoraConfig(
        r=8, lora_alpha=8,
        target_modules=["qkv", "proj", "fc1", "fc2"],
        bias="none",
    )
    model = get_peft_model(model, peft_cfg)
    model.to(device).train()

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
    )
    loss_fn = nn.CrossEntropyLoss()

    x = torch.randn(batch_size, 3, 224, 224, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # Warmup
    for _ in range(warmup):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    if device == "mps":
        torch.mps.synchronize()

    # Baseline memory right before timed run
    proc = psutil.Process()
    rss_peak = proc.memory_info().rss
    if device == "mps":
        # driver_allocated_memory is cumulative; we'll track max across loop
        mps_peak = torch.mps.driver_allocated_memory()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

        # Sample peak memory during the loop
        rss = proc.memory_info().rss
        if rss > rss_peak:
            rss_peak = rss
        if device == "mps":
            m = torch.mps.driver_allocated_memory()
            if m > mps_peak:
                mps_peak = m

    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000
    img_s = batch_size / (mean_ms / 1000)

    if device == "mps":
        peak_mb = mps_peak / (1024 ** 2)
    else:
        peak_mb = rss_peak / (1024 ** 2)

    # Free for the next run
    del model, opt, x, y
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    return {
        "backend": f"pytorch_{device}",
        "arch": arch,
        "bs": batch_size,
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "img_s": round(img_s, 2),
        "peak_mb": round(peak_mb, 1),
    }


# --------------------------------------------------------------------------
#  mlx-vit-tune v0.4
# --------------------------------------------------------------------------

def bench_mlx(
    arch: str,
    batch_size: int,
    warmup: int = 3,
    iters: int = 15,
) -> dict:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx.utils

    from mlx_vit.vit import ViTConfig, VisionTransformer
    from mlx_vit.lora import inject_lora
    from mlx_vit.trainer import cross_entropy_loss

    mx.clear_cache()
    mx.reset_peak_memory()

    if arch == "vit_b":
        cfg = ViTConfig.vit_base_patch16(
            num_classes=10, image_size=224, gradient_checkpointing=True,
        )
    else:
        cfg = ViTConfig.vit_large_patch16(
            num_classes=10, image_size=224, gradient_checkpointing=True,
        )

    mx.random.seed(0)
    model = VisionTransformer(cfg)
    model, _ = inject_lora(model, rank=8, alpha=8.0)
    mx.eval(model.parameters())

    x = mx.random.normal((batch_size, 224, 224, 3)).astype(mx.float32)
    labels = mx.random.randint(0, 10, (batch_size,))
    mx.eval(x, labels)

    loss_fn = nn.value_and_grad(model, cross_entropy_loss)
    opt = optim.AdamW(learning_rate=1e-4)

    for _ in range(warmup):
        (l, _), g = loss_fn(model, x, labels)
        opt.update(model, g)
        mx.eval(l, model.parameters(), opt.state)

    mx.reset_peak_memory()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        (l, _), g = loss_fn(model, x, labels)
        opt.update(model, g)
        mx.eval(l, model.parameters(), opt.state)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    peak_bytes = mx.get_peak_memory()
    mean_ms = statistics.mean(times) * 1000
    std_ms = statistics.stdev(times) * 1000
    img_s = batch_size / (mean_ms / 1000)

    return {
        "backend": "mlx_vit_tune_v04",
        "arch": arch,
        "bs": batch_size,
        "mean_ms": round(mean_ms, 2),
        "std_ms": round(std_ms, 2),
        "img_s": round(img_s, 2),
        "peak_mb": round(peak_bytes / (1024 ** 2), 1),
    }


# --------------------------------------------------------------------------
#  Entry
# --------------------------------------------------------------------------

def main():
    import torch

    # Limit PyTorch CPU threads a bit so it doesn't fight with the rest of the
    # machine; using all 12 threads makes runtime noisy.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) - 2))
    print(
        f"PyTorch {torch.__version__}  |  "
        f"MPS available: {torch.backends.mps.is_available()}  |  "
        f"CPU threads: {torch.get_num_threads()}"
    )

    configs = [
        ("vit_b", 32),
        ("vit_l", 16),
    ]

    all_results: list[dict] = []

    for arch, bs in configs:
        print()
        print(f"=== {arch.upper()}  bs={bs}  (LoRA r=8 all-linear + grad ckpt) ===")
        for name, fn in [
            ("pytorch_cpu", lambda a=arch, b=bs: bench_pytorch(a, b, "cpu")),
            ("pytorch_mps", lambda a=arch, b=bs: bench_pytorch(a, b, "mps")),
            ("mlx_vit_tune_v04", lambda a=arch, b=bs: bench_mlx(a, b)),
        ]:
            print(f"  {name:20s}  ...  ", end="", flush=True)
            try:
                r = fn()
                all_results.append(r)
                print(
                    f"{r['mean_ms']:8.1f} ms ± {r['std_ms']:4.1f}   "
                    f"({r['img_s']:6.1f} img/s,  peak {r['peak_mb']/1024:5.2f} GB)"
                )
            except Exception as e:
                print(f"FAILED: {type(e).__name__}: {e}")
                all_results.append({
                    "backend": name,
                    "arch": arch,
                    "bs": bs,
                    "error": str(e),
                })

    out = "benchmark_results/bench_3way.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print("Summary:")
    print(f"  {'backend':20s}  {'config':15s}  {'img/s':>8s}  {'peak GB':>8s}")
    for r in all_results:
        if "error" in r:
            print(
                f"  {r['backend']:20s}  {r['arch']} bs={r['bs']:<5d}"
                f"  FAILED: {r['error'][:50]}"
            )
            continue
        label = f"{r['arch']} bs={r['bs']}"
        print(
            f"  {r['backend']:20s}  {label:15s}  "
            f"{r['img_s']:8.1f}  {r['peak_mb']/1024:8.2f}"
        )

    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
