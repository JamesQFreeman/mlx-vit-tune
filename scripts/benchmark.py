#!/usr/bin/env python3
"""Benchmark MLX GPU vs PyTorch CPU for ViT-B/16.

Generates comparison plots saved to benchmark_results/.
Run: python scripts/benchmark.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def bench_mlx(batch_size, n_warmup=3, n_inf_runs=20, n_train_runs=10):
    """Benchmark MLX GPU: inference + training."""
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx_vit.vit import VisionTransformer, ViTConfig

    config = ViTConfig.vit_base_patch16(num_classes=10, image_size=224)
    model = VisionTransformer(config)
    x = mx.random.normal((batch_size, 224, 224, 3))
    labels = mx.zeros((batch_size,), dtype=mx.int32)

    # --- Inference ---
    for _ in range(n_warmup):
        mx.eval(model(x))

    inf_times = []
    for _ in range(n_inf_runs):
        t0 = time.perf_counter()
        mx.eval(model(x))
        inf_times.append(time.perf_counter() - t0)

    # --- Training ---
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)

    def loss_fn(model, x, y):
        return nn.losses.cross_entropy(model(x), y, reduction="mean")

    grad_fn = nn.value_and_grad(model, loss_fn)

    for _ in range(n_warmup):
        loss, grads = grad_fn(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

    train_times = []
    for _ in range(n_train_runs):
        t0 = time.perf_counter()
        loss, grads = grad_fn(model, x, labels)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        train_times.append(time.perf_counter() - t0)

    return inf_times, train_times


def bench_pytorch_cpu(batch_size, n_warmup=2, n_inf_runs=10, n_train_runs=5):
    """Benchmark PyTorch CPU: inference + training with a real ViT."""
    import torch
    import torch.nn as torchnn

    # Build a real ViT-B/16 on CPU
    class PatchEmbed(torchnn.Module):
        def __init__(self, img_size=224, patch_size=16, embed_dim=768):
            super().__init__()
            self.proj = torchnn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
            self.num_patches = (img_size // patch_size) ** 2

        def forward(self, x):
            return self.proj(x).flatten(2).transpose(1, 2)

    class Block(torchnn.Module):
        def __init__(self, dim=768, heads=12):
            super().__init__()
            self.norm1 = torchnn.LayerNorm(dim)
            self.attn = torchnn.MultiheadAttention(dim, heads, batch_first=True)
            self.norm2 = torchnn.LayerNorm(dim)
            self.mlp = torchnn.Sequential(
                torchnn.Linear(dim, dim * 4),
                torchnn.GELU(),
                torchnn.Linear(dim * 4, dim),
            )

        def forward(self, x):
            h = self.norm1(x)
            x = x + self.attn(h, h, h, need_weights=False)[0]
            x = x + self.mlp(self.norm2(x))
            return x

    class ViT(torchnn.Module):
        def __init__(self, depth=12, dim=768, heads=12, num_classes=10):
            super().__init__()
            self.patch_embed = PatchEmbed(embed_dim=dim)
            self.cls_token = torchnn.Parameter(torch.zeros(1, 1, dim))
            self.pos_embed = torchnn.Parameter(torch.randn(1, 197, dim) * 0.02)
            self.blocks = torchnn.ModuleList([Block(dim, heads) for _ in range(depth)])
            self.norm = torchnn.LayerNorm(dim)
            self.head = torchnn.Linear(dim, num_classes)

        def forward(self, x):
            x = self.patch_embed(x)
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
            x = x + self.pos_embed
            for blk in self.blocks:
                x = blk(x)
            return self.head(self.norm(x[:, 0]))

    model = ViT().float()
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224)  # PyTorch: channels-first
    labels = torch.zeros(batch_size, dtype=torch.long)

    # --- Inference ---
    with torch.no_grad():
        for _ in range(n_warmup):
            model(x)

        inf_times = []
        for _ in range(n_inf_runs):
            t0 = time.perf_counter()
            model(x)
            inf_times.append(time.perf_counter() - t0)

    # --- Training ---
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    loss_fn = torchnn.CrossEntropyLoss()

    for _ in range(n_warmup):
        optimizer.zero_grad()
        loss_fn(model(x), labels).backward()
        optimizer.step()

    train_times = []
    for _ in range(n_train_runs):
        t0 = time.perf_counter()
        optimizer.zero_grad()
        loss_fn(model(x), labels).backward()
        optimizer.step()
        train_times.append(time.perf_counter() - t0)

    return inf_times, train_times


def plot_results(results, out_dir):
    """Generate benchmark comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    batch_sizes = [r["batch_size"] for r in results.values()]
    mlx_inf = [r["mlx_inf_ips"] for r in results.values()]
    mlx_train = [r["mlx_train_ips"] for r in results.values()]
    cpu_inf = [r["cpu_inf_ips"] for r in results.values()]
    cpu_train = [r["cpu_train_ips"] for r in results.values()]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x_pos = np.arange(len(batch_sizes))
    width = 0.2

    # --- Plot 1: Throughput ---
    ax = axes[0]
    b1 = ax.bar(x_pos - 1.5 * width, mlx_inf, width, label="MLX GPU — Inference", color="#FF6B35")
    b2 = ax.bar(x_pos - 0.5 * width, mlx_train, width, label="MLX GPU — Training", color="#004E89")
    b3 = ax.bar(x_pos + 0.5 * width, cpu_inf, width, label="PyTorch CPU — Inference", color="#CCCCCC", edgecolor="#888")
    b4 = ax.bar(x_pos + 1.5 * width, cpu_train, width, label="PyTorch CPU — Training", color="#888888")

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Throughput (images/sec)", fontsize=12)
    ax.set_title("ViT-B/16 · 224x224 · Apple M4 16GB", fontsize=13, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    for bars in [b1, b2, b3, b4]:
        for bar in bars:
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        f"{h:.0f}", ha="center", va="bottom", fontsize=7)

    # --- Plot 2: Speedup ---
    ax2 = axes[1]
    inf_speedup = [m / max(c, 0.01) for m, c in zip(mlx_inf, cpu_inf)]
    train_speedup = [m / max(c, 0.01) for m, c in zip(mlx_train, cpu_train)]

    b1 = ax2.bar(x_pos - 0.2, inf_speedup, 0.35, label="Inference speedup", color="#FF6B35")
    b2 = ax2.bar(x_pos + 0.2, train_speedup, 0.35, label="Training speedup", color="#004E89")
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax2.set_xlabel("Batch Size", fontsize=12)
    ax2.set_ylabel("Speedup (MLX GPU / PyTorch CPU)", fontsize=12)
    ax2.set_title("MLX GPU Speedup over CPU", fontsize=13, fontweight="bold")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(b) for b in batch_sizes])
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2, h,
                     f"{h:.1f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_path = out_dir / "benchmark.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close()


def main():
    out_dir = Path("benchmark_results")
    out_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("  ViT-B/16 Benchmark: MLX GPU vs PyTorch CPU")
    print("  Apple M4 16GB · 224x224 · float32")
    print("=" * 60)

    results = {}
    for bs in [1, 4, 8, 16]:
        print(f"\n--- batch_size={bs} ---")

        print("  MLX GPU...", end=" ", flush=True)
        mlx_inf, mlx_train = bench_mlx(bs)
        mlx_inf_ips = bs / np.mean(mlx_inf)
        mlx_train_ips = bs / np.mean(mlx_train)
        print(f"inf {mlx_inf_ips:.1f} img/s, train {mlx_train_ips:.1f} img/s")

        print("  PyTorch CPU...", end=" ", flush=True)
        cpu_inf, cpu_train = bench_pytorch_cpu(bs)
        cpu_inf_ips = bs / np.mean(cpu_inf)
        cpu_train_ips = bs / np.mean(cpu_train)
        print(f"inf {cpu_inf_ips:.1f} img/s, train {cpu_train_ips:.1f} img/s")

        inf_speedup = mlx_inf_ips / max(cpu_inf_ips, 0.01)
        train_speedup = mlx_train_ips / max(cpu_train_ips, 0.01)
        print(f"  Speedup: inf {inf_speedup:.1f}x, train {train_speedup:.1f}x")

        results[f"batch_{bs}"] = {
            "batch_size": bs,
            "mlx_inf_ips": round(mlx_inf_ips, 1),
            "mlx_train_ips": round(mlx_train_ips, 1),
            "cpu_inf_ips": round(cpu_inf_ips, 1),
            "cpu_train_ips": round(cpu_train_ips, 1),
            "inf_speedup": round(inf_speedup, 1),
            "train_speedup": round(train_speedup, 1),
        }

    with open(out_dir / "benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir / 'benchmark.json'}")
    plot_results(results, out_dir)


if __name__ == "__main__":
    main()
