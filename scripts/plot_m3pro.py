"""Multi-chip comparison plot: M4 16GB vs M3 Pro 18GB (v0.2 benchmark).

Reads `benchmark_results/benchmark_v02.json` (M4 baseline, already committed)
and `benchmark_results/benchmark_m3pro.json` (this machine), renders a 2x3
grid comparing train throughput and peak memory across batch sizes.
"""

import json
import os

import matplotlib.pyplot as plt

REPO = "."  # run from repo root

with open(f"{REPO}/benchmark_results/benchmark_v02.json") as f:
    m4_raw = json.load(f)
with open(f"{REPO}/benchmark_results/benchmark_m3pro.json") as f:
    m3_raw = json.load(f)


def pick(data, arch, mode, ckpt):
    return sorted(
        [
            r
            for r in data
            if r["arch"] == arch
            and r["mode"] == mode
            and r["checkpoint"] == ckpt
            and not r.get("oom", False)
        ],
        key=lambda r: r["batch_size"],
    )


plt.rcParams.update(
    {
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.family": "sans-serif",
        "font.size": 11,
    }
)

C = {
    "m4": "#58a6ff",
    "m4_ckpt": "#3fb950",
    "m3": "#f0883e",
    "m3_ckpt": "#d2a8ff",
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(
    "mlx-vit-tune v0.2  ·  M4 16 GB vs M3 Pro 18 GB  ·  float32  ·  real images",
    fontsize=16,
    fontweight="bold",
    y=0.97,
)

# --- Row 0: ViT-B training throughput ---
for col, (mode, title) in enumerate(
    [
        ("lora", "ViT-B  LoRA Training"),
        ("full_ft", "ViT-B  Full FT Training"),
    ]
):
    ax = axes[0, col]
    for data, ckpt, color, label in [
        (m4_raw, False, C["m4"], "M4 16GB"),
        (m4_raw, True, C["m4_ckpt"], "M4 16GB + ckpt"),
        (m3_raw, False, C["m3"], "M3 Pro 18GB"),
        (m3_raw, True, C["m3_ckpt"], "M3 Pro 18GB + ckpt"),
    ]:
        d = pick(data, "vit_b", mode, ckpt)
        if d:
            ax.plot(
                [r["batch_size"] for r in d],
                [r["train_img_s"] for r in d],
                "D-" if ckpt else "s-",
                color=color,
                label=label,
                linewidth=2.5 if ckpt else 2,
                markersize=7,
            )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("img / s")
    ax.set_xticks([1, 4, 8, 16, 32])
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

# ViT-B Memory (LoRA)
ax = axes[0, 2]
for data, ckpt, color, label in [
    (m4_raw, False, C["m4"], "M4 LoRA"),
    (m4_raw, True, C["m4_ckpt"], "M4 LoRA+ckpt"),
    (m3_raw, False, C["m3"], "M3 Pro LoRA"),
    (m3_raw, True, C["m3_ckpt"], "M3 Pro LoRA+ckpt"),
]:
    d = pick(data, "vit_b", "lora", ckpt)
    if d:
        ax.plot(
            [r["batch_size"] for r in d],
            [r["peak_memory_mb"] / 1024 for r in d],
            "D-" if ckpt else "s-",
            color=color,
            label=label,
            linewidth=2,
            markersize=7,
        )
ax.axhline(y=10.5, color="#f85149", ls=":", lw=1.5, alpha=0.7, label="~M4 usable")
ax.axhline(y=12.5, color="#f85149", ls="--", lw=1.5, alpha=0.5, label="~M3 Pro usable")
ax.set_title("ViT-B  Peak Memory (LoRA)", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("GB")
ax.set_xticks([1, 4, 8, 16, 32])
ax.legend(fontsize=7, framealpha=0.3)
ax.grid(True, alpha=0.3)

# --- Row 1: ViT-L training throughput ---
for col, (mode, title) in enumerate(
    [
        ("lora", "ViT-L  LoRA Training"),
        ("full_ft", "ViT-L  Full FT Training"),
    ]
):
    ax = axes[1, col]
    for data, ckpt, color, label in [
        (m4_raw, False, C["m4"], "M4 16GB"),
        (m4_raw, True, C["m4_ckpt"], "M4 16GB + ckpt"),
        (m3_raw, False, C["m3"], "M3 Pro 18GB"),
        (m3_raw, True, C["m3_ckpt"], "M3 Pro 18GB + ckpt"),
    ]:
        d = pick(data, "vit_l", mode, ckpt)
        if d:
            ax.plot(
                [r["batch_size"] for r in d],
                [r["train_img_s"] for r in d],
                "D-" if ckpt else "s-",
                color=color,
                label=label,
                linewidth=2.5 if ckpt else 2,
                markersize=7,
            )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("img / s")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True, alpha=0.3)

# ViT-L Memory (LoRA)
ax = axes[1, 2]
for data, ckpt, color, label in [
    (m4_raw, False, C["m4"], "M4 LoRA"),
    (m4_raw, True, C["m4_ckpt"], "M4 LoRA+ckpt"),
    (m3_raw, False, C["m3"], "M3 Pro LoRA"),
    (m3_raw, True, C["m3_ckpt"], "M3 Pro LoRA+ckpt"),
]:
    d = pick(data, "vit_l", "lora", ckpt)
    if d:
        ax.plot(
            [r["batch_size"] for r in d],
            [r["peak_memory_mb"] / 1024 for r in d],
            "D-" if ckpt else "s-",
            color=color,
            label=label,
            linewidth=2,
            markersize=7,
        )
ax.axhline(y=10.5, color="#f85149", ls=":", lw=1.5, alpha=0.7, label="~M4 usable")
ax.axhline(y=12.5, color="#f85149", ls="--", lw=1.5, alpha=0.5, label="~M3 Pro usable")
ax.set_title("ViT-L  Peak Memory (LoRA)", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("GB")
ax.legend(fontsize=7, framealpha=0.3)
ax.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.94])
out = f"{REPO}/benchmark_results/benchmark_comparison.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()
