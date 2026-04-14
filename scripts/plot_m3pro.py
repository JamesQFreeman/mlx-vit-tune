"""Multi-chip comparison bar plot: M4 16GB vs M3 Pro 18GB (v0.2 benchmark).

Reads `benchmark_results/benchmark_v02.json` (M4 baseline, already committed)
and `benchmark_results/benchmark_m3pro.json` (this machine), renders a 2x2
grid of grouped bar charts comparing training throughput across batch sizes
for the four primary configurations. Speedup ratios (M3 Pro / M4) are printed
on top of each M3 Pro bar.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

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
        "xtick.color": "#c9d1d9",
        "ytick.color": "#c9d1d9",
        "grid.color": "#21262d",
        "font.family": "sans-serif",
        "font.size": 11,
    }
)

C_M4 = "#58a6ff"
C_M3 = "#f0883e"

PANELS = [
    ("vit_b", "lora",    True,  "ViT-B  LoRA + ckpt  (training)"),
    ("vit_b", "full_ft", False, "ViT-B  Full FT  (training)"),
    ("vit_l", "lora",    True,  "ViT-L  LoRA + ckpt  (training)"),
    ("vit_l", "full_ft", True,  "ViT-L  Full FT + ckpt  (training)"),
]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "mlx-vit-tune v0.2  ·  M4 16 GB vs M3 Pro 18 GB  ·  float32  ·  real images",
    fontsize=15,
    fontweight="bold",
    y=0.98,
)

for ax, (arch, mode, ckpt, title) in zip(axes.flat, PANELS):
    m4 = pick(m4_raw, arch, mode, ckpt)
    m3 = pick(m3_raw, arch, mode, ckpt)

    # Align on the intersection of batch sizes so bars pair up even if one
    # hardware ran an extra config.
    m4_map = {r["batch_size"]: r["train_img_s"] for r in m4}
    m3_map = {r["batch_size"]: r["train_img_s"] for r in m3}
    batch_sizes = sorted(set(m4_map) & set(m3_map))

    m4_vals = [m4_map[b] for b in batch_sizes]
    m3_vals = [m3_map[b] for b in batch_sizes]

    x = np.arange(len(batch_sizes))
    width = 0.38

    b_m4 = ax.bar(x - width / 2, m4_vals, width, color=C_M4,
                  edgecolor="#0d1117", label="M4 16 GB")
    b_m3 = ax.bar(x + width / 2, m3_vals, width, color=C_M3,
                  edgecolor="#0d1117", label="M3 Pro 18 GB")

    # Value labels on the M4 bars (img/s) and speedup ratio on the M3 Pro bars.
    y_top = max(max(m4_vals), max(m3_vals)) * 1.22
    ax.set_ylim(0, y_top)
    for rect, v in zip(b_m4, m4_vals):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_top * 0.012,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=8, color="#8b949e")
    for rect, m4v, m3v in zip(b_m3, m4_vals, m3_vals):
        if m4v > 0:
            speedup = m3v / m4v
            label = f"{m3v:.1f}\n{speedup:.2f}×"
            color = "#3fb950" if speedup >= 1 else "#f85149"
        else:
            label = f"{m3v:.1f}\n—"
            color = "#c9d1d9"
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_top * 0.012,
                label, ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("img / s")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.3, fontsize=9)

# Footer: summary of hardware spec delta
fig.text(
    0.5, 0.015,
    "M4:  10 GPU cores · 120 GB/s · 16 GB          "
    "M3 Pro:  14 GPU cores · 150 GB/s · 18 GB",
    ha="center", fontsize=10, color="#8b949e",
)

plt.tight_layout(rect=[0, 0.035, 1, 0.955])
out = f"{REPO}/benchmark_results/benchmark_comparison.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()
