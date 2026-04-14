"""v0.4 improvement plot: fp32 (v0.3 baseline) vs bf16 (v0.4 default).

Uses the interleaved high-statistics measurement from scripts/ab_bf16.py
(persisted to benchmark_results/ab_bf16_m3pro.json).
"""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = "."

with open(f"{REPO}/benchmark_results/ab_bf16_m3pro.json") as f:
    data = json.load(f)


def pick(arch):
    return [r for r in data if r["arch"] == arch]


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

C_FP32 = "#8b949e"  # grey
C_BF16 = "#d2a8ff"  # purple — new default

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "mlx-vit-tune v0.4  ·  fp32 (v0.3) vs bf16 (v0.4 default)\n"
    "M3 Pro 18 GB  ·  LoRA + gradient checkpointing  ·  interleaved A/B, n=60 per variant",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

for ax, (arch, title) in zip(axes, [
    ("vit_b", "ViT-B/16  LoRA + ckpt"),
    ("vit_l", "ViT-L/16  LoRA + ckpt"),
]):
    rows = sorted(pick(arch), key=lambda r: r["bs"])
    batch_sizes = [r["bs"] for r in rows]
    fp32 = [r["fp32_ms"] for r in rows]
    bf16 = [r["bf16_ms"] for r in rows]
    fp32_err = [r["fp32_std"] for r in rows]
    bf16_err = [r["bf16_std"] for r in rows]
    speedups = [f / b for f, b in zip(fp32, bf16)]

    x = np.arange(len(batch_sizes))
    width = 0.38

    b_fp32 = ax.bar(x - width / 2, fp32, width, color=C_FP32,
                    edgecolor="#0d1117", yerr=fp32_err, ecolor="#6e7681",
                    capsize=3, label="fp32 (v0.3)")
    b_bf16 = ax.bar(x + width / 2, bf16, width, color=C_BF16,
                    edgecolor="#0d1117", yerr=bf16_err, ecolor="#6e7681",
                    capsize=3, label="bf16 (v0.4)")

    y_top = max(max(fp32), max(bf16)) * 1.28
    ax.set_ylim(0, y_top)

    for rect, val in zip(b_fp32, fp32):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_top * 0.012,
                f"{val:.0f}", ha="center", va="bottom",
                fontsize=8, color="#8b949e")
    for rect, v_fp32, v_bf16, sp in zip(b_bf16, fp32, bf16, speedups):
        color = "#3fb950" if sp >= 1.02 else "#c9d1d9" if sp >= 1.0 else "#f85149"
        label = f"{v_bf16:.0f}\n{sp:.2f}×"
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_top * 0.012,
                label, ha="center", va="bottom",
                fontsize=8, color=color, fontweight="bold")

    ax.set_title(title, fontweight="bold", pad=10)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("training step time  (ms, lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(batch_sizes)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.3, fontsize=9)

all_speedups = [r["fp32_ms"] / r["bf16_ms"] for r in data]
geo = math.exp(sum(math.log(s) for s in all_speedups) / len(all_speedups))
min_s = min(all_speedups)
max_s = max(all_speedups)
fig.text(
    0.5, -0.02,
    f"Geometric mean speedup across 9 configurations: {geo:.3f}×  "
    f"(range {min_s:.3f}×–{max_s:.3f}×)     "
    f"Error bars: ±1 stddev across 60 iterations.",
    ha="center", fontsize=10, color="#8b949e",
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{REPO}/benchmark_results/benchmark_v04_improvement.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()
