"""v0.3 improvement plot: manual attention vs mx.fast.scaled_dot_product_attention.

Uses the interleaved, high-statistics measurement from scripts/ab_sdpa.py
(persisted to benchmark_results/ab_sdpa_m3pro.json). This is a much cleaner
signal than the 10-iter full-sweep benchmark, whose per-config noise floor
(~3-5%) is larger than the SDPA speedup (~1-2%) on ViT workloads.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = "."

with open(f"{REPO}/benchmark_results/ab_sdpa_m3pro.json") as f:
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

C_MANUAL = "#8b949e"  # grey
C_FAST   = "#3fb950"  # green

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle(
    "mlx-vit-tune v0.3  ·  manual attention vs mx.fast.scaled_dot_product_attention\n"
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
    manual = [r["manual_ms"] for r in rows]
    fast = [r["fast_ms"] for r in rows]
    manual_err = [r["manual_std"] for r in rows]
    fast_err = [r["fast_std"] for r in rows]
    speedups = [m / f for m, f in zip(manual, fast)]

    x = np.arange(len(batch_sizes))
    width = 0.38

    b_m = ax.bar(x - width / 2, manual, width, color=C_MANUAL,
                 edgecolor="#0d1117", yerr=manual_err, ecolor="#6e7681",
                 capsize=3, label="manual softmax path")
    b_f = ax.bar(x + width / 2, fast, width, color=C_FAST,
                 edgecolor="#0d1117", yerr=fast_err, ecolor="#6e7681",
                 capsize=3, label="mx.fast SDPA")

    y_top = max(max(manual), max(fast)) * 1.28
    ax.set_ylim(0, y_top)

    for rect, val in zip(b_m, manual):
        ax.text(rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_top * 0.012,
                f"{val:.1f}", ha="center", va="bottom",
                fontsize=8, color="#8b949e")
    for rect, v_m, v_f, sp in zip(b_f, manual, fast, speedups):
        pct = (sp - 1.0) * 100
        color = "#3fb950" if sp >= 1.0 else "#f85149"
        label = f"{v_f:.1f}\n{sp:.3f}×"
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

# Geometric mean of speedups for the footer
all_speedups = [r["manual_ms"] / r["fast_ms"] for r in data]
geo = float(np.exp(np.mean(np.log(all_speedups))))
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
out = f"{REPO}/benchmark_results/benchmark_v03_improvement.png"
os.makedirs(os.path.dirname(out), exist_ok=True)
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()
