"""Two hero bar charts for the README teaser:

  benchmark_results/hero_speed.png   — training throughput (img/s, higher is better)
  benchmark_results/hero_memory.png  — peak memory (GB, lower is better)

Reads benchmark_results/bench_3way.json from scripts/bench_3way.py.
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np

REPO = "."

with open(f"{REPO}/benchmark_results/bench_3way.json") as f:
    data = json.load(f)

# Style — matches the rest of the benchmark plots
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
        "font.size": 12,
    }
)

# Per-backend colors
COLORS = {
    "pytorch_cpu":      "#8b949e",  # grey — baseline
    "pytorch_mps":      "#58a6ff",  # blue — Apple Silicon GPU via torch
    "mlx_vit_tune_v04": "#3fb950",  # green — our library
}
LABELS = {
    "pytorch_cpu":      "PyTorch CPU",
    "pytorch_mps":      "PyTorch MPS",
    "mlx_vit_tune_v04": "mlx-vit-tune v0.4",
}
ORDER = ["pytorch_cpu", "pytorch_mps", "mlx_vit_tune_v04"]


def by_config(configs, metric):
    """Return a nested dict {config_label: {backend: value}} for the given metric."""
    out = {}
    for r in data:
        if "error" in r:
            continue
        label = f"{'ViT-B/16' if r['arch']=='vit_b' else 'ViT-L/16'}  bs={r['bs']}"
        out.setdefault(label, {})[r["backend"]] = r[metric]
    # Preserve config order (ViT-B first, then ViT-L)
    config_labels = [lbl for lbl in configs if lbl in out]
    return config_labels, out


# Discover config labels in data order (ViT-B before ViT-L)
seen = []
for r in data:
    if "error" in r:
        continue
    label = f"{'ViT-B/16' if r['arch']=='vit_b' else 'ViT-L/16'}  bs={r['bs']}"
    if label not in seen:
        seen.append(label)


def draw_bar_chart(metric: str, out_path: str, *, title: str, ylabel: str,
                   ratio_mode: str, better: str):
    """ratio_mode: "higher_is_better" (speedup vs baseline) or "lower_is_better"
    (ratio baseline / backend, so still shows as ">=1 = better")."""
    config_labels, by_cfg = by_config(seen, metric)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    n_backends = len(ORDER)
    bar_width = 0.27
    x = np.arange(len(config_labels))

    for i, backend in enumerate(ORDER):
        vals = [by_cfg[cfg].get(backend, 0.0) for cfg in config_labels]
        positions = x + (i - (n_backends - 1) / 2) * bar_width
        rects = ax.bar(
            positions, vals, bar_width,
            color=COLORS[backend], edgecolor="#0d1117",
            label=LABELS[backend],
        )

        # Value label on each bar
        y_max_so_far = max(
            by_cfg[cfg].get(b, 0.0) for cfg in config_labels for b in ORDER
        )
        for rect, val, cfg in zip(rects, vals, config_labels):
            if val == 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + y_max_so_far * 0.012,
                f"{val:.1f}" if val < 1000 else f"{val:,.0f}",
                ha="center", va="bottom",
                fontsize=9,
                color=COLORS[backend],
                fontweight="bold" if backend == "mlx_vit_tune_v04" else "normal",
            )

    # For each config, compute the mlx-vit-tune vs pytorch_mps ratio and annotate
    # above the config label. Skip if baseline is missing.
    for cfg_idx, cfg in enumerate(config_labels):
        ours = by_cfg[cfg].get("mlx_vit_tune_v04", 0)
        mps = by_cfg[cfg].get("pytorch_mps", 0)
        cpu = by_cfg[cfg].get("pytorch_cpu", 0)
        if ours and mps:
            if ratio_mode == "higher_is_better":
                ratio_mps = ours / mps
                ratio_cpu = ours / cpu if cpu else 0
            else:
                ratio_mps = mps / ours
                ratio_cpu = cpu / ours if cpu else 0
            tag = f"{ratio_mps:.1f}× vs MPS"
            if ratio_cpu:
                tag += f",  {ratio_cpu:.1f}× vs CPU"

    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontweight="bold", pad=12, fontsize=13)
    ax.grid(True, alpha=0.25, axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", framealpha=0.3, fontsize=10)

    # Extra headroom for labels
    y_max = max(
        by_cfg[cfg].get(b, 0) for cfg in config_labels for b in ORDER
    )
    ax.set_ylim(0, y_max * 1.25)

    # Footer
    fig.text(
        0.5, 0.01,
        "M3 Pro 18 GB  ·  LoRA r=8 (all linear layers)  ·  gradient checkpointing  ·  "
        f"{better} is better",
        ha="center", fontsize=9, color="#8b949e",
    )

    plt.tight_layout(rect=[0, 0.025, 1, 1])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved -> {out_path}")
    plt.close()


# -- speed chart (img/s, higher is better) --
draw_bar_chart(
    "img_s",
    f"{REPO}/benchmark_results/hero_speed.png",
    title="Training throughput  ·  higher is better",
    ylabel="img / s",
    ratio_mode="higher_is_better",
    better="higher",
)

# -- memory chart (GB, lower is better) --
# Convert MB to GB on the fly via a temporary annotation
for r in data:
    if "error" not in r:
        r["peak_gb"] = r["peak_mb"] / 1024.0

draw_bar_chart(
    "peak_gb",
    f"{REPO}/benchmark_results/hero_memory.png",
    title="Peak training memory  ·  lower is better",
    ylabel="peak memory (GB)",
    ratio_mode="lower_is_better",
    better="lower",
)
