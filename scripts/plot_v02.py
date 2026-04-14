"""Plot v0.2 benchmark results — 6 panels covering speed + memory for both archs."""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ---------- load data ----------
with open("/Volumes/ExternalHD/mlx-path-foundation-tune/benchmark_results/benchmark_v02.json") as f:
    raw = json.load(f)

# v0.1 CPU baselines (PyTorch CPU, ViT-B LoRA)
cpu_lora = {1: 4.5, 4: 8.2, 8: 8.0, 16: 8.5}

# ---------- helpers ----------
def pick(arch, mode, ckpt):
    return sorted(
        [r for r in raw if r["arch"] == arch and r["mode"] == mode and r["checkpoint"] == ckpt and not r["oom"]],
        key=lambda r: r["batch_size"],
    )

# ---------- style ----------
plt.rcParams.update({
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
})

COLORS = {
    "cpu":       "#8b949e",
    "mlx":       "#58a6ff",
    "ckpt":      "#3fb950",
    "full_mlx":  "#d2a8ff",
    "full_ckpt": "#f0883e",
}

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("mlx-vit-tune v0.2 Benchmark  ·  Apple M4 16 GB  ·  float32", fontsize=16, fontweight="bold", y=0.97)

# ============================================================
# Panel 1: ViT-B LoRA Training Speed (3 lines: CPU, MLX, ckpt)
# ============================================================
ax = axes[0, 0]
bs_cpu = sorted(cpu_lora.keys())
ax.plot(bs_cpu, [cpu_lora[b] for b in bs_cpu], "o--", color=COLORS["cpu"], label="CPU (PyTorch)", linewidth=1.5, markersize=6)

d = pick("vit_b", "lora", False)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "s-", color=COLORS["mlx"], label="MLX", linewidth=2, markersize=7)

d = pick("vit_b", "lora", True)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "D-", color=COLORS["ckpt"], label="mlx-vit-tune", linewidth=2.5, markersize=8)

ax.set_title("ViT-B  LoRA Training Speed", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("img / s")
ax.set_xticks([1, 4, 8, 16, 32])
ax.legend(loc="lower right", fontsize=9, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ============================================================
# Panel 2: ViT-B Full FT Training Speed
# ============================================================
ax = axes[0, 1]
d = pick("vit_b", "full_ft", False)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "s-", color=COLORS["full_mlx"], label="Full FT", linewidth=2, markersize=7)

d = pick("vit_b", "full_ft", True)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "D-", color=COLORS["full_ckpt"], label="Full FT + ckpt", linewidth=2, markersize=7)

d = pick("vit_b", "lora", True)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "D-", color=COLORS["ckpt"], label="LoRA + ckpt", linewidth=2.5, markersize=8, alpha=0.5)

ax.set_title("ViT-B  Full FT vs LoRA", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("img / s")
ax.set_xticks([1, 4, 8, 16, 32])
ax.legend(loc="lower right", fontsize=9, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ============================================================
# Panel 3: ViT-B Peak Memory
# ============================================================
ax = axes[0, 2]
for mode, ckpt, color, label, ls, marker in [
    ("lora", False, COLORS["mlx"], "LoRA", "-", "s"),
    ("lora", True, COLORS["ckpt"], "LoRA + ckpt", "-", "D"),
    ("full_ft", False, COLORS["full_mlx"], "Full FT", "--", "s"),
    ("full_ft", True, COLORS["full_ckpt"], "Full FT + ckpt", "--", "D"),
]:
    d = pick("vit_b", mode, ckpt)
    ax.plot([r["batch_size"] for r in d], [r["peak_memory_mb"] / 1024 for r in d],
            f"{marker}{ls}", color=color, label=label, linewidth=2, markersize=7)

ax.axhline(y=10.5, color="#f85149", linestyle=":", linewidth=1.5, alpha=0.7, label="~Usable limit (16 GB)")
ax.set_title("ViT-B  Peak Memory", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("GB")
ax.set_xticks([1, 4, 8, 16, 32])
ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ============================================================
# Panel 4: ViT-L LoRA Training Speed
# ============================================================
ax = axes[1, 0]
d = pick("vit_l", "lora", False)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "s-", color=COLORS["mlx"], label="MLX", linewidth=2, markersize=7)

d = pick("vit_l", "lora", True)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "D-", color=COLORS["ckpt"], label="mlx-vit-tune", linewidth=2.5, markersize=8)

ax.set_title("ViT-L  LoRA Training Speed", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("img / s")
ax.set_xticks([1, 2, 4, 8])
ax.legend(loc="lower right", fontsize=9, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ============================================================
# Panel 5: ViT-L Full FT Training Speed
# ============================================================
ax = axes[1, 1]
d = pick("vit_l", "full_ft", False)
# Filter out the thrashing point (0.1 img/s at bs=8)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "s-", color=COLORS["full_mlx"], label="Full FT", linewidth=2, markersize=7)

d = pick("vit_l", "full_ft", True)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "D-", color=COLORS["full_ckpt"], label="Full FT + ckpt", linewidth=2, markersize=7)

d = pick("vit_l", "lora", True)
ax.plot([r["batch_size"] for r in d], [r["train_img_s"] for r in d], "D-", color=COLORS["ckpt"], label="LoRA + ckpt", linewidth=2.5, markersize=8, alpha=0.5)

# Annotate the thrashing point
ax.annotate("OOM thrashing\n(0.1 img/s)", xy=(8, 0.1), xytext=(6, 2.5),
            fontsize=8, color="#f85149", ha="center",
            arrowprops=dict(arrowstyle="->", color="#f85149", lw=1.2))

ax.set_title("ViT-L  Full FT vs LoRA", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("img / s")
ax.set_xticks([1, 2, 4, 8])
ax.legend(loc="upper left", fontsize=9, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ============================================================
# Panel 6: ViT-L Peak Memory
# ============================================================
ax = axes[1, 2]
for mode, ckpt, color, label, ls, marker in [
    ("lora", False, COLORS["mlx"], "LoRA", "-", "s"),
    ("lora", True, COLORS["ckpt"], "LoRA + ckpt", "-", "D"),
    ("full_ft", False, COLORS["full_mlx"], "Full FT", "--", "s"),
    ("full_ft", True, COLORS["full_ckpt"], "Full FT + ckpt", "--", "D"),
]:
    d = pick("vit_l", mode, ckpt)
    ax.plot([r["batch_size"] for r in d], [r["peak_memory_mb"] / 1024 for r in d],
            f"{marker}{ls}", color=color, label=label, linewidth=2, markersize=7)

ax.axhline(y=10.5, color="#f85149", linestyle=":", linewidth=1.5, alpha=0.7, label="~Usable limit (16 GB)")
ax.set_title("ViT-L  Peak Memory", fontweight="bold")
ax.set_xlabel("Batch Size")
ax.set_ylabel("GB")
ax.set_xticks([1, 2, 4, 8])
ax.legend(loc="upper left", fontsize=8, framealpha=0.3)
ax.grid(True, alpha=0.3)

# ---------- finalize ----------
plt.tight_layout(rect=[0, 0, 1, 0.94])
out = "/Volumes/ExternalHD/mlx-path-foundation-tune/benchmark_results/benchmark_v02.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved → {out}")
plt.close()
