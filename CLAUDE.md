# CLAUDE.md — Project Handover & Context

## What This Project Is

**mlx-vit-tune** — a generic Vision Transformer fine-tuning tool on Apple Silicon using MLX, with an Unsloth-like API (`FastViTModel`). Published at https://github.com/JamesQFreeman/mlx-vit-tune

There is a **sibling repo** `mlx-path-foundation-tune` (https://github.com/JamesQFreeman/mlx-path-foundation-tune) for pathology-specific fine-tuning. It's empty — waiting for user's dataset. The pathology repo will eventually depend on this generic ViT repo.

## User Context

- **GitHub**: JamesQFreeman (also author of the popular [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) repo)
- **Hardware now**: Apple M4 16GB
- **Hardware incoming**: Apple M5 Pro 64GB (arriving ~late April 2026)
- **Python env**: Use `/opt/anaconda3/envs/mlx/bin/python` (conda env `mlx`, Python 3.12)
- **GitHub CLI**: `gh` is installed, auth requires `GH_CONFIG_DIR=/tmp/gh-config` prefix (because `/Users/sheng/.config/` is root-owned, can't write there)
- **HF token**: In `.env` file but it was expired/invalid at test time. User has gated access to MahmoodLab/conch, MahmoodLab/UNI, paige-ai/Virchow2

## What Was Built (v0.1) — Committed & Pushed

### File Overview

```
mlx_vit/
├── __init__.py        # Exports FastViTModel
├── vit.py             # ViT-B/L/H with SwiGLU FFN, register tokens, LayerScale
│                        - PatchEmbed, Attention, MLP, SwiGLU, TransformerBlock, VisionTransformer
│                        - MODEL_CONFIGS dict with presets: vit_base_patch16_224, vit_large_patch16_224,
│                          vit_huge_patch14_224, conch, uni, uni2_h, virchow2
│                        - global_pool modes: "token" (CLS), "avg", "token+avg" (Virchow2-style)
│
├── lora.py            # LoRA implementation
│                        - LoRALinear(nn.Module): wraps nn.Linear with A,B matrices
│                        - inject_lora(): freezes model, replaces Linear→LoRALinear, unfreezes norms+head
│                        - merge_lora(), save_adapters(), load_adapters()
│                        - DEFAULT_TARGET_MODULES = [q_proj, k_proj, v_proj, out_proj, fc1, fc2]
│                        - SWIGLU_TARGET_MODULES adds fc1_gate
│
├── convert.py         # timm/PyTorch → MLX weight converter
│                        - Handles fused QKV split (timm packs Q,K,V into one weight)
│                        - Handles SwiGLUPacked split (timm packs fc1+gate into w12)
│                        - Handles CONCH CoCa wrapper (extracts visual.trunk.*)
│                        - Conv2d transpose: PyTorch [O,I,H,W] → MLX [O,H,W,I]
│                        - download_and_convert(): auto-download from HuggingFace
│
├── data.py            # Image dataset + transforms
│                        - ImageDataset: directory, CSV, or JSON format
│                        - Augmentations: flip, rot90, brightness, contrast
│                        - create_batches(): yields (mx.array images, mx.array labels)
│                        - IMAGENET_MEAN/STD normalization
│
├── model.py           # FastViTModel — Unsloth-like API
│                        - from_pretrained(): HF download, local path, or architecture name
│                        - get_lora_model(): inject LoRA with "all"/"attention"/"mlp" shortcuts
│                        - save_pretrained(), save_pretrained_merged(), load_adapters()
│                        - HF_MODEL_REGISTRY: maps HF IDs → architecture names
│                        - Default dtype is float32 (fp16 causes NaN with random weights)
│
├── trainer.py         # Training loop
│                        - train(): AdamW + cosine/linear/constant LR schedule
│                        - Gradient accumulation support
│                        - Validation + best model checkpointing
│                        - cross_entropy_loss() with accuracy tracking
│                        - ~16 img/s LoRA training on M4 16GB (ViT-B)
│
scripts/
├── train.py           # CLI entry point (argparse)
├── demo.py            # Self-contained demo with synthetic dataset
├── benchmark.py       # MLX GPU vs PyTorch CPU benchmark + matplotlib plot
│
tests/
├── test_vit.py        # 8 tests: forward, features, ViT-L, SwiGLU, Virchow2 pooling,
│                        LoRA injection, LoRA merge, param count
│
configs/
├── conch.yaml         # Example config for CONCH
│
benchmark_results/
├── benchmark.json     # Raw benchmark numbers
├── benchmark.png      # Plot for README
```

### Key Design Decisions

1. **Attention uses separate Q,K,V projections** (not fused QKV like timm). This makes LoRA injection cleaner — each projection gets its own LoRA adapter. The converter splits timm's fused QKV weight.

2. **MLX channels-last**: MLX Conv2d expects [B,H,W,C], not PyTorch's [B,C,H,W]. The converter transposes conv weights.

3. **LoRA targets ALL linear layers** by default (not just Q,V). Research shows MLP layers are 2/3 of ViT params — attention-only LoRA significantly underperforms on ViTs (unlike LLMs where Q,V is often enough).

4. **Default dtype is float32**, not float16. Float16 with random weights causes NaN loss because 12 layers of random transformations overflow fp16 range. With pretrained weights (where values are well-scaled), fp16 should work fine — this can be revisited in v0.2.

5. **No position embedding interpolation yet** — raises NotImplementedError if input resolution differs from config. Planned for v0.4.

### Bugs Encountered & Fixed

- `model.parameters()` returns nested dict in MLX, not flat. Must use `mlx.utils.tree_flatten()` to count params.
- `mx.tree_map` doesn't exist in this MLX version. Use `mlx.utils.tree_flatten()` + list comprehension instead.
- ViTConfig classmethods (`vit_large_patch16`) set `layer_scale_init=1e-5` as default, which conflicts if the MODEL_CONFIGS lambda also passes it. Fixed by removing duplicate kwargs from lambdas.
- `gh auth login` generates a new code each time it's called. The auth token gets saved to keyring but the config dir write can fail if `~/.config/` is root-owned. Workaround: `GH_CONFIG_DIR=/tmp/gh-config`.

### Benchmark Results (M4 16GB)

| Batch | MLX Inf | MLX Train | CPU Inf | CPU Train | Inf Speedup | Train Speedup |
|-------|---------|-----------|---------|-----------|-------------|---------------|
| 1 | 51 img/s | 5.6 img/s | 18 img/s | 4.5 img/s | 2.9x | 1.3x |
| 4 | 61 img/s | 12 img/s | 26 img/s | 8.2 img/s | 2.4x | 1.4x |
| 8 | 64 img/s | 14 img/s | 27 img/s | 8.0 img/s | 2.4x | 1.8x |
| 16 | 64 img/s | 16 img/s | 26 img/s | 8.5 img/s | 2.4x | 1.9x |

---

## v0.2 — What To Build

**Goal**: Gradient checkpointing + gradient accumulation + memory reporting. This unlocks ViT-L LoRA training on M4 16GB.

### 1. Gradient Checkpointing

MLX has `mx.checkpoint()` which wraps a function to recompute activations during backward instead of storing them.

**Where to apply**: Each `TransformerBlock.__call__()` should be wrappable with checkpoint.

```python
# In VisionTransformer, replace:
for block in self.blocks:
    x = block(x)

# With:
for block in self.blocks:
    if self.gradient_checkpointing:
        x = mx.checkpoint(block)(x)
    else:
        x = block(x)
```

**Config**: Add `gradient_checkpointing: bool = False` to ViTConfig. Enable via `FastViTModel.from_pretrained(..., gradient_checkpointing=True)`.

**Expected impact**: ~2x batch size at ~33% speed cost. ViT-B batch 16→32, ViT-L becomes feasible at batch 1-2 on 16GB.

### 2. Gradient Accumulation

Already partially implemented in `trainer.py` (the `gradient_accumulation_steps` arg exists and the accumulation logic is written), but it should be tested and verified.

**To verify**: Run with `gradient_accumulation_steps=4, batch_size=2` and confirm it produces similar results to `batch_size=8` without accumulation.

### 3. Memory Reporting

Add memory usage tracking using `mx.metal.get_active_memory()` and `mx.metal.get_peak_memory()` (if available in this MLX version).

Print at the start of training:
```
Model memory: X.X GB
Available: ~10-11 GB (M4 16GB)
Suggested batch size: N
```

### 4. Precision Control

Float16 training should work with pretrained weights (where values are well-scaled). Add a tested fp16 training path:
- Load weights in fp32, cast to fp16
- Keep loss computation in fp32 (cast logits before cross_entropy)
- This halves memory usage

### Testing v0.2

1. Enable gradient checkpointing on ViT-B, verify training still works (loss decreases)
2. Compare memory usage with/without checkpointing
3. Train ViT-L with LoRA + checkpointing on M4 16GB — should fit at batch 1-2
4. Verify gradient accumulation produces same results as larger batch
5. Test fp16 training with pretrained weights (not random)

---

## Full Roadmap (for reference)

| Version | Focus | Hardware |
|---------|-------|----------|
| v0.1 ✅ | ViT + LoRA + training pipeline | M4 16GB |
| **v0.2** | **Gradient checkpointing + accumulation** | **M4 16GB** |
| v0.3 | Fused LoRA autograd (custom mx.vjp for QKV+MLP) | M4 16GB |
| v0.4 | Pathology augmentations, multi-res, eval modes | M4 16GB |
| v0.5 | UNI, Virchow2, UNI2-h on M5 Pro 64GB | M5 Pro 64GB |
| v0.6 | QLoRA, DoRA, AdaLoRA, fused CE | M5 Pro 64GB |
| v0.7 | Model zoo, docs, PyPI | Both |

## Research Findings (Key Takeaways)

### LoRA on ViTs
- Works but ~1-3% gap vs full FT (larger gap than LLMs)
- MUST target all linear layers (Q,K,V,O,fc1,fc2), not just attention
- Optimal rank: r=8 default, r=4-8 for domain-pretrained models, r=16 for large domain shift
- LR ~1e-4 (10x higher than full FT), alpha = rank

### mlx-tune (ARahim3/mlx-tune)
- Well-engineered API wrapper around mlx-lm, NOT custom kernels
- No Unsloth-level optimizations (no fused autograd, no custom Metal kernels)
- Value is in Unsloth-compatible API and broad model support

### Unsloth Tricks (transferable to MLX)
- **Fused QKV LoRA autograd** (~40% of speedup): 3 separate backward passes → 1, sharing X.t()
- **Fused MLP LoRA autograd**: same pattern for fc1+fc2
- Both implementable as custom `mx.vjp` in MLX — planned for v0.3
- In-place SwiGLU backward, fused cross-entropy — possible as Metal kernels

### Pathology Foundation Models
- CONCH: ViT-B/16, 90M vision params, 802 MB, CoCa architecture
- UNI: ViT-L/16, 300M params, 1.21 GB
- UNI2-h: ViT-H/14 + SwiGLU + 8 register tokens, 681M params, 2.73 GB
- Virchow2: ViT-H/14 + SwiGLU + 4 register tokens, 632M params, 2.53 GB, token+avg pooling → 2560-dim
- All CC-BY-NC-ND 4.0, gated HuggingFace, require institutional email

### M4 16GB Constraints
- ~10-11 GB available for ML after OS overhead
- ViT-B LoRA: comfortable (batch 8-16)
- ViT-L LoRA: tight (batch 1-2, needs grad checkpointing)
- ViT-H LoRA: marginal (needs checkpointing + accumulation)
- 120 GB/s bandwidth is the bottleneck, not compute

## Environment Notes

```bash
# Python environment
conda activate mlx
# or directly: /opt/anaconda3/envs/mlx/bin/python

# Installed packages: mlx, numpy, pillow, safetensors, huggingface_hub, tqdm, pyyaml, matplotlib, torch

# GitHub CLI (needs config dir workaround)
GH_CONFIG_DIR=/tmp/gh-config gh <command>

# Run tests
/opt/anaconda3/envs/mlx/bin/python tests/test_vit.py

# Run demo
/opt/anaconda3/envs/mlx/bin/python scripts/demo.py

# Run benchmark
/opt/anaconda3/envs/mlx/bin/python scripts/benchmark.py
```
