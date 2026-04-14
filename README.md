# mlx-vit-tune

Fine-tune Vision Transformers on Apple Silicon with MLX. An Unsloth-like API for ViT.

From the creator of [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) — now natively on Mac.

## Why

- No MLX ViT fine-tuning pipeline exists
- Apple Silicon's unified memory lets you fine-tune ViT-B/L/H without a cloud GPU
- 5-20x faster than CPU-only training

## Quick Start

```bash
pip install mlx numpy pillow safetensors huggingface_hub tqdm pyyaml
```

```python
from mlx_vit import FastViTModel
from mlx_vit.data import ImageDataset
from mlx_vit.trainer import TrainingArgs, train

# Load any ViT (from HuggingFace, local path, or random weights)
model = FastViTModel.from_pretrained("vit_base_patch16_224", num_classes=2)

# Add LoRA — targets ALL linear layers (Q,K,V,O,fc1,fc2)
model = FastViTModel.get_lora_model(model, rank=8)

# Train
train_ds = ImageDataset("data/train", image_size=224, augment=True)
val_ds = ImageDataset("data/val", image_size=224, augment=False)

train(model, train_ds, val_ds, TrainingArgs(
    batch_size=8, lr=1e-4, epochs=10
))
```

## Supported Architectures

| Architecture | Params | LoRA on M4 16GB | LoRA on M5 Pro 64GB |
|-------------|--------|----------------|-------------------|
| **ViT-B/16** | 86M | batch 8-16 | batch 32-64 |
| **ViT-L/16** | 304M | batch 1-2 | batch 32-48 |
| **ViT-H/14** | 632M | marginal | batch 16-24 |
| **ViT-H/14 + SwiGLU** | 632-681M | needs v0.2 | batch 16-24 |

All architectures support optional **SwiGLU FFN** and **register tokens**.

## LoRA: Target ALL Layers

Research shows ViT LoRA must target **all linear layers**, not just attention.
MLP layers contain ~2/3 of ViT parameters — attention-only LoRA significantly underperforms.

```python
# Default: targets Q, K, V, output proj, MLP fc1, MLP fc2
model = FastViTModel.get_lora_model(model, rank=8, target_modules="all")

# Or be specific
model = FastViTModel.get_lora_model(model, rank=8, target_modules="attention")  # Q,K,V,O only
model = FastViTModel.get_lora_model(model, rank=8, target_modules="mlp")        # fc1,fc2 only
```

## Loading Pretrained Models

```python
# Random weights (for testing)
model = FastViTModel.from_pretrained("vit_base_patch16_224", num_classes=10)

# From HuggingFace (auto-downloads and converts to MLX)
model = FastViTModel.from_pretrained("MahmoodLab/UNI", num_classes=2, hf_token="hf_xxx")

# From local converted weights
model = FastViTModel.from_pretrained("/path/to/weights", num_classes=2)
```

### Supported HuggingFace Models

| Model | HF ID | Type |
|-------|-------|------|
| CONCH | `MahmoodLab/conch` | ViT-B/16 (CoCa) |
| UNI | `MahmoodLab/UNI` | ViT-L/16 |
| UNI2-h | `MahmoodLab/UNI2-h` | ViT-H/14 + SwiGLU |
| Virchow2 | `paige-ai/Virchow2` | ViT-H/14 + SwiGLU |
| Phikon | `owkin/phikon` | ViT-B/16 |

Any timm-compatible ViT can be converted via `mlx_vit/convert.py`.

## Dataset Format

Directory structure (ImageFolder style):
```
data/
  train/
    class_a/
      img001.png
      img002.jpg
    class_b/
      img003.png
  val/
    class_a/
      img004.png
    class_b/
      img005.png
```

Also supports CSV (`image_path,label`) and JSON formats.

## CLI

```bash
python scripts/train.py \
    --model vit_base_patch16_224 \
    --train_data data/train \
    --val_data data/val \
    --num_classes 2 \
    --lora --lora_rank 8 \
    --batch_size 8 --lr 1e-4 --epochs 10
```

## Saving and Loading

```python
# Save LoRA adapters
FastViTModel.save_pretrained(model, "my_adapters")

# Save merged model (LoRA baked into weights)
FastViTModel.save_pretrained_merged(model, "my_merged_model")

# Load adapters onto a base model
base = FastViTModel.from_pretrained("vit_base_patch16_224", num_classes=2)
model = FastViTModel.load_adapters(base, "my_adapters")
```

## Performance

Measured on Apple M4 16GB, ViT-B/16, LoRA r=8, 224x224 images:

| Metric | Value |
|--------|-------|
| Training throughput | ~16 img/s |
| Trainable params | 1.57% (1.4M / 87M) |
| Peak memory | ~3-4 GB |

## Roadmap

- [x] **v0.1** — ViT-B/L/H + LoRA + training pipeline
- [ ] **v0.2** — Gradient checkpointing + accumulation
- [ ] **v0.3** — Fused LoRA autograd (Unsloth-style ~1.5-2x speedup)
- [ ] **v0.4** — Multi-resolution + evaluation (linear probe, kNN)
- [ ] **v0.5** — Big model validation (ViT-H on M5 Pro 64GB)
- [ ] **v0.6** — QLoRA, DoRA, AdaLoRA
- [ ] **v0.7** — Model zoo, docs, PyPI

## Related Projects

- [LoRA-ViT](https://github.com/JamesQFreeman/LoRA-ViT) — LoRA for ViT in PyTorch (by the same author)
- [mlx-tune](https://github.com/ARahim3/mlx-tune) — LLM fine-tuning on MLX (inspiration for API design)
- [Unsloth](https://github.com/unslothai/unsloth) — Fast LLM fine-tuning (inspiration for optimization roadmap)

## License

Apache-2.0
