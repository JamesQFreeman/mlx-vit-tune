"""LoRA (Low-Rank Adaptation) for Vision Transformers.

Applies LoRA to all linear layers in a ViT: Q, K, V, output projection,
and MLP fc1/fc2. Research shows targeting ALL layers (not just attention)
is critical for ViT performance — MLP layers contain ~2/3 of parameters.
"""

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils

from mlx_vit.vit import VisionTransformer


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Using MLX's [..., in] → [..., out] row-vector convention:

        y = x @ W.T + (x @ A) @ B * scale

    where ``A: [in, rank]``, ``B: [rank, out]``, and ``scale = alpha / rank``.
    The base ``nn.Linear`` is frozen; only ``A`` and ``B`` are trained.
    """

    def __init__(
        self,
        base: nn.Linear,
        rank: int = 8,
        alpha: float = 8.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        in_features = base.weight.shape[1]
        out_features = base.weight.shape[0]

        self.base = base
        self.rank = rank
        self.scale = alpha / rank

        # LoRA matrices — A uses Kaiming init, B is zero-initialized
        self.lora_a = mx.random.normal((in_features, rank)) * (1.0 / rank**0.5)
        self.lora_b = mx.zeros((rank, out_features))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def __call__(self, x: mx.array) -> mx.array:
        base_out = self.base(x)
        lora_input = x
        if self.dropout is not None:
            lora_input = self.dropout(lora_input)
        lora_out = (lora_input @ self.lora_a) @ self.lora_b * self.scale
        return base_out + lora_out

    @property
    def weight(self):
        return self.base.weight

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into the base linear layer."""
        merged_weight = self.base.weight + (self.lora_a @ self.lora_b).T * self.scale
        new_linear = nn.Linear(
            self.base.weight.shape[1], self.base.weight.shape[0]
        )
        new_linear.weight = merged_weight
        if hasattr(self.base, "bias") and self.base.bias is not None:
            new_linear.bias = self.base.bias
        return new_linear


# Default target modules for ViTs — all linear layers in each transformer block
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "out_proj",  # Attention
    "fc1", "fc2",  # MLP
]

# Also include SwiGLU gate for models that use it
SWIGLU_TARGET_MODULES = DEFAULT_TARGET_MODULES + ["fc1_gate"]


def inject_lora(
    model: VisionTransformer,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> tuple[VisionTransformer, int]:
    """Inject LoRA adapters into a ViT model.

    Args:
        model: The ViT model to adapt
        rank: LoRA rank (default 8)
        alpha: LoRA scaling factor (default = rank)
        dropout: Dropout rate for LoRA (default 0.0)
        target_modules: List of module names to target, or None for defaults

    Returns:
        Tuple of (model, num_trainable_params)
    """
    if target_modules is None:
        # Auto-detect: use SwiGLU targets if model uses SwiGLU
        if model.config.use_swiglu:
            target_modules = SWIGLU_TARGET_MODULES
        else:
            target_modules = DEFAULT_TARGET_MODULES

    # Freeze the entire model first
    model.freeze()

    # Find and replace target modules with LoRA versions
    num_lora_params = 0
    num_replaced = 0

    for block_idx, block in enumerate(model.blocks):
        for name in target_modules:
            if name in ("q_proj", "k_proj", "v_proj", "out_proj"):
                base = getattr(block.attn, name, None)
                if base is not None and isinstance(base, nn.Linear):
                    lora_layer = LoRALinear(base, rank, alpha, dropout)
                    setattr(block.attn, name, lora_layer)
                    num_lora_params += base.weight.shape[1] * rank + rank * base.weight.shape[0]
                    num_replaced += 1
            elif name in ("fc1", "fc2", "fc1_gate"):
                base = getattr(block.mlp, name, None)
                if base is not None and isinstance(base, nn.Linear):
                    lora_layer = LoRALinear(base, rank, alpha, dropout)
                    setattr(block.mlp, name, lora_layer)
                    num_lora_params += base.weight.shape[1] * rank + rank * base.weight.shape[0]
                    num_replaced += 1

    # Unfreeze classification head and layer norms
    if model.config.num_classes > 0 and hasattr(model, "head"):
        model.head.unfreeze()
    model.norm.unfreeze()
    for block in model.blocks:
        block.norm1.unfreeze()
        block.norm2.unfreeze()

    all_leaves = mlx.utils.tree_flatten(model.parameters())
    total_params = sum(v.size for _, v in all_leaves)

    # Count trainable (LoRA params + unfrozen norms + head)
    trainable_leaves = mlx.utils.tree_flatten(model.trainable_parameters())
    trainable = sum(v.size for _, v in trainable_leaves)

    print(f"LoRA injected: {num_replaced} layers, rank={rank}, alpha={alpha}")
    print(f"Trainable params: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    return model, trainable


def merge_lora(model: VisionTransformer) -> VisionTransformer:
    """Merge all LoRA adapters back into base weights."""
    for block in model.blocks:
        # Attention layers
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            layer = getattr(block.attn, name, None)
            if isinstance(layer, LoRALinear):
                merged = layer.merge()
                setattr(block.attn, name, merged)

        # MLP layers
        for name in ("fc1", "fc2", "fc1_gate"):
            layer = getattr(block.mlp, name, None)
            if isinstance(layer, LoRALinear):
                merged = layer.merge()
                setattr(block.mlp, name, merged)

    return model


def save_adapters(model: VisionTransformer, path: str):
    """Save only LoRA adapter weights."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    adapters = {}
    config_info = {"rank": None, "alpha": None, "target_modules": []}

    for block_idx, block in enumerate(model.blocks):
        for loc, container in [("attn", block.attn), ("mlp", block.mlp)]:
            for name in ("q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "fc1_gate"):
                layer = getattr(container, name, None)
                if isinstance(layer, LoRALinear):
                    key_prefix = f"blocks.{block_idx}.{loc}.{name}"
                    adapters[f"{key_prefix}.lora_a"] = layer.lora_a
                    adapters[f"{key_prefix}.lora_b"] = layer.lora_b

                    if config_info["rank"] is None:
                        config_info["rank"] = layer.rank
                        config_info["alpha"] = layer.scale * layer.rank
                    if name not in config_info["target_modules"]:
                        config_info["target_modules"].append(name)

    mx.savez(str(path / "adapters.npz"), **adapters)

    with open(path / "adapter_config.json", "w") as f:
        json.dump(config_info, f, indent=2)

    print(f"Saved {len(adapters)} adapter tensors to {path}")


def load_adapters(model: VisionTransformer, path: str) -> VisionTransformer:
    """Load LoRA adapters from disk and inject into model."""
    path = Path(path)

    with open(path / "adapter_config.json") as f:
        config_info = json.load(f)

    # Inject LoRA if not already present
    first_block = model.blocks[0]
    first_target = config_info["target_modules"][0]
    if first_target in ("q_proj", "k_proj", "v_proj", "out_proj"):
        check = getattr(first_block.attn, first_target)
    else:
        check = getattr(first_block.mlp, first_target)

    if not isinstance(check, LoRALinear):
        inject_lora(
            model,
            rank=config_info["rank"],
            alpha=config_info["alpha"],
            target_modules=config_info["target_modules"],
        )

    # Load adapter weights
    adapters = dict(mx.load(str(path / "adapters.npz")))

    for key, value in adapters.items():
        parts = key.split(".")
        # blocks.{idx}.{attn|mlp}.{name}.{lora_a|lora_b}
        block_idx = int(parts[1])
        loc = parts[2]  # "attn" or "mlp"
        name = parts[3]  # "q_proj", "fc1", etc.
        param = parts[4]  # "lora_a" or "lora_b"

        block = model.blocks[block_idx]
        container = block.attn if loc == "attn" else block.mlp
        layer = getattr(container, name)

        if isinstance(layer, LoRALinear):
            setattr(layer, param, value)

    print(f"Loaded adapters from {path}")
    return model
