"""FastViTModel — Unsloth-like API for ViT fine-tuning on MLX.

Usage:
    from mlx_vit import FastViTModel

    # Load a pretrained model
    model = FastViTModel.from_pretrained("MahmoodLab/UNI", num_classes=2)

    # Add LoRA adapters
    model = FastViTModel.get_lora_model(model, rank=8, target_modules="all")

    # Train (see trainer.py)
    # Save
    FastViTModel.save_pretrained(model, "my_model")
"""

import os
from pathlib import Path
from typing import Optional, Union

import mlx.core as mx
import mlx.utils

from mlx_vit.convert import convert_weights, download_and_convert, load_mlx_weights
from mlx_vit.lora import (
    DEFAULT_TARGET_MODULES,
    SWIGLU_TARGET_MODULES,
    inject_lora,
    load_adapters,
    merge_lora,
    save_adapters,
)
from mlx_vit.vit import MODEL_CONFIGS, ViTConfig, VisionTransformer, create_vit


# Known HuggingFace model IDs → architecture mapping
HF_MODEL_REGISTRY = {
    "MahmoodLab/conch": "conch",
    "MahmoodLab/UNI": "uni",
    "MahmoodLab/UNI2-h": "uni2_h",
    "paige-ai/Virchow2": "virchow2",
    "owkin/phikon": "vit_base_patch16_224",
}

# Known model types for weight conversion
HF_MODEL_TYPES = {
    "MahmoodLab/conch": "conch",
    "MahmoodLab/UNI": "timm",
    "MahmoodLab/UNI2-h": "timm",
    "paige-ai/Virchow2": "swiglu",
    "owkin/phikon": "timm",
}


class FastViTModel:
    """Unsloth-like interface for loading, adapting, and saving ViT models."""

    @staticmethod
    def from_pretrained(
        model_name_or_path: str,
        num_classes: int = 0,
        image_size: int = 224,
        arch: Optional[str] = None,
        hf_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        dtype: str = "float32",
        gradient_checkpointing: bool = False,
    ) -> VisionTransformer:
        """Load a pretrained ViT model.

        Args:
            model_name_or_path: HuggingFace model ID, local path, or architecture name.
                Examples:
                    "MahmoodLab/UNI" — download from HuggingFace
                    "/path/to/weights" — local converted weights
                    "conch" — architecture name (random weights)
            num_classes: Number of output classes (0 = feature extraction only)
            image_size: Input image resolution
            arch: Architecture override (e.g., "vit_base_patch16_224")
            hf_token: HuggingFace auth token for gated models
            cache_dir: Directory to cache converted weights
            dtype: Weight dtype ("float16", "bfloat16", "float32")

        Returns:
            VisionTransformer model with loaded weights
        """
        # Resolve HF token from environment if not provided
        if hf_token is None:
            hf_token = os.environ.get("HF_TOKEN")

        local_path = Path(model_name_or_path)

        # Case 1: Local directory with converted weights
        if local_path.is_dir() and (local_path / "model.safetensors").exists():
            print(f"Loading from local path: {local_path}")
            weights, config = load_mlx_weights(str(local_path))
            config.num_classes = num_classes
            config.image_size = image_size
            config.gradient_checkpointing = gradient_checkpointing
            model = VisionTransformer(config)
            model.load_weights(list(weights.items()))
            return _cast_model(model, dtype)

        # Case 2: Architecture name (random weights)
        if model_name_or_path in MODEL_CONFIGS:
            print(f"Creating model with random weights: {model_name_or_path}")
            model = create_vit(
                model_name_or_path, num_classes=num_classes,
                image_size=image_size, gradient_checkpointing=gradient_checkpointing,
            )
            return _cast_model(model, dtype)

        # Case 3: HuggingFace model ID
        if "/" in model_name_or_path:
            # Resolve architecture
            if arch is None:
                arch = HF_MODEL_REGISTRY.get(model_name_or_path)
                if arch is None:
                    raise ValueError(
                        f"Unknown HuggingFace model: {model_name_or_path}. "
                        f"Known models: {list(HF_MODEL_REGISTRY.keys())}. "
                        f"Pass arch= to specify architecture manually."
                    )

            model_type = HF_MODEL_TYPES.get(model_name_or_path)

            # Create config
            config_fn = MODEL_CONFIGS[arch]
            config = config_fn(
                num_classes=num_classes, image_size=image_size,
                gradient_checkpointing=gradient_checkpointing,
            )

            # Determine cache directory
            if cache_dir is None:
                cache_dir = str(Path.home() / ".cache" / "mlx_vit" / model_name_or_path.replace("/", "_"))

            # Download and convert
            weights_dir = download_and_convert(
                model_name_or_path,
                cache_dir,
                config,
                model_type=model_type,
                hf_token=hf_token,
            )

            # Load
            weights, _ = load_mlx_weights(str(weights_dir), config)
            model = VisionTransformer(config)

            # Load weights (skip classification head if not present in checkpoint)
            model_items = [(k, v) for k, v in weights.items()]
            model.load_weights(model_items, strict=False)

            # model.parameters() is NESTED — tree_flatten to walk leaves.
            flat = mlx.utils.tree_flatten(model.parameters())
            total_params = sum(v.size for _, v in flat if isinstance(v, mx.array))
            print(f"Loaded {model_name_or_path}: {total_params:,} parameters")
            return _cast_model(model, dtype)

        raise ValueError(
            f"Cannot resolve model: {model_name_or_path}. "
            f"Provide a HuggingFace ID (e.g., 'MahmoodLab/UNI'), "
            f"a local path, or an architecture name."
        )

    @staticmethod
    def get_lora_model(
        model: VisionTransformer,
        rank: int = 8,
        alpha: Optional[float] = None,
        dropout: float = 0.0,
        target_modules: Union[str, list[str]] = "all",
    ) -> VisionTransformer:
        """Add LoRA adapters to a ViT model.

        Args:
            model: The pretrained ViT model
            rank: LoRA rank (default 8)
            alpha: LoRA scaling factor (default = rank)
            dropout: LoRA dropout rate
            target_modules: Which layers to target:
                "all" — all linear layers (Q,K,V,O,fc1,fc2) [recommended]
                "attention" — attention layers only (Q,K,V,O)
                "mlp" — MLP layers only (fc1,fc2)
                list — explicit list of module names

        Returns:
            Model with LoRA adapters injected
        """
        if alpha is None:
            alpha = float(rank)

        if isinstance(target_modules, str):
            if target_modules == "all":
                if model.config.use_swiglu:
                    target_modules = SWIGLU_TARGET_MODULES
                else:
                    target_modules = DEFAULT_TARGET_MODULES
            elif target_modules == "attention":
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            elif target_modules == "mlp":
                if model.config.use_swiglu:
                    target_modules = ["fc1", "fc1_gate", "fc2"]
                else:
                    target_modules = ["fc1", "fc2"]
            else:
                raise ValueError(f"Unknown target_modules shorthand: {target_modules}")

        model, num_trainable = inject_lora(model, rank, alpha, dropout, target_modules)
        return model

    @staticmethod
    def save_pretrained(model: VisionTransformer, path: str):
        """Save model (adapters only if LoRA, full weights otherwise)."""
        from mlx_vit.lora import LoRALinear
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        has_lora = any(
            isinstance(getattr(block.attn, "q_proj", None), LoRALinear)
            for block in model.blocks
        )

        if has_lora:
            save_adapters(model, str(path))
        else:
            # model.parameters() is nested; savez wants flat name=array kwargs.
            flat = mlx.utils.tree_flatten(model.parameters())
            weights = {name: arr for name, arr in flat if isinstance(arr, mx.array)}
            mx.savez(str(path / "model.npz"), **weights)
        print(f"Saved model → {path}")

    @staticmethod
    def save_pretrained_merged(model: VisionTransformer, path: str):
        """Merge LoRA adapters and save full model weights."""
        model = merge_lora(model)
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        flat = mlx.utils.tree_flatten(model.parameters())
        weights = {name: arr for name, arr in flat if isinstance(arr, mx.array)}
        mx.savez(str(path / "model.npz"), **weights)
        print(f"Saved merged model → {path}")

    @staticmethod
    def load_adapters(model: VisionTransformer, path: str) -> VisionTransformer:
        """Load LoRA adapters from disk."""
        return load_adapters(model, path)


_DTYPE_ALIASES = {
    "float16": mx.float16, "fp16": mx.float16, "half": mx.float16,
    "bfloat16": mx.bfloat16, "bf16": mx.bfloat16,
    "float32": mx.float32, "fp32": mx.float32, "float": mx.float32,
}


def _resolve_dtype(dtype) -> "mx.Dtype":
    """Accept a string alias (``'bf16'``, ``'bfloat16'``, ``'fp32'``, ...) or
    an ``mx.Dtype`` directly, and return the canonical ``mx.Dtype``. Raises
    ``ValueError`` on unknown input — no silent fallback (this used to be a
    bug where unknown strings became ``mx.float16``)."""
    if isinstance(dtype, mx.Dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.lower()
        if key in _DTYPE_ALIASES:
            return _DTYPE_ALIASES[key]
        raise ValueError(
            f"Unknown dtype {dtype!r}. Valid options: "
            f"{sorted(set(_DTYPE_ALIASES.keys()))}."
        )
    raise TypeError(
        f"dtype must be an mx.Dtype or string alias, got {type(dtype).__name__}"
    )


def _cast_model(model: VisionTransformer, dtype) -> VisionTransformer:
    """Cast all model weights in-place to ``dtype`` (string alias or mx.Dtype)."""
    import mlx.utils

    target = _resolve_dtype(dtype)

    flat = mlx.utils.tree_flatten(model.parameters())
    casted = [
        (k, v.astype(target) if isinstance(v, mx.array) and v.dtype != target else v)
        for k, v in flat
    ]
    model.load_weights(casted, strict=False)
    return model
