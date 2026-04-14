"""Convert timm/PyTorch ViT weights to MLX safetensors format.

Handles weight key remapping for CONCH (CoCa wrapper), UNI (timm),
Virchow2 (SwiGLUPacked), and generic timm ViTs.
"""

import json
import re
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

from mlx_vit.vit import ViTConfig


def _remap_timm_keys(state_dict: dict, config: ViTConfig) -> dict:
    """Remap timm ViT state_dict keys to our naming convention.

    timm naming:         Our naming:
    blocks.0.attn.qkv    blocks.0.attn.q_proj / k_proj / v_proj
    blocks.0.attn.proj   blocks.0.attn.out_proj
    blocks.0.mlp.fc1     blocks.0.mlp.fc1
    blocks.0.mlp.fc2     blocks.0.mlp.fc2
    blocks.0.norm1       blocks.0.norm1
    blocks.0.norm2       blocks.0.norm2
    blocks.0.ls1.gamma   blocks.0.ls1.gamma
    blocks.0.ls2.gamma   blocks.0.ls2.gamma
    """
    new_state = {}

    for key, value in state_dict.items():
        new_key = key

        # Handle fused QKV → split into q_proj, k_proj, v_proj
        if ".attn.qkv." in key:
            # timm fuses QKV into one linear: [3*embed_dim, embed_dim]
            base = key.replace(".attn.qkv.", ".attn.")
            suffix = "weight" if "weight" in key else "bias"
            dim = config.embed_dim

            if isinstance(value, np.ndarray):
                q, k, v = np.split(value, 3, axis=0)
            else:
                q, k, v = value[:dim], value[dim : 2 * dim], value[2 * dim :]

            new_state[base.replace(f".{suffix}", f".q_proj.{suffix}")] = q
            new_state[base.replace(f".{suffix}", f".k_proj.{suffix}")] = k
            new_state[base.replace(f".{suffix}", f".v_proj.{suffix}")] = v
            continue

        # Handle SwiGLUPacked → split into fc1 and fc1_gate
        if ".mlp.w12." in key and config.use_swiglu:
            # timm's SwiGLUPacked packs gate and up into [2*hidden, embed_dim]
            base = key.replace(".mlp.w12.", ".mlp.")
            suffix = "weight" if "weight" in key else "bias"
            hidden = value.shape[0] // 2 if isinstance(value, np.ndarray) else len(value) // 2

            if isinstance(value, np.ndarray):
                fc1, gate = np.split(value, 2, axis=0)
            else:
                fc1, gate = value[:hidden], value[hidden:]

            new_state[base.replace(f".{suffix}", f".fc1.{suffix}")] = fc1
            new_state[base.replace(f".{suffix}", f".fc1_gate.{suffix}")] = gate
            continue

        # SwiGLUPacked w3 → fc2
        if ".mlp.w3." in key and config.use_swiglu:
            new_key = key.replace(".mlp.w3.", ".mlp.fc2.")

        # attn.proj → attn.out_proj
        elif ".attn.proj." in key:
            new_key = key.replace(".attn.proj.", ".attn.out_proj.")

        # patch_embed.proj stays the same
        # cls_token stays the same
        # pos_embed stays the same
        # norm stays the same (final norm)

        new_state[new_key] = value

    return new_state


def _remap_conch_keys(state_dict: dict, config: ViTConfig) -> dict:
    """Extract vision encoder from CONCH's CoCa wrapper.

    CONCH wraps the ViT in a CoCa model. The vision encoder keys are:
    visual.trunk.* → *
    """
    vision_state = {}
    prefix = "visual.trunk."

    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            vision_state[new_key] = value

    if not vision_state:
        # Try alternative prefix patterns
        for alt_prefix in ["visual.", "model.visual.trunk.", "model.visual."]:
            for key, value in state_dict.items():
                if key.startswith(alt_prefix):
                    new_key = key[len(alt_prefix) :]
                    vision_state[new_key] = value
            if vision_state:
                break

    if not vision_state:
        raise ValueError(
            "Could not find vision encoder weights in CONCH checkpoint. "
            f"Available key prefixes: {set(k.split('.')[0] for k in state_dict)}"
        )

    return _remap_timm_keys(vision_state, config)


def _transpose_conv_weights(state_dict: dict) -> dict:
    """Transpose Conv2d weights from PyTorch [O,I,H,W] to MLX [O,H,W,I]."""
    new_state = {}
    for key, value in state_dict.items():
        if isinstance(value, np.ndarray) and value.ndim == 4 and "patch_embed" in key:
            # PyTorch Conv2d: [out_channels, in_channels, kH, kW]
            # MLX Conv2d: [out_channels, kH, kW, in_channels]
            value = np.transpose(value, (0, 2, 3, 1))
        new_state[key] = value
    return new_state


def _detect_model_type(state_dict: dict) -> str:
    """Auto-detect model type from state_dict key patterns."""
    keys = set(state_dict.keys())
    key_str = " ".join(list(keys)[:50])

    if any(k.startswith("visual.trunk.") for k in keys):
        return "conch"
    if any(k.startswith("visual.") for k in keys):
        return "conch"
    if any(".mlp.w12." in k for k in keys):
        return "swiglu"  # Virchow2 or UNI2-h
    return "timm"  # Standard timm ViT (UNI, Phikon, etc.)


def convert_weights(
    pytorch_path: str,
    output_path: str,
    config: ViTConfig,
    model_type: Optional[str] = None,
) -> Path:
    """Convert PyTorch ViT weights to MLX safetensors.

    Args:
        pytorch_path: Path to PyTorch .bin or .safetensors file
        output_path: Directory to save MLX weights
        config: ViT configuration
        model_type: "timm", "conch", or "swiglu". Auto-detected if None.

    Returns:
        Path to saved weights directory
    """
    from safetensors import safe_open
    from safetensors.numpy import save_file

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load weights
    pytorch_path = str(pytorch_path)
    if pytorch_path.endswith(".safetensors"):
        state_dict = {}
        with safe_open(pytorch_path, framework="numpy") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    elif pytorch_path.endswith(".bin") or pytorch_path.endswith(".pt"):
        import torch

        checkpoint = torch.load(pytorch_path, map_location="cpu", weights_only=True)
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        state_dict = {k: v.numpy() for k, v in checkpoint.items()}
    else:
        raise ValueError(f"Unsupported weight format: {pytorch_path}")

    # Auto-detect model type
    if model_type is None:
        model_type = _detect_model_type(state_dict)
        print(f"Auto-detected model type: {model_type}")

    # Remap keys
    if model_type == "conch":
        state_dict = _remap_conch_keys(state_dict, config)
    else:
        state_dict = _remap_timm_keys(state_dict, config)

    # Transpose conv weights
    state_dict = _transpose_conv_weights(state_dict)

    # Filter out keys we don't use (text encoder, decoder, etc.)
    valid_prefixes = (
        "patch_embed", "cls_token", "pos_embed", "register_tokens",
        "blocks", "norm", "head",
    )
    filtered = {}
    skipped = []
    for key, value in state_dict.items():
        if any(key.startswith(p) for p in valid_prefixes):
            filtered[key] = value
        else:
            skipped.append(key)

    if skipped:
        print(f"Skipped {len(skipped)} non-vision keys: {skipped[:5]}...")

    print(f"Converting {len(filtered)} tensors to MLX format")

    # Save as safetensors
    save_file(filtered, str(output_dir / "model.safetensors"))

    # Save config
    config_dict = {
        "image_size": config.image_size,
        "patch_size": config.patch_size,
        "embed_dim": config.embed_dim,
        "depth": config.depth,
        "num_heads": config.num_heads,
        "mlp_ratio": config.mlp_ratio,
        "use_swiglu": config.use_swiglu,
        "num_register_tokens": config.num_register_tokens,
        "layer_scale_init": config.layer_scale_init,
        "global_pool": config.global_pool,
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)

    total_params = sum(v.size for v in filtered.values())
    total_mb = sum(v.nbytes for v in filtered.values()) / 1024 / 1024
    print(f"Saved: {total_params:,} parameters ({total_mb:.1f} MB) → {output_dir}")

    return output_dir


def load_mlx_weights(weights_dir: str, config: Optional[ViTConfig] = None):
    """Load MLX weights from a directory.

    Returns:
        Tuple of (weights_dict, config)
    """
    weights_dir = Path(weights_dir)

    # Load config if not provided
    if config is None:
        config_path = weights_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                cfg = json.load(f)
            config = ViTConfig(**cfg)
        else:
            raise ValueError(f"No config.json found in {weights_dir} and no config provided")

    # Load weights
    weights_path = weights_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"No model.safetensors found in {weights_dir}")

    weights = mx.load(str(weights_path))
    return weights, config


def download_and_convert(
    hf_model_id: str,
    output_dir: str,
    config: ViTConfig,
    model_type: Optional[str] = None,
    hf_token: Optional[str] = None,
) -> Path:
    """Download a model from HuggingFace and convert to MLX.

    Args:
        hf_model_id: HuggingFace model ID (e.g., "MahmoodLab/UNI")
        output_dir: Local directory to save converted weights
        config: ViT configuration
        model_type: "timm", "conch", or "swiglu"
        hf_token: HuggingFace auth token for gated models

    Returns:
        Path to converted weights directory
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    output_path = Path(output_dir)
    if (output_path / "model.safetensors").exists():
        print(f"Weights already exist at {output_path}, skipping download")
        return output_path

    print(f"Downloading {hf_model_id}...")

    # Find the weights file
    files = list_repo_files(hf_model_id, token=hf_token)
    weight_file = None
    for candidate in ["model.safetensors", "pytorch_model.bin"]:
        if candidate in files:
            weight_file = candidate
            break

    if weight_file is None:
        raise FileNotFoundError(
            f"No model weights found in {hf_model_id}. Files: {files}"
        )

    local_path = hf_hub_download(
        hf_model_id, weight_file, token=hf_token
    )
    print(f"Downloaded to {local_path}")

    return convert_weights(local_path, output_dir, config, model_type)
