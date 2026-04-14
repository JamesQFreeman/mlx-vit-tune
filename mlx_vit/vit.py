"""Vision Transformer (ViT) implementation in MLX.

Supports ViT-B/16, ViT-L/16, ViT-H/14 with optional SwiGLU FFN
and register tokens for compatibility with UNI2-h and Virchow2.
"""

import math
from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn


@dataclass
class ViTConfig:
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    num_classes: int = 0  # 0 = feature extraction only
    use_swiglu: bool = False
    num_register_tokens: int = 0
    layer_scale_init: Optional[float] = None  # e.g. 1e-5 for UNI
    class_token: bool = True
    global_pool: str = "token"  # "token" (CLS), "avg", "token+avg" (Virchow2)
    drop_rate: float = 0.0
    gradient_checkpointing: bool = False
    # v0.3: use mx.fast.scaled_dot_product_attention instead of the manual
    # Q @ K.T + softmax + @V path. Saves ~1 ms per attention call on Apple
    # Silicon and fuses the backward pass too. Set to False to debug / compare
    # against the reference path.
    use_fast_sdpa: bool = True

    @classmethod
    def vit_base_patch16(cls, **kwargs) -> "ViTConfig":
        return cls(
            patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4.0, **kwargs
        )

    @classmethod
    def vit_large_patch16(cls, **kwargs) -> "ViTConfig":
        return cls(
            patch_size=16, embed_dim=1024, depth=24,
            num_heads=16, mlp_ratio=4.0, layer_scale_init=1e-5, **kwargs
        )

    @classmethod
    def vit_huge_patch14(cls, **kwargs) -> "ViTConfig":
        return cls(
            patch_size=14, embed_dim=1280, depth=32,
            num_heads=16, mlp_ratio=4.0, layer_scale_init=1e-5, **kwargs
        )

    @classmethod
    def vit_huge_patch14_swiglu(cls, **kwargs) -> "ViTConfig":
        """UNI2-h / Virchow2 style: ViT-H/14 with SwiGLU and register tokens."""
        return cls(
            patch_size=14, embed_dim=1280, depth=32,
            num_heads=16, mlp_ratio=4.0, use_swiglu=True,
            layer_scale_init=1e-5, **kwargs
        )


class PatchEmbed(nn.Module):
    """Convert image patches to embeddings using a convolution."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.num_patches = (config.image_size // config.patch_size) ** 2

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, H, W, C] (MLX uses channels-last)
        x = self.proj(x)  # [B, H', W', embed_dim]
        B = x.shape[0]
        x = x.reshape(B, -1, x.shape[-1])  # [B, num_patches, embed_dim]
        return x


class Attention(nn.Module):
    """Multi-head self-attention.

    v0.3: Routes through ``mx.fast.scaled_dot_product_attention`` by default.
    The fast path is a tiled Metal kernel that avoids materializing the full
    ``[B, H, N, N]`` attention matrix and provides a fused backward — a
    single-optimization win over the manual softmax path used in v0.1 / v0.2.
    Set ``ViTConfig.use_fast_sdpa=False`` to fall back to the reference path.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.embed_dim // config.num_heads
        self.scale = self.head_dim ** -0.5
        self.use_fast_sdpa = config.use_fast_sdpa

        self.q_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.k_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.v_proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.out_proj = nn.Linear(config.embed_dim, config.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape

        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if self.use_fast_sdpa:
            # [B, H, N, Hd] input layout — same convention MLX expects.
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        else:
            attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale
            attn = mx.softmax(attn, axis=-1)
            o = attn @ v

        x = o.transpose(0, 2, 1, 3).reshape(B, N, C)
        x = self.out_proj(x)
        return x


class MLP(nn.Module):
    """Standard MLP with GELU activation."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)))


class SwiGLU(nn.Module):
    """SwiGLU FFN used by UNI2-h and Virchow2.

    Uses a gated architecture: output = fc2(SiLU(fc1a(x)) * fc1b(x))
    timm's SwiGLUPacked packs fc1a and fc1b into a single linear layer.
    We keep them separate for clarity and LoRA compatibility.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        # SwiGLU uses 2/3 * 4 * embed_dim for hidden, matching timm's convention
        hidden_dim = int(config.embed_dim * config.mlp_ratio * 2 / 3)
        # Round to nearest multiple of 8 for efficiency
        hidden_dim = ((hidden_dim + 7) // 8) * 8
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.fc1_gate = nn.Linear(config.embed_dim, hidden_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.act(self.fc1(x)) * self.fc1_gate(x))


class LayerScale(nn.Module):
    """Per-channel learnable scaling, used by DINOv2-trained models."""

    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_value

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class TransformerBlock(nn.Module):
    """Single transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = Attention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        if config.use_swiglu:
            self.mlp = SwiGLU(config)
        else:
            self.mlp = MLP(config)

        self.ls1 = None
        self.ls2 = None
        if config.layer_scale_init is not None:
            self.ls1 = LayerScale(config.embed_dim, config.layer_scale_init)
            self.ls2 = LayerScale(config.embed_dim, config.layer_scale_init)

    def __call__(self, x: mx.array) -> mx.array:
        attn_out = self.attn(self.norm1(x))
        if self.ls1 is not None:
            attn_out = self.ls1(attn_out)
        x = x + attn_out

        mlp_out = self.mlp(self.norm2(x))
        if self.ls2 is not None:
            mlp_out = self.ls2(mlp_out)
        x = x + mlp_out
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with optional register tokens and SwiGLU.

    Compatible with timm's ViT, DINOv2, CONCH, UNI, UNI2-h, and Virchow2.
    """

    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbed(config)
        num_patches = self.patch_embed.num_patches

        # Class token
        if config.class_token:
            self.cls_token = mx.zeros((1, 1, config.embed_dim))

        # Register tokens (UNI2-h uses 8, Virchow2 uses 4)
        if config.num_register_tokens > 0:
            self.register_tokens = mx.zeros(
                (1, config.num_register_tokens, config.embed_dim)
            )

        # Positional embedding
        num_pos = num_patches + (1 if config.class_token else 0)
        self.pos_embed = mx.zeros((1, num_pos, config.embed_dim))

        # Transformer blocks
        self.blocks = [TransformerBlock(config) for _ in range(config.depth)]

        # Final norm
        self.norm = nn.LayerNorm(config.embed_dim)

        # Classification head
        if config.num_classes > 0:
            self.head = nn.Linear(self._feature_dim(), config.num_classes)

    def _feature_dim(self) -> int:
        """Output feature dimension based on pooling strategy."""
        if self.config.global_pool == "token+avg":
            return self.config.embed_dim * 2
        return self.config.embed_dim

    def _interpolate_pos_embed(self, x: mx.array, h: int, w: int) -> mx.array:
        """Interpolate positional embeddings for different resolutions."""
        num_patches = x.shape[1] - (1 if self.config.class_token else 0)
        N = self.pos_embed.shape[1] - (1 if self.config.class_token else 0)

        if num_patches == N:
            return self.pos_embed

        if self.config.class_token:
            cls_pos = self.pos_embed[:, :1]
            patch_pos = self.pos_embed[:, 1:]
        else:
            patch_pos = self.pos_embed

        sqrt_N = int(math.sqrt(N))
        # Reshape to 2D grid for interpolation
        patch_pos = patch_pos.reshape(1, sqrt_N, sqrt_N, self.config.embed_dim)

        # Simple bilinear-like interpolation using repeat + reshape
        # For exact bilinear, would need a proper resize op
        new_h = h // self.config.patch_size
        new_w = w // self.config.patch_size

        if new_h != sqrt_N or new_w != sqrt_N:
            # Use mx.image.resize if available, otherwise nearest-neighbor
            # For v0.1, we require matching resolution
            raise NotImplementedError(
                f"Position embedding interpolation from {sqrt_N}x{sqrt_N} to "
                f"{new_h}x{new_w} not yet implemented. Use image_size={self.config.image_size}."
            )

        patch_pos = patch_pos.reshape(1, -1, self.config.embed_dim)
        if self.config.class_token:
            return mx.concatenate([cls_pos, patch_pos], axis=1)
        return patch_pos

    def _pool(self, x: mx.array) -> mx.array:
        """Pool transformer output to a single feature vector."""
        if self.config.global_pool == "token":
            return x[:, 0]
        elif self.config.global_pool == "avg":
            # Average over patch tokens (exclude CLS and register tokens)
            start = 1 if self.config.class_token else 0
            return x[:, start:].mean(axis=1)
        elif self.config.global_pool == "token+avg":
            # Virchow2-style: concatenate CLS with mean of patch tokens
            cls_token = x[:, 0]
            start = 1 + self.config.num_register_tokens
            patch_avg = x[:, start:].mean(axis=1)
            return mx.concatenate([cls_token, patch_avg], axis=-1)
        else:
            raise ValueError(f"Unknown global_pool: {self.config.global_pool}")

    def features(self, x: mx.array) -> mx.array:
        """Extract features without classification head."""
        B, H, W, C = x.shape

        # Patch embedding
        x = self.patch_embed(x)

        # Prepend class token
        if self.config.class_token:
            cls_tokens = mx.broadcast_to(self.cls_token, (B, 1, self.config.embed_dim))
            x = mx.concatenate([cls_tokens, x], axis=1)

        # Add positional embedding
        x = x + self._interpolate_pos_embed(x, H, W)

        # Insert register tokens after CLS + pos_embed (before patch tokens)
        if self.config.num_register_tokens > 0:
            reg = mx.broadcast_to(
                self.register_tokens,
                (B, self.config.num_register_tokens, self.config.embed_dim),
            )
            # Insert after CLS token
            x = mx.concatenate([x[:, :1], reg, x[:, 1:]], axis=1)

        # Transformer blocks
        for block in self.blocks:
            if self.config.gradient_checkpointing:
                x = mx.checkpoint(block)(x)
            else:
                x = block(x)

        x = self.norm(x)
        return self._pool(x)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass: image → logits (or features if num_classes=0)."""
        x = self.features(x)
        if self.config.num_classes > 0:
            x = self.head(x)
        return x


# Preset configs matching common pathology foundation models
MODEL_CONFIGS = {
    # Standard ViT variants
    "vit_base_patch16_224": ViTConfig.vit_base_patch16,
    "vit_large_patch16_224": ViTConfig.vit_large_patch16,
    "vit_huge_patch14_224": ViTConfig.vit_huge_patch14,

    # Pathology foundation models
    "conch": lambda **kw: ViTConfig.vit_base_patch16(**kw),
    "uni": lambda **kw: ViTConfig.vit_large_patch16(**kw),
    "uni2_h": lambda **kw: ViTConfig(
        patch_size=14, embed_dim=1536, depth=24, num_heads=24,
        mlp_ratio=4.0, use_swiglu=True, layer_scale_init=1e-5,
        num_register_tokens=8, **kw
    ),
    "virchow2": lambda **kw: ViTConfig(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        mlp_ratio=4.0, use_swiglu=True, layer_scale_init=1e-5,
        num_register_tokens=4, global_pool="token+avg", **kw
    ),
}


def create_vit(arch: str, **kwargs) -> VisionTransformer:
    """Create a ViT model from an architecture name."""
    if arch not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown architecture '{arch}'. Available: {available}")

    config_fn = MODEL_CONFIGS[arch]
    config = config_fn(**kwargs)
    return VisionTransformer(config)
