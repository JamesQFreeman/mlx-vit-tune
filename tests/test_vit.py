"""Basic tests for ViT architecture and LoRA."""

import sys
sys.path.insert(0, "/Volumes/ExternalHD/mlx-path-foundation-tune")

import mlx.core as mx
import mlx.utils
import numpy as np

from mlx_vit.vit import VisionTransformer, ViTConfig, create_vit
from mlx_vit.lora import inject_lora, merge_lora, LoRALinear


def count_params(model):
    """Count total parameters in an MLX model."""
    leaves = mlx.utils.tree_flatten(model.parameters())
    return sum(v.size for _, v in leaves)


def test_vit_forward():
    """Test basic forward pass with random weights."""
    config = ViTConfig.vit_base_patch16(num_classes=10, image_size=224)
    model = VisionTransformer(config)

    # Random input: [B, H, W, C] (MLX channels-last)
    x = mx.random.normal((2, 224, 224, 3))
    out = model(x)
    mx.eval(out)

    assert out.shape == (2, 10), f"Expected (2, 10), got {out.shape}"
    print(f"[PASS] ViT-B forward: input {x.shape} → output {out.shape}")


def test_vit_feature_extraction():
    """Test feature extraction mode (num_classes=0)."""
    config = ViTConfig.vit_base_patch16(num_classes=0, image_size=224)
    model = VisionTransformer(config)

    x = mx.random.normal((2, 224, 224, 3))
    out = model(x)
    mx.eval(out)

    assert out.shape == (2, 768), f"Expected (2, 768), got {out.shape}"
    print(f"[PASS] ViT-B features: input {x.shape} → output {out.shape}")


def test_vit_large():
    """Test ViT-L architecture."""
    config = ViTConfig.vit_large_patch16(num_classes=5, image_size=224)
    model = VisionTransformer(config)

    x = mx.random.normal((1, 224, 224, 3))
    out = model(x)
    mx.eval(out)

    assert out.shape == (1, 5), f"Expected (1, 5), got {out.shape}"
    print(f"[PASS] ViT-L forward: input {x.shape} → output {out.shape}")


def test_vit_swiglu():
    """Test ViT with SwiGLU FFN (like UNI2-h/Virchow2)."""
    config = ViTConfig(
        patch_size=14, embed_dim=256, depth=4, num_heads=4,
        mlp_ratio=4.0, use_swiglu=True, num_register_tokens=4,
        num_classes=3, image_size=224,
    )
    model = VisionTransformer(config)

    x = mx.random.normal((1, 224, 224, 3))
    out = model(x)
    mx.eval(out)

    assert out.shape == (1, 3), f"Expected (1, 3), got {out.shape}"
    print(f"[PASS] ViT-H SwiGLU forward: input {x.shape} → output {out.shape}")


def test_virchow2_pooling():
    """Test token+avg pooling (Virchow2 style)."""
    config = ViTConfig(
        patch_size=14, embed_dim=256, depth=2,
        num_heads=4, mlp_ratio=4.0,
        use_swiglu=True, num_register_tokens=4,
        global_pool="token+avg", num_classes=3,
        image_size=224,
    )
    model = VisionTransformer(config)

    x = mx.random.normal((1, 224, 224, 3))
    features = model.features(x)
    mx.eval(features)

    # token+avg concatenates CLS (256) + avg_patches (256) = 512
    assert features.shape == (1, 512), f"Expected (1, 512), got {features.shape}"
    print(f"[PASS] Virchow2 pooling: features shape {features.shape}")


def test_lora_injection():
    """Test LoRA injection and parameter counting."""
    config = ViTConfig.vit_base_patch16(num_classes=10, image_size=224)
    model = VisionTransformer(config)

    # Count params before LoRA
    total_before = count_params(model)

    # Inject LoRA
    model, trainable = inject_lora(model, rank=8, alpha=8.0)

    # Verify LoRA layers exist
    first_block = model.blocks[0]
    assert isinstance(first_block.attn.q_proj, LoRALinear), "q_proj should be LoRALinear"
    assert isinstance(first_block.mlp.fc1, LoRALinear), "fc1 should be LoRALinear"

    # Forward should still work
    x = mx.random.normal((2, 224, 224, 3))
    out = model(x)
    mx.eval(out)
    assert out.shape == (2, 10)

    print(f"[PASS] LoRA injection: {trainable:,} trainable params, forward works")


def test_lora_merge():
    """Test that merged LoRA produces same output as unmerged."""
    config = ViTConfig(
        patch_size=16, embed_dim=64, depth=2,
        num_heads=4, num_classes=5, image_size=32,
    )
    model = VisionTransformer(config)

    # Inject LoRA
    model, _ = inject_lora(model, rank=4, alpha=4.0)

    # Forward pass
    x = mx.random.normal((1, 32, 32, 3))
    out_lora = model(x)
    mx.eval(out_lora)

    # Merge and compare
    model = merge_lora(model)
    out_merged = model(x)
    mx.eval(out_merged)

    diff = mx.abs(out_lora - out_merged).max().item()
    assert diff < 1e-3, f"Merge mismatch: max diff = {diff}"
    print(f"[PASS] LoRA merge: max diff = {diff:.2e}")


def test_param_count():
    """Verify parameter counts match expected values for ViT-B."""
    config = ViTConfig.vit_base_patch16(num_classes=0, image_size=224)
    model = VisionTransformer(config)

    total = count_params(model)
    # ViT-B/16 should have ~86M params (without head)
    assert 80_000_000 < total < 95_000_000, f"ViT-B param count {total:,} out of range"
    print(f"[PASS] ViT-B param count: {total:,} (expected ~86M)")


if __name__ == "__main__":
    print("Running ViT + LoRA tests...\n")
    test_vit_forward()
    test_vit_feature_extraction()
    test_vit_large()
    test_vit_swiglu()
    test_virchow2_pooling()
    test_lora_injection()
    test_lora_merge()
    test_param_count()
    print("\nAll tests passed!")
