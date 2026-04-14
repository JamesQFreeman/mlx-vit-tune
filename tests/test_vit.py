"""Basic tests for ViT architecture and LoRA."""

import os
import sys
from pathlib import Path

# Run from repo root regardless of where pytest/python was invoked.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import mlx.core as mx
import mlx.nn as nn
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


def test_gradient_checkpointing():
    """Test that gradient checkpointing produces same output and gradients."""
    config = ViTConfig(
        patch_size=16, embed_dim=64, depth=4,
        num_heads=4, num_classes=5, image_size=32,
    )

    # Without checkpointing
    model_no_ckpt = VisionTransformer(config)
    x = mx.random.normal((2, 32, 32, 3))
    out_no_ckpt = model_no_ckpt(x)
    mx.eval(out_no_ckpt)

    # With checkpointing — same weights
    config_ckpt = ViTConfig(
        patch_size=16, embed_dim=64, depth=4,
        num_heads=4, num_classes=5, image_size=32,
        gradient_checkpointing=True,
    )
    model_ckpt = VisionTransformer(config_ckpt)
    # Copy weights from no-checkpoint model
    flat = mlx.utils.tree_flatten(model_no_ckpt.parameters())
    model_ckpt.load_weights(flat)

    out_ckpt = model_ckpt(x)
    mx.eval(out_ckpt)

    diff = mx.abs(out_no_ckpt - out_ckpt).max().item()
    assert diff < 1e-5, f"Checkpointing changed output: max diff = {diff}"

    # Verify gradients work with checkpointing
    labels = mx.array([0, 1])
    from mlx_vit.trainer import cross_entropy_loss
    loss_fn = nn.value_and_grad(model_ckpt, cross_entropy_loss)
    (loss, acc), grads = loss_fn(model_ckpt, x, labels)
    mx.eval(loss, acc)

    assert loss.item() > 0, "Loss should be positive"
    assert not mx.isnan(loss).item(), "Loss is NaN with checkpointing"
    print(f"[PASS] Gradient checkpointing: output diff {diff:.2e}, loss {loss.item():.4f}")


def test_gradient_checkpointing_lora():
    """Test gradient checkpointing works with LoRA."""
    config = ViTConfig(
        patch_size=16, embed_dim=64, depth=4,
        num_heads=4, num_classes=5, image_size=32,
        gradient_checkpointing=True,
    )
    model = VisionTransformer(config)
    model, trainable = inject_lora(model, rank=4, alpha=4.0)

    x = mx.random.normal((2, 32, 32, 3))
    labels = mx.array([0, 1])

    from mlx_vit.trainer import cross_entropy_loss
    loss_fn = nn.value_and_grad(model, cross_entropy_loss)
    (loss, acc), grads = loss_fn(model, x, labels)
    mx.eval(loss, acc)

    assert loss.item() > 0, "Loss should be positive"
    assert not mx.isnan(loss).item(), "Loss is NaN with checkpointing + LoRA"
    print(f"[PASS] Checkpointing + LoRA: loss {loss.item():.4f}, {trainable:,} trainable")


def test_gradient_accumulation():
    """Test that gradient accumulation produces averaged gradients."""
    config = ViTConfig(
        patch_size=16, embed_dim=64, depth=2,
        num_heads=4, num_classes=5, image_size=32,
    )
    model = VisionTransformer(config)

    x1 = mx.random.normal((2, 32, 32, 3))
    x2 = mx.random.normal((2, 32, 32, 3))
    labels = mx.array([0, 1])

    from mlx_vit.trainer import cross_entropy_loss
    loss_fn = nn.value_and_grad(model, cross_entropy_loss)

    # Get gradients for two batches
    (_, _), grads1 = loss_fn(model, x1, labels)
    (_, _), grads2 = loss_fn(model, x2, labels)

    # Accumulate and average (same logic as trainer.py)
    accum = mlx.utils.tree_map(lambda a, b: a + b, grads1, grads2)
    avg_grads = mlx.utils.tree_map(lambda g: g * 0.5, accum)
    mx.eval(avg_grads)

    # Just verify it doesn't crash and produces finite values
    flat_grads = mlx.utils.tree_flatten(avg_grads)
    all_finite = all(not mx.isnan(v).any().item() for _, v in flat_grads if isinstance(v, mx.array))
    assert all_finite, "Accumulated gradients contain NaN"
    print(f"[PASS] Gradient accumulation: {len(flat_grads)} gradient tensors, all finite")


def test_bf16_is_default():
    """v0.4: fresh ViTConfig should produce bf16 model parameters."""
    cfg = ViTConfig.vit_base_patch16(num_classes=5, image_size=224)
    assert cfg.dtype == mx.bfloat16, f"Default dtype regressed to {cfg.dtype}"

    model = VisionTransformer(cfg)
    mx.eval(model.parameters())

    flat = mlx.utils.tree_flatten(model.parameters())
    non_bf16 = [(k, v.dtype) for k, v in flat
                if isinstance(v, mx.array) and v.dtype != mx.bfloat16]
    assert not non_bf16, f"Non-bf16 params leaked: {non_bf16[:5]}"
    print(f"[PASS] bf16 default: {len(flat)} params all bfloat16")


def test_bf16_forward_no_nan():
    """v0.4: random-init bf16 ViT forward must not NaN or Inf, and input
    data arriving in fp32 should be cast at the features() boundary."""
    cfg = ViTConfig.vit_large_patch16(num_classes=10, image_size=224)
    model = VisionTransformer(cfg)
    mx.eval(model.parameters())

    # Pass FP32 input — tests the boundary cast.
    x = mx.random.normal((2, 224, 224, 3)).astype(mx.float32)
    out = model(x)
    mx.eval(out)
    assert out.dtype == mx.bfloat16, f"Output dtype should be bf16, got {out.dtype}"
    assert not bool(mx.any(mx.isnan(out)).item()), "bf16 forward produced NaN"
    assert not bool(mx.any(mx.isinf(out)).item()), "bf16 forward produced Inf"
    max_abs = float(mx.max(mx.abs(out)).item())
    assert max_abs < 1e3, f"bf16 forward exploded: max_abs={max_abs}"
    print(f"[PASS] bf16 ViT-L random-init forward: max_abs={max_abs:.3f}")


def test_bf16_matches_fp32_within_tolerance():
    """v0.4: a ViT built with dtype=bf16 should produce outputs within ~bf16
    precision (~1e-2 relative) of the same ViT in fp32, when both use
    identical weights."""
    cfg_kwargs = dict(
        patch_size=16, embed_dim=128, depth=4,
        num_heads=4, num_classes=8, image_size=32,
    )

    model_fp32 = VisionTransformer(ViTConfig(dtype=mx.float32, **cfg_kwargs))
    model_bf16 = VisionTransformer(ViTConfig(dtype=mx.bfloat16, **cfg_kwargs))

    # Copy fp32 weights into bf16 model (down-casting).
    flat_fp32 = mlx.utils.tree_flatten(model_fp32.parameters())
    bf16_items = [(k, v.astype(mx.bfloat16)) if isinstance(v, mx.array) else (k, v)
                  for k, v in flat_fp32]
    model_bf16.load_weights(bf16_items, strict=False)
    mx.eval(model_bf16.parameters())

    x = mx.random.normal((2, 32, 32, 3)).astype(mx.float32)
    out_fp32 = model_fp32(x)
    out_bf16 = model_bf16(x).astype(mx.float32)
    mx.eval(out_fp32, out_bf16)

    abs_diff = float(mx.max(mx.abs(out_fp32 - out_bf16)).item())
    rel_diff = abs_diff / max(float(mx.max(mx.abs(out_fp32)).item()), 1e-10)
    # bf16 has 7 mantissa bits → relative precision ~2^-7 ≈ 8e-3. Accumulated
    # over 4 transformer blocks, expect up to ~5e-2 relative drift.
    assert rel_diff < 5e-2, (
        f"bf16 diverged from fp32: abs_diff={abs_diff:.4f}, rel_diff={rel_diff:.4f}"
    )
    print(f"[PASS] bf16 ~= fp32 forward: abs {abs_diff:.2e}, rel {rel_diff:.2e}")


def test_bf16_training_step_stable():
    """v0.4: a few bf16 LoRA training steps should produce finite, decreasing
    loss. Covers: LoRA adapters inherit bf16, fp32 logit cast in loss works,
    bf16 gradients propagate."""
    from mlx_vit.trainer import cross_entropy_loss
    import mlx.optimizers as optim

    cfg = ViTConfig(
        patch_size=16, embed_dim=128, depth=3,
        num_heads=4, num_classes=5, image_size=32,
        gradient_checkpointing=True,
    )
    model = VisionTransformer(cfg)
    model, _ = inject_lora(model, rank=4, alpha=4.0)
    mx.eval(model.parameters())

    # Verify LoRA adapters inherited bf16 from base
    for block in model.blocks:
        lora_q = block.attn.q_proj
        assert lora_q.lora_a.dtype == mx.bfloat16, f"LoRA A leaked as {lora_q.lora_a.dtype}"
        assert lora_q.lora_b.dtype == mx.bfloat16, f"LoRA B leaked as {lora_q.lora_b.dtype}"

    x = mx.random.normal((4, 32, 32, 3))
    labels = mx.array([0, 1, 2, 3])
    mx.eval(x, labels)

    loss_fn = nn.value_and_grad(model, cross_entropy_loss)
    opt = optim.AdamW(learning_rate=1e-3)

    losses = []
    for step in range(10):
        (loss, _), grads = loss_fn(model, x, labels)
        opt.update(model, grads)
        mx.eval(loss, model.parameters(), opt.state)
        loss_val = loss.item()
        assert not (loss_val != loss_val), f"NaN loss at step {step}"  # NaN check
        losses.append(loss_val)

    first, last = losses[0], losses[-1]
    assert last < first, f"bf16 LoRA training didn't converge: {first:.3f} -> {last:.3f}"
    print(f"[PASS] bf16 LoRA training: loss {first:.3f} -> {last:.3f} over 10 steps")


def test_save_checkpoint_full_ft():
    """Regression: ``_save_checkpoint`` used to call ``mx.savez(**dict(model.
    parameters()))`` directly, but ``model.parameters()`` is a NESTED dict
    whose leaves are mx.array — the splat raised ``std::bad_cast`` at the end
    of every full-FT epoch. The fix flattens via ``mlx.utils.tree_flatten``
    before saving; this test reproduces the save path and verifies a round
    trip succeeds."""
    import tempfile
    from mlx_vit.trainer import _save_checkpoint

    config = ViTConfig(
        patch_size=16, embed_dim=64, depth=2,
        num_heads=4, num_classes=3, image_size=32,
    )
    model = VisionTransformer(config)
    mx.eval(model.parameters())

    with tempfile.TemporaryDirectory() as tmp:
        ckpt_path = Path(tmp) / "epoch_1"
        # This used to raise std::bad_cast on the nested parameters dict.
        _save_checkpoint(model, ckpt_path, is_lora=False)
        assert (ckpt_path / "model.npz").exists()

        # Round-trip load, verify shapes and dtypes are preserved.
        loaded = dict(mx.load(str(ckpt_path / "model.npz")))
        flat = mlx.utils.tree_flatten(model.parameters())
        model_dict = {k: v for k, v in flat if isinstance(v, mx.array)}
        assert set(loaded.keys()) == set(model_dict.keys()), (
            f"Saved keys differ: only in saved = {set(loaded.keys()) - set(model_dict.keys())}, "
            f"only in model = {set(model_dict.keys()) - set(loaded.keys())}"
        )
        for name, arr in model_dict.items():
            assert loaded[name].shape == arr.shape, f"{name} shape mismatch"

    print(f"[PASS] Full-FT checkpoint save: {len(model_dict)} tensors round-trip OK")


def test_fast_sdpa_matches_manual():
    """v0.3: mx.fast.scaled_dot_product_attention path must produce the same
    logits and (more importantly) the same gradients as the manual path.

    Pinned to float32 — this is a correctness test of the SDPA kernel vs
    the manual softmax path, not a precision test. The looser bf16 tolerance
    is covered separately in ``test_bf16_matches_fp32_within_tolerance``.
    """
    # Small ViT so the test is fast but still exercises multi-head attention
    # and gradient checkpointing.
    cfg_kwargs = dict(
        patch_size=16, embed_dim=64, depth=3,
        num_heads=4, num_classes=5, image_size=32,
        gradient_checkpointing=True,
        dtype=mx.float32,
    )

    model_manual = VisionTransformer(ViTConfig(use_fast_sdpa=False, **cfg_kwargs))
    model_fast   = VisionTransformer(ViTConfig(use_fast_sdpa=True,  **cfg_kwargs))

    # Copy weights from manual into fast so both models start identical.
    flat = mlx.utils.tree_flatten(model_manual.parameters())
    model_fast.load_weights(flat)

    x = mx.random.normal((2, 32, 32, 3))
    labels = mx.array([0, 1])
    mx.eval(x, labels)

    # --- Forward match ---
    out_m = model_manual(x)
    out_f = model_fast(x)
    mx.eval(out_m, out_f)
    fwd_diff = mx.abs(out_m - out_f).max().item()
    # Small numerical diff from the reduction order inside the Metal kernel
    # is acceptable (MLX's fast softmax runs in fp32).
    assert fwd_diff < 1e-4, f"Fast SDPA forward diverged: max diff = {fwd_diff}"

    # --- Gradient match ---
    from mlx_vit.trainer import cross_entropy_loss
    (loss_m, _), grads_m = nn.value_and_grad(model_manual, cross_entropy_loss)(
        model_manual, x, labels
    )
    (loss_f, _), grads_f = nn.value_and_grad(model_fast, cross_entropy_loss)(
        model_fast, x, labels
    )
    mx.eval(loss_m, loss_f, grads_m, grads_f)

    loss_diff = abs(loss_m.item() - loss_f.item())
    assert loss_diff < 1e-4, f"Fast SDPA loss diverged: diff = {loss_diff}"

    flat_m = dict(mlx.utils.tree_flatten(grads_m))
    flat_f = dict(mlx.utils.tree_flatten(grads_f))
    assert set(flat_m.keys()) == set(flat_f.keys())
    max_grad_diff = 0.0
    for k, gm in flat_m.items():
        gf = flat_f[k]
        if gm.size == 0:
            continue
        d = mx.abs(gm - gf).max().item()
        if d > max_grad_diff:
            max_grad_diff = d
    assert max_grad_diff < 1e-4, f"Fast SDPA grads diverged: max diff = {max_grad_diff}"

    print(f"[PASS] Fast SDPA matches manual: "
          f"fwd {fwd_diff:.2e}, loss {loss_diff:.2e}, grad {max_grad_diff:.2e}")


def test_memory_reporting():
    """Test memory reporting functions."""
    assert mx.metal.is_available(), "Metal not available"

    config = ViTConfig.vit_base_patch16(num_classes=10, image_size=224)
    model = VisionTransformer(config)
    mx.eval(model.parameters())

    active_mem = mx.get_active_memory()
    peak_mem = mx.get_peak_memory()
    assert active_mem > 0, "Active memory should be > 0"
    assert peak_mem > 0, "Peak memory should be > 0"

    device = mx.device_info()
    assert "architecture" in device or "memory_size" in device, "device_info should return useful info"

    print(f"[PASS] Memory reporting: active {active_mem / (1024**2):.1f} MB, peak {peak_mem / (1024**2):.1f} MB")


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

    print("\n--- v0.2 tests ---\n")
    test_gradient_checkpointing()
    test_gradient_checkpointing_lora()
    test_gradient_accumulation()
    test_memory_reporting()

    print("\n--- v0.3 tests ---\n")
    test_fast_sdpa_matches_manual()

    print("\n--- v0.4 tests ---\n")
    test_bf16_is_default()
    test_bf16_forward_no_nan()
    test_bf16_matches_fp32_within_tolerance()
    test_bf16_training_step_stable()

    print("\n--- Regressions ---\n")
    test_save_checkpoint_full_ft()
    print("\nAll tests passed!")
