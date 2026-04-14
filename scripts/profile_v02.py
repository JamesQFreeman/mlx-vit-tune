"""v0.3 baseline profiler.

Measures where time goes in a single LoRA training step on both ViT-B and
ViT-L so we can pick the single highest-leverage optimization for v0.3.

Breakdown levels (each measured with mx.eval barriers so timings are real
wall-clock GPU work, not lazy-graph construction):

  Level 0  — full forward, full backward (value_and_grad)
  Level 1  — forward-only: patch_embed, blocks, norm+head
  Level 2  — one transformer block: norm1, attn, norm2, mlp
  Level 3  — one attention block: q/k/v projections, attn math, out_proj
  Level 4  — attn math alone: manual path vs mx.fast.scaled_dot_product_attention

All times are averaged over 20 iterations after 5 warmups.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
from mlx_vit.vit import VisionTransformer, ViTConfig
from mlx_vit.lora import inject_lora
from mlx_vit.trainer import cross_entropy_loss


WARMUP = 5
ITERS = 20


def _eval_result(out):
    if out is None:
        return
    if isinstance(out, (tuple, list)):
        mx.eval(*out)
    else:
        mx.eval(out)


def timed(fn, *args, warmup=WARMUP, iters=ITERS, **kwargs):
    """Run fn(*args, **kwargs), force eval, return mean seconds over iters.

    fn is expected to either return a fully-eval'd result itself (train_step
    and its friends above) or return a lazy graph we can eval here.
    """
    for _ in range(warmup):
        _eval_result(fn(*args, **kwargs))

    t0 = time.perf_counter()
    for _ in range(iters):
        _eval_result(fn(*args, **kwargs))
    t1 = time.perf_counter()
    return (t1 - t0) / iters


def build_lora_model(arch: str, checkpoint: bool = True):
    if arch == "vit_b":
        cfg = ViTConfig.vit_base_patch16(num_classes=10, image_size=224,
                                         gradient_checkpointing=checkpoint)
    else:
        cfg = ViTConfig.vit_large_patch16(num_classes=10, image_size=224,
                                          gradient_checkpointing=checkpoint)
    model = VisionTransformer(cfg)
    model, _ = inject_lora(model, rank=8, alpha=8.0)
    mx.eval(model.parameters())
    return model, cfg


def profile_arch(arch: str, batch_size: int):
    print()
    print("=" * 80)
    print(f"  {arch.upper()}  bs={batch_size}  (LoRA, fp32, with gradient checkpointing)")
    print("=" * 80)

    model, cfg = build_lora_model(arch, checkpoint=True)
    D = cfg.embed_dim
    H = cfg.num_heads
    Hd = D // H
    N = (cfg.image_size // cfg.patch_size) ** 2 + (1 if cfg.class_token else 0)

    x = mx.random.normal((batch_size, cfg.image_size, cfg.image_size, 3))
    labels = mx.random.randint(0, cfg.num_classes, (batch_size,))
    mx.eval(x, labels)

    # ---- Level 0: full train step ----
    loss_fn = nn.value_and_grad(model, cross_entropy_loss)
    opt = optim.AdamW(learning_rate=1e-4)
    # one warmup of value_and_grad to build the graph
    (loss, _), grads = loss_fn(model, x, labels)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state)

    def train_step():
        (l, _), g = loss_fn(model, x, labels)
        opt.update(model, g)
        # Force eval of the full gradient tree AND the updated parameters/state.
        mx.eval(l, model.parameters(), opt.state)
        return l

    def fwd_only():
        out = model(x)
        mx.eval(out)
        return out

    def fwd_and_grad():
        (l, _), g = loss_fn(model, x, labels)
        mx.eval(l, g)
        return l

    t_train = timed(train_step)
    t_fwd   = timed(fwd_only)
    t_fwdg  = timed(fwd_and_grad)
    t_bwd_plus_opt = t_train - t_fwdg

    print(f"  full training step : {t_train*1000:8.2f} ms   ({batch_size/t_train:6.1f} img/s)")
    print(f"    forward only     : {t_fwd*1000:8.2f} ms   ({t_fwd/t_train*100:5.1f}%)")
    print(f"    fwd + grad       : {t_fwdg*1000:8.2f} ms")
    print(f"    bwd + opt.update : {t_bwd_plus_opt*1000:8.2f} ms   ({t_bwd_plus_opt/t_train*100:5.1f}%)")

    # ---- Level 1: forward components ----
    print()
    print("  Forward breakdown (single forward call):")

    x_patches = model.patch_embed(x)
    cls = mx.broadcast_to(model.cls_token, (batch_size, 1, D))
    x_tok = mx.concatenate([cls, x_patches], axis=1) + model.pos_embed
    mx.eval(x_patches, x_tok)

    def run_patch_embed():
        return model.patch_embed(x)

    def run_all_blocks():
        y = x_tok
        for block in model.blocks:
            y = block(y)
        return y

    def run_norm_head():
        y = model.norm(x_tok)
        return model.head(y[:, 0])

    t_patch = timed(run_patch_embed)
    t_blocks = timed(run_all_blocks)
    t_norm_head = timed(run_norm_head)
    total = t_patch + t_blocks + t_norm_head
    print(f"    patch_embed    : {t_patch*1000:8.2f} ms   ({t_patch/total*100:5.1f}%)")
    print(f"    all {cfg.depth:2d} blocks : {t_blocks*1000:8.2f} ms   ({t_blocks/total*100:5.1f}%)")
    print(f"    norm + head    : {t_norm_head*1000:8.2f} ms   ({t_norm_head/total*100:5.1f}%)")

    # ---- Level 2: one block breakdown ----
    print()
    print("  Single block breakdown (block[0]):")
    block = model.blocks[0]
    y = x_tok

    def run_norm1():
        return block.norm1(y)

    def run_attn_only():
        return block.attn(block.norm1(y))

    def run_norm2():
        return block.norm2(y)

    def run_mlp_only():
        return block.mlp(block.norm2(y))

    t_norm1 = timed(run_norm1)
    t_attn  = timed(run_attn_only)
    t_norm2 = timed(run_norm2)
    t_mlp   = timed(run_mlp_only)
    t_block = t_norm1 + t_attn + t_norm2 + t_mlp
    # minus norm contributions from attn and mlp to avoid double-count in %
    t_attn_pure = t_attn - t_norm1
    t_mlp_pure = t_mlp - t_norm2
    t_bk = t_norm1 + t_attn_pure + t_norm2 + t_mlp_pure
    print(f"    norm1          : {t_norm1*1000:8.2f} ms   ({t_norm1/t_bk*100:5.1f}%)")
    print(f"    attn (pure)    : {t_attn_pure*1000:8.2f} ms   ({t_attn_pure/t_bk*100:5.1f}%)")
    print(f"    norm2          : {t_norm2*1000:8.2f} ms   ({t_norm2/t_bk*100:5.1f}%)")
    print(f"    mlp (pure)     : {t_mlp_pure*1000:8.2f} ms   ({t_mlp_pure/t_bk*100:5.1f}%)")
    print(f"    block sum      : {t_bk*1000:8.2f} ms  (x{cfg.depth} blocks ~ {t_bk*cfg.depth*1000:.1f} ms / forward)")

    # ---- Level 3: attention subcomponents ----
    print()
    print("  Attention subcomponents (inside block[0].attn):")
    attn = block.attn
    x_in = block.norm1(y)
    mx.eval(x_in)
    B, Nseq, C = x_in.shape

    def run_q_proj():
        return attn.q_proj(x_in)

    def run_kv_proj():
        return attn.k_proj(x_in), attn.v_proj(x_in)

    def run_attn_math_manual():
        q = attn.q_proj(x_in).reshape(B, Nseq, H, Hd).transpose(0, 2, 1, 3)
        k = attn.k_proj(x_in).reshape(B, Nseq, H, Hd).transpose(0, 2, 1, 3)
        v = attn.v_proj(x_in).reshape(B, Nseq, H, Hd).transpose(0, 2, 1, 3)
        a = (q @ k.transpose(0, 1, 3, 2)) * attn.scale
        a = mx.softmax(a, axis=-1)
        return (a @ v).transpose(0, 2, 1, 3).reshape(B, Nseq, C)

    def run_attn_math_fast():
        q = attn.q_proj(x_in).reshape(B, Nseq, H, Hd).transpose(0, 2, 1, 3)
        k = attn.k_proj(x_in).reshape(B, Nseq, H, Hd).transpose(0, 2, 1, 3)
        v = attn.v_proj(x_in).reshape(B, Nseq, H, Hd).transpose(0, 2, 1, 3)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=attn.scale)
        return o.transpose(0, 2, 1, 3).reshape(B, Nseq, C)

    def run_out_proj():
        return attn.out_proj(x_in)  # dummy input for shape-matching

    t_q = timed(run_q_proj)
    t_kv = timed(run_kv_proj)
    t_attn_manual = timed(run_attn_math_manual)
    t_attn_fast = timed(run_attn_math_fast)
    t_out = timed(run_out_proj)

    print(f"    q_proj (1 LoRA)     : {t_q*1000:8.2f} ms")
    print(f"    k_proj + v_proj     : {t_kv*1000:8.2f} ms  (two LoRA linears)")
    print(f"    attn math (manual)  : {t_attn_manual*1000:8.2f} ms  <-- q/k/v+softmax+@V")
    print(f"    attn math (fast SDP): {t_attn_fast*1000:8.2f} ms  <-- mx.fast.scaled_dot_product_attention")
    print(f"    out_proj (1 LoRA)   : {t_out*1000:8.2f} ms")
    speedup = t_attn_manual / t_attn_fast if t_attn_fast > 0 else 0
    delta_ms = (t_attn_manual - t_attn_fast) * 1000
    print(f"    -> SDPA speedup     : {speedup:.2f}x  ({delta_ms:+.2f} ms saved per attn call)")
    print(f"    -> per forward save : ~{delta_ms * cfg.depth:.1f} ms  (x{cfg.depth} blocks)")
    # approximate fraction of forward
    if t_blocks > 0:
        frac = (delta_ms * cfg.depth / 1000) / t_blocks
        print(f"    -> % of forward     : ~{frac*100:.1f}%")

    # ---- Level 4: LayerNorm fast path ----
    print()
    print("  LayerNorm fast-path comparison:")

    ln = block.norm1
    weight = ln.weight
    bias = ln.bias
    eps = ln.eps if hasattr(ln, "eps") else 1e-5

    def run_ln_module():
        return ln(y)

    def run_ln_fast():
        return mx.fast.layer_norm(y, weight, bias, eps)

    t_ln_mod = timed(run_ln_module)
    t_ln_fast = timed(run_ln_fast)
    print(f"    nn.LayerNorm(x)     : {t_ln_mod*1000:8.3f} ms")
    print(f"    mx.fast.layer_norm  : {t_ln_fast*1000:8.3f} ms")
    ln_speedup = t_ln_mod / t_ln_fast if t_ln_fast > 0 else 0
    print(f"    -> speedup          : {ln_speedup:.2f}x")

    return {
        "arch": arch,
        "batch_size": batch_size,
        "t_train_step_ms": t_train * 1000,
        "t_forward_ms": t_fwd * 1000,
        "t_backward_ms": t_bwd_plus_opt * 1000,
        "t_blocks_ms": t_blocks * 1000,
        "t_attn_manual_ms": t_attn_manual * 1000,
        "t_attn_fast_ms": t_attn_fast * 1000,
        "attn_speedup": speedup,
        "per_fwd_save_ms": delta_ms * cfg.depth,
    }


def main():
    print("=" * 80)
    print("v0.3 Profiling — where does time go in a LoRA training step?")
    print("=" * 80)

    device = mx.device_info()
    print(f"Device: {device.get('architecture', '?')}  "
          f"({device.get('memory_size', 0)/1e9:.0f} GB)")

    results = []
    for arch, bs in [("vit_b", 8), ("vit_l", 8)]:
        results.append(profile_arch(arch, bs))

    # ---- Summary ----
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'arch':8s} {'bs':>3s} {'train_ms':>10s} {'fwd_ms':>9s} {'bwd_ms':>9s} "
          f"{'attn_man':>9s} {'attn_fast':>10s} {'SDPA_save':>11s}")
    for r in results:
        print(f"{r['arch']:8s} {r['batch_size']:3d} "
              f"{r['t_train_step_ms']:10.1f} {r['t_forward_ms']:9.1f} "
              f"{r['t_backward_ms']:9.1f} {r['t_attn_manual_ms']:9.2f} "
              f"{r['t_attn_fast_ms']:10.2f} {r['per_fwd_save_ms']:10.1f}ms")

    print()
    print("Verdict: if SDPA saves a large fraction of forward time, that is the")
    print("v0.3 target. Otherwise, LoRA / LayerNorm paths are the alternatives.")


if __name__ == "__main__":
    main()
