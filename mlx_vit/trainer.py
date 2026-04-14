"""Training loop for ViT fine-tuning with MLX.

Supports both full fine-tuning and LoRA, with AdamW optimizer,
cosine learning rate schedule, and basic metric tracking.
"""

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_vit.data import ImageDataset, create_batches
from mlx_vit.lora import save_adapters, LoRALinear
from mlx_vit.vit import VisionTransformer


@dataclass
class TrainingArgs:
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_schedule: str = "cosine"  # "cosine", "linear", "constant"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    save_every: int = 0  # Save checkpoint every N epochs (0 = only best)
    output_dir: str = "outputs"
    log_every: int = 10  # Log every N steps
    eval_every: int = 0  # Evaluate every N steps (0 = end of epoch only)
    seed: int = 42


def _has_lora(model: VisionTransformer) -> bool:
    """Check if model has LoRA adapters."""
    for block in model.blocks:
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            if isinstance(getattr(block.attn, name, None), LoRALinear):
                return True
    return False


def _cosine_schedule(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def _linear_schedule(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    """Linear learning rate schedule with warmup."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * (1.0 - progress)


def cross_entropy_loss(model: VisionTransformer, images: mx.array, labels: mx.array):
    """Compute cross-entropy loss and accuracy."""
    logits = model(images)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    predictions = mx.argmax(logits, axis=-1)
    accuracy = mx.mean(predictions == labels)
    return loss, accuracy


def train(
    model: VisionTransformer,
    train_dataset: ImageDataset,
    val_dataset: Optional[ImageDataset] = None,
    args: Optional[TrainingArgs] = None,
):
    """Train a ViT model.

    Args:
        model: The ViT model (with or without LoRA)
        train_dataset: Training dataset
        val_dataset: Optional validation dataset
        args: Training arguments
    """
    if args is None:
        args = TrainingArgs()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_lora = _has_lora(model)

    # Calculate steps
    steps_per_epoch = math.ceil(len(train_dataset) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    effective_batch = args.batch_size * args.gradient_accumulation_steps

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"Mode:            {'LoRA' if is_lora else 'Full fine-tuning'}")
    print(f"Train samples:   {len(train_dataset):,}")
    if val_dataset:
        print(f"Val samples:     {len(val_dataset):,}")
    print(f"Batch size:      {args.batch_size} (effective: {effective_batch})")
    print(f"Epochs:          {args.epochs}")
    print(f"Total steps:     {total_steps:,}")
    print(f"Learning rate:   {args.lr}")
    print(f"LR schedule:     {args.lr_schedule}")
    print(f"Weight decay:    {args.weight_decay}")
    print(f"Output dir:      {output_dir}")
    print(f"{'='*60}\n")

    # Optimizer
    optimizer = optim.AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    # Loss and grad function
    loss_and_grad_fn = nn.value_and_grad(model, cross_entropy_loss)

    best_val_acc = 0.0
    global_step = 0

    for epoch in range(args.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        epoch_samples = 0
        epoch_start = time.time()

        accum_grads = None

        for batch_idx, (images, labels) in enumerate(
            create_batches(train_dataset, args.batch_size, shuffle=True)
        ):
            # Update learning rate
            if args.lr_schedule == "cosine":
                lr = _cosine_schedule(global_step, total_steps, args.warmup_steps, args.lr)
            elif args.lr_schedule == "linear":
                lr = _linear_schedule(global_step, total_steps, args.warmup_steps, args.lr)
            else:
                lr = args.lr
            optimizer.learning_rate = mx.array(lr)

            # Forward + backward
            (loss, acc), grads = loss_and_grad_fn(model, images, labels)

            # Gradient accumulation
            if args.gradient_accumulation_steps > 1:
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = mx.tree_map(lambda a, b: a + b, accum_grads, grads)

                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Average accumulated gradients
                    scale = 1.0 / args.gradient_accumulation_steps
                    accum_grads = mx.tree_map(lambda g: g * scale, accum_grads)
                    optimizer.update(model, accum_grads)
                    accum_grads = None
                    mx.eval(model.parameters(), optimizer.state)
            else:
                optimizer.update(model, grads)
                mx.eval(model.parameters(), optimizer.state)

            batch_size = images.shape[0]
            epoch_loss += loss.item() * batch_size
            epoch_acc += acc.item() * batch_size
            epoch_samples += batch_size
            global_step += 1

            # Logging
            if global_step % args.log_every == 0:
                avg_loss = epoch_loss / epoch_samples
                avg_acc = epoch_acc / epoch_samples
                elapsed = time.time() - epoch_start
                imgs_per_sec = epoch_samples / elapsed
                print(
                    f"  step {global_step:5d} | "
                    f"loss {loss.item():.4f} | "
                    f"acc {acc.item():.3f} | "
                    f"lr {lr:.2e} | "
                    f"{imgs_per_sec:.1f} img/s"
                )

            # Mid-epoch evaluation
            if args.eval_every > 0 and global_step % args.eval_every == 0 and val_dataset:
                val_loss, val_acc = evaluate(model, val_dataset, args.batch_size)
                print(f"  [eval] loss {val_loss:.4f} | acc {val_acc:.3f}")
                model.train()

        # End of epoch stats
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / max(epoch_samples, 1)
        avg_acc = epoch_acc / max(epoch_samples, 1)
        print(
            f"\nEpoch {epoch+1}/{args.epochs} | "
            f"train_loss {avg_loss:.4f} | "
            f"train_acc {avg_acc:.3f} | "
            f"time {epoch_time:.1f}s"
        )

        # Validation
        if val_dataset:
            val_loss, val_acc = evaluate(model, val_dataset, args.batch_size)
            print(f"  val_loss {val_loss:.4f} | val_acc {val_acc:.3f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                _save_checkpoint(model, output_dir / "best", is_lora)
                print(f"  New best val_acc: {val_acc:.3f}")
            model.train()

        # Periodic checkpoint
        if args.save_every > 0 and (epoch + 1) % args.save_every == 0:
            _save_checkpoint(model, output_dir / f"epoch_{epoch+1}", is_lora)

    # Save final model
    _save_checkpoint(model, output_dir / "final", is_lora)
    print(f"\nTraining complete. Best val_acc: {best_val_acc:.3f}")
    return model


def evaluate(
    model: VisionTransformer,
    dataset: ImageDataset,
    batch_size: int = 32,
) -> tuple[float, float]:
    """Evaluate model on a dataset.

    Returns:
        Tuple of (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    for images, labels in create_batches(dataset, batch_size, shuffle=False):
        logits = model(images)
        loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
        predictions = mx.argmax(logits, axis=-1)
        acc = mx.mean(predictions == labels)

        batch_size_actual = images.shape[0]
        total_loss += loss.item() * batch_size_actual
        total_acc += acc.item() * batch_size_actual
        total_samples += batch_size_actual

    avg_loss = total_loss / max(total_samples, 1)
    avg_acc = total_acc / max(total_samples, 1)
    return avg_loss, avg_acc


def _save_checkpoint(model: VisionTransformer, path: Path, is_lora: bool):
    """Save model checkpoint."""
    path.mkdir(parents=True, exist_ok=True)
    if is_lora:
        save_adapters(model, str(path))
    else:
        weights = dict(model.parameters())
        mx.savez(str(path / "model.npz"), **weights)
    print(f"  Saved checkpoint → {path}")
