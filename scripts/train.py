#!/usr/bin/env python3
"""CLI entry point for ViT fine-tuning."""

import argparse
import yaml

from mlx_vit import FastViTModel
from mlx_vit.data import ImageDataset
from mlx_vit.trainer import TrainingArgs, train


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a ViT model with MLX")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--model", type=str, default="MahmoodLab/conch", help="Model name or path")
    parser.add_argument("--train_data", type=str, required=True, help="Training data path")
    parser.add_argument("--val_data", type=str, default=None, help="Validation data path")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_targets", type=str, default="all", help="LoRA target modules")
    parser.add_argument("--full_ft", action="store_true", help="Full fine-tuning (default)")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_accumulation", type=int, default=1, help="Gradient accumulation steps")
    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        # Override with CLI args where provided
        model_cfg = config.get("model", {})
        lora_cfg = config.get("lora", {})
        train_cfg = config.get("training", {})

        if args.model == "MahmoodLab/conch":  # default, may be overridden by config
            args.model = model_cfg.get("name", args.model)
        args.num_classes = model_cfg.get("num_classes", args.num_classes)
        args.image_size = model_cfg.get("image_size", args.image_size)

    # Load model
    print(f"\nLoading model: {args.model}")
    model = FastViTModel.from_pretrained(
        args.model,
        num_classes=args.num_classes,
        image_size=args.image_size,
        hf_token=args.hf_token,
        dtype=args.dtype,
    )

    # Apply LoRA if requested
    if args.lora:
        model = FastViTModel.get_lora_model(
            model,
            rank=args.lora_rank,
            target_modules=args.lora_targets,
        )

    # Load datasets
    print(f"\nLoading training data: {args.train_data}")
    train_dataset = ImageDataset(
        args.train_data,
        image_size=args.image_size,
        augment=True,
    )

    val_dataset = None
    if args.val_data:
        print(f"Loading validation data: {args.val_data}")
        val_dataset = ImageDataset(
            args.val_data,
            image_size=args.image_size,
            augment=False,
        )

    # Training args
    training_args = TrainingArgs(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_dir=args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation,
    )

    # Train
    train(model, train_dataset, val_dataset, training_args)


if __name__ == "__main__":
    main()
