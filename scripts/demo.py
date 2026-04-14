#!/usr/bin/env python3
"""Quick demo: Fine-tune a ViT-B/16 with LoRA on a synthetic dataset.

No downloads needed — uses random weights and auto-generated images.
Run: python scripts/demo.py
"""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_dataset(root: str, n_train: int = 100, n_val: int = 20):
    """Create a toy 2-class dataset: reddish vs bluish patches."""
    root = Path(root)
    np.random.seed(42)

    for split, n in [("train", n_train), ("val", n_val)]:
        for cls_name, channel in [("class_0", 0), ("class_1", 2)]:
            d = root / split / cls_name
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n // 2):
                img = np.random.randint(80, 180, (224, 224, 3), dtype=np.uint8)
                img[:, :, channel] = np.clip(img[:, :, channel] + 80, 0, 255)
                Image.fromarray(img).save(d / f"{i:04d}.png")

    print(f"Created dataset: {n_train} train, {n_val} val, 2 classes → {root}")


def main():
    from mlx_vit import FastViTModel
    from mlx_vit.data import ImageDataset
    from mlx_vit.trainer import TrainingArgs, train

    data_dir = "demo_data"
    print("=" * 60)
    print("  mlx-vit-tune Demo")
    print("  Fine-tune ViT-B/16 with LoRA on Apple Silicon")
    print("=" * 60)

    # Step 1: Create dataset
    print("\n[1/4] Creating synthetic dataset...")
    create_synthetic_dataset(data_dir)

    # Step 2: Load model
    print("\n[2/4] Loading ViT-B/16 (random weights)...")
    model = FastViTModel.from_pretrained("vit_base_patch16_224", num_classes=2)

    # Step 3: Add LoRA
    print("\n[3/4] Injecting LoRA (rank=8, all linear layers)...")
    model = FastViTModel.get_lora_model(model, rank=8, target_modules="all")

    # Step 4: Train
    print("\n[4/4] Training for 5 epochs...")
    train_ds = ImageDataset(f"{data_dir}/train", image_size=224, augment=True)
    val_ds = ImageDataset(f"{data_dir}/val", image_size=224, augment=False)

    start = time.time()
    train(model, train_ds, val_ds, TrainingArgs(
        batch_size=8,
        lr=1e-3,
        epochs=5,
        output_dir="demo_outputs",
        log_every=5,
        warmup_steps=10,
    ))
    elapsed = time.time() - start

    print(f"\nTotal time: {elapsed:.1f}s")
    print("\nDone! Check demo_outputs/ for saved adapters.")


if __name__ == "__main__":
    main()
