"""Image data pipeline for ViT fine-tuning.

Supports directory-based datasets (root/class_name/img.png) and
CSV/JSON formats. Uses Pillow for image loading — no torchvision dependency.
"""

import json
import math
import random
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image

# ImageNet normalization stats
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Pathology-specific normalization (from typical H&E statistics)
PATHOLOGY_MEAN = np.array([0.70, 0.55, 0.70], dtype=np.float32)
PATHOLOGY_STD = np.array([0.17, 0.20, 0.17], dtype=np.float32)

NORM_STATS = {
    "imagenet": (IMAGENET_MEAN, IMAGENET_STD),
    "pathology": (PATHOLOGY_MEAN, PATHOLOGY_STD),
    "none": (np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)),
}


def load_image(path: str, size: int = 224) -> np.ndarray:
    """Load and preprocess a single image.

    Returns: [H, W, 3] float32 array in [0, 1] range.
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def augment_image(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Apply basic augmentations for training.

    - Random horizontal flip
    - Random vertical flip
    - Random 90-degree rotation (important for pathology tiles)
    - Random color jitter (brightness, contrast)
    """
    # Random horizontal flip
    if rng.random() > 0.5:
        img = img[:, ::-1, :]

    # Random vertical flip
    if rng.random() > 0.5:
        img = img[::-1, :, :]

    # Random 90-degree rotation
    k = rng.randint(0, 3)
    if k > 0:
        img = np.rot90(img, k=k, axes=(0, 1))

    # Random brightness adjustment
    if rng.random() > 0.5:
        factor = rng.uniform(0.8, 1.2)
        img = np.clip(img * factor, 0.0, 1.0)

    # Random contrast adjustment
    if rng.random() > 0.5:
        factor = rng.uniform(0.8, 1.2)
        mean = img.mean()
        img = np.clip((img - mean) * factor + mean, 0.0, 1.0)

    return np.ascontiguousarray(img)


def normalize(img: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Normalize image with mean and std."""
    return (img - mean) / std


class ImageDataset:
    """Simple image dataset supporting directory and CSV/JSON formats.

    Directory format:
        root/
            class_a/
                img1.png
                img2.jpg
            class_b/
                img3.png

    CSV/JSON format:
        [{"image_path": "path/to/img.png", "label": 0}, ...]
    """

    def __init__(
        self,
        root: str,
        image_size: int = 224,
        normalize_type: str = "imagenet",
        augment: bool = False,
        seed: int = 42,
    ):
        self.image_size = image_size
        self.augment = augment
        self.rng = random.Random(seed)

        mean, std = NORM_STATS.get(normalize_type, NORM_STATS["imagenet"])
        self.mean = mean
        self.std = std

        root = Path(root)
        if root.is_dir():
            self._load_from_directory(root)
        elif root.suffix == ".json":
            self._load_from_json(root)
        elif root.suffix == ".csv":
            self._load_from_csv(root)
        else:
            raise ValueError(f"Unsupported dataset format: {root}")

    def _load_from_directory(self, root: Path):
        """Load from directory structure: root/class_name/image_file."""
        self.samples = []
        self.class_names = sorted([
            d.name for d in root.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        for class_name, idx in self.class_to_idx.items():
            class_dir = root / class_name
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
                    self.samples.append((str(img_path), idx))

        print(f"Loaded {len(self.samples)} images, {len(self.class_names)} classes")

    def _load_from_json(self, path: Path):
        """Load from JSON file: [{"image_path": ..., "label": ...}, ...]"""
        with open(path) as f:
            data = json.load(f)

        self.samples = [(d["image_path"], d["label"]) for d in data]
        labels = set(d["label"] for d in data)
        self.class_names = [str(i) for i in sorted(labels)]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        print(f"Loaded {len(self.samples)} images from JSON")

    def _load_from_csv(self, path: Path):
        """Load from CSV: image_path,label"""
        self.samples = []
        labels = set()
        with open(path) as f:
            header = f.readline()  # skip header
            for line in f:
                parts = line.strip().split(",")
                img_path, label = parts[0], int(parts[1])
                self.samples.append((img_path, label))
                labels.add(label)

        self.class_names = [str(i) for i in sorted(labels)]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        print(f"Loaded {len(self.samples)} images from CSV")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        img_path, label = self.samples[idx]
        img = load_image(img_path, self.image_size)

        if self.augment:
            img = augment_image(img, self.rng)

        img = normalize(img, self.mean, self.std)
        return img, label


def create_batches(
    dataset: ImageDataset,
    batch_size: int,
    shuffle: bool = True,
    drop_last: bool = False,
):
    """Yield batches of (images, labels) as mx.arrays.

    Args:
        dataset: ImageDataset instance
        batch_size: Number of images per batch
        shuffle: Whether to shuffle the dataset
        drop_last: Whether to drop the last incomplete batch

    Yields:
        Tuple of (images: mx.array [B,H,W,C], labels: mx.array [B])
    """
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    num_batches = len(indices) // batch_size
    if not drop_last and len(indices) % batch_size != 0:
        num_batches += 1

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, len(indices))
        batch_indices = indices[start:end]

        images = []
        labels = []
        for idx in batch_indices:
            img, label = dataset[idx]
            images.append(img)
            labels.append(label)

        images = mx.array(np.stack(images))  # [B, H, W, C]
        labels = mx.array(np.array(labels))  # [B]

        yield images, labels
