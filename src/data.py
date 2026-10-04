"""
Dataset loading, deduplication, and preprocessing pipeline.
"""

import hashlib
from collections import Counter
from pathlib import Path
from typing import Tuple, List

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from src.config import (
    DATASET_DIR,
    CLASS_NAMES,
    IMAGE_SIZE,
    BATCH_SIZE,
    RANDOM_SEED,
)

AUTOTUNE = tf.data.AUTOTUNE


def is_valid_image(path: Path) -> bool:
    """Verify image extension and ensure no OS hidden files."""
    return (
        path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not path.name.startswith(".")
    )


def load_and_preprocess_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Reads, decodes, resizes, and casts an image to float32."""
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    return image, label


def build_dataset(paths: List[str], labels: List[int], shuffle: bool = False) -> tf.data.Dataset:
    """Creates a high-performance tf.data.Dataset pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(paths), seed=RANDOM_SEED)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return dataset


def prepare_datasets(dataset_dir: Path = DATASET_DIR):
    """
    Scans dataset, filters exact MD5 duplicates, and performs stratified 80/10/10 split.
    """
    valid_files = [
        path for path in dataset_dir.rglob("*") if is_valid_image(path)
    ]

    unique_files = []
    seen_hashes = set()

    for path in sorted(valid_files):
        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
        if file_hash in seen_hashes:
            continue
        seen_hashes.add(file_hash)
        unique_files.append(path)

    class_to_idx = {name: idx for idx, name in enumerate(CLASS_NAMES)}
    all_paths = [str(p) for p in unique_files]
    all_targets = [class_to_idx[Path(p).parent.name] for p in unique_files]

    # Train / (Val + Test) (80% / 20%)
    train_paths, temp_paths, train_targets, temp_targets = train_test_split(
        all_paths,
        all_targets,
        test_size=0.2,
        stratify=all_targets,
        random_state=RANDOM_SEED,
    )

    # Val / Test (10% / 10%)
    val_paths, test_paths, val_targets, test_targets = train_test_split(
        temp_paths,
        temp_targets,
        test_size=0.5,
        stratify=temp_targets,
        random_state=RANDOM_SEED,
    )

    train_ds = build_dataset(train_paths, train_targets, shuffle=True)
    val_ds = build_dataset(val_paths, val_targets)
    test_ds = build_dataset(test_paths, test_targets)

    return (
        (train_ds, val_ds, test_ds),
        (train_targets, val_targets, test_targets),
        (train_paths, val_paths, test_paths),
    )
