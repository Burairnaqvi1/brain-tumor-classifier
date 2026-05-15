"""
preprocessing.py
----------------
Handles all image loading and preprocessing.
Each step is a pure function. The full pipeline is composed in `preprocess_image`.
Raw data is never modified — processed arrays are returned in memory.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from src.config import (
    IMAGE_SIZE, APPLY_CLAHE, CLAHE_CLIP_LIMIT,
    CLAHE_TILE_SIZE, APPLY_BLUR, CLASSES,
    CLASS_TO_IDX, RAW_TRAIN_DIR, RAW_TEST_DIR
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# INDIVIDUAL STEPS
# ─────────────────────────────────────────────

def load_grayscale(img_path: str) -> Optional[np.ndarray]:
    """Load image as grayscale. Returns None if file is unreadable."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning(f"Could not read image: {img_path}")
    return img


def resize_image(img: np.ndarray, size: Tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Resize to target size using area interpolation (best for downscaling)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Enhances local contrast without amplifying noise globally.
    Critical for MRI images where tumor boundaries may be low-contrast.
    """
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_TILE_SIZE
    )
    return clahe.apply(img)


def normalize(img: np.ndarray) -> np.ndarray:
    """Normalize pixel values to [0.0, 1.0] as float32."""
    return (img / 255.0).astype(np.float32)


# ─────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────

def preprocess_image(img_path: str) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for a single image.
    Steps: Load → Resize → CLAHE (optional) → Normalize
    Gaussian blur is intentionally excluded: LBP and GLCM
    depend on micro-texture that blur would degrade.

    Returns:
        Preprocessed float32 numpy array of shape IMAGE_SIZE, or None if unreadable.
    """
    img = load_grayscale(img_path)
    if img is None:
        return None

    img = resize_image(img)

    if APPLY_CLAHE:
        img = apply_clahe(img)

    img = normalize(img)
    return img


# ─────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────

def load_dataset(split: str = "train") -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load all images for a given split and return arrays.

    Args:
        split: "train" or "test"

    Returns:
        images : np.ndarray of shape (N, H, W)
        labels : np.ndarray of shape (N,) — integer class indices
        paths  : list of Path objects (for error analysis later)
    """
    base_dir = RAW_TRAIN_DIR if split == "train" else RAW_TEST_DIR

    images, labels, paths = [], [], []
    skipped = 0

    for class_name in CLASSES:
        class_dir = base_dir / class_name
        if not class_dir.exists():
            logger.warning(f"Class folder not found: {class_dir}")
            continue

        img_files = list(class_dir.glob("*.jpg")) + \
                    list(class_dir.glob("*.jpeg")) + \
                    list(class_dir.glob("*.png"))

        for img_path in img_files:
            img = preprocess_image(img_path)
            if img is None:
                skipped += 1
                continue
            images.append(img)
            labels.append(CLASS_TO_IDX[class_name])
            paths.append(img_path)

    if skipped > 0:
        logger.warning(f"Skipped {skipped} unreadable images in {split} split.")

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32),
        paths
    )


# ─────────────────────────────────────────────
# DATASET VERIFICATION
# ─────────────────────────────────────────────

def verify_dataset() -> dict:
    """
    Scan raw data folders and return class distribution for both splits.
    Prints a formatted report and returns counts dict.
    """
    report = {"train": {}, "test": {}}

    for split, base_dir in [("train", RAW_TRAIN_DIR), ("test", RAW_TEST_DIR)]:
        for class_name in CLASSES:
            class_dir = base_dir / class_name
            if class_dir.exists():
                count = len(list(class_dir.glob("*.jpg"))) + \
                        len(list(class_dir.glob("*.jpeg"))) + \
                        len(list(class_dir.glob("*.png")))
                report[split][class_name] = count
            else:
                report[split][class_name] = 0

    _print_verification_report(report)
    return report


def _print_verification_report(report: dict):
    """Print a clean dataset summary table to console."""
    print("\n" + "=" * 55)
    print("  DATASET VERIFICATION REPORT")
    print("=" * 55)
    print(f"  {'Class':<15} {'Train':>10} {'Test':>10} {'Total':>10}")
    print("-" * 55)

    for class_name in CLASSES:
        train_n = report["train"].get(class_name, 0)
        test_n  = report["test"].get(class_name, 0)
        print(f"  {class_name:<15} {train_n:>10} {test_n:>10} {train_n + test_n:>10}")

    total_train = sum(report["train"].values())
    total_test  = sum(report["test"].values())
    print("-" * 55)
    print(f"  {'TOTAL':<15} {total_train:>10} {total_test:>10} {total_train + total_test:>10}")
    print("=" * 55 + "\n")