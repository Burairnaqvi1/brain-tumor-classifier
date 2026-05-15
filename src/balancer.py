"""
balancer.py
-----------
Handles class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).
SMOTE is applied exclusively on the training set — the test set is never touched.

If the dataset is already balanced (min/max class ratio >= BALANCE_THRESHOLD),
SMOTE is skipped automatically with a clear log message. This prevents the
pipeline from reporting SMOTE "applied" when it had no actual effect.
"""

import numpy as np
import logging
from collections import Counter

from imblearn.over_sampling import SMOTE

from src.config import (
    APPLY_SMOTE, SMOTE_STRATEGY, SMOTE_K,
    RANDOM_SEED, IDX_TO_CLASS
)

logger = logging.getLogger(__name__)

# If the smallest class is at least this fraction of the largest,
# the dataset is considered balanced and SMOTE is skipped.
BALANCE_THRESHOLD = 0.90


def balance_classes(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the training set if class imbalance is detected.

    Decision logic:
      - Compute min_class_count / max_class_count ratio
      - If ratio >= BALANCE_THRESHOLD → dataset already balanced → skip SMOTE
      - If ratio <  BALANCE_THRESHOLD → apply SMOTE

    SMOTE generates synthetic feature vectors by interpolating between
    real minority-class samples. It does NOT duplicate existing samples,
    which reduces overfitting compared to random oversampling.

    Args:
        X_train : Training feature matrix (N, F)
        y_train : Training labels (N,)

    Returns:
        X_resampled : Feature matrix (balanced or original)
        y_resampled : Labels (balanced or original)
    """
    _print_distribution("Class Distribution — Training Set", y_train)

    if not APPLY_SMOTE:
        logger.info("SMOTE disabled in config.py — skipping.")
        return X_train, y_train

    counts = Counter(y_train)
    min_count = min(counts.values())
    max_count = max(counts.values())
    balance_ratio = min_count / max_count

    if balance_ratio >= BALANCE_THRESHOLD:
        logger.info(
            f"Dataset is already balanced "
            f"(min/max ratio = {balance_ratio:.3f} >= threshold {BALANCE_THRESHOLD}). "
            f"SMOTE skipped — no synthetic samples needed."
        )
        print(f"\n  ✔ SMOTE skipped: dataset is already balanced "
              f"(ratio = {balance_ratio:.3f})\n")
        return X_train, y_train

    logger.info(
        f"Class imbalance detected (ratio = {balance_ratio:.3f}). "
        f"Applying SMOTE..."
    )

    smote = SMOTE(
        sampling_strategy=SMOTE_STRATEGY,
        k_neighbors=SMOTE_K,
        random_state=RANDOM_SEED
    )

    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    _print_distribution("Class Distribution — After SMOTE", y_resampled)

    return X_resampled, y_resampled


def _print_distribution(title: str, y: np.ndarray):
    """Print a clean class distribution table."""
    counts = Counter(y)
    total  = len(y)

    print(f"\n  {title}")
    print("  " + "-" * 42)
    print(f"  {'Class':<15} {'Count':>8} {'Share':>8}")
    print("  " + "-" * 42)

    for idx in sorted(counts.keys()):
        name  = IDX_TO_CLASS.get(idx, str(idx))
        count = counts[idx]
        share = count / total * 100
        print(f"  {name:<15} {count:>8} {share:>7.1f}%")

    print("  " + "-" * 42)
    print(f"  {'Total':<15} {total:>8}\n")