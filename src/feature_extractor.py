"""
feature_extractor.py
--------------------
Unified feature extraction class combining HOG, LBP, and GLCM.
Each extractor is isolated and togglable via config.py.
Features are extracted once and saved to CSV — never recomputed during training.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

from skimage.feature import hog, local_binary_pattern, graycomatrix, graycoprops

from src.config import (
    USE_HOG, USE_LBP, USE_GLCM,
    HOG_ORIENTATIONS, HOG_PIXELS_PER_CELL, HOG_CELLS_PER_BLOCK,
    LBP_RADIUS, LBP_N_POINTS, LBP_METHOD, LBP_N_BINS,
    GLCM_DISTANCES, GLCM_ANGLES, GLCM_PROPERTIES,
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH,
    CLASSES, CLASS_TO_IDX
)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts and concatenates HOG, LBP, and GLCM features from preprocessed images.
    All extractors operate on float32 grayscale images normalized to [0, 1].
    """

    # ─────────────────────────────────────────────
    # INDIVIDUAL EXTRACTORS
    # ─────────────────────────────────────────────

    @staticmethod
    def _extract_hog(img: np.ndarray) -> np.ndarray:
        """
        Histogram of Oriented Gradients.
        Captures edge directions and structural shape of tumor regions.
        """
        features = hog(
            img,
            orientations=HOG_ORIENTATIONS,
            pixels_per_cell=HOG_PIXELS_PER_CELL,
            cells_per_block=HOG_CELLS_PER_BLOCK,
            visualize=False,
            feature_vector=True
        )
        return features.astype(np.float32)

    @staticmethod
    def _extract_lbp(img: np.ndarray) -> np.ndarray:
        """
        Local Binary Patterns.
        Captures local micro-texture — distinguishes tumor tissue from healthy tissue.
        Returns normalized histogram so it is scale-invariant.
        """
        # LBP expects uint8
        img_uint8 = (img * 255).astype(np.uint8)
        lbp = local_binary_pattern(img_uint8, P=LBP_N_POINTS, R=LBP_RADIUS, method=LBP_METHOD)
        hist, _ = np.histogram(lbp.ravel(), bins=LBP_N_BINS, range=(0, LBP_N_BINS))
        # Normalize to sum = 1 (probability distribution)
        hist = hist.astype(np.float32)
        total = hist.sum()
        if total > 0:
            hist /= total
        return hist

    @staticmethod
    def _extract_glcm(img: np.ndarray) -> np.ndarray:
        """
        Gray Level Co-occurrence Matrix features.
        Captures statistical texture: contrast, correlation, energy, homogeneity, dissimilarity.
        Computed at 4 angles and averaged — makes descriptor rotation-invariant.
        """
        img_uint8 = (img * 255).astype(np.uint8)
        glcm = graycomatrix(
            img_uint8,
            distances=GLCM_DISTANCES,
            angles=GLCM_ANGLES,
            symmetric=True,
            normed=True
        )
        features = []
        for prop in GLCM_PROPERTIES:
            values = graycoprops(glcm, prop)   # shape: (len(distances), len(angles))
            features.append(values.mean())     # average over all angles → rotation invariant
        return np.array(features, dtype=np.float32)

    # ─────────────────────────────────────────────
    # COMBINED EXTRACTION
    # ─────────────────────────────────────────────

    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract and concatenate all enabled features for a single image.

        Args:
            img: Preprocessed float32 numpy array, shape (H, W), values in [0, 1]

        Returns:
            1D feature vector as float32
        """
        parts = []

        if USE_HOG:
            parts.append(self._extract_hog(img))
        if USE_LBP:
            parts.append(self._extract_lbp(img))
        if USE_GLCM:
            parts.append(self._extract_glcm(img))

        if not parts:
            raise ValueError("All feature extractors are disabled in config.py.")

        return np.concatenate(parts)

    def extract_dataset(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        paths: list
    ) -> pd.DataFrame:
        """
        Extract features from an entire dataset.

        Args:
            images : np.ndarray (N, H, W)
            labels : np.ndarray (N,)
            paths  : list of Path objects

        Returns:
            DataFrame where each row is one image's feature vector + label + path
        """
        all_features = []
        n = len(images)

        for i, (img, label) in enumerate(zip(images, labels)):
            vec = self.extract(img)
            all_features.append(vec)

            if (i + 1) % 500 == 0 or (i + 1) == n:
                logger.info(f"  Extracted features: {i + 1}/{n}")

        feature_matrix = np.vstack(all_features)
        n_features = feature_matrix.shape[1]
        col_names = [f"f{i}" for i in range(n_features)]

        df = pd.DataFrame(feature_matrix, columns=col_names)
        df["label"] = labels
        df["path"]  = [str(p) for p in paths]

        return df


# ─────────────────────────────────────────────
# SAVE / LOAD HELPERS
# ─────────────────────────────────────────────

def save_features(df: pd.DataFrame, split: str = "train"):
    """Save feature DataFrame to CSV."""
    path = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH
    df.to_csv(path, index=False)
    logger.info(f"Saved {split} features → {path}  ({len(df)} samples, {len(df.columns)-2} features)")


def load_features(split: str = "train") -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load pre-saved feature CSV.

    Returns:
        X     : feature matrix (N, F)
        y     : label array (N,)
        paths : list of image path strings
    """
    path = TRAIN_FEATURES_PATH if split == "train" else TEST_FEATURES_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Feature file not found: {path}\n"
            "Run main.py with --extract flag first."
        )

    df    = pd.read_csv(path)
    paths = df["path"].tolist()
    y     = df["label"].values.astype(np.int32)
    X     = df.drop(columns=["label", "path"]).values.astype(np.float32)

    logger.info(f"Loaded {split} features: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y, paths


def features_exist() -> bool:
    """Check whether both train and test feature CSVs already exist."""
    return TRAIN_FEATURES_PATH.exists() and TEST_FEATURES_PATH.exists()