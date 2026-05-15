"""
main.py
-------
Single entry point for the Brain Tumor MRI Classification project.
Runs the full pipeline in sequence:
  Verify → Extract Features → Balance → Scale → PCA → Train → Evaluate → Visualize

Usage:
  python main.py                  # Full pipeline
  python main.py --skip-extract   # Skip feature extraction (use saved CSVs)
  python main.py --skip-train     # Skip training (use saved models)
"""

import argparse
import logging
import time
import sys
from pathlib import Path

# ─── ensure project root is on PYTHONPATH when run directly ───
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import CLASSES, IDX_TO_CLASS
from src.preprocessing import verify_dataset, load_dataset
from src.feature_extractor import (
    FeatureExtractor, save_features, load_features, features_exist
)
from src.balancer import balance_classes
from src.trainer import (
    fit_scaler_and_transform, fit_pca_and_transform,
    train_all_models, load_model, load_scaler, load_pca
)
from src.evaluator import evaluate_all_models, export_metrics_table, get_misclassified_samples
from src.visualizer import (
    plot_class_distribution, plot_sample_images,
    plot_all_confusion_matrices, plot_roc_curves,
    plot_metrics_comparison, plot_feature_importance,
    plot_training_time, plot_error_analysis, plot_cv_results
)
from src.config import APPLY_PCA, MODEL_FILENAMES

# ─────────────────────────────────────────────
# LOGGING SETUP
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", mode="w", encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# PIPELINE STAGES
# ─────────────────────────────────────────────

def stage(name: str):
    """Print a clean stage header."""
    print(f"\n{'═' * 60}")
    print(f"  STAGE: {name}")
    print(f"{'═' * 60}")


def run(skip_extract: bool = False, skip_train: bool = False):
    total_start = time.time()

    # ── STAGE 1: Dataset Verification ─────────────────────────
    stage("1 / 7  —  Dataset Verification")
    report = verify_dataset()

    # ── STAGE 2: Feature Extraction ───────────────────────────
    stage("2 / 7  —  Feature Extraction")

    if skip_extract and features_exist():
        logger.info("--skip-extract: loading saved feature CSVs.")
        X_train, y_train, train_paths = load_features("train")
        X_test,  y_test,  test_paths  = load_features("test")
    else:
        logger.info("Loading and preprocessing training images...")
        train_images, y_train, train_paths = load_dataset("train")
        logger.info("Loading and preprocessing test images...")
        test_images,  y_test,  test_paths  = load_dataset("test")

        extractor = FeatureExtractor()

        logger.info("Extracting features from training set...")
        train_df = extractor.extract_dataset(train_images, y_train, train_paths)
        save_features(train_df, "train")

        logger.info("Extracting features from test set...")
        test_df = extractor.extract_dataset(test_images, y_test, test_paths)
        save_features(test_df, "test")

        X_train = train_df.drop(columns=["label", "path"]).values
        X_test  = test_df.drop(columns=["label", "path"]).values

    # ── STAGE 3: Visualize Raw Distribution + Samples ─────────
    stage("3 / 7  —  Exploratory Visualization")
    plot_sample_images(n_per_class=4)
    logger.info("Sample images plot saved.")

    # ── STAGE 4: Scale → PCA → SMOTE ──────────────────────────
    stage("4 / 7  —  Preprocessing: Scale / PCA / Balance")

    logger.info("Fitting StandardScaler on training set...")
    X_train_scaled, X_test_scaled, scaler = fit_scaler_and_transform(X_train, X_test)

    logger.info("Applying PCA...")
    X_train_pca, X_test_pca, pca = fit_pca_and_transform(X_train_scaled, X_test_scaled)

    logger.info("Applying SMOTE on training set only...")
    before_dist = {cls: int((y_train == i).sum()) for i, cls in IDX_TO_CLASS.items()}
    X_train_balanced, y_train_balanced = balance_classes(X_train_pca, y_train)
    after_dist  = {cls: int((y_train_balanced == i).sum()) for i, cls in IDX_TO_CLASS.items()}

    plot_class_distribution(
        {"train": before_dist},
        after_dist
    )

    # ── STAGE 5: Train ─────────────────────────────────────────
    stage("5 / 7  —  Model Training")

    if skip_train:
        logger.info("--skip-train: loading saved models.")
        cv_results = {}   # CV results unavailable when skipping
    else:
        cv_results = train_all_models(X_train_balanced, y_train_balanced)

    # ── STAGE 6: Evaluate ──────────────────────────────────────
    stage("6 / 7  —  Evaluation")

    all_results = evaluate_all_models(X_test_pca, y_test, cv_results)
    export_metrics_table(all_results, cv_results)

    # ── STAGE 7: Visualize Results ─────────────────────────────
    stage("7 / 7  —  Visualization")

    plot_all_confusion_matrices(all_results)
    plot_roc_curves(all_results, y_test)
    plot_metrics_comparison(all_results)
    plot_cv_results(cv_results)
    plot_training_time(cv_results)

    # Feature importance from Random Forest
    rf_model = load_model("RandomForest")
    plot_feature_importance(rf_model, top_n=20)

    # Error analysis for every model
    for name, res in all_results.items():
        misclassified = get_misclassified_samples(
            y_test, res["y_pred"], test_paths
        )
        plot_error_analysis(misclassified, name)

    # ── Done ───────────────────────────────────────────────────
    elapsed = time.time() - total_start
    print(f"\n{'═' * 60}")
    print(f"  PIPELINE COMPLETE  —  {elapsed:.1f}s total")
    print(f"  Results saved to:  results/")
    print(f"  Models saved to:   models/")
    print(f"{'═' * 60}\n")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Brain Tumor MRI Classification — Full ML Pipeline"
    )
    parser.add_argument(
        "--skip-extract", action="store_true",
        help="Skip feature extraction and load saved CSVs from data/features/"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip model training and load saved .pkl files from models/"
    )
    args = parser.parse_args()
    run(skip_extract=args.skip_extract, skip_train=args.skip_train)