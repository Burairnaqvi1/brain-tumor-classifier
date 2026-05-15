"""
visualizer.py
-------------
All visualization functions for the project.
Every plot is saved as a high-resolution PNG to results/.
Consistent style, palette, and DPI across all figures.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2
import logging
from pathlib import Path

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from src.config import (
    CLASSES, CLASS_DISPLAY_NAMES, NUM_CLASSES,
    PLOT_DPI, PLOT_STYLE, PLOT_PALETTE, FIGURE_SIZE, CMAP_CONFUSION,
    CONFUSION_MATRIX_DIR, ROC_CURVES_DIR, PLOTS_DIR,
    RAW_TRAIN_DIR
)

logger = logging.getLogger(__name__)

try:
    plt.style.use(PLOT_STYLE)
except:
    plt.style.use("seaborn-v0_8-whitegrid")

DISPLAY_NAMES = [CLASS_DISPLAY_NAMES.get(c, c) for c in CLASSES]
COLORS        = sns.color_palette(PLOT_PALETTE, NUM_CLASSES)
MODEL_COLORS  = sns.color_palette("tab10", 5)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, tight: bool = True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved plot → {path}")


# ─────────────────────────────────────────────
# 1. CLASS DISTRIBUTION
# ─────────────────────────────────────────────

def plot_class_distribution(report_before: dict, report_after: dict = None):
    """Bar chart of class counts before (and optionally after) SMOTE."""
    fig, axes = plt.subplots(1, 2 if report_after else 1, figsize=(14 if report_after else 8, 5))

    if report_after is None:
        axes = [axes]

    def _draw(ax, data, title):
        counts = [data.get(cls, 0) for cls in CLASSES]
        bars   = ax.bar(DISPLAY_NAMES, counts, color=COLORS, edgecolor="white", linewidth=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Class", fontsize=11)
        ax.set_ylabel("Sample Count", fontsize=11)
        ax.set_ylim(0, max(counts) * 1.15)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                    str(count), ha="center", va="bottom", fontsize=10, fontweight="bold")

    _draw(axes[0], report_before["train"], "Training Set — Original Distribution")
    if report_after:
        _draw(axes[1], report_after, "Training Set — After SMOTE")

    _save(fig, PLOTS_DIR / "class_distribution.png")


# ─────────────────────────────────────────────
# 2. SAMPLE IMAGES GRID
# ─────────────────────────────────────────────

def plot_sample_images(n_per_class: int = 4):
    """Grid of sample MRI images — one row per class."""
    fig, axes = plt.subplots(NUM_CLASSES, n_per_class, figsize=(n_per_class * 3, NUM_CLASSES * 3))
    fig.suptitle("Sample MRI Images by Class", fontsize=14, fontweight="bold", y=1.01)

    for row, cls in enumerate(CLASSES):
        cls_dir = RAW_TRAIN_DIR / cls
        imgs    = list(cls_dir.glob("*.jpg"))[:n_per_class]

        for col in range(n_per_class):
            ax = axes[row][col]
            ax.axis("off")
            if col < len(imgs):
                img = cv2.imread(str(imgs[col]), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    ax.imshow(img, cmap="gray")
            if col == 0:
                ax.set_ylabel(CLASS_DISPLAY_NAMES.get(cls, cls),
                              fontsize=11, fontweight="bold", rotation=0,
                              labelpad=60, va="center")

    _save(fig, PLOTS_DIR / "sample_images.png")


# ─────────────────────────────────────────────
# 3. CONFUSION MATRICES
# ─────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, model_name: str):
    """Normalized and raw confusion matrix heatmap for one model."""
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Raw Counts", "Normalized (Row %)"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt,
            xticklabels=DISPLAY_NAMES,
            yticklabels=DISPLAY_NAMES,
            cmap=CMAP_CONFUSION,
            linewidths=0.5, linecolor="white",
            ax=ax, cbar=True
        )
        ax.set_title(title, fontsize=11, pad=10)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.tick_params(axis="x", rotation=30)
        ax.tick_params(axis="y", rotation=0)

    fname = model_name.lower().replace(" ", "_")
    _save(fig, CONFUSION_MATRIX_DIR / f"cm_{fname}.png")


def plot_all_confusion_matrices(all_results: dict):
    for name, res in all_results.items():
        plot_confusion_matrix(res["cm"], name)


# ─────────────────────────────────────────────
# 4. ROC CURVES
# ─────────────────────────────────────────────

def plot_roc_curves(all_results: dict, y_test: np.ndarray):
    """One ROC plot per model (OvR multiclass), plus all models on one combined plot."""
    y_bin = label_binarize(y_test, classes=list(range(NUM_CLASSES)))

    # Individual ROC per model
    for name, res in all_results.items():
        y_prob = res["y_prob"]
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)

        for i, (cls, color) in enumerate(zip(CLASSES, COLORS)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
            roc_auc     = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{CLASS_DISPLAY_NAMES.get(cls, cls)} (AUC = {roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(f"ROC Curves — {name}", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)

        fname = name.lower().replace(" ", "_")
        _save(fig, ROC_CURVES_DIR / f"roc_{fname}.png")

    # Combined: one curve per model (macro-average AUC)
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    for (name, res), color in zip(all_results.items(), MODEL_COLORS):
        y_prob   = res["y_prob"]
        macro_fpr, macro_tpr, macro_auc_val = _macro_roc(y_bin, y_prob)
        ax.plot(macro_fpr, macro_tpr, color=color, lw=2,
                label=f"{name} (Macro AUC = {macro_auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models (Macro Average)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    _save(fig, PLOTS_DIR / "roc_all_models.png")


def _macro_roc(y_bin, y_prob):
    """Compute macro-average ROC curve."""
    all_fpr = np.unique(np.concatenate([
        roc_curve(y_bin[:, i], y_prob[:, i])[0] for i in range(NUM_CLASSES)
    ]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(NUM_CLASSES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        mean_tpr   += np.interp(all_fpr, fpr, tpr)
    mean_tpr /= NUM_CLASSES
    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


# ─────────────────────────────────────────────
# 5. METRICS BAR CHART
# ─────────────────────────────────────────────

def plot_metrics_comparison(all_results: dict):
    """Grouped bar chart — all models × all metrics side by side."""
    metrics      = ["accuracy", "precision", "recall", "f1", "auc"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]
    model_names  = list(all_results.keys())
    n_models     = len(model_names)
    n_metrics    = len(metrics)

    x     = np.arange(n_metrics)
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, color) in enumerate(zip(model_names, MODEL_COLORS)):
        values = [all_results[name][m] for m in metrics]
        bars   = ax.bar(x + i * width, values, width, label=name, color=color,
                        edgecolor="white", linewidth=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, rotation=45)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(metric_names, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Performance Comparison — All Metrics", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.7, alpha=0.5)

    _save(fig, PLOTS_DIR / "metrics_comparison.png")


# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Random Forest)
# ─────────────────────────────────────────────

def plot_feature_importance(rf_model, top_n: int = 20):
    """Bar chart of top N feature importances from the Random Forest model."""
    importances = rf_model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_values  = importances[indices]
    top_labels  = [f"f{i}" for i in indices]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(top_n), top_values, color=sns.color_palette("Blues_r", top_n),
           edgecolor="white")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(top_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Importance Score", fontsize=11)
    ax.set_title(f"Top {top_n} Feature Importances — Random Forest", fontsize=13, fontweight="bold")

    _save(fig, PLOTS_DIR / "feature_importance.png")


# ─────────────────────────────────────────────
# 7. TRAINING TIME
# ─────────────────────────────────────────────

def plot_training_time(cv_results: dict):
    """Horizontal bar chart of training time per model."""
    names = list(cv_results.keys())
    times = [cv_results[n].get("train_time_sec", 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(names, times, color=MODEL_COLORS[:len(names)], edgecolor="white")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_title("Training Time per Model", fontsize=13, fontweight="bold")
    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", fontsize=10)

    _save(fig, PLOTS_DIR / "training_time.png")


# ─────────────────────────────────────────────
# 8. ERROR ANALYSIS
# ─────────────────────────────────────────────

def plot_error_analysis(misclassified: list, model_name: str):
    """
    Show misclassified MRI images with true vs predicted labels.
    Helps explain WHY the model made mistakes (e.g., visual similarity between classes).
    """
    n = len(misclassified)
    if n == 0:
        logger.info(f"No misclassified samples to display for {model_name}.")
        return

    fig, axes = plt.subplots(1, n, figsize=(n * 3.5, 4))
    if n == 1:
        axes = [axes]

    fig.suptitle(f"Error Analysis — {model_name}\n(Misclassified Samples)",
                 fontsize=12, fontweight="bold")

    for ax, sample in zip(axes, misclassified):
        img = cv2.imread(str(sample["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            ax.axis("off")
            continue
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        true_d = CLASS_DISPLAY_NAMES.get(sample["true"], sample["true"])
        pred_d = CLASS_DISPLAY_NAMES.get(sample["predicted"], sample["predicted"])
        ax.set_title(f"True: {true_d}\nPred: {pred_d}",
                     fontsize=9, color="red", fontweight="bold")

    fname = model_name.lower().replace(" ", "_")
    _save(fig, PLOTS_DIR / f"error_analysis_{fname}.png")


# ─────────────────────────────────────────────
# 9. CV RESULTS (mean ± std)
# ─────────────────────────────────────────────

def plot_cv_results(cv_results: dict):
    """Bar chart with error bars showing CV mean ± std accuracy per model."""
    names  = list(cv_results.keys())
    means  = [cv_results[n].get("accuracy_mean", 0) for n in names]
    stds   = [cv_results[n].get("accuracy_std", 0) for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, capsize=6,
                  color=MODEL_COLORS[:len(names)], edgecolor="white",
                  error_kw={"elinewidth": 2, "ecolor": "black"})
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("CV Accuracy (Mean ± Std)", fontsize=11)
    ax.set_title(f"{5}-Fold Stratified CV Accuracy", fontsize=13, fontweight="bold")
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + std + 0.01,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    _save(fig, PLOTS_DIR / "cv_accuracy.png")