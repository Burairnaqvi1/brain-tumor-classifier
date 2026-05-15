"""
evaluator.py
------------
Full evaluation suite for all trained models.
Computes per-model and per-class metrics, confusion matrices,
ROC curves, and error analysis (misclassified samples).
All results exported to results/metrics_table.csv.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix
)

from src.config import (
    CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, CLASS_DISPLAY_NAMES,
    METRICS_TABLE_PATH, MODEL_FILENAMES, MODELS_DIR,
    ERROR_ANALYSIS_SAMPLES
)
from src.trainer import load_model

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SINGLE MODEL EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> dict:
    """
    Evaluate a single model on the test set.

    Returns a dict with:
      - accuracy, precision, recall, f1, auc (weighted)
      - per-class precision, recall, f1
      - confusion matrix
      - predicted labels and probabilities
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    result = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "auc":       round(roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"), 4),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "cm":        confusion_matrix(y_test, y_pred),
    }

    # Per-class metrics
    per_class_p = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_r = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class_f = f1_score(y_test, y_pred, average=None, zero_division=0)

    for i, cls in enumerate(CLASSES):
        display = CLASS_DISPLAY_NAMES.get(cls, cls)
        result[f"precision_{display}"] = round(per_class_p[i], 4)
        result[f"recall_{display}"]    = round(per_class_r[i], 4)
        result[f"f1_{display}"]        = round(per_class_f[i], 4)

    _print_model_results(model_name, result, y_test, y_pred)
    return result


def _print_model_results(name: str, result: dict, y_test: np.ndarray, y_pred: np.ndarray):
    print(f"\n  {'─'*50}")
    print(f"  Results: {name}")
    print(f"  {'─'*50}")
    print(f"  Accuracy  : {result['accuracy']:.4f}")
    print(f"  Precision : {result['precision']:.4f}")
    print(f"  Recall    : {result['recall']:.4f}")
    print(f"  F1-Score  : {result['f1']:.4f}")
    print(f"  AUC       : {result['auc']:.4f}")
    print()
    display_names = [CLASS_DISPLAY_NAMES.get(c, c) for c in CLASSES]
    print(classification_report(
        y_test, y_pred,
        target_names=display_names,
        zero_division=0
    ))


# ─────────────────────────────────────────────
# EVALUATE ALL MODELS
# ─────────────────────────────────────────────

def evaluate_all_models(
    X_test: np.ndarray,
    y_test: np.ndarray,
    cv_results: dict
) -> dict:
    """
    Load and evaluate all saved models on the test set.

    Args:
        X_test     : Scaled (and optionally PCA-reduced) test features
        y_test     : True test labels
        cv_results : CV metrics returned by trainer.train_all_models()

    Returns:
        all_results: dict mapping model_name → evaluation result dict
    """
    all_results = {}

    for model_name in MODEL_FILENAMES:
        model = load_model(model_name)
        result = evaluate_model(model, X_test, y_test, model_name)
        # Merge CV results into evaluation result
        result.update(cv_results.get(model_name, {}))
        all_results[model_name] = result

    return all_results


# ─────────────────────────────────────────────
# METRICS TABLE EXPORT
# ─────────────────────────────────────────────

def export_metrics_table(all_results: dict, cv_results: dict):
    """
    Export a clean comparison table as CSV.
    Rows = models, Columns = metrics (test set + CV mean ± std).
    """
    rows = []
    for name, res in all_results.items():
        cv = cv_results.get(name, {})
        row = {
            "Model":           name,
            "Accuracy":        res["accuracy"],
            "Precision":       res["precision"],
            "Recall":          res["recall"],
            "F1-Score":        res["f1"],
            "AUC":             res["auc"],
            "CV Acc Mean":     cv.get("accuracy_mean", "-"),
            "CV Acc Std":      cv.get("accuracy_std", "-"),
            "CV F1 Mean":      cv.get("f1_mean", "-"),
            "CV F1 Std":       cv.get("f1_std", "-"),
            "Train Time (s)":  cv.get("train_time_sec", "-"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(METRICS_TABLE_PATH, index=False)
    logger.info(f"Metrics table saved → {METRICS_TABLE_PATH}")

    # Pretty print to console
    print("\n" + "=" * 90)
    print("  FINAL PERFORMANCE COMPARISON TABLE")
    print("=" * 90)
    print(df.to_string(index=False))
    print("=" * 90 + "\n")


# ─────────────────────────────────────────────
# ERROR ANALYSIS
# ─────────────────────────────────────────────

def get_misclassified_samples(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    paths: list,
    n: int = ERROR_ANALYSIS_SAMPLES
) -> list:
    """
    Return up to n misclassified sample records for error analysis visualization.

    Returns list of dicts with: path, true_label, predicted_label
    """
    misclassified = []
    for i, (true, pred) in enumerate(zip(y_test, y_pred)):
        if true != pred:
            misclassified.append({
                "path":      paths[i],
                "true":      IDX_TO_CLASS[true],
                "predicted": IDX_TO_CLASS[pred],
            })
        if len(misclassified) >= n:
            break
    return misclassified