"""
trainer.py
----------
Trains all five classifiers on the processed feature set.
- StandardScaler fit on train only, applied to both train and test
- Optional PCA fit on train only
- GridSearchCV for SVM and XGBoost (most hyperparameter-sensitive)
- StratifiedKFold cross-validation for all models
- All artifacts (models, scaler, PCA) saved as .pkl files
"""

import numpy as np
import joblib
import logging
import time
from pathlib import Path

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    MODELS_DIR, RANDOM_SEED, CV_FOLDS,
    APPLY_PCA, PCA_VARIANCE_RATIO,
    SVM_PARAM_GRID, SVM_DEFAULT_PARAMS,
    RF_PARAMS, KNN_PARAMS, NB_PARAMS,
    XGB_PARAM_GRID, XGB_DEFAULT_PARAMS,
    GRID_SEARCH_CV, GRID_SEARCH_SCORING, GRID_SEARCH_JOBS,
    MODEL_FILENAMES, SCALER_FILENAME, PCA_FILENAME
)

logger = logging.getLogger(__name__)

# Metrics computed during cross-validation for all models
CV_SCORING = {
    "accuracy":  "accuracy",
    "precision": "precision_weighted",
    "recall":    "recall_weighted",
    "f1":        "f1_weighted",
}


# ─────────────────────────────────────────────
# SCALER + PCA
# ─────────────────────────────────────────────

def fit_scaler_and_transform(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data only, then apply to both splits.
    Scaler is saved to disk for reproducibility and inference use.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    scaler_path = MODELS_DIR / SCALER_FILENAME
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved → {scaler_path}")

    return X_train_scaled, X_test_scaled, scaler


def fit_pca_and_transform(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA on training data only, retaining VARIANCE_RATIO of explained variance.
    Applied to test data using the train-fitted components.
    """
    if not APPLY_PCA:
        logger.info("PCA disabled in config — skipping.")
        return X_train, X_test, None

    pca = PCA(n_components=PCA_VARIANCE_RATIO, random_state=RANDOM_SEED)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    pca_path = MODELS_DIR / PCA_FILENAME
    joblib.dump(pca, pca_path)

    logger.info(
        f"PCA: {X_train.shape[1]} → {X_train_pca.shape[1]} components "
        f"({pca.explained_variance_ratio_.sum() * 100:.1f}% variance retained)"
    )
    logger.info(f"PCA saved → {pca_path}")

    return X_train_pca, X_test_pca, pca


# ─────────────────────────────────────────────
# MODEL BUILDERS
# ─────────────────────────────────────────────

def _build_svm(X_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """SVM with GridSearchCV over C, gamma, kernel."""
    logger.info("  Running GridSearchCV for SVM...")
    base = SVC(**SVM_DEFAULT_PARAMS)
    cv   = StratifiedKFold(n_splits=GRID_SEARCH_CV, shuffle=True, random_state=RANDOM_SEED)
    gs   = GridSearchCV(
        base, SVM_PARAM_GRID,
        cv=cv, scoring=GRID_SEARCH_SCORING,
        n_jobs=GRID_SEARCH_JOBS, verbose=0
    )
    gs.fit(X_train, y_train)
    logger.info(f"  SVM best params: {gs.best_params_}  |  CV score: {gs.best_score_:.4f}")
    return gs.best_estimator_


def _build_rf() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


def _build_knn() -> KNeighborsClassifier:
    return KNeighborsClassifier(**KNN_PARAMS)


def _build_nb() -> GaussianNB:
    return GaussianNB(**NB_PARAMS)


def _build_xgboost(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """XGBoost with GridSearchCV over n_estimators, learning_rate, max_depth."""
    logger.info("  Running GridSearchCV for XGBoost...")
    base = XGBClassifier(**XGB_DEFAULT_PARAMS)
    cv   = StratifiedKFold(n_splits=GRID_SEARCH_CV, shuffle=True, random_state=RANDOM_SEED)
    gs   = GridSearchCV(
        base, XGB_PARAM_GRID,
        cv=cv, scoring=GRID_SEARCH_SCORING,
        n_jobs=GRID_SEARCH_JOBS, verbose=0
    )
    gs.fit(X_train, y_train)
    logger.info(f"  XGBoost best params: {gs.best_params_}  |  CV score: {gs.best_score_:.4f}")
    return gs.best_estimator_


# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────

def run_cross_validation(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str
) -> dict:
    """
    Run StratifiedKFold CV and return mean ± std for each metric.
    Uses the already-fitted scaler/PCA data (X_train is already transformed).
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    scores = cross_validate(model, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=-1)

    result = {}
    for metric in CV_SCORING:
        key    = f"test_{metric}"
        mean   = scores[key].mean()
        std    = scores[key].std()
        result[f"{metric}_mean"] = round(mean, 4)
        result[f"{metric}_std"]  = round(std, 4)
        logger.info(f"  CV {metric:<12}: {mean:.4f} ± {std:.4f}")

    return result


# ─────────────────────────────────────────────
# MAIN TRAINING ENTRY POINT
# ─────────────────────────────────────────────

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    """
    Train all five models. For each model:
      1. Build with optimal hyperparameters
      2. Run StratifiedKFold CV → record mean ± std metrics
      3. Fit on full training set
      4. Save .pkl to models/

    Args:
        X_train : Scaled (and optionally PCA-reduced) training features
        y_train : Training labels (post-SMOTE)

    Returns:
        cv_results : dict mapping model_name → CV metric dict
    """
    builders = {
        "SVM":          lambda: _build_svm(X_train, y_train),
        "RandomForest": _build_rf,
        "KNN":          _build_knn,
        "NaiveBayes":   _build_nb,
        "XGBoost":      lambda: _build_xgboost(X_train, y_train),
    }

    cv_results = {}

    for name, builder in builders.items():
        print(f"\n{'─'*50}")
        print(f"  Training: {name}")
        print(f"{'─'*50}")

        t0    = time.time()
        model = builder()

        logger.info(f"  Running {CV_FOLDS}-fold StratifiedKFold CV...")
        cv_results[name] = run_cross_validation(model, X_train, y_train, name)

        # Final fit on full training set
        model.fit(X_train, y_train)
        elapsed = time.time() - t0
        cv_results[name]["train_time_sec"] = round(elapsed, 2)

        # Save model
        model_path = MODELS_DIR / MODEL_FILENAMES[name]
        joblib.dump(model, model_path)
        logger.info(f"  Saved → {model_path}  ({elapsed:.1f}s)")

    print(f"\n{'═'*50}")
    print("  All models trained and saved.")
    print(f"{'═'*50}\n")

    return cv_results


def load_model(model_name: str):
    """Load a saved model by name."""
    path = MODELS_DIR / MODEL_FILENAMES[model_name]
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def load_scaler() -> StandardScaler:
    path = MODELS_DIR / SCALER_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Scaler not found: {path}")
    return joblib.load(path)


def load_pca() -> PCA:
    path = MODELS_DIR / PCA_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"PCA not found: {path}")
    return joblib.load(path)