"""
config.py
---------
Central configuration for the Brain Tumor MRI Classification project.
All hyperparameters, paths, and settings are defined here.
No other file should contain hardcoded values.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# PROJECT ROOT
# ─────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent


# ─────────────────────────────────────────────
# DATA PATHS
# ─────────────────────────────────────────────
DATA_DIR            = ROOT_DIR / "data"
RAW_DOWNLOAD_DIR    = DATA_DIR / "raw_download"
RAW_TRAIN_DIR       = RAW_DOWNLOAD_DIR / "Training"
RAW_TEST_DIR        = RAW_DOWNLOAD_DIR / "Testing"
FEATURES_DIR        = DATA_DIR / "features"

TRAIN_FEATURES_PATH = FEATURES_DIR / "train_features.csv"
TEST_FEATURES_PATH  = FEATURES_DIR / "test_features.csv"


# ─────────────────────────────────────────────
# OUTPUT PATHS
# ─────────────────────────────────────────────
MODELS_DIR           = ROOT_DIR / "models"
RESULTS_DIR          = ROOT_DIR / "results"
CONFUSION_MATRIX_DIR = RESULTS_DIR / "confusion_matrices"
ROC_CURVES_DIR       = RESULTS_DIR / "roc_curves"
PLOTS_DIR            = RESULTS_DIR / "plots"
METRICS_TABLE_PATH   = RESULTS_DIR / "metrics_table.csv"
DATA_REPORT_PATH     = RESULTS_DIR / "data_report.txt"


# ─────────────────────────────────────────────
# CLASS CONFIGURATION
# Matches exact folder names in the dataset
# ─────────────────────────────────────────────
CLASSES     = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls for cls, idx in CLASS_TO_IDX.items()}

# Human-readable display names for plots and reports
CLASS_DISPLAY_NAMES = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "notumor":    "No Tumor",
    "pituitary":  "Pituitary",
}


# ─────────────────────────────────────────────
# REPRODUCIBILITY
# ─────────────────────────────────────────────
RANDOM_SEED = 42


# ─────────────────────────────────────────────
# DATASET — already split by provider
# We use their split as-is (no re-splitting)
# but we still verify stratification
# ─────────────────────────────────────────────
DATASET_PRE_SPLIT = True   # masoudnickparvar dataset comes with Training/Testing folders


# ─────────────────────────────────────────────
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────
IMAGE_SIZE       = (128, 128)
APPLY_CLAHE      = True
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE  = (8, 8)
# Gaussian blur intentionally disabled:
# LBP and GLCM rely on micro-texture — blur degrades these features
APPLY_BLUR       = False


# ─────────────────────────────────────────────
# FEATURE EXTRACTION TOGGLES
# ─────────────────────────────────────────────
USE_HOG  = True
USE_LBP  = True
USE_GLCM = True

# HOG
HOG_ORIENTATIONS    = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)

# LBP
LBP_RADIUS   = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD   = "uniform"
LBP_N_BINS   = LBP_N_POINTS + 2

# GLCM
GLCM_DISTANCES  = [1]
GLCM_ANGLES     = [0, 0.785, 1.571, 2.356]
GLCM_PROPERTIES = ["contrast", "correlation", "energy", "homogeneity", "dissimilarity"]


# ─────────────────────────────────────────────
# DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────
APPLY_PCA          = True
PCA_VARIANCE_RATIO = 0.95


# ─────────────────────────────────────────────
# CLASS BALANCING
# ─────────────────────────────────────────────
APPLY_SMOTE    = True
SMOTE_STRATEGY = "auto"
SMOTE_K        = 5


# ─────────────────────────────────────────────
# CROSS VALIDATION
# ─────────────────────────────────────────────
CV_FOLDS = 5


# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
SVM_PARAM_GRID = {
    "C":      [0.1, 1, 10, 100],
    "gamma":  ["scale", "auto"],
    "kernel": ["rbf", "poly"],
}
SVM_DEFAULT_PARAMS = {
    "probability":  True,
    "random_state": RANDOM_SEED,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth":    None,
    "max_features": "sqrt",
    "n_jobs":       -1,
    "random_state": RANDOM_SEED,
}

KNN_PARAMS = {
    "n_neighbors": 7,
    "metric":      "euclidean",
    "weights":     "distance",
    "n_jobs":      -1,
}

NB_PARAMS = {
    "var_smoothing": 1e-9,
}

XGB_PARAM_GRID = {
    "n_estimators":  [100, 200],
    "learning_rate": [0.05, 0.1],
    "max_depth":     [4, 6],
}
XGB_DEFAULT_PARAMS = {
    "eval_metric":  "mlogloss",
    "random_state": RANDOM_SEED,
    "n_jobs":       -1,
}


# ─────────────────────────────────────────────
# GRID SEARCH
# ─────────────────────────────────────────────
GRID_SEARCH_CV      = 3
GRID_SEARCH_SCORING = "f1_weighted"
GRID_SEARCH_JOBS    = -1


# ─────────────────────────────────────────────
# VISUALIZATION
# ─────────────────────────────────────────────
PLOT_DPI               = 150
PLOT_STYLE             = "seaborn-v0_8-whitegrid"
PLOT_PALETTE           = "Set2"
FIGURE_SIZE            = (10, 7)
CMAP_CONFUSION         = "Blues"
ERROR_ANALYSIS_SAMPLES = 5


# ─────────────────────────────────────────────
# SAVED ARTIFACT FILENAMES
# ─────────────────────────────────────────────
MODEL_FILENAMES = {
    "SVM":          "svm.pkl",
    "RandomForest": "random_forest.pkl",
    "KNN":          "knn.pkl",
    "NaiveBayes":   "naive_bayes.pkl",
    "XGBoost":      "xgboost.pkl",
}
SCALER_FILENAME = "scaler.pkl"
PCA_FILENAME    = "pca.pkl"


# ─────────────────────────────────────────────
# RUNTIME — ensure all output dirs exist
# ─────────────────────────────────────────────
def _ensure_dirs():
    for d in [FEATURES_DIR, MODELS_DIR, CONFUSION_MATRIX_DIR, ROC_CURVES_DIR, PLOTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

_ensure_dirs()