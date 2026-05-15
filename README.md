# 🧠 Brain Tumor MRI Classification

A complete, production-style **traditional Machine Learning pipeline** for classifying brain MRI images into four categories using handcrafted feature extraction and five classical ML algorithms.

---

## 📋 Project Overview

| Detail | Info |
|--------|------|
| **Course** | AIL-301 — Machine Learning |
| **University** | Bahria University, Islamabad |
| **Dataset** | Brain Tumor MRI Dataset (Cheng et al., 2017) |
| **Classes** | Glioma, Meningioma, No Tumor, Pituitary |
| **Best Model** | SVM — 91.87% Accuracy, AUC 0.977 |

---

## 🏗️ Pipeline Architecture

```
Dataset (7,200 MRI images)
    ↓
Image Preprocessing (Grayscale → Resize 128×128 → CLAHE → Normalize)
    ↓
Feature Extraction (HOG + LBP + GLCM → 1,779 features)
    ↓
StandardScaler (fit on train only)
    ↓
PCA (1,779 → 377 components, 95% variance retained)
    ↓
SMOTE Check (skipped — dataset already balanced)
    ↓
Train 5 Models (GridSearchCV + StratifiedKFold CV)
    ↓
Evaluate (Accuracy, Precision, Recall, F1, AUC)
    ↓
Visualize (Confusion Matrices, ROC Curves, Error Analysis)
```

---

## 📊 Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|--------|----------|-----|
| **SVM** | **0.9187** | **0.9221** | **0.9188** | **0.9164** | **0.9772** |
| KNN | 0.8750 | 0.8773 | 0.8750 | 0.8705 | 0.9616 |
| XGBoost | 0.8662 | 0.8668 | 0.8662 | 0.8620 | 0.9677 |
| Random Forest | 0.8581 | 0.8569 | 0.8581 | 0.8539 | 0.9654 |
| Naive Bayes | 0.7250 | 0.7337 | 0.7250 | 0.7269 | 0.8795 |

---

## 📁 Project Structure

```
brain_tumor_classifier/
├── src/
│   ├── config.py            # All hyperparameters and paths
│   ├── preprocessing.py     # Image loading and preprocessing
│   ├── feature_extractor.py # HOG, LBP, GLCM feature extraction
│   ├── balancer.py          # SMOTE class balancing
│   ├── trainer.py           # Model training and cross-validation
│   ├── evaluator.py         # Metrics and evaluation
│   └── visualizer.py        # All plots and visualizations
├── data/
│   ├── raw_download/        # Dataset (not tracked in git)
│   └── features/            # Extracted feature CSVs (not tracked)
├── models/                  # Saved .pkl files (not tracked)
├── results/                 # Generated plots and metrics (not tracked)
├── main.py                  # Single entry point
└── requirements.txt
```

---

## ⚙️ Setup & Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/brain-tumor-classifier.git
cd brain-tumor-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
```bash
kaggle datasets download -d masoudnickparvar/brain-tumor-mri-dataset --unzip -p data/raw_download
```

### 4. Run the full pipeline
```bash
python main.py
```

### 5. Skip re-extraction and re-training (if already run once)
```bash
python main.py --skip-extract --skip-train
```

---

## 🔬 Feature Extraction

| Descriptor | What It Captures | Features |
|------------|-----------------|----------|
| **HOG** | Shape, edges, gradient directions | ~1,740 |
| **LBP** | Local micro-texture | 10 |
| **GLCM** | Statistical texture (contrast, correlation, energy, homogeneity, dissimilarity) | 5 |
| **Combined** | All three perspectives | 1,779 |

---

## 🧪 Key Design Decisions

- **No data leakage** — StandardScaler and PCA fitted on training data only
- **Stratified splits** — class distribution preserved across train/test
- **SMOTE with intelligence** — auto-skips when dataset is already balanced
- **GridSearchCV** — optimal hyperparameters for SVM and XGBoost
- **StratifiedKFold CV** — mean ± std metrics for all 5 models
- **Modular architecture** — each stage is an independent, reusable module

---

## 📦 Dependencies

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
scikit-image==0.23.2
imbalanced-learn==0.12.3
xgboost==2.0.3
opencv-python==4.9.0.80
matplotlib==3.8.4
seaborn==0.13.2
joblib==1.4.2
```

---

## 👥 Authors

| Name | Enrollment |
|------|-----------|
| Syed Muhammad Burair Abbas | 01-134232-176 |
| Muhammad Reyan Riaz | 01-134232-205 |

**Bahria University, Islamabad — Department of Computer Science**

---

## 📚 References

1. Cheng, J. et al. (2017). Brain tumor dataset. *Figshare*. https://doi.org/10.6084/m9.figshare.1512427.v5
2. Nickparvar, M. (2021). Brain Tumor MRI Dataset. *Kaggle*.
3. Dalal, N. & Triggs, B. (2005). Histograms of oriented gradients for human detection. *CVPR*.
4. Ojala, T. et al. (2002). Multiresolution gray-scale and rotation invariant texture with LBP. *IEEE TPAMI*.
5. Haralick, R.M. et al. (1973). Textural features for image classification. *IEEE Trans. SMC*.
