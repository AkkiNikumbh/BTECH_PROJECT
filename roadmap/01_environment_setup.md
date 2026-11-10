# 🔧 Stage 1: Environment Setup & Code Testing
## Period: 8 October – 21 October 2026 (2 weeks)

> **← Back to** [Master Roadmap](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/00_master_roadmap.md)
> **→ Next Stage**: [Stage 2 — Data Documentation](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/02_data_documentation.md)

---

## 🎯 Objectives

1. Set up a reproducible Python development environment on all three team laptops
2. Install and verify every required library
3. Initialize version control with fixed random seeds
4. Validate the full pipeline (train → explain → measure) on a small toy dataset where results can be checked by hand
5. Establish project folder structure and coding conventions

---

## 📋 Prerequisites

| Item | Requirement |
|------|-------------|
| Hardware | 3 personal laptops, 8–16 GB RAM |
| OS | Windows / Linux / macOS |
| Python | 3.9 or later already installed |
| Internet | Required for downloading packages and datasets |

---

## 📁 Project Folder Structure to Create

```
BTP/
├── roadmap/                    # ← You are here (roadmap docs)
├── data/
│   ├── raw/                    # Original downloaded files (never modify)
│   │   ├── nhanes_2015_2016/
│   │   ├── nhanes_2017_2018/
│   │   ├── uci_heart/         # UCI Heart Disease (dataset 45) — for testing
│   │   └── cdc_diabetes/      # CDC Diabetes Health Indicators (UCI 891)
│   ├── processed/             # Cleaned, merged datasets
│   └── results/               # Experiment outputs
├── src/
│   ├── __init__.py
│   ├── config.py              # Global config, random seeds, paths
│   ├── data_loader.py         # Data loading and preprocessing
│   ├── models.py              # Model training (LogReg, RF)
│   ├── explanations.py        # SHAP & LIME explanation generation
│   ├── adversarial.py         # Adversarial simulation engine
│   ├── metrics.py             # Measurement & comparison utilities
│   └── visualization.py       # Plotting functions
├── notebooks/
│   └── 01_sanity_check.ipynb  # Quick exploratory analysis
├── tests/
│   ├── test_models.py
│   ├── test_explanations.py
│   └── test_adversarial.py
├── scripts/
│   ├── run_trial.py           # Trial run script (Stage 5)
│   └── run_final.py           # Final experiment script (Stage 7)
├── results/
│   ├── figures/               # Generated plots
│   └── tables/                # Generated tables (CSV/LaTeX)
├── docs/
│   ├── column_mapping.md      # NHANES column documentation (Stage 2)
│   └── analysis_plan.md       # Pre-registered plan (Stage 6)
├── requirements.txt
├── README.md
├── .gitignore
└── setup.py
```

---

## 🔨 Step-by-Step Implementation

### Task 1.1 — Create the Python Environment

```bash
# Create a virtual environment
python -m venv btp_env

# Activate it
# Windows:
btp_env\Scripts\activate
# Linux/Mac:
source btp_env/bin/activate

# Verify Python version
python --version  # Should be 3.9+
```

### Task 1.2 — Install All Required Libraries

Create `requirements.txt`:

```txt
# Data handling
pandas>=2.0.0
numpy>=1.24.0

# Machine learning
scikit-learn>=1.3.0

# Explainability
shap>=0.42.0
lime>=0.2.0

# Statistics
scipy>=1.11.0
statsmodels>=0.14.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Testing
pytest>=7.4.0

# Utilities
tqdm>=4.65.0
```

```bash
pip install -r requirements.txt
```

### Task 1.3 — Create the Global Config with Fixed Seeds

Create `src/config.py`:

```python
"""
Global configuration for GP-109 BTP project.
All random seeds are fixed here for reproducibility.
"""
import os
import random
import numpy as np

# ============================================================
# RANDOM SEEDS — Fixed for reproducibility
# ============================================================
GLOBAL_SEED = 42
SEED_MODEL_LOGREG = 42
SEED_MODEL_RF = 123
SEED_ADVERSARIAL = 456
SEED_TRAIN_TEST_SPLIT = 789

def set_all_seeds(seed=GLOBAL_SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# ============================================================
# PATHS
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
DATA_RESULTS = os.path.join(PROJECT_ROOT, 'data', 'results')
RESULTS_FIGURES = os.path.join(PROJECT_ROOT, 'results', 'figures')
RESULTS_TABLES = os.path.join(PROJECT_ROOT, 'results', 'tables')

# ============================================================
# MODEL PARAMETERS
# ============================================================
TEST_SIZE = 0.2
N_ESTIMATORS_RF = 100  # Random Forest trees
MAX_DEPTH_RF = 10

# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================
N_BLIND_ATTEMPTS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
# ^ The number of blind queries we test against one explanation
```

### Task 1.4 — Initialize Git Repository

```bash
git init
git add .
git commit -m "Initial project structure with fixed seeds and requirements"
```

Create `.gitignore`:

```
btp_env/
__pycache__/
*.pyc
.ipynb_checkpoints/
data/raw/
*.pkl
*.joblib
.DS_Store
```

### Task 1.5 — Download and Test with UCI Heart Disease Dataset

This is the "toy dataset" for verifying the pipeline works end-to-end.

```python
"""
scripts/test_pipeline.py
Sanity check: run full pipeline on UCI Heart Disease dataset.
Results should be verifiable by hand.
"""
import sys
sys.path.append('..')

from src.config import set_all_seeds, SEED_TRAIN_TEST_SPLIT
set_all_seeds()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import shap
import lime
import lime.lime_tabular

# ---- Step 1: Load UCI Heart Disease ----
from sklearn.datasets import fetch_openml
heart = fetch_openml(data_id=45, as_frame=True)
X = heart.data
y = (heart.target.astype(int) > 0).astype(int)  # Binary: disease or not

print(f"Dataset shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# ---- Step 2: Train/Test Split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED_TRAIN_TEST_SPLIT, stratify=y
)

# ---- Step 3: Train Models ----
# Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
print(f"\nLogistic Regression Accuracy: {accuracy_score(y_test, lr.predict(X_test_scaled)):.4f}")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=123)
rf.fit(X_train, y_train)
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.4f}")

# ---- Step 4: Generate SHAP Explanations ----
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

print(f"\nSHAP values shape: {np.array(shap_values).shape}")
print(f"Top 3 features by mean |SHAP|:")
mean_shap = np.abs(shap_values[1]).mean(axis=0)
top_features = np.argsort(mean_shap)[-3:][::-1]
for idx in top_features:
    print(f"  {X.columns[idx]}: {mean_shap[idx]:.4f}")

# ---- Step 5: Generate LIME Explanation for 1 sample ----
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns.tolist(),
    class_names=['No Disease', 'Disease'],
    mode='classification',
    random_state=42
)

sample_idx = 0
lime_exp = lime_explainer.explain_instance(
    X_test.iloc[sample_idx].values,
    rf.predict_proba,
    num_features=5
)

print(f"\nLIME explanation for sample {sample_idx}:")
for feat, weight in lime_exp.as_list():
    print(f"  {feat}: {weight:+.4f}")

# ---- Step 6: Simple adversarial test ----
# Manually change the most important feature and observe score change
sample = X_test.iloc[sample_idx].copy()
original_score = rf.predict_proba(sample.values.reshape(1, -1))[0, 1]
print(f"\nOriginal disease risk score: {original_score:.4f}")

# Change top SHAP feature by 1 std
top_feat_idx = top_features[0]
top_feat_name = X.columns[top_feat_idx]
change = X_train[top_feat_name].std()
sample[top_feat_name] -= change  # Reduce to lower risk

new_score = rf.predict_proba(sample.values.reshape(1, -1))[0, 1]
print(f"After changing '{top_feat_name}' by -1 std: {new_score:.4f}")
print(f"Score reduction: {original_score - new_score:.4f}")

print("\n✅ Pipeline sanity check PASSED — all components working!")
```

### Task 1.6 — Write Basic Tests

Create `tests/test_models.py`:

```python
"""Basic tests to verify model training works correctly."""
import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def test_rf_reproducibility():
    """Two RF models with same seed should give identical predictions."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    rf1 = RandomForestClassifier(n_estimators=10, random_state=123)
    rf2 = RandomForestClassifier(n_estimators=10, random_state=123)
    rf1.fit(X, y)
    rf2.fit(X, y)
    assert np.array_equal(rf1.predict(X), rf2.predict(X))

def test_lr_reproducibility():
    """Two LR models with same seed should give identical predictions."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    lr1 = LogisticRegression(random_state=42, max_iter=1000)
    lr2 = LogisticRegression(random_state=42, max_iter=1000)
    lr1.fit(X, y)
    lr2.fit(X, y)
    assert np.allclose(lr1.predict_proba(X), lr2.predict_proba(X))

def test_shap_runs():
    """SHAP should produce explanations without error."""
    import shap
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    rf = RandomForestClassifier(n_estimators=10, random_state=123)
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X[:5])
    assert shap_values is not None
```

Run with:
```bash
pytest tests/ -v
```

---

## ✅ Completion Checklist

- [ ] Python 3.9+ virtual environment created on **all 3 laptops**
- [ ] All libraries in `requirements.txt` install without errors
- [ ] Git repository initialized with `.gitignore`
- [ ] Fixed random seeds verified (same results on different machines)
- [ ] Project folder structure created as specified above
- [ ] `src/config.py` written with all seeds and paths
- [ ] UCI Heart Disease dataset downloaded and loadable
- [ ] `test_pipeline.py` runs end-to-end without errors
- [ ] SHAP explanations generated successfully on toy data
- [ ] LIME explanations generated successfully on toy data
- [ ] Adversarial score change computed on at least 1 sample
- [ ] `pytest tests/ -v` passes all tests
- [ ] First commit pushed to shared Git repository
- [ ] All team members can pull, run, and reproduce identical results

---

## 📤 Outputs → Stage 2

| Output | Format | Used By |
|--------|--------|---------|
| Working virtual environment | `requirements.txt` | All subsequent stages |
| Project folder structure | Directory tree | All subsequent stages |
| `src/config.py` with seeds | Python module | Stages 3–8 |
| Verified pipeline on toy data | `test_pipeline.py` results | Confidence check |
| Git repo with first commit | `.git/` | All subsequent stages |

---

## ⚠️ Common Pitfalls

> [!WARNING]
> - **SHAP version compatibility**: `shap>=0.42` requires specific NumPy versions. If installation fails, try `pip install shap --no-build-isolation`
> - **LIME random state**: LIME does NOT always respect `random_state` in all internal sampling. Document any non-deterministic behavior.
> - **Windows path issues**: Use `os.path.join()` everywhere, never hardcode `/` or `\\`
> - **Git Large Files**: Add `data/raw/` to `.gitignore` immediately — NHANES files can be large
