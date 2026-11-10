"""
Global configuration for GP-109 BTP project.
Can an AI Explanation Be Misused? Testing Whether SHAP and LIME
Explanations Help Someone Cheat a Clinical Risk Model.

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
    print(f"[CONFIG] All seeds set to {seed}")

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
N_ESTIMATORS_RF = 100   # Random Forest trees
MAX_DEPTH_RF = 10       # Random Forest max depth

# ============================================================
# EXPERIMENT PARAMETERS
# ============================================================
N_BLIND_ATTEMPTS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]
# ^ The number of blind queries we test against one explanation

# ============================================================
# FINAL CONFIG — Locked after pre-registration (Stage 6)
# Do NOT modify after publishing the analysis plan.
# ============================================================
FINAL_CONFIG = {
    # Data
    'development_cycle': '2015_2016',    # Already analyzed
    'holdout_cycle': '2017_2018',        # NOT YET TOUCHED

    # Models
    'model_types': ['rf', 'lr'],
    'rf_n_estimators': N_ESTIMATORS_RF,
    'rf_max_depth': MAX_DEPTH_RF,
    'rf_random_state': SEED_MODEL_RF,
    'lr_random_state': SEED_MODEL_LOGREG,
    'lr_max_iter': 1000,

    # Split
    'test_size': TEST_SIZE,
    'split_random_state': SEED_TRAIN_TEST_SPLIT,

    # Adversary
    'lie_budget_method': 'percentile',
    'lie_budget_percentile': 95,
    'blind_attempts': N_BLIND_ATTEMPTS,

    # Explanations
    'shap_method': 'tree',          # TreeSHAP for RF
    'lime_n_samples': 500,
    'lime_n_features': 5,
    'lime_random_state': 42,

    # Statistics
    'bootstrap_n_iterations': 1000,
    'confidence_level': 0.95,
    'adversarial_seed': SEED_ADVERSARIAL,
}
