"""
Global configuration for GP-109 BTP project.
Can an AI Explanation Be Misused? Testing Whether SHAP and LIME
Explanations Help Someone Cheat a Clinical Risk Model.

All random seeds are fixed here for reproducibility.
"""
import os
import sys
import random
import numpy as np

# The scripts print symbols like ≈ ✅ ❌ →. A default Windows console
# (cp1252) cannot encode them and Python crashes with UnicodeEncodeError.
# Switch stdout/stderr to UTF-8 so the scripts run in any terminal.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        pass

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
# Per-config results of run_final.py. A NEW folder for every version of the
# experiment code, so results from older code ('final_run_cache' = original,
# 'final_run_cache_v2_fixed' = v2) are never re-loaded by mistake on resume.
FINAL_RUN_CACHE = os.path.join(DATA_PROCESSED, 'final_run_cache_v3')

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
# GAMEABLE FEATURES — what a patient can actually lie about
# ============================================================
# Only SELF-REPORTED height/weight are gameable (see docs/column_mapping.md
# and docs/analysis_plan.md §4). Nurse-measured height/weight
# (measured_height_cm, measured_weight_kg) are NOT gameable: the patient
# cannot change what the nurse measures. So Model A (measured inputs) has
# no gameable features and cannot be attacked with a height/weight lie.
GAMEABLE_FEATURES = ['self_height_cm', 'self_weight_kg']


def get_gameable_features(feature_names):
    """Return the features in `feature_names` a patient can lie about.

    Returns an empty list for the measured-input model (Model A).
    """
    return [f for f in feature_names if f in GAMEABLE_FEATURES]


# ============================================================
# ATTACK + MODEL SETTINGS (v2 fixes — see docs/fixes_v2.md)
# Each one is a named switch so the choice is visible and can be
# reverted / used for a sensitivity analysis.
# ============================================================

# FIX 7 — Which way may a patient lie?
#   'realistic': only in the direction people actually misreport, taken
#                from the 2015-16 data (lie_budget[...]['mean_direction']):
#                height is OVER-reported (+), weight is UNDER-reported (-).
#   'both'     : either direction (the old behaviour).
LIE_DIRECTION = 'realistic'

# FIX 6 — How big is each lie? The SAME rule for both attackers, so the
#   comparison measures the information in the explanation, not a
#   difference in lie size.
#   'uniform': every lie (explanation-guided AND each blind attempt) has a
#              size drawn uniformly from (0, max_lie]. The explanation only
#              tells the attacker WHICH feature and WHICH direction.
#   'max'    : every lie uses the full max_lie (old explanation behaviour,
#              now also applied to blind attempts).
LIE_MAGNITUDE = 'uniform'

# FIX 10 — class imbalance (~13% diabetic). 'balanced' re-weights classes
#   so the models do not simply predict "no diabetes" for everyone.
#   None = old behaviour.
CLASS_WEIGHT = 'balanced'

# FIX 9 — LIME shows ALL features (None), so height/weight are never cut
#   off from the explanation. Old value: 5 (of 7).
LIME_N_FEATURES = None

# FIX 11 — SHAP LinearExplainer background: a seeded RANDOM sample of
#   this many training rows (old: the first 100 rows).
SHAP_BACKGROUND_N = 100
SEED_SHAP_BACKGROUND = 2024

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

    'class_weight': CLASS_WEIGHT,               # v2 (fix 10); was None

    # Adversary
    'lie_budget_method': 'percentile',
    'lie_budget_percentile': 95,
    'blind_attempts': N_BLIND_ATTEMPTS,
    'lie_direction': LIE_DIRECTION,             # v2 (fix 7); was 'both'
    'lie_magnitude': LIE_MAGNITUDE,             # v2 (fix 6); was max vs uniform
    'per_patient_seeding': True,                # v2 (fix 8)

    # Explanations
    'shap_method': 'tree',          # TreeSHAP for RF, LinearExplainer for LR
    'shap_background_n': SHAP_BACKGROUND_N,     # v2 (fix 11): random sample
    'lime_n_samples': 500,
    'lime_n_features': LIME_N_FEATURES,         # v2 (fix 9); was 5
    'lime_random_state': 42,

    # Primary metric
    'nstar_method': 'mean_curve',               # v2 (fix 4): as pre-registered

    # Statistics
    'bootstrap_n_iterations': 1000,
    'confidence_level': 0.95,
    'adversarial_seed': SEED_ADVERSARIAL,
}
