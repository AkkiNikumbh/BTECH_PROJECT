"""
Stage review validation script.
Validates Stages 1, 2, and 3 completeness and correctness.
Run: python scripts/review_stages.py
"""
import sys, os, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
issues = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        issues.append(label)


# ============================================================
print("\n" + "="*60)
print("  STAGE 1: Environment & Configuration")
print("="*60)

# Libraries
try:
    import shap, lime, seaborn, statsmodels, pytest, tqdm
    import sklearn, scipy, pandas, matplotlib
    import importlib.metadata as _meta
    def _ver(pkg):
        try: return _meta.version(pkg)
        except Exception: return "installed"
    check("shap installed",        True, shap.__version__)
    check("lime installed",        True, _ver('lime'))
    check("scikit-learn installed",True, sklearn.__version__)
    check("pandas installed",      True, pandas.__version__)
    check("numpy installed",       True, np.__version__)
    check("scipy installed",       True, scipy.__version__)
    check("seaborn installed",     True, seaborn.__version__)
    check("statsmodels installed", True, statsmodels.__version__)
    check("pytest installed",      True, pytest.__version__)
    check("tqdm installed",        True, tqdm.__version__)
except ImportError as e:
    check(f"Library import", False, str(e))

# Config
from src.config import (
    GLOBAL_SEED, SEED_MODEL_RF, SEED_MODEL_LOGREG,
    SEED_ADVERSARIAL, SEED_TRAIN_TEST_SPLIT, N_BLIND_ATTEMPTS,
    DATA_RAW, DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES, FINAL_CONFIG
)
check("GLOBAL_SEED defined",     GLOBAL_SEED == 42, str(GLOBAL_SEED))
check("SEED_MODEL_RF defined",   SEED_MODEL_RF == 123, str(SEED_MODEL_RF))
check("SEED_MODEL_LR defined",   SEED_MODEL_LOGREG == 42, str(SEED_MODEL_LOGREG))
check("SEED_ADVERSARIAL defined",SEED_ADVERSARIAL == 456, str(SEED_ADVERSARIAL))
check("SEED_SPLIT defined",      SEED_TRAIN_TEST_SPLIT == 789, str(SEED_TRAIN_TEST_SPLIT))
check("N_BLIND_ATTEMPTS correct",len(N_BLIND_ATTEMPTS) == 10, str(N_BLIND_ATTEMPTS))
check("FINAL_CONFIG present",    len(FINAL_CONFIG) > 0)
check("Holdout cycle NOT touched",FINAL_CONFIG['holdout_cycle'] == '2017_2018')

# Source files exist
src_files = ['config.py','data_loader.py','gap_analysis.py',
             'models.py','explanations.py','adversarial.py',
             'metrics.py','visualization.py','__init__.py']
for f in src_files:
    check(f"src/{f} exists", os.path.exists(f"src/{f}"))

# Git
check(".gitignore exists",   os.path.exists('.gitignore'))
check("README.md exists",    os.path.exists('README.md'))
check("requirements.txt exists", os.path.exists('requirements.txt'))
check(".git exists (repo init)", os.path.exists('.git'))

# Folder structure
for folder in ['data/raw','data/processed','results/figures',
               'results/tables','scripts','tests','docs','notebooks']:
    check(f"{folder}/ exists", os.path.exists(folder))

# Pytest
check("tests/test_pipeline.py exists", os.path.exists('tests/test_pipeline.py'))


# ============================================================
print("\n" + "="*60)
print("  STAGE 2: NHANES Data Download & Column Mapping")
print("="*60)

cycles = {'2015_2016': 'I', '2017_2018': 'J'}
components = ['DEMO', 'BMX', 'WHQ', 'GHB', 'BPX', 'SMQ', 'PAQ']

for cycle, sfx in cycles.items():
    for comp in components:
        fname = f"data/raw/nhanes_{cycle}/{comp}_{sfx}.XPT"
        exists = os.path.exists(fname)
        if exists:
            sz = os.path.getsize(fname)
            valid = sz > 50_000
            check(f"{comp}_{sfx}.XPT valid", valid, f"{sz//1024} KB")
        else:
            check(f"{comp}_{sfx}.XPT exists", False)

check("docs/column_mapping.md exists", os.path.exists('docs/column_mapping.md'))
check("scripts/explore_columns.py exists", os.path.exists('scripts/explore_columns.py'))

# Verify column mapping content
if os.path.exists('docs/column_mapping.md'):
    content = open('docs/column_mapping.md', encoding='utf-8').read()
    check("Column map has WHD010 (self-height)", 'WHD010' in content)
    check("Column map has WHD020 (self-weight)", 'WHD020' in content)
    check("Column map has BMXHT (measured)",     'BMXHT' in content)
    check("Column map has LBXGH (HbA1c target)", 'LBXGH' in content)
    check("Column map has gameable label YES",   'YES' in content)
    check("Column map has unit conversion",      '2.54' in content)
    check("Column map has ADA threshold",        '6.5' in content)

# Test-download leftover deleted
leftover = 'data/raw/nhanes_2015_2016/DEMO_I_test.XPT'
if os.path.exists(leftover):
    print(f"  {WARN} Leftover test file found: {leftover} — deleting")
    os.remove(leftover)
    print(f"       Deleted.")


# ============================================================
print("\n" + "="*60)
print("  STAGE 3: Data Preparation, Gap Analysis & Feature Sets")
print("="*60)

pkl_path = 'data/processed/nhanes_2015_2016_prepared.pkl'
check("Processed pickle exists", os.path.exists(pkl_path))

if os.path.exists(pkl_path):
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)

    X_m  = d['X_measured']
    X_r  = d['X_reported']
    y    = d['y']
    fn_m = d['feature_names_measured']
    fn_r = d['feature_names_reported']
    lb   = d['lie_budget']

    # Sample size
    check("Patients >= 4000",    len(y) >= 4000, str(len(y)))
    check("Diabetes 10-20%",     0.10 <= y.mean() <= 0.20, f"{y.mean()*100:.1f}%")

    # Shape checks
    check("X_measured has 7 features",  X_m.shape[1] == 7, str(X_m.shape))
    check("X_reported has 7 features",  X_r.shape[1] == 7, str(X_r.shape))
    check("Same number of patients",    X_m.shape[0] == X_r.shape[0] == len(y))

    # NaN checks (should be zero — we already dropped them)
    check("No NaN in X_measured", np.isnan(X_m).sum() == 0, f"{np.isnan(X_m).sum()} NaNs")
    check("No NaN in X_reported", np.isnan(X_r).sum() == 0, f"{np.isnan(X_r).sum()} NaNs")
    check("No NaN in y",          np.isnan(y).sum() == 0,   f"{np.isnan(y).sum()} NaNs")

    # Feature name checks
    check("measured_height_cm in fn_m", 'measured_height_cm' in fn_m)
    check("measured_weight_kg in fn_m", 'measured_weight_kg' in fn_m)
    check("self_height_cm in fn_r",     'self_height_cm' in fn_r)
    check("self_weight_kg in fn_r",     'self_weight_kg' in fn_r)

    # Verify common features are IDENTICAL across models
    n_common = 5  # RIDAGEYR, RIAGENDR, BPXSY1, SMQ020, PAQ650
    check("Common features identical",
          np.allclose(X_m[:, :n_common], X_r[:, :n_common]),
          "first 5 cols")

    # Verify height/weight ARE different
    h_idx_m = fn_m.index('measured_height_cm')
    h_idx_r = fn_r.index('self_height_cm')
    check("Heights differ (self vs measured)",
          not np.allclose(X_m[:, h_idx_m], X_r[:, h_idx_r]))

    # Lie budget sanity
    check("Lie budget has height key", 'height' in lb)
    check("Lie budget has weight key", 'weight' in lb)
    check("Height budget 5-15 cm",  5 <= lb['height']['max_lie'] <= 15,
          f"{lb['height']['max_lie']:.2f} cm")
    check("Weight budget 5-20 kg",  5 <= lb['weight']['max_lie'] <= 20,
          f"{lb['weight']['max_lie']:.2f} kg")
    check("Height direction saved", 'mean_direction' in lb['height'])
    check("Weight direction saved", 'mean_direction' in lb['weight'])
    check("Height — people overestimate",
          lb['height']['mean_direction'] == 'positive',
          lb['height']['mean_direction'])
    check("Weight — people underestimate",
          lb['weight']['mean_direction'] == 'negative',
          lb['weight']['mean_direction'])

    # df_complete saved
    check("df_complete saved in pickle", 'df_complete' in d)
    df_c = d['df_complete']
    check("df_complete has gap columns",
          'height_gap_cm' in df_c.columns and 'weight_gap_kg' in df_c.columns)

# Figures
check("reporting_gaps.png saved", os.path.exists('results/figures/reporting_gaps.png'))
check("scripts/prepare_data.py exists", os.path.exists('scripts/prepare_data.py'))


# ============================================================
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
if issues:
    print(f"\n  {FAIL} {len(issues)} issue(s) found:")
    for i in issues:
        print(f"    - {i}")
else:
    print(f"\n  {PASS} ALL checks passed. Stages 1, 2, 3 are fully complete.")
print()
