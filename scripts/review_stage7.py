"""
Stage 7 Review: Pre-flight check before running the final experiment.
Verifies every pre-condition is met before touching held-out data.
Run: python scripts/review_stage7.py
"""
import os, sys, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
issues = []

def check(label, condition, detail="", warn=False):
    if condition:
        print(f"  {PASS} {label}" + (f" — {detail}" if detail else ""))
    elif warn:
        print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        issues.append(label)

print("\n" + "="*65)
print("  STAGE 7: Final Experiment Pre-Flight Check")
print("="*65)

# ----------------------------------------------------------------
# A: Pre-Registration gate
# ----------------------------------------------------------------
print("\n  [A] Pre-Registration Gate")

result = subprocess.run(["git", "tag"], capture_output=True, text=True)
tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
check("Git tag v0.1-preregistration exists", "v0.1-preregistration" in tags)

check("docs/analysis_plan.md exists",            os.path.exists("docs/analysis_plan.md"))
check("docs/changelog_trial_to_final.md exists", os.path.exists("docs/changelog_trial_to_final.md"))

if os.path.exists("docs/analysis_plan.md"):
    plan = open("docs/analysis_plan.md", encoding="utf-8").read()
    check("Plan has pre-registered hypothesis",  "Primary Hypothesis" in plan)
    check("Plan has interpretation thresholds",  "N* < 5" in plan or "N\\* < 5" in plan)
    check("Plan has locked rules",               "Will NOT" in plan)

# ----------------------------------------------------------------
# B: Holdout data safety
# ----------------------------------------------------------------
print("\n  [B] Holdout Data Safety")

check("2017-2018 XPT files downloaded",
      all(os.path.exists(f"data/raw/nhanes_2017_2018/{f}.XPT")
          for f in ["DEMO_J","BMX_J","WHQ_J","GHB_J","BPX_J","SMQ_J","PAQ_J"]))

check("2017-2018 pickle NOT yet processed",
      not os.path.exists("data/processed/nhanes_2017_2018_prepared.pkl"),
      "Safe to proceed" if not os.path.exists("data/processed/nhanes_2017_2018_prepared.pkl")
      else "Already exists — note this in paper")

check("2015-2016 pickle exists (development data)",
      os.path.exists("data/processed/nhanes_2015_2016_prepared.pkl"))

# ----------------------------------------------------------------
# C: Scripts ready
# ----------------------------------------------------------------
print("\n  [C] Scripts & Source Modules")
scripts = ["scripts/run_final.py", "scripts/review_stage7.py"]
modules = ["src/config.py","src/data_loader.py","src/gap_analysis.py",
           "src/models.py","src/explanations.py","src/adversarial.py",
           "src/metrics.py","src/visualization.py"]
for f in scripts + modules:
    check(f"{f} exists", os.path.exists(f))

# ----------------------------------------------------------------
# D: FINAL_CONFIG integrity
# ----------------------------------------------------------------
print("\n  [D] FINAL_CONFIG Integrity")
from src.config import FINAL_CONFIG, N_BLIND_ATTEMPTS

required_keys = [
    'development_cycle', 'holdout_cycle', 'model_types',
    'rf_n_estimators', 'rf_max_depth', 'rf_random_state',
    'lr_random_state', 'test_size', 'split_random_state',
    'lie_budget_method', 'lie_budget_percentile', 'blind_attempts',
    'lime_n_samples', 'lime_n_features', 'lime_random_state',
    'bootstrap_n_iterations', 'confidence_level', 'adversarial_seed',
]
for k in required_keys:
    check(f"FINAL_CONFIG['{k}'] present", k in FINAL_CONFIG,
          str(FINAL_CONFIG.get(k, "MISSING")))

check("holdout cycle is 2017_2018",  FINAL_CONFIG['holdout_cycle'] == '2017_2018')
check("blind_attempts has 10 values", len(FINAL_CONFIG['blind_attempts']) == 10)
check("N_BLIND_ATTEMPTS matches config",
      FINAL_CONFIG['blind_attempts'] == N_BLIND_ATTEMPTS)

# ----------------------------------------------------------------
# E: Quick import smoke-test
# ----------------------------------------------------------------
print("\n  [E] Import & Smoke Test")
try:
    from src.data_loader import merge_nhanes, clean_nhanes, create_diabetes_target, prepare_feature_sets
    check("data_loader imports OK", True)
except Exception as e:
    check("data_loader imports OK", False, str(e))

try:
    from src.adversarial import AdversarialSimulator
    check("adversarial imports OK", True)
    # Verify blind_attack_multi_n exists (critical fix)
    check("blind_attack_multi_n method present",
          hasattr(AdversarialSimulator, 'blind_attack_multi_n'))
except Exception as e:
    check("adversarial imports OK", False, str(e))

try:
    from src.metrics import compute_equivalence, compute_confidence_intervals
    check("metrics imports OK", True)
except Exception as e:
    check("metrics imports OK", False, str(e))

try:
    from src.visualization import plot_equivalence_curve, plot_model_comparison
    import matplotlib
    check("visualization imports OK", True, f"backend={matplotlib.get_backend()}")
    check("matplotlib using Agg backend",
          matplotlib.get_backend().lower() == 'agg',
          matplotlib.get_backend())
except Exception as e:
    check("visualization imports OK", False, str(e))

# ----------------------------------------------------------------
# F: Result output directories
# ----------------------------------------------------------------
print("\n  [F] Output Directories")
from src.config import RESULTS_FIGURES, RESULTS_TABLES, DATA_PROCESSED
dirs = [RESULTS_FIGURES, RESULTS_TABLES,
        os.path.join(DATA_PROCESSED, 'final_run_cache')]
for d in dirs:
    os.makedirs(d, exist_ok=True)
    check(f"{os.path.relpath(d)} ready", os.path.exists(d))

# ----------------------------------------------------------------
# G: Git state
# ----------------------------------------------------------------
print("\n  [G] Git State")
log = subprocess.run(["git","log","--oneline","-1"], capture_output=True, text=True)
check("Repo has commits", bool(log.stdout.strip()), log.stdout.strip().split('\n')[0])

status = subprocess.run(["git","status","--short"], capture_output=True, text=True)
uncommitted = status.stdout.strip()
check("No uncommitted changes",
      not bool(uncommitted),
      "OK" if not uncommitted else f"uncommitted: {uncommitted[:80]}",
      warn=bool(uncommitted))

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)

if issues:
    print(f"\n  {FAIL} {len(issues)} blocking issue(s). Fix before running final experiment:")
    for i in issues:
        print(f"    - {i}")
else:
    print(f"\n  {PASS} ALL pre-flight checks passed!")
    print("\n  *** READY TO RUN FINAL EXPERIMENT ***")
    print("\n  Command (run overnight):")
    print("    python scripts/run_final.py")
    print("\n  Estimated runtime: 6-11 hours")
    print("  Results will save to: results/tables/final_results.csv")
    print("                        results/tables/final_results.json")
print()
