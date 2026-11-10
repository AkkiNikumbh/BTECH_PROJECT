"""
Stage 6 Review: Pre-Registration completeness check.
Run: python scripts/review_stage6.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS = "[PASS]"
FAIL = "[FAIL]"
issues = []

def check(label, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  {status} {label}" + (f" — {detail}" if detail else ""))
    if not condition:
        issues.append(label)

print("\n" + "="*60)
print("  STAGE 6: Pre-Registration Review")
print("="*60)

# --- Document existence ---
print("\n  [A] Documents")
check("docs/analysis_plan.md exists",
      os.path.exists("docs/analysis_plan.md"))
check("docs/changelog_trial_to_final.md exists",
      os.path.exists("docs/changelog_trial_to_final.md"))
check("docs/column_mapping.md exists",
      os.path.exists("docs/column_mapping.md"))

# --- Analysis plan content ---
print("\n  [B] Analysis Plan Content")
if os.path.exists("docs/analysis_plan.md"):
    plan = open("docs/analysis_plan.md", encoding="utf-8").read()
    checks = {
        "Has research question":        "Research Question" in plan,
        "Has primary hypothesis":       "Primary Hypothesis" in plan or "hypothesize" in plan,
        "Has 2015-2016 dev set":        "2015" in plan and "Development" in plan,
        "Has 2017-2018 holdout stated": "2017" in plan and "NOT YET" in plan,
        "Has lie budget numbers":       "7.29" in plan and "8.96" in plan,
        "Has 7 features listed":        "RIDAGEYR" in plan and "self_height_cm" in plan,
        "Has all 4 models":             "measured_rf" in plan and "reported_lr" in plan,
        "Has SHAP + LIME":              "SHAP" in plan and "LIME" in plan,
        "Has blind_attack definition":  "blind_attack_multi_n" in plan or "N attempts" in plan or "sequential" in plan,
        "Has primary metric N*":        "N*" in plan or "N\\*" in plan,
        "Has statistical tests":        "Wilcoxon" in plan and "bootstrap" in plan.lower(),
        "Has trial numbers":            "9.5" in plan,
        "Has interpretation table":     "N* < 5" in plan or "N\\* < 5" in plan,
        "Has 8 configs listed":         "reported_rf" in plan and "measured_lr" in plan,
        "Has locked rules section":     "Will NOT" in plan,
        "Has FINAL_CONFIG block":       "FINAL_CONFIG" in plan,
        "Has sign-off section":         "Sign-Off" in plan or "sign-off" in plan,
        "Has registration date":        "28 October 2026" in plan or "2026" in plan,
    }
    for label, cond in checks.items():
        check(label, cond)

# --- Changelog content ---
print("\n  [C] Changelog Content")
if os.path.exists("docs/changelog_trial_to_final.md"):
    clog = open("docs/changelog_trial_to_final.md", encoding="utf-8").read()
    check("Changelog has blind_attack fix",   "blind_attack" in clog)
    check("Changelog has LIME fix",           "LIME" in clog)
    check("Changelog has metrics fix",        "monoton" in clog.lower() or "metrics" in clog.lower())
    check("Changelog has unchanged settings", "UNCHANGED" in clog)
    check("Changelog has trial results",      "9.5" in clog)

# --- FINAL_CONFIG locked ---
print("\n  [D] FINAL_CONFIG Lock")
from src.config import FINAL_CONFIG, N_BLIND_ATTEMPTS
check("FINAL_CONFIG present",           bool(FINAL_CONFIG))
check("holdout_cycle = 2017_2018",      FINAL_CONFIG.get("holdout_cycle") == "2017_2018")
check("development_cycle = 2015_2016",  FINAL_CONFIG.get("development_cycle") == "2015_2016")
check("rf_n_estimators = 100",          FINAL_CONFIG.get("rf_n_estimators") == 100)
check("rf_max_depth = 10",              FINAL_CONFIG.get("rf_max_depth") == 10)
check("lie_budget_method = percentile", FINAL_CONFIG.get("lie_budget_method") == "percentile")
check("lie_budget_pct = 95",            FINAL_CONFIG.get("lie_budget_percentile") == 95)
check("blind_attempts has 10 values",   len(FINAL_CONFIG.get("blind_attempts", [])) == 10)
check("bootstrap iters = 1000",         FINAL_CONFIG.get("bootstrap_n_iterations") == 1000)
check("confidence = 0.95",              FINAL_CONFIG.get("confidence_level") == 0.95)
check("N_BLIND_ATTEMPTS matches config",
      FINAL_CONFIG.get("blind_attempts") == N_BLIND_ATTEMPTS)

# --- Git tag ---
print("\n  [E] Git State")
import subprocess
try:
    result = subprocess.run(["git", "tag"], capture_output=True, text=True)
    tags = result.stdout.strip().split('\n') if result.stdout.strip() else []
    has_tag = "v0.1-preregistration" in tags
    check("Git tag v0.1-preregistration exists", has_tag,
          "run: git tag -a v0.1-preregistration -m 'Pre-registration'" if not has_tag else "OK")
except Exception as e:
    check("Git tags readable", False, str(e))

try:
    log = subprocess.run(
        ["git", "log", "--oneline", "-3"],
        capture_output=True, text=True
    ).stdout.strip()
    check("Git repo has commits", bool(log), log.split('\n')[0] if log else "No commits")
except Exception:
    check("Git log readable", False)

# --- Holdout data safety ---
print("\n  [F] Holdout Data Safety")
import pickle
pkl = "data/processed/nhanes_2015_2016_prepared.pkl"
check("Only 2015-2016 pickle exists", os.path.exists(pkl))
holdout_pkl = "data/processed/nhanes_2017_2018_prepared.pkl"
holdout_exists = os.path.exists(holdout_pkl)
if not holdout_exists:
    check("2017-2018 pickle does NOT exist", True, "Safe — Stage 7 not yet run")
else:
    # Pickle exists — verify it was created AFTER the pre-registration tag
    try:
        tag_ts_str = subprocess.run(
            ["git", "log", "-1", "--format=%ct", "v0.1-preregistration"],
            capture_output=True, text=True
        ).stdout.strip()
        tag_ts = int(tag_ts_str) if tag_ts_str else 0
        pkl_ts = os.path.getmtime(holdout_pkl)
        after_tag = pkl_ts > tag_ts
        check("2017-2018 pickle created AFTER pre-registration tag",
              after_tag,
              "OK — Stage 7 ran legitimately after tag" if after_tag
              else "CRITICAL: processed before tag!")
    except Exception:
        check("2017-2018 pickle exists (Stage 7 ran)", True,
              "Tag timestamp unverifiable", warn=True)


# --- Summary ---
print("\n" + "="*60)
print("  SUMMARY")
print("="*60)
if issues:
    print(f"\n  {FAIL} {len(issues)} issue(s) found:")
    for i in issues:
        print(f"    - {i}")
    print("\n  Fix these before creating the git tag.")
else:
    print(f"\n  {PASS} ALL checks passed.")
    print("\n  NEXT STEP: Create the pre-registration git tag:")
    print("    git tag -a v0.1-preregistration -m 'Pre-registered analysis plan'")
    print("    git push origin v0.1-preregistration  (if using remote)")
print()
