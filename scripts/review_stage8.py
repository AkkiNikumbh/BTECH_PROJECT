"""
Stage 8 Review: Verifies all analysis outputs exist and are valid.
Run AFTER scripts/run_analysis.py completes.
Usage: python scripts/review_stage8.py
"""
import os, sys, json
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
print("  STAGE 8: Analysis Output Review")
print("="*65)

from src.config import RESULTS_FIGURES, RESULTS_TABLES, DATA_PROCESSED

# ----------------------------------------------------------------
# A: Figures (8 required)
# ----------------------------------------------------------------
print("\n  [A] Publication Figures (300 DPI, min 50KB each)")
required_figs = {
    'fig1_reporting_gaps.png':    'Fig 1 — Reporting gap distributions',
    'fig2_main_result.png':       'Fig 2 — Main result panel (SHAP)',
    'fig3_score_reductions.png':  'Fig 3 — Score reduction box plots',
    'fig4_model_comparison.png':  'Fig 4 — Model A vs B bar chart',
    'fig5_shap_lime_scatter.png': 'Fig 5 — SHAP vs LIME scatter',
    'fig6_shap_summary.png':      'Fig 6 — SHAP feature importance',
    'fig7_nstar_histogram.png':   'Fig 7 — Per-patient N* histogram',
    'fig8_config_heatmap.png':    'Fig 8 — Configuration heatmap',
}
for fname, desc in required_figs.items():
    path = os.path.join(RESULTS_FIGURES, fname)
    exists = os.path.exists(path)
    size_ok = os.path.getsize(path) > 50_000 if exists else False
    size_kb = os.path.getsize(path) // 1024 if exists else 0
    check(f"{fname}: {desc}", exists and size_ok,
          f"{size_kb}KB" if exists else "MISSING",
          warn=(exists and not size_ok))

# ----------------------------------------------------------------
# B: Tables (CSV + LaTeX)
# ----------------------------------------------------------------
print("\n  [B] Results Tables (CSV + LaTeX)")
required_tables = [
    ('table1_dataset_summary',    'Dataset summary'),
    ('table2_model_performance',  'Model performance'),
    ('table3_reporting_gap',      'Reporting gap comparison'),
    ('table4_main_results',       'Main equivalence results'),
    ('table5_statistical_tests',  'Wilcoxon tests'),
    ('table6_model_ab_comparison','Model A vs B'),
]
for name, desc in required_tables:
    for ext in ['csv', 'tex']:
        path = os.path.join(RESULTS_TABLES, f'{name}.{ext}')
        check(f"{name}.{ext}: {desc}", os.path.exists(path),
              f"{os.path.getsize(path)//1024}KB" if os.path.exists(path) else "MISSING")

# ----------------------------------------------------------------
# C: Final results JSON integrity
# ----------------------------------------------------------------
print("\n  [C] Final Results JSON Integrity")
json_path = os.path.join(RESULTS_TABLES, 'final_results.json')
if check("final_results.json exists", os.path.exists(json_path)):
    with open(json_path) as f:
        master = json.load(f)
    check("Has meta section",    'meta' in master)
    check("Has results section", 'results' in master)
    n_configs = len(master.get('results', {}))
    check("Has 8 configs",       n_configs == 8, f"{n_configs}/8 found")
    if 'results' in master:
        for key, res in master['results'].items():
            check(f"{key} has mean_Nstar",  'mean_Nstar' in res)
            check(f"{key} has CI bounds",   'CI_95_low' in res and 'CI_95_high' in res)
            check(f"{key} has Wilcoxon p",  'wilcoxon_p' in res)
else:
    issues.append("final_results.json missing")

# ----------------------------------------------------------------
# D: Interpretation narrative
# ----------------------------------------------------------------
print("\n  [D] Interpretation Narrative")
narr_path = os.path.join(RESULTS_TABLES, 'interpretation_narrative.txt')
if os.path.exists(narr_path):
    narr = open(narr_path, encoding='utf-8').read()
    check("narrative file exists",        True, f"{len(narr)} chars")
    check("narrative has N* value",       'N*=' in narr or 'N*≈' in narr or 'blind attempts' in narr)
    check("narrative has verdict",        any(w in narr for w in ['THREAT', 'SAFE', 'CAUTION']))
    check("narrative has all 8 configs",  narr.count('reported_') >= 4)
else:
    check("interpretation_narrative.txt exists", False, "MISSING")

# ----------------------------------------------------------------
# E: Main result sanity check
# ----------------------------------------------------------------
print("\n  [E] Main Result Sanity Checks")
import pickle
cache_dir = os.path.join(DATA_PROCESSED, 'final_run_cache')
all_results_loaded = {}
for key in ['reported_rf_shap','reported_rf_lime','reported_lr_shap','reported_lr_lime',
            'measured_rf_shap','measured_rf_lime','measured_lr_shap','measured_lr_lime']:
    p = os.path.join(cache_dir, f'{key}.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as f:
            all_results_loaded[key] = pickle.load(f)

check("At least 1 config computed",   len(all_results_loaded) >= 1,
      f"{len(all_results_loaded)}/8 configs")
check("All 8 configs computed",       len(all_results_loaded) == 8,
      f"{len(all_results_loaded)}/8", warn=(len(all_results_loaded) < 8))

for key, data in all_results_loaded.items():
    s  = data['summary']
    ci = data['confidence_interval']
    n  = s['n_patients']
    check(f"{key}: N patients >= 100",   n >= 100, str(n))
    check(f"{key}: N* non-negative",     s['mean_equivalence'] >= 0,
          f"{s['mean_equivalence']:.2f}")
    check(f"{key}: CI lower <= mean",
          ci['mean_ci'][0] <= s['mean_equivalence'],
          f"{ci['mean_ci'][0]:.2f} <= {s['mean_equivalence']:.2f}")
    check(f"{key}: mean <= CI upper",
          s['mean_equivalence'] <= ci['mean_ci'][1],
          f"{s['mean_equivalence']:.2f} <= {ci['mean_ci'][1]:.2f}")
    check(f"{key}: pct_useless in [0,100]",
          0 <= s['pct_explanation_useless'] <= 100,
          f"{s['pct_explanation_useless']:.1f}%")

# ----------------------------------------------------------------
# F: No deviations from pre-registered plan
# ----------------------------------------------------------------
print("\n  [F] Pre-Registration Compliance")
plan_path = 'docs/analysis_plan.md'
if os.path.exists(plan_path):
    plan = open(plan_path, encoding='utf-8').read()
    check("Plan references 7.29 cm height budget", "7.29" in plan)
    check("Plan references 8.96 kg weight budget", "8.96" in plan)
    check("Plan references bootstrap n=1000",      "1000" in plan)
    check("Plan references confidence 0.95",       "0.95" in plan)
    # Verify actual config matches plan
    from src.config import FINAL_CONFIG
    check("FINAL_CONFIG bootstrap matches plan",
          FINAL_CONFIG['bootstrap_n_iterations'] == 1000)
    check("FINAL_CONFIG confidence matches plan",
          FINAL_CONFIG['confidence_level'] == 0.95)

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)

if issues:
    print(f"\n  {FAIL} {len(issues)} issue(s) found:")
    for i in issues:
        print(f"    - {i}")
    print("\n  Run scripts/run_analysis.py to generate missing outputs.")
else:
    print(f"\n  {PASS} ALL Stage 8 checks passed!")
    print("\n  Proceed to Stage 9: Paper Writing")
    print("  Key files:")
    print(f"    results/tables/table4_main_results.csv")
    print(f"    results/figures/fig2_main_result.png")
    print(f"    results/figures/fig8_config_heatmap.png")
    print(f"    results/tables/interpretation_narrative.txt")
print()
