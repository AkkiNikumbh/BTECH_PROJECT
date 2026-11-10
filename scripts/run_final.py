"""
scripts/run_final.py
====================
THE FINAL EXPERIMENT — GP-109 BTP Project
NHANES 2017-2018 (held-out data, first time touched)

Run EXACTLY as pre-registered in docs/analysis_plan.md
Git tag: v0.1-preregistration

Usage:
    python scripts/run_final.py

This script:
  1. Processes 2017-2018 NHANES data for the first time
  2. Compares reporting gap with 2015-2016
  3. Trains all 4 models (locked hyperparameters from FINAL_CONFIG)
  4. Runs all 8 adversarial configurations on the FULL test set
  5. Computes equivalence metrics + bootstrap CIs
  6. Runs all pre-registered statistical tests
  7. Saves ALL results, figures, and tables
  8. Produces a final_results.json summary

Estimated runtime: 6-11 hours (run overnight)
Supports RESUMPTION: completed configs are skipped automatically.
"""
import os, sys, pickle, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    set_all_seeds, DATA_RAW, DATA_PROCESSED,
    RESULTS_FIGURES, RESULTS_TABLES, FINAL_CONFIG, N_BLIND_ATTEMPTS
)
from src.data_loader import merge_nhanes, clean_nhanes, create_diabetes_target, prepare_feature_sets
from src.gap_analysis import compute_reporting_gap, define_lie_budget, plot_reporting_gaps
from src.models import train_all_models
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence, compute_confidence_intervals, run_statistical_tests
from src.visualization import plot_equivalence_curve, plot_score_reductions, plot_model_comparison

# ============================================================
# SETUP
# ============================================================
set_all_seeds()
os.makedirs(RESULTS_FIGURES, exist_ok=True)
os.makedirs(RESULTS_TABLES, exist_ok=True)

RESUME_DIR = os.path.join(DATA_PROCESSED, 'final_run_cache')
os.makedirs(RESUME_DIR, exist_ok=True)

BLIND_ATTEMPTS = FINAL_CONFIG['blind_attempts']

print("=" * 70)
print("  FINAL EXPERIMENT — NHANES 2017-2018 (HELD-OUT DATA)")
print("  Pre-registered plan: docs/analysis_plan.md")
print("  Git tag: v0.1-preregistration")
print("=" * 70)
print(f"\n  Blind attempts: {BLIND_ATTEMPTS}")
print(f"  Lie budget method: {FINAL_CONFIG['lie_budget_method']} ({FINAL_CONFIG['lie_budget_percentile']}th pct)")
print(f"  Bootstrap iterations: {FINAL_CONFIG['bootstrap_n_iterations']}")

# ============================================================
# SAFETY CHECK: Confirm holdout data has not been pre-analyzed
# ============================================================
holdout_pkl = os.path.join(DATA_PROCESSED, 'nhanes_2017_2018_prepared.pkl')
# Note: it is OK to create this file NOW (first time) — but we check
# the cache doesn't contain pre-registered results from before tag
tag_date_unix = 1761609600  # approx 28 Oct 2026 00:00 UTC
if os.path.exists(holdout_pkl):
    mtime = os.path.getmtime(holdout_pkl)
    if mtime < tag_date_unix:
        print("\n[WARNING] Holdout pickle is OLDER than pre-registration tag!")
        print("          This suggests the held-out data was analyzed before registration.")
        print("          Proceeding, but note this deviation in the paper.")
    else:
        print("\n[OK] Holdout pickle was created AFTER pre-registration. Integrity preserved.")

# ============================================================
# STEP 1: Process 2017-2018 data
# ============================================================
print("\n" + "=" * 70)
print("  [1/8] Processing 2017-2018 NHANES data (FIRST TIME)")
print("=" * 70)

t_start = time.time()

df_2018 = merge_nhanes(cycle='2017_2018')
df_2018 = clean_nhanes(df_2018)
df_2018 = create_diabetes_target(df_2018)
df_2018 = compute_reporting_gap(df_2018)

lie_budget_2018 = define_lie_budget(
    df_2018,
    method=FINAL_CONFIG['lie_budget_method'],
    pct=FINAL_CONFIG['lie_budget_percentile']
)

# Save gap plots for 2017-2018
plot_reporting_gaps(df_2018, save_dir=RESULTS_FIGURES)
# rename to avoid overwriting 2015-2016 gaps
old_path = os.path.join(RESULTS_FIGURES, 'reporting_gaps.png')
new_path = os.path.join(RESULTS_FIGURES, 'reporting_gaps_2017_2018.png')
if os.path.exists(old_path):
    os.replace(old_path, new_path)

# ============================================================
# STEP 2: Compare gaps with 2015-2016
# ============================================================
print("\n" + "=" * 70)
print("  [2/8] Comparing reporting gaps: 2015-2016 vs 2017-2018")
print("=" * 70)

dev_pkl = os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl')
with open(dev_pkl, 'rb') as f:
    dev_data = pickle.load(f)

lie_budget_dev = dev_data['lie_budget']
gap_comparison = {}

for feature in ['height', 'weight']:
    dev_lie = lie_budget_dev[feature]['max_lie']
    new_lie = lie_budget_2018[feature]['max_lie']
    diff = new_lie - dev_lie
    gap_comparison[feature] = {
        'dev_2015_2016': round(dev_lie, 3),
        'holdout_2017_2018': round(new_lie, 3),
        'absolute_difference': round(abs(diff), 3),
        'pct_change': round((diff / dev_lie) * 100, 1)
    }
    unit = 'cm' if feature == 'height' else 'kg'
    print(f"  {feature:6s}: dev={dev_lie:.2f}{unit} | holdout={new_lie:.2f}{unit} "
          f"| diff={diff:+.2f}{unit} ({diff/dev_lie*100:+.1f}%)")

print("\n  NOTE: We use the 2017-2018 lie budget for the final experiment")
print("  (pre-registered: lie budget is computed from the test cycle data)")

# ============================================================
# STEP 3: Prepare feature sets
# ============================================================
print("\n" + "=" * 70)
print("  [3/8] Preparing feature sets")
print("=" * 70)

X_meas, X_rep, y, fn_meas, fn_rep, df_complete = prepare_feature_sets(df_2018)

print(f"\n  Final N = {len(y):,}  |  Diabetes = {y.mean()*100:.1f}% ({int(y.sum())})")

# Save holdout processed data
holdout_save = {
    'X_measured': X_meas, 'X_reported': X_rep,
    'y': y,
    'feature_names_measured': fn_meas,
    'feature_names_reported': fn_rep,
    'lie_budget': lie_budget_2018,
    'df_complete': df_complete,
    'gap_comparison': gap_comparison,
}
with open(holdout_pkl, 'wb') as f:
    pickle.dump(holdout_save, f)
print(f"  Saved: {holdout_pkl}")

# ============================================================
# STEP 4: Train models (locked hyperparameters)
# ============================================================
print("\n" + "=" * 70)
print("  [4/8] Training all 4 models (locked FINAL_CONFIG)")
print("=" * 70)

models, split = train_all_models(X_meas, X_rep, y, fn_meas, fn_rep)

# Save model training metrics
model_metrics = {k: m.metrics for k, m in models.items()}
metrics_path = os.path.join(RESULTS_TABLES, 'final_model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(model_metrics, f, indent=2)
print(f"\n  Model metrics saved: {metrics_path}")

# ============================================================
# STEP 5: Run ALL 8 adversarial simulations (full patient set)
# ============================================================
print("\n" + "=" * 70)
print("  [5/8] Running adversarial simulations — ALL PATIENTS, 8 CONFIGS")
print("  This is the main experiment. No subsets. No shortcuts.")
print("=" * 70)

all_results = {}

for input_type in ['reported', 'measured']:   # reported first (main hypothesis)
    X = X_rep if input_type == 'reported' else X_meas
    fn = fn_rep if input_type == 'reported' else fn_meas
    gameable = [f for f in fn if 'self' in f or 'height' in f or 'weight' in f
                if 'measured' not in f]
    if not gameable:
        # fallback: any height or weight feature
        gameable = [f for f in fn if 'height' in f or 'weight' in f]

    X_train = X[split['idx_train']]
    X_test  = X[split['idx_test']]

    for model_type in FINAL_CONFIG['model_types']:
        model_key = f"{input_type}_{model_type}"
        model = models[model_key]
        explainer = ExplanationGenerator(model, X_train, fn)
        simulator = AdversarialSimulator(
            model, explainer, lie_budget_2018, gameable, fn
        )

        for exp_method in ['shap', 'lime']:
            run_key = f"{model_key}_{exp_method}"

            # RESUMPTION: skip if already computed
            cache_path = os.path.join(RESUME_DIR, f'{run_key}.pkl')
            if os.path.exists(cache_path):
                print(f"\n  [SKIP] {run_key} — already in cache, loading...")
                with open(cache_path, 'rb') as f:
                    all_results[run_key] = pickle.load(f)
                continue

            print(f"\n  >> CONFIG: {run_key}")
            print(f"     N_patients = {len(X_test)} (FULL test set)")
            print(f"     Gameable: {gameable}")
            t_run = time.time()

            results = simulator.run_comparison(
                X_test,
                n_patients=None,           # ALL — as pre-registered
                blind_attempts_list=BLIND_ATTEMPTS,
                explanation_method=exp_method
            )

            equivalences, summary = compute_equivalence(results, BLIND_ATTEMPTS)
            ci = compute_confidence_intervals(
                equivalences,
                n_bootstrap=FINAL_CONFIG['bootstrap_n_iterations'],
                confidence=FINAL_CONFIG['confidence_level']
            )

            run_data = {
                'results': results,
                'equivalences': equivalences,
                'summary': summary,
                'confidence_interval': ci,
                'runtime_s': time.time() - t_run,
            }
            all_results[run_key] = run_data

            # Save immediately after each config (for resumption)
            with open(cache_path, 'wb') as f:
                pickle.dump(run_data, f)

            # Plot equivalence curve for this config
            plot_equivalence_curve(
                results, BLIND_ATTEMPTS,
                title=f"Final: {run_key.upper().replace('_',' ')}",
                save_path=os.path.join(RESULTS_FIGURES, f'final_equiv_{run_key}.png')
            )
            print(f"     Done in {(time.time()-t_run)/60:.1f} min")

elapsed_total = time.time() - t_start
print(f"\n  Total experiment time so far: {elapsed_total/3600:.2f} hours")

# ============================================================
# STEP 6: Statistical tests (pre-registered)
# ============================================================
print("\n" + "=" * 70)
print("  [6/8] Running pre-registered statistical tests")
print("=" * 70)

stat_results = run_statistical_tests(all_results)

# ============================================================
# STEP 7: Build summary table
# ============================================================
print("\n" + "=" * 70)
print("  [7/8] Building results table")
print("=" * 70)

rows = []
for run_key, data in all_results.items():
    s   = data['summary']
    ci  = data['confidence_interval']
    st  = stat_results.get(run_key, {})
    parts = run_key.split('_')
    # run_key = e.g. reported_rf_shap
    exp_method = parts[-1]
    model_type = parts[-2].upper()
    input_type = '_'.join(parts[:-2])

    rows.append({
        'Config':        run_key,
        'Input':         input_type,
        'Model':         model_type,
        'Explanation':   exp_method.upper(),
        'N_patients':    s['n_patients'],
        'Mean_Nstar':    round(s['mean_equivalence'], 2),
        'Median_Nstar':  round(s['median_equivalence'], 2),
        'Std_Nstar':     round(s['std_equivalence'], 2),
        'CI_low':        round(ci['mean_ci'][0], 2),
        'CI_high':       round(ci['mean_ci'][1], 2),
        'Pct_useless':   round(s['pct_explanation_useless'], 1),
        'Pct_worth_10p': round(s['pct_explanation_worth_10plus'], 1),
        'Wilcoxon_p':    round(st.get('wilcoxon_pval', 1.0), 6),
        'Cohens_d':      round(st.get('cohens_d', 0.0), 3),
        'Significant':   st.get('significant', False),
    })

df_results = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_TABLES, 'final_results.csv')
df_results.to_csv(csv_path, index=False)

print("\n" + "=" * 70)
print("  FINAL RESULTS TABLE")
print("=" * 70)
print(df_results.to_string(index=False))

# ============================================================
# STEP 8: Save master results JSON + comparison figures
# ============================================================
print("\n" + "=" * 70)
print("  [8/8] Saving master results and comparison figures")
print("=" * 70)

# Model comparison figure
plot_model_comparison(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'final_model_comparison.png')
)

# Score reduction box plots (reported_rf_shap — main config)
main_key = 'reported_rf_shap'
if main_key in all_results:
    plot_score_reductions(
        all_results[main_key]['results'],
        save_path=os.path.join(RESULTS_FIGURES, 'final_score_reductions.png')
    )

# Master JSON
master = {
    'meta': {
        'experiment': 'GP-109 Final Experiment',
        'date': time.strftime('%Y-%m-%d %H:%M'),
        'git_tag': 'v0.1-preregistration',
        'cycle': '2017_2018',
        'n_patients_total': int(len(y)),
        'diabetes_prevalence_pct': round(float(y.mean()) * 100, 2),
        'lie_budget': {k: {kk: round(float(vv), 3) for kk, vv in v.items()}
                       for k, v in lie_budget_2018.items()},
        'gap_comparison': gap_comparison,
        'total_runtime_hours': round(elapsed_total / 3600, 2),
    },
    'results': {
        k: {
            'mean_Nstar':    round(float(d['summary']['mean_equivalence']), 3),
            'median_Nstar':  round(float(d['summary']['median_equivalence']), 3),
            'std_Nstar':     round(float(d['summary']['std_equivalence']), 3),
            'CI_95_low':     round(float(d['confidence_interval']['mean_ci'][0]), 3),
            'CI_95_high':    round(float(d['confidence_interval']['mean_ci'][1]), 3),
            'pct_useless':   round(float(d['summary']['pct_explanation_useless']), 1),
            'pct_worth_10p': round(float(d['summary']['pct_explanation_worth_10plus']), 1),
            'wilcoxon_p':    round(float(stat_results.get(k, {}).get('wilcoxon_pval', 1.0)), 6),
            'cohens_d':      round(float(stat_results.get(k, {}).get('cohens_d', 0.0)), 3),
            'significant':   bool(stat_results.get(k, {}).get('significant', False)),
        }
        for k, d in all_results.items()
    },
    'interpretation': {
        'threshold_low':  5,
        'threshold_high': 20,
        'conclusion': 'TBD — filled during Stage 8 analysis'
    }
}

json_path = os.path.join(RESULTS_TABLES, 'final_results.json')
with open(json_path, 'w') as f:
    json.dump(master, f, indent=2)

print(f"\n  Results CSV:  {csv_path}")
print(f"  Results JSON: {json_path}")
print(f"  Figures dir:  {RESULTS_FIGURES}")

print("\n" + "=" * 70)
print("  FINAL EXPERIMENT COMPLETE")
print("  Proceed to Stage 8: Analysis & Paper Writing")
print("=" * 70)
