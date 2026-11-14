"""
CONFIRMATORY run on NHANES August 2021 - August 2023 (suffix L).

Pre-registered in docs/analysis_plan_v5_confirmatory.md, git tag
v0.2-preregistration. This cycle was not analysed by anyone on the team
before that tag was pushed (only file column names and the public CDC
codebooks were read to define the variable mapping).

Everything is fixed in advance: the v5 code at the tag, FINAL_CONFIG,
the 2015-2016 lie budget, the primary configuration (Model B, RF, SHAP),
the primary metric (N* from the mean curve) and the interpretation
thresholds. Whatever this script prints is reported, unchanged.

Usage: python scripts/run_confirmatory.py
Outputs: results/tables/confirmatory_*.{csv,json,tex},
         results/figures/confirmatory_*.png
"""
import os
import sys
import json
import time
import pickle
import subprocess
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (set_all_seeds, FINAL_CONFIG, DATA_PROCESSED,
                        RESULTS_FIGURES, RESULTS_TABLES, get_gameable_features)
from src.data_loader import (merge_nhanes, clean_nhanes, create_diabetes_target,
                             prepare_feature_sets)
from src.gap_analysis import compute_reporting_gap, define_lie_budget
from src.models import train_all_models
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import (compute_equivalence, compute_confidence_intervals,
                         run_statistical_tests, paired_auc_difference)
from src.visualization import plot_equivalence_curve

CYCLE = '2021_2023'
PRIMARY = 'reported_rf_shap'
NS = FINAL_CONFIG['blind_attempts']
CACHE = os.path.join(DATA_PROCESSED, 'confirmatory_cache_v5')

set_all_seeds()
os.makedirs(CACHE, exist_ok=True)
os.makedirs(RESULTS_TABLES, exist_ok=True)
t0 = time.time()

try:
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
except Exception:
    commit = 'unknown'

print("=" * 70)
print("  CONFIRMATORY RUN — NHANES 2021-2023 (pre-registered, v0.2-preregistration)")
print(f"  Code commit: {commit}")
print("=" * 70)

# --- Data ---
df = merge_nhanes(cycle=CYCLE)
df = clean_nhanes(df)
df = create_diabetes_target(df)
df = compute_reporting_gap(df)
gap_observed = define_lie_budget(df, method=FINAL_CONFIG['lie_budget_method'],
                                 pct=FINAL_CONFIG['lie_budget_percentile'])
X_meas, X_rep, y, fn_meas, fn_rep, df_complete = prepare_feature_sets(df)

# Pre-registered lie budget: NHANES 2015-2016 only
with open(os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl'), 'rb') as f:
    lie_budget = pickle.load(f)['lie_budget']
print(f"\n  Lie budget used (2015-16): height {lie_budget['height']['max_lie']:.2f} cm, "
      f"weight {lie_budget['weight']['max_lie']:.2f} kg")

# --- Models (same locked hyperparameters, 80/20 split of this cycle) ---
models, split = train_all_models(X_meas, X_rep, y, fn_meas, fn_rep)

all_results = {}
for input_type, X, fn in [('reported', X_rep, fn_rep), ('measured', X_meas, fn_meas)]:
    gameable = get_gameable_features(fn)
    X_train, X_test = X[split['idx_train']], X[split['idx_test']]
    for model_type in FINAL_CONFIG['model_types']:
        model = models[f'{input_type}_{model_type}']
        expl = ExplanationGenerator(model, X_train, fn)
        sim = AdversarialSimulator(model, expl, lie_budget, gameable, fn)
        for method in ['shap', 'lime']:
            key = f'{input_type}_{model_type}_{method}'
            cache_path = os.path.join(CACHE, f'{key}.pkl')
            if os.path.exists(cache_path):
                with open(cache_path, 'rb') as f:
                    all_results[key] = pickle.load(f)
                print(f"  [SKIP] {key} (cached)")
                continue
            print(f"\n  >> {key}  (N = {len(X_test)})")
            res = sim.run_comparison(X_test, blind_attempts_list=NS,
                                     explanation_method=method)
            eq, s = compute_equivalence(res, NS)
            ci = compute_confidence_intervals(
                eq, n_bootstrap=FINAL_CONFIG['bootstrap_n_iterations'],
                confidence=FINAL_CONFIG['confidence_level'],
                results=res, blind_attempts_list=NS)
            all_results[key] = {'results': res, 'equivalences': eq,
                                'summary': s, 'confidence_interval': ci}
            with open(cache_path, 'wb') as f:
                pickle.dump(all_results[key], f)
            plot_equivalence_curve(
                res, NS, title=f"Confirmatory 2021-23: {key.upper().replace('_', ' ')}",
                save_path=os.path.join(RESULTS_FIGURES, f'confirmatory_equiv_{key}.png'))

stats = run_statistical_tests(all_results)

rows = []
for key, d in all_results.items():
    s, ci, st, res = d['summary'], d['confidence_interval'], stats[key], d['results']
    raw = np.array([r.get('explanation_raw_reduction', r['explanation_reduction']) for r in res])
    inp, mt, meth = key.split('_')
    rows.append({
        'Config': key, 'Input': inp, 'Model': mt.upper(), 'Explanation': meth.upper(),
        'N_patients': s['n_patients'],
        'Nstar': round(s['nstar'], 2),
        'Nstar_CI_low': round(ci['nstar_ci'][0], 2),
        'Nstar_CI_high': round(ci['nstar_ci'][1], 2),
        'Mean_exp_reduction': round(s['mean_explanation_reduction'], 4),
        'Mean_blind_1': round(s['mean_blind_curve'][1], 4),
        'Mean_blind_10': round(s['mean_blind_curve'][10], 4),
        'Pct_lie_undone': round(float((raw < 0).mean() * 100), 1),
        'PerPatient_Mean_Nstar': round(s['mean_equivalence'], 2),
        'Pct_useless': round(s['pct_explanation_useless'], 1),
        'Pct_worth_10p': round(s['pct_explanation_worth_10plus'], 1),
        'Wilcoxon_p': round(st['wilcoxon_pval'], 6),
        'Wilcoxon_p_Holm': round(st['wilcoxon_pval_holm'], 6),
        'Cohens_d': round(st['cohens_d'], 3),
        'Significant': st['significant'],
    })
table = pd.DataFrame(rows)
table.to_csv(os.path.join(RESULTS_TABLES, 'confirmatory_results.csv'), index=False)

# Model A vs B accuracy on the same test patients
auc_rows = []
te = split['idx_test']
for mt in FINAL_CONFIG['model_types']:
    d = paired_auc_difference(split['y_test'],
                              models[f'measured_{mt}'].predict_proba(X_meas[te]),
                              models[f'reported_{mt}'].predict_proba(X_rep[te]))
    auc_rows.append({'Model': mt.upper(), 'AUC_A_measured': round(d['auc_a'], 4),
                     'AUC_B_reported': round(d['auc_b'], 4), 'Diff_A_minus_B': round(d['diff'], 4),
                     'CI_low': round(d['ci'][0], 4), 'CI_high': round(d['ci'][1], 4)})
pd.DataFrame(auc_rows).to_csv(os.path.join(RESULTS_TABLES, 'confirmatory_input_source_effect.csv'),
                              index=False)

nstar = all_results[PRIMARY]['summary']['nstar']
verdict = ('LOW THREAT' if nstar < 5 else 'MODERATE THREAT' if nstar <= 20 else 'HIGH THREAT')
summary = {
    'cycle': CYCLE, 'code_commit': commit, 'preregistration_tag': 'v0.2-preregistration',
    'n_complete': int(len(y)), 'n_test': int(len(te)),
    'diabetes_prevalence_pct': round(float(y.mean()) * 100, 2),
    'lie_budget_used_2015_2016': {k: {kk: (float(vv) if not isinstance(vv, str) else vv)
                                      for kk, vv in v.items()} for k, v in lie_budget.items()},
    'reporting_gap_observed_2021_2023': {k: {kk: (float(vv) if not isinstance(vv, str) else vv)
                                             for kk, vv in v.items()} for k, v in gap_observed.items()},
    'model_metrics': {k: m.metrics for k, m in models.items()},
    'primary': {'config': PRIMARY, 'nstar': float(nstar),
                'ci': [float(x) for x in all_results[PRIMARY]['confidence_interval']['nstar_ci']],
                'verdict': verdict},
    'results': rows, 'input_source_effect': auc_rows,
    'runtime_min': round((time.time() - t0) / 60, 1),
}
with open(os.path.join(RESULTS_TABLES, 'confirmatory_results.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, indent=2, default=float)

print("\n" + table.to_string(index=False))
print("\n" + pd.DataFrame(auc_rows).to_string(index=False))
print(f"\n  PRIMARY ({PRIMARY}): N* = {nstar:.2f}  "
      f"95% CI {summary['primary']['ci']}  -> {verdict}")
print(f"  Done in {summary['runtime_min']} min")
