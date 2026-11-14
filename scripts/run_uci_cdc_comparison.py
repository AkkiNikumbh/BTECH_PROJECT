"""
Research plan §6.4 — comparison dataset: CDC Diabetes Health Indicators
(UCI id 891; BRFSS 2015, 253,680 respondents).

Runs the SAME explanation-vs-blind experiment as the NHANES final run, on a
second, much larger self-report survey:

  * features mirror NHANES: Age, Sex, HighBP, Smoker, PhysActivity, BMI
  * BMI is computed from SELF-REPORTED height/weight in BRFSS, so it is the
    only gameable feature; there is no nurse-measured version, so this is a
    "Model B only" comparison
  * the realistic BMI lie comes from the NHANES 2015-2016 gap
    (self-reported BMI minus measured BMI; 95th percentile, mean direction),
    i.e. the same rule as the main budget and no use of 2017-2018 data
  * same models, seeds, lie rules, N values, N* metric and tests as
    FINAL_CONFIG
  * a seeded random sample of N_PATIENTS test respondents is attacked
    (the full 50k test set would take ~1 day; the sample keeps the CI narrow)

Limitation: the BRFSS diabetes label is self-reported (NHANES uses HbA1c).

Usage: python scripts/run_uci_cdc_comparison.py
Outputs: results/tables/table8_uci891_comparison.{csv,tex,json}
"""
import os
import sys
import io
import json
import time
import pickle
import contextlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (set_all_seeds, DATA_PROCESSED, RESULTS_TABLES,
                        FINAL_CONFIG, TEST_SIZE, SEED_TRAIN_TEST_SPLIT,
                        SEED_ADVERSARIAL)
from src.supporting_data import (load_cdc_diabetes, nhanes_bmi_lie_budget,
                                 UCI891_CSV, UCI891_GAMEABLE)
from src.models import DiabetesModel
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import (compute_equivalence, compute_confidence_intervals,
                         run_statistical_tests)

N_PATIENTS = 1000
NS = FINAL_CONFIG['blind_attempts']

if not os.path.exists(UCI891_CSV):
    sys.exit(f"Missing {UCI891_CSV}. Download data.csv from "
             "https://archive.ics.uci.edu/static/public/891/data.csv")

set_all_seeds()
t0 = time.time()
print("=" * 70)
print("  Comparison dataset: CDC Diabetes Health Indicators (UCI 891)")
print("=" * 70)

# --- Lie budget from NHANES 2015-2016 (development) ---
with open(os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl'), 'rb') as f:
    dev = pickle.load(f)
budget = nhanes_bmi_lie_budget(dev['df_complete'], pct=FINAL_CONFIG['lie_budget_percentile'])
b = budget['BMI']
print(f"  BMI lie budget (NHANES 2015-16, n={b['n']}): max {b['max_lie']:.2f} kg/m2, "
      f"mean gap {b['mean_gap']:+.2f} -> people {('over' if b['mean_direction'] == 'positive' else 'under')}-report")

# --- Data ---
X, y, fn = load_cdc_diabetes()
print(f"  Respondents: {len(y)} | diabetes (self-reported): {y.mean()*100:.1f}%")
idx = np.arange(len(y))
idx_tr, idx_te = train_test_split(idx, test_size=TEST_SIZE,
                                  random_state=SEED_TRAIN_TEST_SPLIT, stratify=y)
X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
sample = np.random.RandomState(SEED_ADVERSARIAL).choice(len(X_te), N_PATIENTS, replace=False)
X_attack = X_te[sample]
print(f"  Train {len(y_tr)} | test {len(y_te)} | attacked sample {N_PATIENTS}")

all_results, metrics = {}, {}
for model_type in FINAL_CONFIG['model_types']:
    with contextlib.redirect_stdout(io.StringIO()):
        m = DiabetesModel(model_type, 'reported')
        m.train(X_tr, y_tr, feature_names=fn)
        metrics[model_type] = m.evaluate(X_te, y_te)
    print(f"\n  [{model_type.upper()}] AUC {metrics[model_type]['auc_roc']:.4f} | "
          f"recall {metrics[model_type]['recall']:.3f}")
    # LIME/SHAP background: a seeded 5,000-row sample of the training data
    bg = X_tr[np.random.RandomState(2024).choice(len(X_tr), 5000, replace=False)]
    expl = ExplanationGenerator(m, bg, fn)
    sim = AdversarialSimulator(m, expl, budget, list(UCI891_GAMEABLE), fn)
    for method in ['shap', 'lime']:
        key = f'uci891_{model_type}_{method}'
        t = time.time()
        res = sim.run_comparison(X_attack, blind_attempts_list=NS,
                                 explanation_method=method)
        with contextlib.redirect_stdout(io.StringIO()):
            eq, s = compute_equivalence(res, NS)
            ci = compute_confidence_intervals(
                eq, n_bootstrap=FINAL_CONFIG['bootstrap_n_iterations'],
                confidence=FINAL_CONFIG['confidence_level'],
                results=res, blind_attempts_list=NS)
        all_results[key] = {'results': res, 'summary': s, 'confidence_interval': ci}
        print(f"    {key}: N*={s['nstar']:.2f} [{ci['nstar_ci'][0]:.2f}, {ci['nstar_ci'][1]:.2f}]"
              f" | 1 expl {s['mean_explanation_reduction']:.4f} vs 1 blind "
              f"{s['mean_blind_curve'][1]:.4f}, 10 blind {s['mean_blind_curve'][10]:.4f}"
              f" ({(time.time()-t)/60:.1f} min)")

print()
stats = run_statistical_tests(all_results)

rows = []
for key, d in all_results.items():
    s, ci, st, res = d['summary'], d['confidence_interval'], stats[key], d['results']
    _, mt, meth = key.split('_')
    raw = np.array([r['explanation_raw_reduction'] for r in res])
    rows.append({
        'Model': mt.upper(), 'Expl.': meth.upper(), 'N': s['n_patients'],
        'N*': round(s['nstar'], 2),
        'N* 95% CI': f"[{ci['nstar_ci'][0]:.2f}, {ci['nstar_ci'][1]:.2f}]",
        '1 explanation': round(s['mean_explanation_reduction'], 4),
        '1 blind': round(s['mean_blind_curve'][1], 4),
        '10 blind': round(s['mean_blind_curve'][10], 4),
        'Per-patient mean N*': round(s['mean_equivalence'], 2),
        '% useless': round(s['pct_explanation_useless'], 1),
        '% lie undone': round(float((raw < 0).mean() * 100), 1),
        'Wilcoxon p': round(st['wilcoxon_pval'], 4),
        "Cohen's d": round(st['cohens_d'], 3),
    })
df = pd.DataFrame(rows)
os.makedirs(RESULTS_TABLES, exist_ok=True)
base = os.path.join(RESULTS_TABLES, 'table8_uci891_comparison')
df.to_csv(base + '.csv', index=False, encoding='utf-8')
with open(base + '.tex', 'w', encoding='utf-8') as f:
    f.write(df.to_latex(index=False, escape=True,
                        caption='Comparison dataset (CDC Diabetes Health Indicators, '
                                'UCI 891): one explanation vs.\\ blind attempts, '
                                'self-reported BMI as the only gameable feature',
                        label='tab:uci891'))
with open(base + '.json', 'w', encoding='utf-8') as f:
    json.dump({'lie_budget': budget, 'n_attacked': N_PATIENTS,
               'model_metrics': metrics, 'table': rows,
               'runtime_min': round((time.time() - t0) / 60, 1)}, f, indent=2)

print("\n" + df.to_string(index=False))
print(f"\n  Saved: {base}.csv / .tex / .json  ({(time.time()-t0)/60:.1f} min)")
