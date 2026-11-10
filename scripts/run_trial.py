"""
Stage 5: Trial run — full adversarial simulation on 2015-2016 data.
Usage: python scripts/run_trial.py
"""
import os, sys, pickle, time, json
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import set_all_seeds, DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES, N_BLIND_ATTEMPTS
from src.models import train_all_models
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence, compute_confidence_intervals, run_statistical_tests
from src.visualization import plot_equivalence_curve, plot_score_reductions, plot_model_comparison

set_all_seeds()
os.makedirs(RESULTS_FIGURES, exist_ok=True)
os.makedirs(RESULTS_TABLES, exist_ok=True)

TRIAL_N_PATIENTS = 200      # Use subset for speed; set None for all
EXPLANATION_METHODS = ['shap', 'lime']

print("="*60)
print("  Stage 5: Trial Run on NHANES 2015-2016")
print("="*60)

# Load data
pkl = os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl')
if not os.path.exists(pkl):
    print("ERROR: Run scripts/prepare_data.py first!")
    sys.exit(1)

with open(pkl, 'rb') as f:
    data = pickle.load(f)

X_meas   = data['X_measured']
X_rep    = data['X_reported']
y        = data['y']
fn_meas  = data['feature_names_measured']
fn_rep   = data['feature_names_reported']
lie_budget = data['lie_budget']

print(f"\n[DATA] {len(y)} patients | Diabetes: {y.mean()*100:.1f}%")
print(f"[DATA] Lie budget: height +/-{lie_budget['height']['max_lie']:.2f}cm | "
      f"weight +/-{lie_budget['weight']['max_lie']:.2f}kg")

# Train models
print("\n[MODELS] Training all 4 models...")
models, split = train_all_models(X_meas, X_rep, y, fn_meas, fn_rep)

# Gameable features (self-reported height/weight columns)
gameable_meas = [f for f in fn_meas if 'self' in f]
gameable_rep  = [f for f in fn_rep  if 'self' in f]
if not gameable_rep:
    gameable_rep = [f for f in fn_rep if 'height' in f or 'weight' in f]
if not gameable_meas:
    gameable_meas = [f for f in fn_meas if 'height' in f or 'weight' in f]

print(f"\n[ADVERSARY] Gameable features (reported model): {gameable_rep}")

# Run simulations
all_results = {}
t0 = time.time()

for input_type, X, fn, gameable in [
    ('reported', X_rep,  fn_rep,  gameable_rep),
    ('measured', X_meas, fn_meas, gameable_meas),
]:
    for model_type in ['rf', 'lr']:
        key = f"{input_type}_{model_type}"
        model = models[key]
        X_train = X[split['idx_train']]
        X_test  = X[split['idx_test']]

        explainer = ExplanationGenerator(model, X_train, fn)
        simulator = AdversarialSimulator(model, explainer, lie_budget, gameable, fn)

        for exp_method in EXPLANATION_METHODS:
            run_key = f"{key}_{exp_method}"
            print(f"\n  >> {run_key} (n={TRIAL_N_PATIENTS or len(X_test)} patients)")
            results = simulator.run_comparison(
                X_test,
                n_patients=TRIAL_N_PATIENTS,
                blind_attempts_list=N_BLIND_ATTEMPTS,
                explanation_method=exp_method
            )
            equivalences, summary = compute_equivalence(results, N_BLIND_ATTEMPTS)
            ci = compute_confidence_intervals(equivalences)

            all_results[run_key] = {
                'results'            : results,
                'equivalences'       : equivalences,
                'summary'            : summary,
                'confidence_interval': ci,
            }

            # Save equivalence curve
            plot_equivalence_curve(
                results, N_BLIND_ATTEMPTS,
                title=f"Equivalence Curve: {run_key.upper()}",
                save_path=os.path.join(RESULTS_FIGURES, f'trial_equiv_{run_key}.png')
            )

elapsed = time.time() - t0
print(f"\n[TIME] Total simulation: {elapsed/60:.1f} min")

# Model comparison plot
plot_model_comparison(all_results, save_path=os.path.join(RESULTS_FIGURES, 'trial_model_comparison.png'))

# Score reduction box plots — first config
first_key = list(all_results.keys())[0]
plot_score_reductions(
    all_results[first_key]['results'],
    save_path=os.path.join(RESULTS_FIGURES, 'trial_score_reductions.png')
)

# Summary table
rows = []
for run_key, d in all_results.items():
    s = d['summary']
    ci = d['confidence_interval']
    parts = run_key.split('_')
    rows.append({
        'Config'         : run_key,
        'Input'          : parts[0],
        'Model'          : parts[1].upper(),
        'Explanation'    : parts[2].upper(),
        'Mean N*'        : round(s['mean_equivalence'], 2),
        'Median N*'      : round(s['median_equivalence'], 2),
        'CI_mean_lo'     : round(ci['mean_ci'][0], 2),
        'CI_mean_hi'     : round(ci['mean_ci'][1], 2),
        '% useless'      : round(s['pct_explanation_useless'], 1),
        '% worth 10+'    : round(s['pct_explanation_worth_10plus'], 1),
        'N patients'     : s['n_patients'],
    })

df_summary = pd.DataFrame(rows)
csv_path = os.path.join(RESULTS_TABLES, 'trial_summary.csv')
df_summary.to_csv(csv_path, index=False)

print("\n" + "="*60)
print("  TRIAL SUMMARY TABLE")
print("="*60)
print(df_summary.to_string(index=False))
print(f"\nSaved: {csv_path}")

# Statistical tests
print("\n[STATS] Running statistical tests...")
run_statistical_tests(all_results)

print("\n[DONE] Stage 5 Trial Run complete!")
print(f"       Figures: {RESULTS_FIGURES}")
print(f"       Tables : {RESULTS_TABLES}")
