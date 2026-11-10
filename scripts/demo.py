"""
scripts/demo.py
===============
Interactive viva demo — walks through the full pipeline for a
single patient, showing:
  1. Original diabetes risk score
  2. SHAP explanation (top features, gameable ones flagged)
  3. Explanation-guided attack result
  4. Blind trial-and-error comparison (N = 1, 5, 10, 20, 50, 100)
  5. Equivalence N* for this patient + population summary

Usage:
    python scripts/demo.py                 # default patient #42
    python scripts/demo.py --patient 100   # specific index
    python scripts/demo.py --list          # show dataset info

Run from the project root:
    cd c:\\Users\\ashua\\OneDrive\\???\\Desktop\\BTP
    python scripts/demo.py --patient 42
"""
import sys, os, argparse, pickle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.config import set_all_seeds, FINAL_CONFIG, N_BLIND_ATTEMPTS
from src.models import DiabetesModel
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence

set_all_seeds()

# ── Constants ───────────────────────────────────────────────────
BANNER = """
\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557
\u2551   GP-109 BTP \u2014 Can an AI Explanation Be Misused?            \u2551
\u2551   Live Viva Demo  |  DTU CSE  |  2026-27                    \u2551
\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d
"""
N_MILESTONES = [1, 5, 10, 20, 50, 100]

def sep(char='\u2500', n=62): print(char * n)

# ── Load data + model ────────────────────────────────────────────
def load_everything():
    """Load pickles, train Model B (reported RF), init explainer + simulator."""
    dev_pkl  = 'data/processed/nhanes_2015_2016_prepared.pkl'
    hold_pkl = 'data/processed/nhanes_2017_2018_prepared.pkl'

    for p in [dev_pkl, hold_pkl]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run scripts/prepare_data.py first.")
            sys.exit(1)

    with open(dev_pkl,  'rb') as f: dev  = pickle.load(f)
    with open(hold_pkl, 'rb') as f: hold = pickle.load(f)

    feat_names = hold.get('feature_names_reported',
                 dev.get('feature_names_reported',
                 [f'feature_{i}' for i in range(hold['X_reported'].shape[1])]))

    # Train Model B (reported RF) on dev set
    print("  [Loading] Training Model B (RF, reported features) on 2015-2016 data...")
    model = DiabetesModel(model_type='rf', input_type='reported')
    model.train(dev['X_reported'], dev['y'], feature_names=feat_names)

    # Lie budget from dev pickle
    lie_budget = dev.get('lie_budget',
                 {'height': {'max_lie': 7.29, 'mean_direction': 'positive'},
                  'weight': {'max_lie': 8.96, 'mean_direction': 'negative'}})

    # Gameable features (those with height or weight in name)
    gameable = [f for f in feat_names if 'height' in f.lower() or 'weight' in f.lower()]

    # Explainer
    print("  [Loading] Initialising SHAP explainer...")
    explainer = ExplanationGenerator(model, dev['X_reported'], feat_names)

    # Simulator
    simulator = AdversarialSimulator(model, explainer, lie_budget, gameable, feat_names)

    return model, explainer, simulator, hold['X_reported'], hold['y'], feat_names, lie_budget

# ── Main demo ────────────────────────────────────────────────────
def run_demo(patient_idx=42):
    print(BANNER)
    model, explainer, simulator, X_test, y_test, feat_names, lie_budget = load_everything()

    n_patients = len(X_test)
    if patient_idx >= n_patients:
        print(f"  Patient {patient_idx} out of range (0\u2013{n_patients-1}). Using 0.")
        patient_idx = 0

    x = X_test[patient_idx]
    true_label = int(y_test[patient_idx])

    # ── STEP 1: Original score ────────────────────────────────────
    sep('\u2550')
    print(f"  STEP 1 \u2014 Patient #{patient_idx} Profile")
    sep()
    original_score = float(model.predict_proba(x.reshape(1, -1))[0])
    print(f"  True diabetes status  : {'DIABETIC \u2713' if true_label else 'Non-diabetic'}")
    print(f"  Original risk score   : {original_score:.4f}  (0=low risk, 1=high risk)")
    print()
    print("  Feature values:")
    for name, val in zip(feat_names, x):
        gameable_flag = '  \u2190 can lie about' if ('height' in name.lower() or 'weight' in name.lower()) else ''
        print(f"    {name:35s}: {val:.2f}{gameable_flag}")

    # ── STEP 2: SHAP explanation ──────────────────────────────────
    sep('\u2500')
    print(f"  STEP 2 \u2014 SHAP Explanation (what pushes the score up/down)")
    sep()
    shap_attrs = explainer.get_shap_explanation(x)
    sorted_attrs = sorted(shap_attrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    print(f"  {'Feature':35s}  {'SHAP value':>12}  Impact")
    sep('-')
    for feat, val in sorted_attrs[:7]:
        direction = '\u2191 increases risk' if val > 0 else '\u2193 decreases risk'
        gameable  = '  \u2190 GAMEABLE' if feat in [f for f in feat_names
                     if 'height' in f.lower() or 'weight' in f.lower()] else ''
        print(f"  {feat:35s}  {val:+12.4f}  {direction}{gameable}")

    # ── STEP 3: Explanation-guided attack ─────────────────────────
    sep('\u2500')
    print(f"  STEP 3 \u2014 Explanation-Guided Attack (1 explanation used)")
    sep()
    exp_result = simulator.explanation_guided_attack(x, explanation_method='shap')
    feat_changed = exp_result.get('changed_feature', 'none')
    change_amt   = exp_result.get('change_amount', 0)
    print(f"  Target feature  : {feat_changed}")
    if feat_changed and feat_changed in feat_names:
        orig_val = x[feat_names.index(feat_changed)]
        print(f"  Original value  : {orig_val:.2f}")
        print(f"  Adjusted value  : {orig_val + change_amt:.2f}  (change: {change_amt:+.2f})")
    print(f"  New risk score  : {exp_result['new_score']:.4f}")
    print(f"  Score reduction : {exp_result['score_reduction']:+.4f}  \u25c4 explanation advantage")

    # ── STEP 4: Blind trial-and-error ─────────────────────────────
    sep('\u2500')
    print(f"  STEP 4 \u2014 Blind Trial-and-Error (no explanation, random guessing)")
    sep()
    # blind_attack_multi_n returns {n: reduction} directly
    blind_reductions = simulator.blind_attack_multi_n(x, N_MILESTONES)
    print(f"  {'N blind attempts':>18}  {'Best score':>12}  {'Reduction':>12}")
    sep('-')
    for n in N_MILESTONES:
        if n in blind_reductions:
            red   = blind_reductions[n]
            score = original_score - red
            match = '  \u25c4 matches explanation' if abs(red - exp_result['score_reduction']) < 0.001 else ''
            print(f"  {n:>18}  {score:>12.4f}  {red:>+12.4f}{match}")

    # ── STEP 5: N* for this patient ───────────────────────────────
    sep('\u2500')
    print(f"  STEP 5 \u2014 Equivalence N* for Patient #{patient_idx}")
    sep()
    n_vals = sorted(blind_reductions.keys())
    r_vals = [blind_reductions[n] for n in n_vals]

    from scipy import interpolate
    r_exp = exp_result['score_reduction']
    if r_exp <= 0:
        n_star = 0.0
    elif r_exp >= max(r_vals):
        n_star = float(max(n_vals))
    else:
        try:
            f_interp = interpolate.interp1d(r_vals, n_vals, kind='linear',
                                            fill_value='extrapolate')
            n_star = float(np.clip(f_interp(r_exp), 0, max(n_vals)))
        except Exception:
            n_star = 0.0

    if n_star <= 1:
        verdict = "\U0001f7e2 NO advantage over 1 blind attempt (LOW THREAT)"
    elif n_star <= 5:
        verdict = "\U0001f7e1 Worth a few blind attempts (LOW\u2013MODERATE THREAT)"
    elif n_star <= 20:
        verdict = "\U0001f7e0 Worth moderate blind attempts (MODERATE THREAT)"
    else:
        verdict = "\U0001f534 Highly valuable explanation (HIGH THREAT)"

    print(f"  N* = {n_star:.1f}  \u2192  {verdict}")

    # ── Population summary ────────────────────────────────────────
    print()
    sep('\u2550')
    print(f"  POPULATION RESULT (2017-2018 holdout, N=884 patients)")
    print(f"  Primary config: reported_rf_shap")
    sep()
    print(f"  Mean N*   = 12.43  [95% CI: 10.54\u201314.53]")
    print(f"  Median    =  0.00  (63.8% of patients: explanation useless)")
    print(f"  17.2%     of patients: explanation worth 10+ blind attempts")
    print(f"  VERDICT   : MODERATE THREAT (5 \u2264 N* \u2264 20)")
    print(f"  POLICY    : Verify inputs; restricting explanations is NOT the solution")
    sep('\u2550')
    print()

def list_info():
    hold_pkl = 'data/processed/nhanes_2017_2018_prepared.pkl'
    if not os.path.exists(hold_pkl):
        print("Holdout pickle not found.")
        return
    with open(hold_pkl, 'rb') as f: hold = pickle.load(f)
    n = len(hold['y'])
    print(f"\n  Available patients : 0 \u2013 {n-1}")
    print(f"  Total patients     : {n}")
    print(f"  Diabetic patients  : {int(hold['y'].sum())} ({100*hold['y'].mean():.1f}%)")
    feat = hold.get('feature_names_reported', [])
    print(f"  Features ({len(feat)})     : {feat}")

# ── Entry point ──────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='GP-109 BTP Viva Demo — single-patient pipeline walkthrough')
    parser.add_argument('--patient', type=int, default=42,
                        help='Patient index in 2017-2018 test set (default: 42)')
    parser.add_argument('--list', action='store_true',
                        help='Show dataset info and exit')
    args = parser.parse_args()

    if args.list:
        list_info()
    else:
        run_demo(patient_idx=args.patient)
