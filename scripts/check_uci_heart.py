"""
Research plan, phase P1: "test the code on a small dataset where results can
be checked by hand" — UCI Heart Disease (UCI id 45, Cleveland, 297 patients).

The real pipeline classes (DiabetesModel, ExplanationGenerator,
AdversarialSimulator, metrics) are run on a logistic regression, where every
number has a closed form, and each output is compared with a by-hand
calculation:

  1. risk score    p = 1 / (1 + exp(-(w . z + b))),  z = (x - mean) / sd
  2. SHAP value    phi_j = w_j * (z_j - mean of z_j over the SHAP background)
  3. explanation-guided lie: the gameable feature with the largest |phi|,
     moved by the full lie budget TOWARDS the average patient if phi > 0
     and AWAY from it if phi < 0 (fix 13), then undone if it raised the
     score. (This check is what found the old "phi > 0 -> decrease" bug.)
  4. blind attacker: with fixed lie sizes there are only 4 possible lies
     (2 features x 2 directions), so after 100 random attempts the best
     reduction must equal the best of those 4, floored at 0
  5. N* on a curve small enough to interpolate by hand

Resting blood pressure (trestbps) and cholesterol (chol) are TREATED as
"self-reported" here only to exercise the code; this is not a claim about
patients.

Usage: python scripts/check_uci_heart.py
"""
import os
import sys
import io
import contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.supporting_data import load_uci_heart, UCI45_ZIP
from src.models import DiabetesModel
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence

FEATURES = ['age', 'sex', 'trestbps', 'chol', 'thalach']
GAMEABLE = ['trestbps', 'chol']
BUDGET = {'trestbps': {'max_lie': 10.0, 'mean_direction': 'negative'},
          'chol':     {'max_lie': 20.0, 'mean_direction': 'negative'}}
N_SHOW = 5
TOL = 1e-6


def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def run_hand_check(verbose=True):
    df = load_uci_heart()
    X = df[FEATURES].to_numpy(dtype=float)
    y = df['target'].to_numpy()
    X_train, X_test = X[:237], X[237:]          # fixed 80/20 split, no shuffling
    y_train = y[:237]

    with contextlib.redirect_stdout(io.StringIO()):
        model = DiabetesModel('lr', 'reported')
        model.train(X_train, y_train, feature_names=FEATURES)
        expl = ExplanationGenerator(model, X_train, FEATURES)
    sim = AdversarialSimulator(model, expl, BUDGET, GAMEABLE, FEATURES,
                               lie_direction='both', lie_magnitude='max',
                               keep_best=True)

    mu, sd = model.scaler.mean_, model.scaler.scale_
    w, b = model.model.coef_[0], model.model.intercept_[0]
    bg = np.asarray(expl.shap_explainer.masker.data)
    bg_mean = bg.mean(axis=0)
    ref = expl.shap_reference            # raw-unit mean of the background

    def hand_score(x):
        return _sigmoid(np.dot(w, (x - mu) / sd) + b)

    checks = {}
    rows = []
    max_err = {'score': 0.0, 'shap': 0.0, 'expl': 0.0, 'blind': 0.0}
    for i in range(len(X_test)):
        x = X_test[i]
        # 1. score
        p_hand = hand_score(x)
        p_code = float(model.predict_proba(x.reshape(1, -1))[0])
        max_err['score'] = max(max_err['score'], abs(p_hand - p_code))
        # 2. SHAP
        phi_hand = w * ((x - mu) / sd - bg_mean)
        phi_code = expl.get_shap_explanation(x)
        phi_code = np.array([phi_code[f] for f in FEATURES])
        max_err['shap'] = max(max_err['shap'], float(np.max(np.abs(phi_hand - phi_code))))
        # 3. explanation-guided lie
        g_idx = [FEATURES.index(f) for f in GAMEABLE]
        top = max(g_idx, key=lambda j: abs(phi_hand[j]))
        towards = 1 if ref[top] > x[top] else -1      # towards the average
        direction = towards if phi_hand[top] > 0 else -towards
        x_lie = x.copy()
        x_lie[top] += direction * BUDGET[FEATURES[top]]['max_lie']
        red_hand = max(0.0, p_hand - hand_score(x_lie))
        r = sim.explanation_guided_attack(x, 'shap', patient_idx=i)
        max_err['expl'] = max(max_err['expl'], abs(red_hand - r['score_reduction']))
        # 4. blind: best of the 4 possible lies
        best4 = 0.0
        for j in g_idx:
            for d in (-1, 1):
                xl = x.copy()
                xl[j] += d * BUDGET[FEATURES[j]]['max_lie']
                best4 = max(best4, p_hand - hand_score(xl))
        blind = sim.blind_attack_multi_n(x, [1, 10, 100], patient_idx=i)
        max_err['blind'] = max(max_err['blind'], abs(best4 - blind[100]))
        # for a linear model, moving towards/away from the mean as SHAP says
        # always lowers the score, so the lie is never undone
        n_undone = checks.get('_undone', 0) + (r['raw_reduction'] < 0)
        checks['_undone'] = n_undone
        if i < N_SHOW:
            rows.append((i, x, p_hand, phi_hand, FEATURES[top], direction,
                         red_hand, best4))

    checks['score matches closed form'] = max_err['score'] < TOL
    checks['SHAP matches w*(z - mean z_bg)'] = max_err['shap'] < TOL
    checks['explanation lie matches hand rule'] = max_err['expl'] < TOL
    checks['blind(N=100) = best of the 4 lies'] = max_err['blind'] < TOL
    checks['LR: SHAP-guided lie never raises the score'] = checks.pop('_undone') == 0

    # 5. N* by hand: explanation reduction 0.03; blind curve
    #    N=1: 0.01, N=2: 0.02, N=3: 0.04  -> N* = 2 + (0.03-0.02)/(0.04-0.02) = 2.5
    toy = [{'explanation_reduction': 0.03,
            'blind_reductions': {1: 0.01, 2: 0.02, 3: 0.04}}]
    with contextlib.redirect_stdout(io.StringIO()):
        _, s = compute_equivalence(toy, [1, 2, 3])
    checks['N* interpolation (hand: 2.5)'] = abs(s['nstar'] - 2.5) < TOL

    if verbose:
        print("=" * 72)
        print("  P1 hand check — UCI Heart Disease (Cleveland), logistic regression")
        print("=" * 72)
        print(f"  Patients: {len(X)} (train {len(X_train)}, test {len(X_test)})")
        print(f"  Intercept b = {b:+.6f}")
        for f, m_, s_, w_, g in zip(FEATURES, mu, sd, w, bg_mean):
            print(f"  {f:9s} mean={m_:8.3f} sd={s_:7.3f}  w={w_:+.6f}  "
                  f"mean z (SHAP background)={g:+.4f}")
        print("\n  First test patients (recompute with a calculator):")
        for i, x, p, phi, feat, d, red, best4 in rows:
            xs = ', '.join(f'{f}={v:g}' for f, v in zip(FEATURES, x))
            ph = ', '.join(f'{f}={v:+.4f}' for f, v in zip(FEATURES, phi))
            print(f"\n  #{i}: {xs}")
            print(f"      score p = {p:.4f}   SHAP: {ph}")
            print(f"      explanation lie: {feat} {'+' if d > 0 else '-'}"
                  f"{BUDGET[feat]['max_lie']:g} -> reduction {red:.4f}"
                  f"   | best of 4 lies: {best4:.4f}")
        print("\n  Largest |code - hand| differences: "
              + ', '.join(f'{k}={v:.1e}' for k, v in max_err.items()))
        print()
        for k, ok in checks.items():
            print(f"  [{'PASS' if ok else 'FAIL'}] {k}")
    return checks


if __name__ == '__main__':
    if not os.path.exists(UCI45_ZIP):
        sys.exit(f"Missing {UCI45_ZIP}. Download heart+disease.zip from "
                 "https://archive.ics.uci.edu/dataset/45/heart+disease")
    results = run_hand_check()
    sys.exit(0 if all(results.values()) else 1)
