"""
Regression tests for the v3 fixes (4, 6, 7, 8, 9, 10, 11) — see docs/fixes_v2.md.
Run with: python -m pytest tests/ -v
"""
import numpy as np
import pytest

from tests.test_fixes import _synthetic, FEATS_REP, BUDGET

NS = [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]


def _res(exp, curve):
    return {'explanation_reduction': exp, 'blind_reductions': dict(zip(NS, curve))}


def _setup(model_type='rf', **sim_kw):
    from src.models import DiabetesModel
    from src.explanations import ExplanationGenerator
    from src.adversarial import AdversarialSimulator
    from src.config import get_gameable_features
    X, y = _synthetic()
    m = DiabetesModel(model_type, 'reported')
    m.train(X[:500], y[:500], feature_names=FEATS_REP)
    exp = ExplanationGenerator(m, X[:500], FEATS_REP)
    sim = AdversarialSimulator(m, exp, BUDGET, get_gameable_features(FEATS_REP),
                               FEATS_REP, **sim_kw)
    return X, m, exp, sim


# ------------------------------------------------------------------
# FIX 4 — primary N* from the MEAN curve (pre-registered)
# ------------------------------------------------------------------
def test_population_nstar_uses_mean_curve():
    """Two patients: per-patient N* are 0 and 101 (mean 50.5), but the
    pre-registered N* interpolates on the AVERAGED curve."""
    from src.metrics import population_nstar, compute_equivalence
    flat = [0.10] * 10                          # blind reaches 0.10 at N=1
    ramp = [0.00, 0, 0, 0, 0, 0, 0, 0, 0, 0.0]  # blind never improves
    results = [_res(0.05, flat), _res(0.05, ramp)]
    # mean exp = 0.05, mean curve = 0.05 at every N -> reached at N=1 -> N*=0
    assert population_nstar(results, NS) == 0.0
    eq, s = compute_equivalence(results, NS)
    assert list(eq) == [0.0, 101.0]
    assert s['mean_equivalence'] == 50.5      # old headline number
    assert s['nstar'] == 0.0                  # pre-registered headline


def test_population_nstar_interpolates():
    from src.metrics import population_nstar
    curve = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
    # mean exp 0.045 lies between N=5 (0.04) and N=10 (0.05) -> 7.5
    assert population_nstar([_res(0.045, curve)], NS) == pytest.approx(7.5)
    assert population_nstar([_res(0.5, curve)], NS) == 101.0
    assert population_nstar([_res(0.0, curve)], NS) == 0.0


def test_nstar_ci_brackets_estimate():
    from src.metrics import compute_equivalence, compute_confidence_intervals
    rng = np.random.RandomState(0)
    results = []
    for _ in range(80):
        curve = np.sort(rng.uniform(0, 0.1, 10))
        results.append(_res(float(rng.uniform(0, 0.1)), list(curve)))
    eq, s = compute_equivalence(results, NS)
    ci = compute_confidence_intervals(eq, n_bootstrap=200, results=results,
                                      blind_attempts_list=NS)
    assert ci['nstar_ci'][0] <= s['nstar'] <= ci['nstar_ci'][1]


def test_wilcoxon_all_zero_gives_p_one_not_nan():
    from src.metrics import run_statistical_tests
    res = [_res(0.0, [0.0] * 10) for _ in range(5)]
    out = run_statistical_tests({'measured_rf_shap': {'results': res}})
    assert out['measured_rf_shap']['wilcoxon_pval'] == 1.0


# ------------------------------------------------------------------
# FIX 6 + 7 — same lie-size rule for both attackers; realistic direction
# ------------------------------------------------------------------
def test_realistic_direction_from_budget():
    _, _, _, sim = _setup()
    assert sim.realistic_direction('self_height_cm') == +1   # over-reported
    assert sim.realistic_direction('self_weight_kg') == -1   # under-reported


@pytest.mark.parametrize('method', ['shap', 'lime'])
def test_explanation_lies_only_realistically(method):
    X, _, _, sim = _setup('lr', lie_direction='realistic', lie_magnitude='uniform')
    for i in range(500, 530):
        r = sim.explanation_guided_attack(X[i], method, patient_idx=i)
        if r['changed_feature'] == 'self_height_cm':
            assert 0 < r['change_amount'] <= 7.29
        elif r['changed_feature'] == 'self_weight_kg':
            assert -8.96 <= r['change_amount'] < 0


def test_blind_lies_only_realistically_and_uniformly(monkeypatch):
    X, model, _, sim = _setup('rf', lie_direction='realistic', lie_magnitude='uniform')
    seen = []
    orig = model.predict_proba

    def spy(Xq):
        seen.append(np.array(Xq, dtype=float).reshape(-1, len(FEATS_REP)).copy())
        return orig(Xq)
    monkeypatch.setattr(model, 'predict_proba', spy)
    x = X[510]
    sim.blind_attack_multi_n(x, NS, patient_idx=3)
    deltas = np.vstack(seen[1:]) - x       # skip the original-score call
    dh, dw = deltas[:, 5], deltas[:, 6]
    assert np.all(dh >= 0) and np.all(dw <= 0)
    assert np.all(dh <= 7.29 + 1e-9) and np.all(-dw <= 8.96 + 1e-9)
    sizes = np.concatenate([dh[dh > 0], -dw[dw < 0]])
    assert len(sizes) == 100
    assert sizes.min() < 0.5 * 8.96     # not always the max (uniform)


def test_max_magnitude_mode_uses_full_lie_for_both():
    X, _, _, sim = _setup('lr', lie_direction='realistic', lie_magnitude='max')
    r = sim.explanation_guided_attack(X[505], 'shap', patient_idx=5)
    if r['changed_feature']:
        assert abs(r['change_amount']) in (pytest.approx(7.29), pytest.approx(8.96))


def test_both_directions_mode_still_available():
    X, _, _, sim = _setup('lr', lie_direction='both')
    assert sim.allowed_directions('self_weight_kg') == [-1, 1]


# ------------------------------------------------------------------
# FIX 8 — per-patient seeding: SHAP and LIME face the SAME blind attempts
# ------------------------------------------------------------------
def test_same_blind_baseline_for_shap_and_lime():
    X, _, _, sim = _setup('rf')
    shap_res = sim.run_comparison(X[500:506], blind_attempts_list=NS,
                                  explanation_method='shap')
    lime_res = sim.run_comparison(X[500:506], blind_attempts_list=NS,
                                  explanation_method='lime')
    for a, b in zip(shap_res, lime_res):
        assert a['blind_reductions'] == b['blind_reductions']


def test_results_do_not_depend_on_patient_order():
    X, _, _, sim = _setup('lr')
    r_alone = sim.explanation_guided_attack(X[520], 'lime', patient_idx=7)
    for i in range(500, 510):          # explain other patients first
        sim.explanation_guided_attack(X[i], 'lime', patient_idx=i)
    r_again = sim.explanation_guided_attack(X[520], 'lime', patient_idx=7)
    assert r_alone['change_amount'] == r_again['change_amount']
    assert r_alone['new_score'] == r_again['new_score']


# ------------------------------------------------------------------
# FIX 9, 10, 11
# ------------------------------------------------------------------
def test_lime_shows_all_features():
    X, _, exp, _ = _setup('lr')
    assert len(exp.get_lime_explanation(X[505], seed=1)) == len(FEATS_REP)


def test_models_use_balanced_class_weight_and_flag_more_positives():
    from src.models import DiabetesModel
    X, y = _synthetic(n=1500, seed=3)
    for mt in ['rf', 'lr']:
        m = DiabetesModel(mt, 'reported')
        assert m.model.get_params()['class_weight'] == 'balanced'
        m.train(X[:1200], y[:1200])
        assert m.predict(X[1200:]).mean() > 0.0


def test_rf_predicts_single_threaded_after_training():
    from src.models import DiabetesModel
    X, y = _synthetic()
    m = DiabetesModel('rf', 'reported')
    m.train(X[:500], y[:500])
    assert m.model.n_jobs == 1


def test_shap_background_is_random_sample_not_first_rows():
    from src.models import DiabetesModel
    from src.explanations import ExplanationGenerator
    X, y = _synthetic()
    m = DiabetesModel('lr', 'reported')
    m.train(X[:500], y[:500])
    exp = ExplanationGenerator(m, X[:500], FEATS_REP)
    bg = np.asarray(exp.shap_explainer.data if hasattr(exp.shap_explainer, 'data')
                    else exp.shap_explainer.masker.data)
    first100 = m.scaler.transform(X[:100])
    assert bg.shape[0] == 100
    assert not np.allclose(bg, first100)
