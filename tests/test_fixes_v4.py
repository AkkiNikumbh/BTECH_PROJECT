"""
Regression tests for fix 12 (explanation attacker keeps the better of its
lie and its true answers, like the blind attacker) — see docs/fixes_v2.md.
Run with: python -m pytest tests/ -v
"""
import numpy as np

from tests.test_fixes_v3 import _setup


def test_explanation_reduction_never_negative_with_keep_best():
    X, m, exp, sim = _setup('rf', keep_best=True)
    for i in range(60):
        for method in ('shap', 'lime'):
            r = sim.explanation_guided_attack(X[i], method, patient_idx=i)
            assert r['score_reduction'] >= 0.0
            assert r['score_reduction'] == max(0.0, r['raw_reduction'])
            if r['raw_reduction'] < 0:
                assert not r['lie_kept']
                assert np.allclose(r['new_x'], X[i])


def test_keep_best_only_floors_same_lie():
    """keep_best changes nothing but the floor: same lie, same raw effect."""
    X, m, exp, sim_on = _setup('rf', keep_best=True)
    _, _, _, sim_off = _setup('rf', keep_best=False)
    for i in range(40):
        a = sim_on.explanation_guided_attack(X[i], 'shap', patient_idx=i)
        b = sim_off.explanation_guided_attack(X[i], 'shap', patient_idx=i)
        assert a['changed_feature'] == b['changed_feature']
        assert a['change_amount'] == b['change_amount']
        assert np.isclose(a['raw_reduction'], b['score_reduction'])
        assert np.isclose(a['score_reduction'], max(0.0, b['score_reduction']))


def test_run_comparison_stores_raw_reduction():
    X, m, exp, sim = _setup('lr', keep_best=True)
    res = sim.run_comparison(X[:10], blind_attempts_list=[1, 2, 5],
                             explanation_method='shap')
    for r in res:
        assert 'explanation_raw_reduction' in r
        assert r['explanation_reduction'] == max(0.0, r['explanation_raw_reduction'])


# ------------------------------------------------------------------
# FIX 13 — SHAP sign read relative to the average patient
# ------------------------------------------------------------------
import pytest


@pytest.mark.parametrize('phi, x, ref, expected', [
    (+0.1, 160.0, 170.0, [+1]),   # short, "height raises risk" -> taller
    (+0.1, 180.0, 170.0, [-1]),   # tall, "height raises risk"  -> shorter
    (-0.1, 180.0, 170.0, [+1]),   # tall, "height lowers risk"  -> even taller
    (-0.1, 160.0, 170.0, [-1]),   # short, "height lowers risk" -> even shorter
    (0.0, 160.0, 170.0, []),
    (+0.1, 170.0, 170.0, []),     # exactly average: no direction
])
def test_shap_directions_toward_mean(phi, x, ref, expected):
    from src.explanations import shap_directions
    assert shap_directions(phi, x, ref, 'toward_mean') == expected


def test_shap_directions_old_sign_rule_kept_as_option():
    from src.explanations import shap_directions
    assert shap_directions(+0.1, 160.0, 170.0, 'sign') == [-1]
    assert shap_directions(-0.1, 160.0, 170.0, 'sign') == [+1]


def test_lr_shap_guided_lie_never_raises_score():
    """For a linear model, following SHAP towards/away from the mean always
    lowers (or keeps) the score, so no lie is ever undone."""
    X, m, exp, sim = _setup('lr', keep_best=True)
    for i in range(80):
        r = sim.explanation_guided_attack(X[i], 'shap', patient_idx=i)
        assert r['raw_reduction'] >= -1e-12


def test_holm_correction_added_and_not_smaller_than_raw_p():
    from src.metrics import run_statistical_tests
    rng = np.random.RandomState(0)
    all_results = {}
    for k, shift in [('a', 0.02), ('b', 0.0), ('c', -0.01)]:
        res = [{'explanation_reduction': float(0.05 + shift + rng.normal(0, 0.01)),
                'blind_reductions': {10: float(0.05 + rng.normal(0, 0.01))}} for _ in range(60)]
        all_results[k] = {'results': res}
    tests = run_statistical_tests(all_results)
    for t in tests.values():
        assert t['wilcoxon_pval_holm'] >= t['wilcoxon_pval'] - 1e-12
        assert t['wilcoxon_pval_holm'] <= 1.0
