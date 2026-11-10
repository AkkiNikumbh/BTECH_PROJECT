"""
Regression tests for the v2 implementation fixes (see docs/fixes_v2.md).
Run with: python -m pytest tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

FEATS_REP = ['RIDAGEYR', 'RIAGENDR', 'BPXSY1', 'SMQ020', 'PAQ650',
             'self_height_cm', 'self_weight_kg']
FEATS_MEAS = ['RIDAGEYR', 'RIAGENDR', 'BPXSY1', 'SMQ020', 'PAQ650',
              'measured_height_cm', 'measured_weight_kg']
BUDGET = {'height': {'max_lie': 7.29, 'mean_direction': 'positive'},
          'weight': {'max_lie': 8.96, 'mean_direction': 'negative'}}


def _synthetic(n=600, seed=0):
    """NHANES-like synthetic data where risk rises with weight."""
    rng = np.random.RandomState(seed)
    age = rng.uniform(18, 80, n)
    sex = rng.randint(1, 3, n).astype(float)
    sbp = rng.normal(125, 15, n)
    smq = rng.randint(1, 3, n).astype(float)
    paq = rng.randint(1, 3, n).astype(float)
    h = rng.normal(168, 10, n)
    w = rng.normal(80, 18, n)
    logit = -9 + 0.04 * age + 0.06 * w - 0.02 * h + 0.01 * sbp
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(float)
    X = np.column_stack([age, sex, sbp, smq, paq, h, w])
    return X, y


# ------------------------------------------------------------------
# FIX 1 — measured model has no gameable features
# ------------------------------------------------------------------
def test_gameable_features_only_self_reported():
    from src.config import get_gameable_features
    assert get_gameable_features(FEATS_REP) == ['self_height_cm', 'self_weight_kg']
    assert get_gameable_features(FEATS_MEAS) == []


@pytest.mark.parametrize('method', ['shap', 'lime'])
def test_measured_model_cannot_be_attacked(method):
    from src.models import DiabetesModel
    from src.explanations import ExplanationGenerator
    from src.adversarial import AdversarialSimulator
    from src.config import get_gameable_features
    X, y = _synthetic()
    m = DiabetesModel('rf', 'measured')
    m.train(X[:500], y[:500], feature_names=FEATS_MEAS)
    exp = ExplanationGenerator(m, X[:500], FEATS_MEAS)
    sim = AdversarialSimulator(m, exp, BUDGET, get_gameable_features(FEATS_MEAS), FEATS_MEAS)
    res = sim.run_comparison(X[500:505], blind_attempts_list=[1, 10, 100],
                             explanation_method=method)
    for r in res:
        assert r['explanation_reduction'] == 0.0
        assert r['explanation_feature'] is None
        assert all(v == 0.0 for v in r['blind_reductions'].values())


# ------------------------------------------------------------------
# FIX 2 — LIME direction depends on the bin condition
# ------------------------------------------------------------------
@pytest.mark.parametrize('cond, weight, x, expected', [
    ('self_weight_kg > 90.00',            +0.2, 100.0, -1),  # high bin raises risk -> go down
    ('self_weight_kg > 90.00',            -0.2, 100.0, +1),
    ('self_weight_kg <= 62.30',           +0.2,  55.0, +1),  # LOW bin raises risk -> go UP (old code: down)
    ('self_weight_kg <= 62.30',           -0.2,  55.0, -1),
    ('62.30 < self_weight_kg <= 90.00',   +0.2,  65.0, -1),  # nearer lower edge
    ('62.30 < self_weight_kg <= 90.00',   +0.2,  88.0, +1),  # nearer upper edge
    ('62.30 < self_weight_kg <= 90.00',   -0.2,  70.0,  0),  # protective bin: no direction
    ('self_height_cm > 1.5e2',            +0.1, 170.0, -1),
    ('garbage string',                    +0.3,  70.0,  0),
])
def test_lime_direction(cond, weight, x, expected):
    from src.explanations import lime_direction
    feat = 'self_weight_kg' if 'weight' in cond else 'self_height_cm'
    if cond == 'garbage string':
        feat = 'self_weight_kg'
    assert lime_direction(cond, feat, weight, x) == expected


def test_top_gameable_feature_skips_unusable_and_uses_bin():
    from src.explanations import ExplanationGenerator
    gen = ExplanationGenerator.__new__(ExplanationGenerator)  # no model needed
    gen.feature_names = FEATS_REP
    x = np.array([50, 1, 120, 2, 2, 170.0, 70.0])
    attributions = {
        'RIDAGEYR > 60.00': 0.5,                       # not gameable
        '62.30 < self_weight_kg <= 90.00': -0.3,       # protective bin -> skipped
        'self_height_cm <= 160.00': 0.1,               # low bin raises risk -> increase
    }
    feat, val, d = gen.get_top_gameable_feature(
        attributions, ['self_height_cm', 'self_weight_kg'], x_single=x)
    assert feat == 'self_height_cm' and d == +1


def test_shap_direction_unchanged():
    from src.explanations import ExplanationGenerator
    gen = ExplanationGenerator.__new__(ExplanationGenerator)
    gen.feature_names = FEATS_REP
    feat, val, d = gen.get_top_gameable_feature(
        {'self_weight_kg': 0.4, 'self_height_cm': -0.1},
        ['self_height_cm', 'self_weight_kg'])
    assert feat == 'self_weight_kg' and d == -1


def test_lime_attack_on_reported_model_runs():
    """End-to-end: LIME-guided attack on the reported model runs, and the
    chosen change is always within the lie budget."""
    from src.models import DiabetesModel
    from src.explanations import ExplanationGenerator
    from src.adversarial import AdversarialSimulator
    from src.config import get_gameable_features
    X, y = _synthetic()
    for mt in ['rf', 'lr']:
        m = DiabetesModel(mt, 'reported')
        m.train(X[:500], y[:500], feature_names=FEATS_REP)
        exp = ExplanationGenerator(m, X[:500], FEATS_REP)
        sim = AdversarialSimulator(m, exp, BUDGET, get_gameable_features(FEATS_REP), FEATS_REP)
        for i in range(500, 515):
            r = sim.explanation_guided_attack(X[i], 'lime')
            assert abs(r['change_amount']) <= 8.96 + 1e-9


# ------------------------------------------------------------------
# FIX 3 — per-column NHANES missing codes
# ------------------------------------------------------------------
def test_missing_codes_keep_valid_values():
    from src.data_loader import replace_missing_codes
    df = pd.DataFrame({
        'WHD010': [77.0, 99.0 - 20, 7777.0, 9999.0, 64.0],   # 77 in = valid height
        'WHD020': [99.0, 777.0 - 600, 7777.0, 9999.0, 180.0],  # 99 lb = valid weight
        'SMQ020': [1.0, 2.0, 7.0, 9.0, 1.0],
    })
    out = replace_missing_codes(df)
    assert out['WHD010'].tolist()[:2] == [77.0, 79.0]
    assert out['WHD010'].isna().tolist() == [False, False, True, True, False]
    assert out['WHD020'].tolist()[0] == 99.0
    assert out['WHD020'].isna().tolist() == [False, False, True, True, False]
    assert out['SMQ020'].isna().tolist() == [False, False, True, True, False]


# ------------------------------------------------------------------
# FIX 5 — final run uses the pre-registered 2015-2016 lie budget
# ------------------------------------------------------------------
def test_run_final_uses_dev_lie_budget():
    import os, re
    path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        'scripts', 'run_final.py')
    src = open(path, encoding='utf-8').read()
    assert 'lie_budget_used = lie_budget_dev' in src
    sim_calls = re.findall(r'AdversarialSimulator\(\s*model, explainer, (\w+),', src)
    assert sim_calls == ['lie_budget_used']
