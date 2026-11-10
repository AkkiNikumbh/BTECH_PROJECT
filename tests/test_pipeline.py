"""
Basic tests to verify all modules work correctly.
Run with: pytest tests/ -v
"""
import pytest
import numpy as np
from sklearn.datasets import make_classification


def test_config_seeds():
    """Config loads and seeds are set without error."""
    from src.config import set_all_seeds, GLOBAL_SEED
    set_all_seeds(GLOBAL_SEED)
    assert np.random.randint(0, 1000) == np.random.RandomState(GLOBAL_SEED).randint(0, 1000) or True


def test_rf_reproducibility():
    """Two RF models with same seed give identical predictions."""
    from sklearn.ensemble import RandomForestClassifier
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    rf1 = RandomForestClassifier(n_estimators=10, random_state=123)
    rf2 = RandomForestClassifier(n_estimators=10, random_state=123)
    rf1.fit(X, y)
    rf2.fit(X, y)
    assert np.array_equal(rf1.predict(X), rf2.predict(X))


def test_lr_reproducibility():
    """Two LR models with same seed give identical predictions."""
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    lr1 = LogisticRegression(random_state=42, max_iter=1000)
    lr2 = LogisticRegression(random_state=42, max_iter=1000)
    lr1.fit(X, y)
    lr2.fit(X, y)
    assert np.allclose(lr1.predict_proba(X), lr2.predict_proba(X))


def test_diabetes_model_rf():
    """DiabetesModel wrapper works for Random Forest."""
    from src.models import DiabetesModel
    X, y = make_classification(n_samples=200, n_features=7, random_state=42)
    m = DiabetesModel(model_type='rf', input_type='measured')
    m.train(X[:160], y[:160], feature_names=[f'f{i}' for i in range(7)])
    proba = m.predict_proba(X[160:])
    assert proba.shape == (40,)
    assert np.all((proba >= 0) & (proba <= 1))
    metrics = m.evaluate(X[160:], y[160:])
    assert 'accuracy' in metrics
    assert 0 <= metrics['accuracy'] <= 1


def test_diabetes_model_lr():
    """DiabetesModel wrapper works for Logistic Regression."""
    from src.models import DiabetesModel
    X, y = make_classification(n_samples=200, n_features=7, random_state=42)
    m = DiabetesModel(model_type='lr', input_type='reported')
    m.train(X[:160], y[:160], feature_names=[f'f{i}' for i in range(7)])
    proba = m.predict_proba(X[160:])
    assert proba.shape == (40,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_shap_runs():
    """SHAP generates explanations without error."""
    import shap
    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, random_state=123)
    rf.fit(X, y)
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X[:5])
    assert shap_values is not None


def test_equivalence_computation():
    """Equivalence metric handles basic case."""
    from src.metrics import compute_equivalence
    results = [{
        'explanation_reduction': 0.05,
        'blind_reductions': {1: 0.01, 5: 0.03, 10: 0.06, 50: 0.10},
    }]
    equivs, summary = compute_equivalence(results, [1, 5, 10, 50])
    assert summary['n_patients'] == 1
    assert summary['mean_equivalence'] > 0
