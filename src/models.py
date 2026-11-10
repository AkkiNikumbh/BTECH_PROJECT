"""
Model training for the diabetes risk prediction experiment.
Trains two model types (Logistic Regression, Random Forest) on two
feature sets (measured, self-reported) for a total of 4 models.
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from src.config import (
    SEED_MODEL_LOGREG, SEED_MODEL_RF,
    SEED_TRAIN_TEST_SPLIT, TEST_SIZE,
    N_ESTIMATORS_RF, MAX_DEPTH_RF
)


class DiabetesModel:
    """
    Wrapper for a diabetes risk model.
    Handles training, prediction, and evaluation with consistent interfaces.
    """

    def __init__(self, model_type='rf', input_type='measured'):
        """
        Parameters
        ----------
        model_type : str — 'rf' (Random Forest) or 'lr' (Logistic Regression)
        input_type : str — 'measured' or 'reported'
        """
        self.model_type = model_type
        self.input_type = input_type
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.metrics = {}

        if model_type == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=N_ESTIMATORS_RF,
                max_depth=MAX_DEPTH_RF,
                random_state=SEED_MODEL_RF,
                n_jobs=-1
            )
        elif model_type == 'lr':
            self.scaler = StandardScaler()
            self.model = LogisticRegression(
                random_state=SEED_MODEL_LOGREG,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    def train(self, X_train, y_train, feature_names=None):
        """Train the model on the given data."""
        self.feature_names = feature_names
        if self.scaler:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        """Get probability of diabetes (class 1)."""
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        """Get binary prediction."""
        if self.scaler:
            X = self.scaler.transform(X)
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Compute and store evaluation metrics."""
        if self.scaler:
            X_scaled = self.scaler.transform(X_test)
        else:
            X_scaled = X_test

        y_pred = self.model.predict(X_scaled)
        y_prob = self.model.predict_proba(X_scaled)[:, 1]

        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc_roc': roc_auc_score(y_test, y_prob),
        }

        print(f"\n{'='*50}")
        print(f"  Model: {self.model_type.upper()} | Input: {self.input_type}")
        print(f"{'='*50}")
        for metric, value in self.metrics.items():
            print(f"  {metric:12s}: {value:.4f}")

        return self.metrics

    def __repr__(self):
        return f"DiabetesModel(type={self.model_type}, input={self.input_type})"


def train_all_models(X_measured, X_reported, y, fn_measured, fn_reported):
    """
    Train all four models and return them in a dict.

    Returns
    -------
    models : dict — keys: 'measured_rf', 'measured_lr', 'reported_rf', 'reported_lr'
    split : dict — train/test indices and labels
    """
    # Same train/test split for both feature sets
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices, test_size=TEST_SIZE,
        random_state=SEED_TRAIN_TEST_SPLIT,
        stratify=y
    )

    models = {}

    for input_type, X, fn in [
        ('measured', X_measured, fn_measured),
        ('reported', X_reported, fn_reported)
    ]:
        X_train, X_test = X[idx_train], X[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        for model_type in ['rf', 'lr']:
            key = f"{input_type}_{model_type}"
            print(f"\n[TRAIN] Training {key}...")
            m = DiabetesModel(model_type=model_type, input_type=input_type)
            m.train(X_train, y_train, feature_names=fn)
            m.evaluate(X_test, y_test)
            models[key] = m

    split = {
        'idx_train': idx_train,
        'idx_test': idx_test,
        'y_train': y[idx_train],
        'y_test': y[idx_test],
    }

    return models, split
