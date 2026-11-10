"""
SHAP and LIME explanation generation for model predictions.
Provides a unified interface for both explanation methods.
"""
import numpy as np
import shap
import lime
import lime.lime_tabular


class ExplanationGenerator:
    """
    Generate feature attributions using SHAP and LIME.
    Provides a consistent interface for both explanation methods.
    """

    def __init__(self, model, X_train, feature_names):
        """
        Parameters
        ----------
        model : DiabetesModel instance
        X_train : training data (for background/reference)
        feature_names : list of feature name strings
        """
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

        # Initialize SHAP explainer
        if model.model_type == 'rf':
            self.shap_explainer = shap.TreeExplainer(model.model)
        elif model.model_type == 'lr':
            if model.scaler:
                X_bg = model.scaler.transform(X_train[:100])
            else:
                X_bg = X_train[:100]
            self.shap_explainer = shap.LinearExplainer(model.model, X_bg)

        # Initialize LIME explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=feature_names,
            class_names=['No Diabetes', 'Diabetes'],
            mode='classification',
            random_state=42
        )

    def get_shap_explanation(self, x_single):
        """
        Get SHAP values for a single instance.

        Returns
        -------
        dict: {feature_name: shap_value} sorted by absolute value (descending)
        """
        import numpy as np

        x_2d = x_single.reshape(1, -1)

        if self.model.model_type == 'rf':
            sv = self.shap_explainer.shap_values(x_2d)

            # shap >= 0.41 may return an Explanation object
            if hasattr(sv, 'values'):
                sv = sv.values

            # sv can be: list[array], 3-D array (n, feats, classes), or 2-D
            if isinstance(sv, list):
                # list of [class0_array, class1_array]
                vals = np.array(sv[1]).flatten()
            elif isinstance(sv, np.ndarray):
                if sv.ndim == 3:
                    # shape (n_samples, n_features, n_classes) — take class 1
                    vals = sv[0, :, 1]
                elif sv.ndim == 2:
                    vals = sv[0]
                else:
                    vals = sv.flatten()
            else:
                vals = np.array(sv).flatten()

        else:  # Logistic Regression
            if self.model.scaler:
                x_scaled = self.model.scaler.transform(x_2d)
            else:
                x_scaled = x_2d
            sv = self.shap_explainer.shap_values(x_scaled)
            if hasattr(sv, 'values'):
                sv = sv.values
            if isinstance(sv, np.ndarray):
                vals = sv[0] if sv.ndim == 2 else sv.flatten()
            else:
                vals = np.array(sv).flatten()

        vals = np.array(vals, dtype=float).flatten()
        attributions = dict(zip(self.feature_names, vals))
        attributions = dict(sorted(
            attributions.items(),
            key=lambda item: abs(float(item[1])),
            reverse=True
        ))
        return attributions

    def get_lime_explanation(self, x_single, num_features=5):
        """
        Get LIME explanation for a single instance.

        Returns
        -------
        dict: {feature_name: weight} from LIME
        """
        predict_fn = lambda X: np.column_stack([
            1 - self.model.predict_proba(X),
            self.model.predict_proba(X)
        ])

        exp = self.lime_explainer.explain_instance(
            x_single,
            predict_fn,
            num_features=num_features,
            num_samples=500
        )

        return dict(exp.as_list())

    def get_top_gameable_feature(self, attributions, gameable_features,
                                 x_single=None):
        """
        Find the most influential feature that the adversary CAN change,
        and the direction that should LOWER the risk score.

        SHAP returns exact feature names as keys; the value is a signed
        contribution, so positive -> decrease the feature, negative -> increase.

        LIME (with its default discretize_continuous=True) returns BIN
        conditions like:
          'self_height_cm > 170.18'
          '0.5 < self_weight_kg <= 2.1'
          'self_weight_kg <= 62.3'
        and a weight meaning "being in this bin raises (+) / lowers (-) risk".
        The direction therefore depends on the bin, not just the sign
        (FIX 2: the old code always decreased the feature for a positive
        weight, which is wrong for '<=' bins). See lime_direction().

        Features whose explanation gives no usable direction are skipped and
        the next gameable feature (in attribution order) is tried.

        Parameters
        ----------
        attributions : dict of {feature_or_condition: importance}
        gameable_features : list of exact feature names that can be changed
        x_single : np.array, the patient's feature vector (needed for LIME
                   range bins, to pick the nearer edge to leave the bin)

        Returns
        -------
        (feature_name, attribution_value, direction) where direction is
        -1 (decrease) or +1 (increase); or (None, None, 0) if nothing usable.
        """
        for attr_key, val in attributions.items():
            key = str(attr_key)
            # Direct match (SHAP exact names)
            if key in gameable_features:
                val = float(val)
                direction = -1 if val > 0 else (1 if val < 0 else 0)
                if direction != 0:
                    return key, val, direction
                continue
            # LIME condition strings
            for feat in gameable_features:
                if feat in key:
                    x_value = None
                    if x_single is not None and feat in self.feature_names:
                        x_value = float(x_single[self.feature_names.index(feat)])
                    direction = lime_direction(key, feat, float(val), x_value)
                    if direction != 0:
                        return feat, float(val), direction
                    break
        return None, None, 0


# ------------------------------------------------------------------
# LIME condition parsing (FIX 2)
# ------------------------------------------------------------------
_NUM = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'


def lime_direction(condition, feature, weight, x_value=None):
    """
    Turn one LIME (condition, weight) pair into the direction to change
    `feature` so the risk score goes DOWN.

    Rules (weight > 0 means "being in this bin raises risk"):
      'f > a'  / 'f >= a' : weight>0 -> decrease (-1); weight<0 -> increase (+1)
      'f <= a' / 'f < a'  : weight>0 -> increase (+1); weight<0 -> decrease (-1)
      'a < f <= b'        : weight>0 -> leave the bin via the nearer edge
                            (needs x_value; -1 if nearer a, +1 if nearer b);
                            weight<0 -> 0 (the explanation says the patient is
                            already in a protective bin: no direction given)
    Returns -1, +1, or 0 when no usable direction can be read.
    """
    import re
    if weight == 0:
        return 0
    f = re.escape(feature)
    cond = condition.strip()

    # Range bin: a < f <= b  (LIME's format; also accept other <, <= mixes)
    m = re.fullmatch(_NUM + r'\s*<=?\s*' + f + r'\s*<=?\s*' + _NUM, cond)
    if m:
        if weight < 0:
            return 0
        lo, hi = float(m.group(1)), float(m.group(2))
        if x_value is None:
            return 0
        return -1 if (x_value - lo) <= (hi - x_value) else 1

    # One-sided bin: f > a, f >= a, f < a, f <= a
    m = re.fullmatch(f + r'\s*(>=|>|<=|<)\s*' + _NUM, cond)
    if m:
        op = m.group(1)
        if op.startswith('>'):          # in the HIGH bin
            return -1 if weight > 0 else 1
        else:                           # in the LOW bin
            return 1 if weight > 0 else -1

    return 0
