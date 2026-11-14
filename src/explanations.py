"""
SHAP and LIME explanation generation for model predictions.
Provides a unified interface for both explanation methods.
"""
import numpy as np
import shap
import lime
import lime.lime_tabular
from src.config import (LIME_N_FEATURES, SHAP_BACKGROUND_N,
                        SEED_SHAP_BACKGROUND, SHAP_DIRECTION)


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
        self.shap_direction = SHAP_DIRECTION
        # FIX 13: the "average patient" each SHAP value is measured against
        # (TreeSHAP: the training data; LinearExplainer: its background
        # sample, set below). Raw (unscaled) feature units.
        self.shap_reference = np.asarray(X_train, dtype=float).mean(axis=0)

        # Initialize SHAP explainer
        if model.model_type == 'rf':
            self.shap_explainer = shap.TreeExplainer(model.model)
        elif model.model_type == 'lr':
            # FIX 11: seeded RANDOM background sample (was X_train[:100],
            # i.e. whatever rows happened to come first)
            n_bg = min(SHAP_BACKGROUND_N, len(X_train))
            bg_idx = np.random.RandomState(SEED_SHAP_BACKGROUND).choice(
                len(X_train), size=n_bg, replace=False)
            X_bg = X_train[bg_idx]
            self.shap_reference = np.asarray(X_bg, dtype=float).mean(axis=0)
            if model.scaler:
                X_bg = model.scaler.transform(X_bg)
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

    def get_lime_explanation(self, x_single, num_features=None, seed=None):
        """
        Get LIME explanation for a single instance.

        num_features : None -> ALL features (FIX 9: with the old value of 5
                       of 7, height/weight could be cut off, and the attack
                       then did nothing).
        seed         : if given, LIME's random sampling is re-seeded for this
                       patient (FIX 8), so the explanation for a patient does
                       not depend on which patients were explained before.

        Returns
        -------
        dict: {feature_condition: weight} from LIME
        """
        if num_features is None:
            num_features = LIME_N_FEATURES or len(self.feature_names)

        if seed is not None:
            rs = np.random.RandomState(seed)
            self.lime_explainer.random_state = rs
            self.lime_explainer.base.random_state = rs

        def predict_fn(X):
            p = self.model.predict_proba(X)
            return np.column_stack([1 - p, p])

        exp = self.lime_explainer.explain_instance(
            x_single,
            predict_fn,
            num_features=num_features,
            num_samples=500
        )

        return dict(exp.as_list())

    def get_top_gameable_feature(self, attributions, gameable_features,
                                 x_single=None, allowed_directions=None):
        """
        Find the most influential feature that the adversary CAN change,
        and the direction that should LOWER the risk score.

        SHAP returns exact feature names as keys; the value is a signed
        contribution relative to the average patient (shap_reference).
        FIX 13 ('toward_mean'): positive -> move the value TOWARDS the
        average, negative -> move it further AWAY. Old rule ('sign'):
        positive -> decrease, negative -> increase. See shap_directions().

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
        allowed_directions : optional {feature: [allowed directions]}. A
                   feature whose risk-lowering direction is not allowed (e.g.
                   an unrealistic lie, FIX 7) is skipped.

        Returns
        -------
        (feature_name, attribution_value, direction) where direction is
        -1 (decrease) or +1 (increase); or (None, None, 0) if nothing usable.
        """
        def pick(feat, candidates):
            ok = (allowed_directions or {}).get(feat, [-1, 1])
            for d in candidates:
                if d in ok:
                    return d
            return 0

        for attr_key, val in attributions.items():
            key = str(attr_key)
            # Direct match (SHAP exact names)
            if key in gameable_features:
                val = float(val)
                x_value = ref = None
                reference = getattr(self, 'shap_reference', None)
                if x_single is not None and reference is not None:
                    j = self.feature_names.index(key)
                    x_value = float(x_single[j])
                    ref = float(reference[j])
                rule = getattr(self, 'shap_direction', SHAP_DIRECTION)
                direction = pick(key, shap_directions(val, x_value, ref, rule))
                if direction != 0:
                    return key, val, direction
                continue
            # LIME condition strings
            for feat in gameable_features:
                if feat in key:
                    x_value = None
                    if x_single is not None and feat in self.feature_names:
                        x_value = float(x_single[self.feature_names.index(feat)])
                    direction = pick(feat, lime_directions(key, feat, float(val), x_value))
                    if direction != 0:
                        return feat, float(val), direction
                    break
        return None, None, 0


# ------------------------------------------------------------------
# SHAP sign reading (FIX 13)
# ------------------------------------------------------------------
def shap_directions(shap_value, x_value=None, reference=None, rule='toward_mean'):
    """Direction(s) to change a feature so the risk score goes DOWN,
    read from its SHAP value. Empty list = no direction.

    'toward_mean': SHAP > 0 (value raises risk vs. the average patient)
                   -> move towards the average; SHAP < 0 -> move away from it.
                   Needs x_value and reference; falls back to 'sign' without.
    'sign'       : SHAP > 0 -> decrease; SHAP < 0 -> increase.
    """
    if shap_value == 0:
        return []
    if rule == 'toward_mean' and x_value is not None and reference is not None:
        if x_value == reference:
            return []
        towards = 1 if reference > x_value else -1
        return [towards] if shap_value > 0 else [-towards]
    return [-1] if shap_value > 0 else [1]


# ------------------------------------------------------------------
# LIME condition parsing (FIX 2)
# ------------------------------------------------------------------
_NUM = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)'


def lime_direction(condition, feature, weight, x_value=None):
    """Preferred risk-lowering direction (-1/+1), or 0. See lime_directions."""
    dirs = lime_directions(condition, feature, weight, x_value)
    return dirs[0] if dirs else 0


def lime_directions(condition, feature, weight, x_value=None):
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
    Returns the list of directions that leave a risk-raising bin / move
    further into a protective one, PREFERRED FIRST (for range bins both
    edges leave the bin: nearer edge first). Empty list = no direction.
    """
    import re
    if weight == 0:
        return []
    f = re.escape(feature)
    cond = condition.strip()

    # Range bin: a < f <= b  (LIME's format; also accept other <, <= mixes)
    m = re.fullmatch(_NUM + r'\s*<=?\s*' + f + r'\s*<=?\s*' + _NUM, cond)
    if m:
        if weight < 0:
            return []
        lo, hi = float(m.group(1)), float(m.group(2))
        if x_value is None:
            return [-1, 1]
        return [-1, 1] if (x_value - lo) <= (hi - x_value) else [1, -1]

    # One-sided bin: f > a, f >= a, f < a, f <= a
    m = re.fullmatch(f + r'\s*(>=|>|<=|<)\s*' + _NUM, cond)
    if m:
        op = m.group(1)
        if op.startswith('>'):          # in the HIGH bin
            return [-1] if weight > 0 else [1]
        else:                           # in the LOW bin
            return [1] if weight > 0 else [-1]

    return []
