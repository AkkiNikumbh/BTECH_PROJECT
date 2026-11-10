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

    def get_top_gameable_feature(self, attributions, gameable_features):
        """
        Find the most influential feature that the adversary CAN change.

        SHAP returns exact feature names as keys.
        LIME returns condition strings like:
          'self_height_cm > 1.23'
          '0.5 < self_weight_kg <= 2.1'
          'self_weight_kg <= 0.8'
        We match by checking if any gameable feature name appears
        as a substring of the attribution key (handles all LIME formats).

        Parameters
        ----------
        attributions : dict of {feature_or_condition: importance}
        gameable_features : list of exact feature names that can be changed

        Returns
        -------
        (feature_name, attribution_value) or (None, None)
        """
        for attr_key, val in attributions.items():
            # Direct match (SHAP exact names)
            if attr_key in gameable_features:
                return attr_key, val
            # Substring match (LIME condition strings)
            for feat in gameable_features:
                if feat in str(attr_key):
                    return feat, val
        return None, None
