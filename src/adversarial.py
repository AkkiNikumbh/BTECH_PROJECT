"""
Adversarial simulation engine — the core of the experiment.

Implements two strategies:
1. EXPLANATION-GUIDED: Uses SHAP/LIME to decide which feature to change
2. BLIND TRIAL-AND-ERROR: Randomly perturbs features within lie budget

Compares both strategies on the same patients to measure
how much an explanation is worth in blind-attempt equivalents.
"""
import numpy as np
from tqdm import tqdm
from src.config import SEED_ADVERSARIAL, N_BLIND_ATTEMPTS


class AdversarialSimulator:
    """
    Simulates an adversary trying to reduce their diabetes risk score
    by lying about their self-reported features.
    """

    def __init__(self, model, explanation_generator, lie_budget,
                 gameable_features, feature_names):
        """
        Parameters
        ----------
        model : DiabetesModel — the model being attacked
        explanation_generator : ExplanationGenerator instance
        lie_budget : dict — {feature: {'max_lie': float, 'mean_direction': str}}
        gameable_features : list — features the adversary can change
        feature_names : list — all feature names in order
        """
        self.model = model
        self.explainer = explanation_generator
        self.lie_budget = lie_budget
        self.gameable_features = gameable_features
        self.feature_names = feature_names
        self.rng = np.random.RandomState(SEED_ADVERSARIAL)

    def get_feature_index(self, feature_name):
        """Get the index of a feature by name."""
        return self.feature_names.index(feature_name)

    def _get_budget_key(self, feature_name):
        """Map a feature name to its lie budget key (height or weight)."""
        if 'height' in feature_name.lower():
            return 'height'
        elif 'weight' in feature_name.lower():
            return 'weight'
        else:
            return feature_name

    # =========================================================
    # STRATEGY 1: EXPLANATION-GUIDED
    # =========================================================

    def explanation_guided_attack(self, x_original, explanation_method='shap'):
        """
        Use an explanation to determine which gameable feature to change
        and in which direction to REDUCE the diabetes risk score.

        Steps:
        1. Get explanation (SHAP or LIME)
        2. Find the top gameable feature (highest absolute attribution)
        3. Change it in the direction that reduces risk
        4. Constrain the change by the realistic lie budget

        Returns
        -------
        dict: new_x, original_score, new_score, score_reduction,
              changed_feature, change_amount, method
        """
        original_score = float(self.model.predict_proba(
            x_original.reshape(1, -1)
        )[0])

        # Get feature attributions
        if explanation_method == 'shap':
            attributions = self.explainer.get_shap_explanation(x_original)
        else:
            attributions = self.explainer.get_lime_explanation(x_original)

        # Find the most important gameable feature
        top_feat, top_val = self.explainer.get_top_gameable_feature(
            attributions, self.gameable_features
        )

        if top_feat is None:
            return {
                'new_x': x_original.copy(),
                'original_score': original_score,
                'new_score': original_score,
                'score_reduction': 0.0,
                'changed_feature': None,
                'change_amount': 0.0,
                'method': explanation_method,
            }

        feat_idx = self.get_feature_index(top_feat)
        budget_key = self._get_budget_key(top_feat)
        max_change = self.lie_budget[budget_key]['max_lie']

        # Direction: positive attribution → feature pushes score UP → decrease it
        top_val = float(top_val)
        direction = -1 if top_val > 0 else 1
        change = direction * max_change

        new_x = x_original.copy()
        new_x[feat_idx] += change

        new_score = float(self.model.predict_proba(new_x.reshape(1, -1))[0])

        return {
            'new_x': new_x,
            'original_score': original_score,
            'new_score': new_score,
            'score_reduction': original_score - new_score,
            'changed_feature': top_feat,
            'change_amount': change,
            'method': explanation_method,
        }

    # =========================================================
    # STRATEGY 2: BLIND TRIAL-AND-ERROR (statistically correct)
    # =========================================================

    def blind_attack_multi_n(self, x_original, blind_attempts_list):
        """
        Run ONE continuous blind simulation up to max(blind_attempts_list),
        tracking the best score found SO FAR at each N milestone.

        WHY THIS IS CORRECT:
        The adversary accumulates attempts sequentially and always keeps the
        best result seen so far. At N=10 they have also seen N=1..9 results.
        Running separate independent simulations per N would OVER-estimate
        blind performance (each fresh run gets a new lucky random draw),
        making N* (equivalence) artificially inflated and wrong.

        Parameters
        ----------
        x_original : np.array — patient feature vector
        blind_attempts_list : list of int — N milestones to record at

        Returns
        -------
        dict: {n: score_reduction} for each n in blind_attempts_list
        """
        original_score = float(self.model.predict_proba(
            x_original.reshape(1, -1)
        )[0])

        max_n = max(blind_attempts_list)
        milestones = set(blind_attempts_list)

        best_score = original_score
        reductions_at_n = {}

        for attempt in range(1, max_n + 1):
            # Random feature and perturbation within lie budget
            feat = self.rng.choice(self.gameable_features)
            feat_idx = self.get_feature_index(feat)
            budget_key = self._get_budget_key(feat)
            max_change = self.lie_budget[budget_key]['max_lie']

            change = self.rng.uniform(-max_change, max_change)
            trial_x = x_original.copy()
            trial_x[feat_idx] += change

            trial_score = float(self.model.predict_proba(
                trial_x.reshape(1, -1)
            )[0])

            # Keep best score seen so far (adversary is rational)
            if trial_score < best_score:
                best_score = trial_score

            # Record the best reduction achieved up to this N
            if attempt in milestones:
                reductions_at_n[attempt] = original_score - best_score

        return reductions_at_n

    # =========================================================
    # MAIN EXPERIMENT RUNNER
    # =========================================================

    def run_comparison(self, X_test, n_patients=None,
                       blind_attempts_list=None,
                       explanation_method='shap'):
        """
        Run the full comparison experiment for each patient.

        For each patient:
        1. Run explanation-guided attack (1 explanation call)
        2. Run ONE continuous blind simulation, recording best
           score reduction at each N milestone
        3. Record all results for equivalence metric computation

        Returns
        -------
        results : list of dicts, one per patient
        """
        if blind_attempts_list is None:
            blind_attempts_list = N_BLIND_ATTEMPTS

        if n_patients is None:
            n_patients = len(X_test)
        else:
            n_patients = min(n_patients, len(X_test))

        results = []

        for i in tqdm(range(n_patients),
                      desc=f"Adversarial sim ({explanation_method})"):
            x = X_test[i]

            # Strategy 1: 1 explanation → best gameable feature change
            exp_result = self.explanation_guided_attack(x, explanation_method)

            # Strategy 2: continuous blind up to max_N, track best at each N
            blind_reductions = self.blind_attack_multi_n(x, blind_attempts_list)

            results.append({
                'patient_idx': i,
                'original_score': exp_result['original_score'],
                'explanation_reduction': exp_result['score_reduction'],
                'explanation_feature': exp_result['changed_feature'],
                'blind_reductions': blind_reductions,
            })

        return results
