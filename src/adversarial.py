"""
Adversarial simulation engine — the core of the experiment.

Implements two strategies:
1. EXPLANATION-GUIDED: Uses SHAP/LIME to decide which feature to change
   and in which direction
2. BLIND TRIAL-AND-ERROR: Randomly perturbs features within lie budget

Compares both strategies on the same patients to measure
how much an explanation is worth in blind-attempt equivalents.

v2 rules (see docs/fixes_v2.md), all set in src/config.py:
  * FIX 6 — both attackers draw the SIZE of each lie from the same rule
            (LIE_MAGNITUDE), so the explanation only contributes information
            (which feature, which direction), not a bigger lie.
  * FIX 7 — lies only go the way people really misreport (LIE_DIRECTION =
            'realistic': height up, weight down, from the 2015-16 gap data).
  * FIX 8 — every patient gets its own fixed random seed, so SHAP and LIME
            (and Model A vs B) are compared against the SAME blind attempts
            and the same lie sizes for each patient.
"""
import numpy as np
from tqdm import tqdm
from src.config import (SEED_ADVERSARIAL, N_BLIND_ATTEMPTS,
                        LIE_DIRECTION, LIE_MAGNITUDE)

# Offset so the explanation's lie-size stream never overlaps a blind stream
_EXP_SEED_OFFSET = 1_000_003
_LIME_SEED_BASE = 42


class AdversarialSimulator:
    """
    Simulates an adversary trying to reduce their diabetes risk score
    by lying about their self-reported features.
    """

    def __init__(self, model, explanation_generator, lie_budget,
                 gameable_features, feature_names,
                 lie_direction=LIE_DIRECTION, lie_magnitude=LIE_MAGNITUDE,
                 seed=SEED_ADVERSARIAL):
        """
        Parameters
        ----------
        model : DiabetesModel — the model being attacked
        explanation_generator : ExplanationGenerator instance
        lie_budget : dict — {feature: {'max_lie': float, 'mean_direction': str}}
        gameable_features : list — features the adversary can change
        feature_names : list — all feature names in order
        lie_direction : 'realistic' or 'both'   (FIX 7)
        lie_magnitude : 'uniform' or 'max'      (FIX 6)
        seed : base random seed; patient i uses seed + i (FIX 8)
        """
        if lie_direction not in ('realistic', 'both'):
            raise ValueError(f"lie_direction must be 'realistic' or 'both', got {lie_direction}")
        if lie_magnitude not in ('uniform', 'max'):
            raise ValueError(f"lie_magnitude must be 'uniform' or 'max', got {lie_magnitude}")
        self.model = model
        self.explainer = explanation_generator
        self.lie_budget = lie_budget
        self.gameable_features = gameable_features
        self.feature_names = feature_names
        self.lie_direction = lie_direction
        self.lie_magnitude = lie_magnitude
        self.seed = seed
        # Used only when no patient index is given (e.g. scripts/demo.py)
        self.rng = np.random.RandomState(seed)

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

    # ---------------------------------------------------------
    # Lie rules (shared by both attackers)
    # ---------------------------------------------------------

    def realistic_direction(self, feature_name):
        """+1 if people tend to OVER-report this feature, -1 if UNDER."""
        info = self.lie_budget[self._get_budget_key(feature_name)]
        return 1 if info.get('mean_direction', 'positive') == 'positive' else -1

    def allowed_directions(self, feature_name):
        """Directions a lie on this feature may take (FIX 7)."""
        if self.lie_direction == 'realistic':
            return [self.realistic_direction(feature_name)]
        return [-1, 1]

    def lie_size(self, feature_name, rng):
        """Size of one lie — same rule for both attackers (FIX 6)."""
        max_lie = float(self.lie_budget[self._get_budget_key(feature_name)]['max_lie'])
        if self.lie_magnitude == 'max':
            return max_lie
        # uniform on (0, max_lie]
        return max_lie * (1.0 - rng.uniform())

    def _rngs(self, patient_idx):
        """(blind_rng, explanation_rng) for this patient (FIX 8)."""
        if patient_idx is None:
            return self.rng, self.rng
        return (np.random.RandomState(self.seed + patient_idx),
                np.random.RandomState(self.seed + _EXP_SEED_OFFSET + patient_idx))

    # =========================================================
    # STRATEGY 1: EXPLANATION-GUIDED
    # =========================================================

    def explanation_guided_attack(self, x_original, explanation_method='shap',
                                  patient_idx=None):
        """
        Use an explanation to decide WHICH gameable feature to change and
        in WHICH direction to reduce the diabetes risk score.

        Steps:
        1. Get explanation (SHAP or LIME)
        2. Go through the gameable features in order of importance and take
           the first one whose risk-lowering direction is an allowed lie
           (realistic direction only, FIX 7)
        3. Lie in that direction; the size follows the same rule as the
           blind attacker (FIX 6), within the realistic lie budget

        Returns
        -------
        dict: new_x, original_score, new_score, score_reduction,
              changed_feature, change_amount, method
        """
        original_score = float(self.model.predict_proba(
            x_original.reshape(1, -1)
        )[0])

        no_change = {
            'new_x': x_original.copy(),
            'original_score': original_score,
            'new_score': original_score,
            'score_reduction': 0.0,
            'changed_feature': None,
            'change_amount': 0.0,
            'method': explanation_method,
        }

        # FIX 1: nothing the patient can lie about (Model A) -> no attack.
        # Skip the (slow) explanation call entirely.
        if not self.gameable_features:
            return no_change

        _, exp_rng = self._rngs(patient_idx)

        # Get feature attributions
        if explanation_method == 'shap':
            attributions = self.explainer.get_shap_explanation(x_original)
        else:
            lime_seed = None if patient_idx is None else _LIME_SEED_BASE + patient_idx
            attributions = self.explainer.get_lime_explanation(x_original, seed=lime_seed)

        # Most important gameable feature whose risk-lowering direction is
        # an allowed lie (FIX 2: direction read from the LIME bin; FIX 7:
        # only realistic directions)
        allowed = {f: self.allowed_directions(f) for f in self.gameable_features}
        top_feat, top_val, direction = self.explainer.get_top_gameable_feature(
            attributions, self.gameable_features, x_single=x_original,
            allowed_directions=allowed
        )

        if top_feat is None:
            # The explanation says no allowed lie would lower the risk
            return no_change

        feat_idx = self.get_feature_index(top_feat)
        change = direction * self.lie_size(top_feat, exp_rng)

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

    def blind_attack_multi_n(self, x_original, blind_attempts_list,
                             patient_idx=None):
        """
        Run ONE continuous blind simulation up to max(blind_attempts_list),
        tracking the best score found SO FAR at each N milestone.

        Each attempt: pick a gameable feature at random, pick an allowed
        direction at random (only the realistic one under FIX 7), and draw
        the lie size with the same rule as the explanation attacker (FIX 6).

        WHY ONE CONTINUOUS RUN:
        The adversary accumulates attempts sequentially and always keeps the
        best result seen so far. At N=10 they have also seen N=1..9 results.

        Parameters
        ----------
        x_original : np.array — patient feature vector
        blind_attempts_list : list of int — N milestones to record at
        patient_idx : int — fixes this patient's random stream (FIX 8)

        Returns
        -------
        dict: {n: score_reduction} for each n in blind_attempts_list
        """
        # FIX 1: a model with no gameable features (Model A, nurse-measured
        # inputs) cannot be attacked — every attempt leaves the score as is.
        if not self.gameable_features:
            return {n: 0.0 for n in blind_attempts_list}

        rng, _ = self._rngs(patient_idx)

        original_score = float(self.model.predict_proba(
            x_original.reshape(1, -1)
        )[0])

        max_n = max(blind_attempts_list)
        milestones = set(blind_attempts_list)

        best_score = original_score
        reductions_at_n = {}

        for attempt in range(1, max_n + 1):
            feat = self.gameable_features[rng.randint(len(self.gameable_features))]
            feat_idx = self.get_feature_index(feat)
            dirs = self.allowed_directions(feat)
            direction = dirs[rng.randint(len(dirs))] if len(dirs) > 1 else dirs[0]
            change = direction * self.lie_size(feat, rng)

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

        For each patient i (with its own random seed, FIX 8):
        1. Run explanation-guided attack (1 explanation)
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

            exp_result = self.explanation_guided_attack(
                x, explanation_method, patient_idx=i)
            blind_reductions = self.blind_attack_multi_n(
                x, blind_attempts_list, patient_idx=i)

            results.append({
                'patient_idx': i,
                'original_score': exp_result['original_score'],
                'explanation_reduction': exp_result['score_reduction'],
                'explanation_feature': exp_result['changed_feature'],
                'explanation_change': exp_result['change_amount'],
                'blind_reductions': blind_reductions,
            })

        return results
