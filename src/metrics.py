"""
Metrics computation for the adversarial experiment.
Computes the key metric: '1 explanation ≈ X blind attempts'
along with confidence intervals and statistical tests.

PRIMARY metric (pre-registered, docs/analysis_plan.md §8):
    N* = the number of blind attempts at which the MEAN blind score-reduction
    curve (averaged over all patients) reaches the MEAN explanation-guided
    score reduction. -> summary['nstar'], CI in ci['nstar_ci'].

SECONDARY metrics (§9): per-patient N* values, their mean / median,
    % of patients where the explanation is useless (N* <= 1) or worth 10+.
    -> summary['mean_equivalence'], ['median_equivalence'], ci['mean_ci'] ...
"""
import numpy as np
from scipy import interpolate, stats


def _monotone(rs):
    """Running maximum, so the blind curve never goes down."""
    out, running = [], 0.0
    for r in rs:
        running = max(running, r)
        out.append(running)
    return out


def _nstar_from_curve(r_exp, ns, rs):
    """
    Find N* where the (monotone) blind curve rs(ns) reaches r_exp.

    Rules (same for per-patient and population N*, as pre-registered):
      * r_exp <= 0 or r_exp <= blind(N=1)  -> 0   (explanation no better
        than a single blind attempt)
      * r_exp >= blind(N=max)              -> max(N) + 1  (censored)
      * otherwise linear interpolation between the two bracketing N values
    """
    rs = _monotone(rs)
    if r_exp <= 0 or r_exp <= rs[0]:
        return 0.0
    if r_exp >= rs[-1]:
        return float(max(ns) + 1)
    # unique (r, n) pairs — keep the first N at which each r value is reached
    seen = {}
    for r_val, n_val in zip(rs, ns):
        if r_val not in seen:
            seen[r_val] = n_val
    rs_uniq = sorted(seen.keys())
    ns_uniq = [seen[r] for r in rs_uniq]
    f = interpolate.interp1d(rs_uniq, ns_uniq, kind='linear',
                             bounds_error=False,
                             fill_value=(ns_uniq[0], ns_uniq[-1]))
    return max(0.0, float(f(r_exp)))


def population_nstar(results, blind_attempts_list=None):
    """
    PRE-REGISTERED primary metric (analysis_plan.md §8, steps 1-5):
    average the explanation reductions and the blind curve over ALL
    patients first, then interpolate ONE N* on the averaged curve.
    """
    if not results:
        return 0.0
    ns = sorted(blind_attempts_list or results[0]['blind_reductions'].keys())
    mean_exp = float(np.mean([r['explanation_reduction'] for r in results]))
    mean_curve = [float(np.mean([r['blind_reductions'][n] for r in results]))
                  for n in ns]
    return _nstar_from_curve(mean_exp, ns, mean_curve)


def compute_equivalence(results, blind_attempts_list):
    """
    Compute the PRIMARY population N* (pre-registered) and the SECONDARY
    per-patient N* values.

    Returns
    -------
    equivalences : np.array of per-patient N* values (secondary)
    summary : dict — 'nstar' is the primary pre-registered metric
    """
    ns = sorted(blind_attempts_list)

    equivalences = []
    for result in results:
        rs = [result['blind_reductions'][n] for n in ns]
        try:
            equivalences.append(_nstar_from_curve(result['explanation_reduction'], ns, rs))
        except Exception:
            equivalences.append(np.nan)
    equivalences = np.array(equivalences, dtype=float)
    valid = equivalences[~np.isnan(equivalences)]

    nstar = population_nstar(results, ns)
    mean_exp = float(np.mean([r['explanation_reduction'] for r in results])) if results else 0.0
    mean_curve = {n: float(np.mean([r['blind_reductions'][n] for r in results]))
                  for n in ns} if results else {}

    summary = {
        # PRIMARY (pre-registered)
        'nstar': nstar,
        'mean_explanation_reduction': mean_exp,
        'mean_blind_curve': mean_curve,
        # SECONDARY (per-patient)
        'mean_equivalence': np.mean(valid) if len(valid) > 0 else 0,
        'median_equivalence': np.median(valid) if len(valid) > 0 else 0,
        'std_equivalence': np.std(valid) if len(valid) > 0 else 0,
        'pct_explanation_useless': (valid <= 1).mean() * 100 if len(valid) > 0 else 0,
        'pct_explanation_worth_10plus': (valid >= 10).mean() * 100 if len(valid) > 0 else 0,
        'pct_censored': (valid > max(ns)).mean() * 100 if len(valid) > 0 else 0,
        'n_patients': len(valid),
    }

    print(f"\n{'='*60}")
    print(f"  MAIN RESULT: EXPLANATION EQUIVALENCE")
    print(f"{'='*60}")
    print(f"  PRIMARY  N* (pre-registered, mean curve): 1 explanation ≈ {nstar:.1f} blind attempts")
    print(f"  Mean explanation reduction: {mean_exp:.4f}")
    print(f"  Secondary, per-patient N*: mean {summary['mean_equivalence']:.1f} | "
          f"median {summary['median_equivalence']:.1f} | std {summary['std_equivalence']:.1f}")
    print(f"  Explanation useless (≤1 attempt): {summary['pct_explanation_useless']:.1f}%")
    print(f"  Explanation worth 10+: {summary['pct_explanation_worth_10plus']:.1f}%")
    print(f"  Censored (> {max(ns)} attempts): {summary['pct_censored']:.1f}%")
    print(f"  N patients: {summary['n_patients']}")

    return equivalences, summary


def compute_confidence_intervals(equivalences, n_bootstrap=1000, confidence=0.95,
                                 results=None, blind_attempts_list=None):
    """
    Bootstrap confidence intervals (resampling PATIENTS).

    Returns
    -------
    dict with
      'nstar_ci'  — CI of the PRIMARY population N* (only if `results` given)
      'mean_ci', 'median_ci' — CIs of the per-patient N* mean / median
    """
    alpha = 1 - confidence
    out = {'mean_ci': (0, 0), 'median_ci': (0, 0), 'nstar_ci': (0, 0)}

    valid = equivalences[~np.isnan(equivalences)]
    rng = np.random.RandomState(42)
    if len(valid) > 0:
        boot_means, boot_medians = [], []
        for _ in range(n_bootstrap):
            sample = rng.choice(valid, size=len(valid), replace=True)
            boot_means.append(np.mean(sample))
            boot_medians.append(np.median(sample))
        out['mean_ci'] = (np.percentile(boot_means, 100 * alpha / 2),
                          np.percentile(boot_means, 100 * (1 - alpha / 2)))
        out['median_ci'] = (np.percentile(boot_medians, 100 * alpha / 2),
                            np.percentile(boot_medians, 100 * (1 - alpha / 2)))

    if results:
        ns = sorted(blind_attempts_list or results[0]['blind_reductions'].keys())
        exp = np.array([r['explanation_reduction'] for r in results], dtype=float)
        curves = np.array([[r['blind_reductions'][n] for n in ns] for r in results],
                          dtype=float)
        rng_p = np.random.RandomState(43)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng_p.randint(0, len(results), size=len(results))
            boot.append(_nstar_from_curve(float(exp[idx].mean()), ns,
                                          list(curves[idx].mean(axis=0))))
        out['nstar_ci'] = (np.percentile(boot, 100 * alpha / 2),
                           np.percentile(boot, 100 * (1 - alpha / 2)))
        print(f"  95% CI (PRIMARY N*): [{out['nstar_ci'][0]:.2f}, {out['nstar_ci'][1]:.2f}]")

    print(f"  95% CI (per-patient mean):   [{out['mean_ci'][0]:.2f}, {out['mean_ci'][1]:.2f}]")
    print(f"  95% CI (per-patient median): [{out['median_ci'][0]:.2f}, {out['median_ci'][1]:.2f}]")
    return out


def run_statistical_tests(all_results):
    """
    Run all pre-registered statistical tests.

    For each configuration, compare explanation vs blind(N=10)
    using paired Wilcoxon signed-rank test and Cohen's d.

    Returns
    -------
    dict of test results per configuration
    """
    tests = {}

    for run_key, data in all_results.items():
        results = data['results']
        exp_reds = np.array([r['explanation_reduction'] for r in results])

        # Compare against blind with 10 attempts
        if 10 in results[0]['blind_reductions']:
            blind_10 = np.array([r['blind_reductions'][10] for r in results])
        else:
            # Use the closest available
            available = sorted(results[0]['blind_reductions'].keys())
            closest = min(available, key=lambda x: abs(x - 10))
            blind_10 = np.array([r['blind_reductions'][closest] for r in results])

        diff = exp_reds - blind_10
        # Paired Wilcoxon signed-rank test. If every difference is zero
        # (e.g. Model A, which cannot be attacked) there is nothing to test:
        # report p = 1.0 instead of NaN.
        if np.all(diff == 0):
            stat, pval = 0.0, 1.0
        else:
            try:
                stat, pval = stats.wilcoxon(exp_reds, blind_10, alternative='greater')
                if np.isnan(pval):
                    stat, pval = 0.0, 1.0
            except Exception:
                stat, pval = 0.0, 1.0

        # Effect size (Cohen's d)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0

        tests[run_key] = {
            'wilcoxon_stat': stat,
            'wilcoxon_pval': pval,
            'cohens_d': cohens_d,
            'significant': pval < 0.05,
        }

        sig = "✅ YES" if pval < 0.05 else "❌ NO"
        print(f"  {run_key}: p={pval:.6f}, Cohen's d={cohens_d:.3f} → Significant: {sig}")

    return tests
