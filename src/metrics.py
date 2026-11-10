"""
Metrics computation for the adversarial experiment.
Computes the key metric: '1 explanation ≈ X blind attempts'
along with confidence intervals and statistical tests.
"""
import numpy as np
from scipy import interpolate, stats


def compute_equivalence(results, blind_attempts_list):
    """
    For each patient, find how many blind attempts match the
    explanation-guided score reduction.

    Method:
    - The explanation gives a score reduction R_exp
    - Blind with N attempts gives R_blind(N)
    - Find N* such that R_blind(N*) ≈ R_exp via interpolation
    - N* is the 'explanation equivalent' for this patient

    Returns
    -------
    equivalences : np.array of N* values (one per patient)
    summary : dict with mean, median, std, etc.
    """
    equivalences = []

    for result in results:
        r_exp = result['explanation_reduction']

        # Build the curve: N → R_blind(N)
        # blind_attack_multi_n guarantees non-decreasing reductions (best-so-far)
        ns = sorted(result['blind_reductions'].keys())
        rs = [result['blind_reductions'][n] for n in ns]

        # Enforce monotonicity explicitly (cumulative max)
        rs_mono = []
        running_max = 0.0
        for r in rs:
            running_max = max(running_max, r)
            rs_mono.append(running_max)

        if r_exp <= 0:
            equivalences.append(0)
            continue

        if r_exp <= rs_mono[0]:
            # Explanation does no better than 1 blind attempt
            equivalences.append(0)
            continue

        if r_exp >= rs_mono[-1]:
            # Explanation better than max blind attempts → censored at max_N+1
            equivalences.append(float(max(ns) + 1))
            continue

        # Interpolate: rs_mono is strictly non-decreasing, deduplicate for interp
        try:
            # Build unique (r, n) pairs — keep first occurrence per r value
            seen_r = {}
            for r_val, n_val in zip(rs_mono, ns):
                if r_val not in seen_r:
                    seen_r[r_val] = n_val
            rs_uniq = sorted(seen_r.keys())
            ns_uniq = [seen_r[r] for r in rs_uniq]

            f = interpolate.interp1d(rs_uniq, ns_uniq, kind='linear',
                                     bounds_error=False,
                                     fill_value=(ns_uniq[0], ns_uniq[-1]))
            n_star = float(f(r_exp))
            equivalences.append(max(0.0, n_star))
        except Exception:
            equivalences.append(np.nan)

    equivalences = np.array(equivalences)
    valid = equivalences[~np.isnan(equivalences)]

    summary = {
        'mean_equivalence': np.mean(valid) if len(valid) > 0 else 0,
        'median_equivalence': np.median(valid) if len(valid) > 0 else 0,
        'std_equivalence': np.std(valid) if len(valid) > 0 else 0,
        'pct_explanation_useless': (valid <= 1).mean() * 100 if len(valid) > 0 else 0,
        'pct_explanation_worth_10plus': (valid >= 10).mean() * 100 if len(valid) > 0 else 0,
        'n_patients': len(valid),
    }

    print(f"\n{'='*60}")
    print(f"  MAIN RESULT: EXPLANATION EQUIVALENCE")
    print(f"{'='*60}")
    print(f"  1 explanation ≈ {summary['mean_equivalence']:.1f} blind attempts (mean)")
    print(f"  1 explanation ≈ {summary['median_equivalence']:.1f} blind attempts (median)")
    print(f"  Std dev: {summary['std_equivalence']:.1f}")
    print(f"  Explanation useless (≤1 attempt): {summary['pct_explanation_useless']:.1f}%")
    print(f"  Explanation worth 10+: {summary['pct_explanation_worth_10plus']:.1f}%")
    print(f"  N patients: {summary['n_patients']}")

    return equivalences, summary


def compute_confidence_intervals(equivalences, n_bootstrap=1000, confidence=0.95):
    """
    Bootstrap confidence intervals for the equivalence metric.

    Returns
    -------
    dict with mean_ci and median_ci tuples
    """
    valid = equivalences[~np.isnan(equivalences)]
    if len(valid) == 0:
        return {'mean_ci': (0, 0), 'median_ci': (0, 0)}

    boot_means = []
    boot_medians = []

    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(valid, size=len(valid), replace=True)
        boot_means.append(np.mean(sample))
        boot_medians.append(np.median(sample))

    alpha = 1 - confidence
    ci_mean = (np.percentile(boot_means, 100 * alpha / 2),
               np.percentile(boot_means, 100 * (1 - alpha / 2)))
    ci_median = (np.percentile(boot_medians, 100 * alpha / 2),
                 np.percentile(boot_medians, 100 * (1 - alpha / 2)))

    print(f"  95% CI (mean):   [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
    print(f"  95% CI (median): [{ci_median[0]:.2f}, {ci_median[1]:.2f}]")

    return {'mean_ci': ci_mean, 'median_ci': ci_median}


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

        # Paired Wilcoxon signed-rank test
        try:
            stat, pval = stats.wilcoxon(exp_reds, blind_10, alternative='greater')
        except Exception:
            stat, pval = 0, 1.0

        # Effect size (Cohen's d)
        diff = exp_reds - blind_10
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
