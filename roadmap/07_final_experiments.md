# 🚀 Stage 7: Final Experiments on 2017–2018 Data
## Period: 8 January – 7 March 2027 (8 weeks)

> **← Previous**: [Stage 6 — Pre-Registration](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/06_preregistration.md)
> **← Back to** [Master Roadmap](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/00_master_roadmap.md)
> **→ Next Stage**: [Stage 8 — Analysis & Graphs](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/08_analysis.md)

---

> [!IMPORTANT]
> This is where the **held-out 2017–2018 data** is used for the first and only time. The pre-registered analysis plan (Stage 6) is now locked. Follow it exactly — **no deviations**.

---

## 🎯 Objectives

1. Process the 2017–2018 NHANES data using the exact same pipeline developed for 2015–2016
2. Re-compute the reported-vs-measured gap from 2017–2018 (compare with 2015–2016 gap)
3. Train fresh models on 2017–2018 data with locked hyperparameters
4. Execute all adversarial simulations as specified in the pre-registered plan
5. Compute the **final** equivalence metric and all secondary metrics
6. Run all pre-specified statistical tests

---

## 📋 Prerequisites (from Stage 6)

- [x] Pre-registered analysis plan published with timestamp
- [x] All experiment parameters locked in `FINAL_CONFIG`
- [x] Full pipeline verified on 2015–2016 data
- [x] 2017–2018 data downloaded but completely untouched

---

## 🔨 Step-by-Step Implementation

### Task 7.1 — Process 2017–2018 Data

```python
"""
scripts/run_final.py
THE FINAL EXPERIMENT — run exactly as pre-registered.
"""
import sys, os, pickle, time, json
import numpy as np
sys.path.append('..')

from src.config import set_all_seeds, DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES, FINAL_CONFIG
from src.data_loader import merge_nhanes, clean_nhanes, create_diabetes_target, prepare_feature_sets
from src.gap_analysis import compute_reporting_gap, define_lie_budget, plot_reporting_gaps
from src.models import train_all_models
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence, compute_confidence_intervals, run_statistical_tests
from src.visualization import *

set_all_seeds()

print("="*70)
print("FINAL EXPERIMENT — NHANES 2017-2018 (HELD-OUT DATA)")
print("="*70)
print(f"Using FINAL_CONFIG (pre-registered, locked)")
print(f"Config: {json.dumps(FINAL_CONFIG, indent=2)}")

# ============================================================
# STEP 1: Process held-out data (FIRST TIME TOUCHING IT)
# ============================================================
print("\n[1/8] Processing 2017-2018 data for the FIRST time...")

df_2018 = merge_nhanes(cycle='2017_2018')
df_2018 = clean_nhanes(df_2018)
df_2018 = create_diabetes_target(df_2018)

# Compute gap on new data and COMPARE with development data
df_2018 = compute_reporting_gap(df_2018)
lie_budget_2018 = define_lie_budget(
    df_2018, 
    method=FINAL_CONFIG['lie_budget_method'],
    pct=FINAL_CONFIG['lie_budget_percentile']
)

# Compare with 2015-2016 gap
with open(os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl'), 'rb') as f:
    dev_data = pickle.load(f)

print("\n  GAP COMPARISON (2015-2016 vs 2017-2018):")
for feature in ['height', 'weight']:
    dev_lie = dev_data['lie_budget'][feature]['max_lie']
    new_lie = lie_budget_2018[feature]['max_lie']
    print(f"    {feature}: dev={dev_lie:.2f} vs held-out={new_lie:.2f} "
          f"(diff={abs(new_lie-dev_lie):.2f})")

# ============================================================
# STEP 2: Prepare features
# ============================================================
print("\n[2/8] Preparing feature sets...")
X_meas, X_rep, y, fn_meas, fn_rep, df_complete = prepare_feature_sets(df_2018, {})

print(f"  Final N = {len(y)}, Diabetes prevalence = {y.mean()*100:.1f}%")

# ============================================================
# STEP 3: Train models (locked hyperparameters)
# ============================================================
print("\n[3/8] Training models with locked hyperparameters...")
models, split = train_all_models(X_meas, X_rep, y, fn_meas, fn_rep)

# ============================================================
# STEP 4: Run ALL adversarial simulations (full patient set)
# ============================================================
print("\n[4/8] Running adversarial simulations (FULL — this may take hours)...")

all_results = {}
start = time.time()

for input_type in ['measured', 'reported']:
    for model_type in FINAL_CONFIG['model_types']:
        model_key = f"{input_type}_{model_type}"
        model = models[model_key]
        
        X = X_meas if input_type == 'measured' else X_rep
        fn = fn_meas if input_type == 'measured' else fn_rep
        X_test = X[split['idx_test']]
        X_train = X[split['idx_train']]
        
        gameable = [f for f in fn if 'height' in f.lower() or 'weight' in f.lower()]
        
        explainer = ExplanationGenerator(model, X_train, fn)
        simulator = AdversarialSimulator(
            model, explainer, lie_budget_2018, gameable, fn
        )
        
        for exp_method in ['shap', 'lime']:
            run_key = f"{model_key}_{exp_method}"
            print(f"\n  Running: {run_key} (ALL test patients)...")
            
            results = simulator.run_comparison(
                X_test,
                n_patients=None,  # ALL patients — no subset
                blind_attempts_list=FINAL_CONFIG['blind_attempts'],
                explanation_method=exp_method
            )
            
            equivalences, summary = compute_equivalence(
                results, FINAL_CONFIG['blind_attempts']
            )
            
            # Confidence intervals (pre-registered)
            ci = compute_confidence_intervals(
                equivalences,
                n_bootstrap=FINAL_CONFIG['bootstrap_n_iterations'],
                confidence=FINAL_CONFIG['confidence_level']
            )
            
            all_results[run_key] = {
                'results': results,
                'equivalences': equivalences,
                'summary': summary,
                'confidence_interval': ci,
            }

elapsed = time.time() - start
print(f"\n  Total computation time: {elapsed/3600:.1f} hours")

# ============================================================
# STEP 5: Statistical tests
# ============================================================
print("\n[5/8] Running statistical tests...")
stat_results = run_statistical_tests(all_results)

# ============================================================
# STEP 6-8: Save everything (results, figures, tables)
# ============================================================
# ... (save all results to disk)

print("\n✅ FINAL EXPERIMENT COMPLETE")
```

### Task 7.2 — Confidence Intervals and Statistical Tests

```python
"""
src/metrics.py (additions for final experiment)
"""
from scipy import stats

def compute_confidence_intervals(equivalences, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence intervals for the equivalence metric."""
    valid = equivalences[~np.isnan(equivalences)]
    boot_means = []
    boot_medians = []
    
    rng = np.random.RandomState(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(valid, size=len(valid), replace=True)
        boot_means.append(np.mean(sample))
        boot_medians.append(np.median(sample))
    
    alpha = 1 - confidence
    ci_mean = (np.percentile(boot_means, 100*alpha/2),
               np.percentile(boot_means, 100*(1-alpha/2)))
    ci_median = (np.percentile(boot_medians, 100*alpha/2),
                 np.percentile(boot_medians, 100*(1-alpha/2)))
    
    return {
        'mean_ci': ci_mean,
        'median_ci': ci_median,
    }

def run_statistical_tests(all_results):
    """Run all pre-registered statistical tests."""
    tests = {}
    
    # For each model, compare explanation vs blind(N=10)
    for run_key, data in all_results.items():
        results = data['results']
        exp_reds = [r['explanation_reduction'] for r in results]
        blind_10 = [r['blind_reductions'][10] for r in results]
        
        # Paired Wilcoxon signed-rank test
        stat, pval = stats.wilcoxon(exp_reds, blind_10, alternative='greater')
        
        # Effect size (Cohen's d)
        diff = np.array(exp_reds) - np.array(blind_10)
        cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        
        tests[run_key] = {
            'wilcoxon_stat': stat,
            'wilcoxon_pval': pval,
            'cohens_d': cohens_d,
            'significant': pval < 0.05,
        }
        
        print(f"  {run_key}: Wilcoxon p={pval:.6f}, Cohen's d={cohens_d:.3f}")
    
    return tests
```

---

## ⏱️ Expected Runtime

| Task | Estimated Time |
|------|---------------|
| Data processing (2017–2018) | ~5 minutes |
| Model training (4 models) | ~5 minutes |
| SHAP explanations (all test patients) | ~10–30 minutes |
| LIME explanations (all test patients) | ~1–3 hours |
| Adversarial simulation (8 configs × all patients × 10 N values) | ~4–8 hours |
| Statistical tests | ~5 minutes |
| **Total** | **~6–11 hours (leave running overnight)** |

---

## ✅ Completion Checklist

- [ ] 2017–2018 NHANES data processed for the first time
- [ ] Reporting gap compared between 2015–2016 and 2017–2018
- [ ] All 4 models trained with locked hyperparameters
- [ ] All 8 configurations simulated (2 inputs × 2 models × 2 explanations)
- [ ] Full patient set used (no subset)
- [ ] Equivalence metrics computed for each configuration
- [ ] 95% bootstrap confidence intervals computed
- [ ] Wilcoxon signed-rank tests completed
- [ ] Cohen's d effect sizes computed
- [ ] All raw results saved to disk
- [ ] Runtime logged
- [ ] **No deviations from the pre-registered plan**
