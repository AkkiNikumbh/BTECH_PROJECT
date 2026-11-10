# 🧪 Stage 5: Trial Run on 2015–2016 Data
## Period: 1 December – 14 December 2026 (2 weeks)

> **← Previous**: [Stage 4 — Experiment Code](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/04_experiment_code.md)
> **← Back to** [Master Roadmap](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/00_master_roadmap.md)
> **→ Next Stage**: [Stage 6 — Pre-Registration](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/06_preregistration.md)

---

## 🎯 Objectives

1. Execute the full experimental pipeline end-to-end on **2015–2016 data** (the development set)
2. Produce the first real adversarial simulation results and graphs
3. Identify and fix any bugs, edge cases, or performance bottlenecks
4. Determine optimal hyperparameters and experiment settings for the final run
5. Produce preliminary results to calibrate expectations for the pre-registration (Stage 6)

> [!IMPORTANT]
> This is a **trial run**, not the final experiment. The 2015–2016 data is the development set. Results from this stage inform the analysis plan but are NOT the final reported results. The final experiment (Stage 7) uses the held-out 2017–2018 data.

---

## 📋 Prerequisites (from Stage 4)

- [x] All 4 models trainable and evaluated
- [x] SHAP/LIME explanation generation working
- [x] Adversarial simulation engine implemented (both strategies)
- [x] Equivalence metric calculator ready
- [x] All unit tests passing

---

## 🔨 Step-by-Step Implementation

### Task 5.1 — Full Trial Run Script

```python
"""
scripts/run_trial.py
Execute the complete experimental pipeline on 2015-2016 NHANES data.
This is the trial run — results here inform the pre-registration.
"""
import sys
import os
import pickle
import time
import json
sys.path.append('..')

from src.config import set_all_seeds, DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES
from src.models import train_all_models
from src.explanations import ExplanationGenerator
from src.adversarial import AdversarialSimulator
from src.metrics import compute_equivalence
from src.visualization import (
    plot_equivalence_curve, plot_score_reductions, 
    plot_model_comparison, plot_shap_summary
)

# ============================================================
# CONFIGURATION
# ============================================================
set_all_seeds()

TRIAL_CONFIG = {
    'n_patients': 200,          # Subset for trial (full run = all)
    'explanation_methods': ['shap', 'lime'],
    'blind_attempts': [1, 2, 3, 5, 10, 15, 20, 30, 50, 100],
    'model_types': ['rf', 'lr'],
    'input_types': ['measured', 'reported'],
}

print("="*60)
print("TRIAL RUN — NHANES 2015-2016")
print("="*60)
print(f"Config: {json.dumps(TRIAL_CONFIG, indent=2)}")

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n[1/6] Loading prepared data...")
with open(os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl'), 'rb') as f:
    data = pickle.load(f)

X_measured = data['X_measured']
X_reported = data['X_reported']
y = data['y']
fn_measured = data['feature_names_measured']
fn_reported = data['feature_names_reported']
lie_budget = data['lie_budget']

print(f"  Patients: {len(y)}")
print(f"  Diabetes prevalence: {y.mean()*100:.1f}%")
print(f"  Lie budget: {lie_budget}")

# ============================================================
# STEP 2: TRAIN MODELS
# ============================================================
print("\n[2/6] Training all models...")
models, split = train_all_models(X_measured, X_reported, y, fn_measured, fn_reported)

# ============================================================
# STEP 3: GENERATE EXPLANATIONS & RUN SIMULATION
# ============================================================
print("\n[3/6] Running adversarial simulations...")

all_results = {}
start_time = time.time()

for input_type in TRIAL_CONFIG['input_types']:
    for model_type in TRIAL_CONFIG['model_types']:
        model_key = f"{input_type}_{model_type}"
        model = models[model_key]
        
        X = X_measured if input_type == 'measured' else X_reported
        fn = fn_measured if input_type == 'measured' else fn_reported
        X_test = X[split['idx_test']]
        X_train = X[split['idx_train']]
        
        gameable = [f for f in fn if 'height' in f.lower() or 'weight' in f.lower()]
        
        explainer = ExplanationGenerator(model, X_train, fn)
        simulator = AdversarialSimulator(
            model, explainer, lie_budget, gameable, fn
        )
        
        for exp_method in TRIAL_CONFIG['explanation_methods']:
            run_key = f"{model_key}_{exp_method}"
            print(f"\n  Running: {run_key}")
            
            results = simulator.run_comparison(
                X_test,
                n_patients=TRIAL_CONFIG['n_patients'],
                blind_attempts_list=TRIAL_CONFIG['blind_attempts'],
                explanation_method=exp_method
            )
            
            equivalences, summary = compute_equivalence(
                results, TRIAL_CONFIG['blind_attempts']
            )
            
            all_results[run_key] = {
                'results': results,
                'equivalences': equivalences,
                'summary': summary,
            }

elapsed = time.time() - start_time
print(f"\n  Total simulation time: {elapsed/60:.1f} minutes")

# ============================================================
# STEP 4: GENERATE VISUALIZATIONS
# ============================================================
print("\n[4/6] Generating plots...")
os.makedirs(RESULTS_FIGURES, exist_ok=True)

# Plot 1: Equivalence curves for each model
for run_key, data in all_results.items():
    plot_equivalence_curve(
        data['results'], 
        TRIAL_CONFIG['blind_attempts'],
        title=f"Equivalence Curve — {run_key}",
        save_path=os.path.join(RESULTS_FIGURES, f'equiv_{run_key}.png')
    )

# Plot 2: Cross-model comparison
plot_model_comparison(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'model_comparison.png')
)

# ============================================================
# STEP 5: SAVE RESULTS
# ============================================================
print("\n[5/6] Saving results...")
os.makedirs(RESULTS_TABLES, exist_ok=True)

# Summary table
import pandas as pd
summary_rows = []
for run_key, data in all_results.items():
    s = data['summary']
    parts = run_key.split('_')
    summary_rows.append({
        'Input Type': parts[0],
        'Model': parts[1].upper(),
        'Explanation': parts[2].upper(),
        'Mean Equiv.': f"{s['mean_equivalence']:.1f}",
        'Median Equiv.': f"{s['median_equivalence']:.1f}",
        '% Useless (≤1)': f"{s['pct_explanation_useless']:.1f}%",
        '% Worth 10+': f"{s['pct_explanation_worth_10plus']:.1f}%",
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(os.path.join(RESULTS_TABLES, 'trial_summary.csv'), index=False)
print(df_summary.to_string(index=False))

# ============================================================
# STEP 6: PRELIMINARY CONCLUSIONS
# ============================================================
print("\n[6/6] Preliminary conclusions...")
print("="*60)

# Find the median equivalence across all runs
all_equivs = [d['summary']['median_equivalence'] for d in all_results.values()]
overall_median = np.median(all_equivs)

print(f"\n  OVERALL MEDIAN EQUIVALENCE: {overall_median:.1f} blind attempts")
print(f"\n  INTERPRETATION:")
if overall_median < 5:
    print("  → Explanations provide SMALL advantage")
    print("  → Consistent with hypothesis: hiding explanations is not useful")
elif overall_median < 20:
    print("  → Explanations provide MODERATE advantage")
    print("  → Worth investigating which features drive this")
else:
    print("  → Explanations provide LARGE advantage")
    print("  → This contradicts our initial hypothesis")
    
print(f"\n  These are TRIAL results (2015-2016 data).")
print(f"  Final results will use 2017-2018 held-out data.")
print(f"\n✅ Trial run complete!")
```

### Task 5.2 — Visualizations to Generate

```python
"""
src/visualization.py
Plotting functions for the experiment results.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_equivalence_curve(results, blind_attempts_list, title, save_path):
    """
    Plot the average score reduction vs. number of blind attempts,
    with a horizontal line showing the explanation-guided reduction.
    
    This is THE key visualization of the project.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Average explanation reduction
    exp_reductions = [r['explanation_reduction'] for r in results]
    mean_exp = np.mean(exp_reductions)
    
    # Average blind reduction for each N
    blind_means = []
    blind_stds = []
    for n in blind_attempts_list:
        reductions = [r['blind_reductions'][n] for r in results]
        blind_means.append(np.mean(reductions))
        blind_stds.append(np.std(reductions))
    
    # Plot blind curve
    ax.plot(blind_attempts_list, blind_means, 'b-o', linewidth=2, 
            markersize=6, label='Blind Trial-and-Error', zorder=3)
    ax.fill_between(blind_attempts_list,
                     np.array(blind_means) - np.array(blind_stds),
                     np.array(blind_means) + np.array(blind_stds),
                     alpha=0.2, color='blue')
    
    # Plot explanation line
    ax.axhline(y=mean_exp, color='red', linestyle='--', linewidth=2,
               label=f'Explanation-Guided (mean = {mean_exp:.4f})')
    
    # Find crossing point
    for i in range(len(blind_means) - 1):
        if blind_means[i] <= mean_exp <= blind_means[i+1]:
            # Linear interpolation
            frac = (mean_exp - blind_means[i]) / (blind_means[i+1] - blind_means[i])
            crossing_n = blind_attempts_list[i] + frac * (
                blind_attempts_list[i+1] - blind_attempts_list[i]
            )
            ax.axvline(x=crossing_n, color='green', linestyle=':', linewidth=2,
                       label=f'Equivalence point: {crossing_n:.1f} attempts')
            ax.plot(crossing_n, mean_exp, 'g*', markersize=15, zorder=5)
            break
    
    ax.set_xlabel('Number of Blind Attempts', fontsize=12)
    ax.set_ylabel('Mean Score Reduction', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_score_reductions(results, save_path):
    """Box plots comparing score reductions across strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    exp_reds = [r['explanation_reduction'] for r in results]
    data_to_plot = [exp_reds]
    labels = ['Explanation']
    
    for n in [1, 5, 10, 20, 50, 100]:
        if n in results[0]['blind_reductions']:
            reds = [r['blind_reductions'][n] for r in results]
            data_to_plot.append(reds)
            labels.append(f'Blind (N={n})')
    
    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['#e74c3c'] + ['#3498db'] * (len(labels) - 1)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Score Reduction', fontsize=12)
    ax.set_title('Score Reduction: Explanation vs. Blind Attempts', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
```

---

## 📊 What to Look for in Trial Results

| Observation | What It Means | Action |
|-------------|--------------|--------|
| Equivalence ≈ 2–5 attempts | Explanations give small advantage | Consistent with hypothesis; proceed |
| Equivalence > 50 attempts | Explanations very powerful | Re-check for bugs; may need revised hypothesis |
| Many patients with 0 reduction | Model is robust to lies | Good — document this |
| Measured model ≈ Reported model | Input source doesn't matter | Interesting null result |
| Measured model ≠ Reported model | Input verification matters | Strong actionable finding |
| SHAP ≠ LIME results | Explanation method matters | Report both; discuss why |

---

## 📤 Deliverables

| # | Deliverable | Location |
|---|-------------|----------|
| 1 | Trial summary table | `results/tables/trial_summary.csv` |
| 2 | Equivalence curve plots | `results/figures/equiv_*.png` |
| 3 | Score reduction box plots | `results/figures/score_reductions.png` |
| 4 | Model comparison chart | `results/figures/model_comparison.png` |
| 5 | Preliminary conclusions | Documented in run output |
| 6 | Performance benchmarks | Processing time per model/config |

---

## ✅ Completion Checklist

- [ ] Full pipeline runs end-to-end without errors on 2015–2016 data
- [ ] All 8 configurations tested (2 input types × 2 models × 2 explanation methods)
- [ ] Equivalence metric computed for each configuration
- [ ] At least 4 publication-quality plots generated
- [ ] Trial summary table produced
- [ ] Performance bottlenecks identified and addressed
- [ ] Preliminary equivalence number recorded (needed for Stage 6)
- [ ] Results reviewed by all team members
- [ ] All code and results committed to Git
- [ ] Bug fixes applied and re-verified
- [ ] **Still have NOT touched 2017–2018 data**
