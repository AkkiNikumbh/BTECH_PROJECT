# 📋 Stage 6: Pre-Registration & Analysis Plan Publication
## Period: 1 January – 14 January 2027 (2 weeks)

> **← Previous**: [Stage 5 — Trial Run](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/05_trial_run.md)
> **← Back to** [Master Roadmap](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/00_master_roadmap.md)
> **→ Next Stage**: [Stage 7 — Final Experiments](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/07_final_experiments.md)

---

> [!CAUTION]
> **THIS IS THE COMMITMENT POINT.** After publishing the analysis plan, the methodology, metrics, and expected results are **locked**. No post-hoc adjustments to the experimental design are allowed. This ensures scientific integrity and prevents p-hacking or HARKing (Hypothesizing After Results are Known).

---

## 🎯 Objectives

1. Finalize all experiment settings based on the trial run (Stage 5) results
2. Write a formal analysis plan document specifying the exact methodology, metrics, and expected outcomes
3. Publish the plan online with an **immutable timestamp** before running the final experiment
4. Declare the expected result publicly so the conclusion cannot be adjusted afterwards

---

## 📋 Prerequisites (from Stage 5)

- [x] Trial run complete with preliminary results
- [x] All bugs fixed and edge cases handled
- [x] Performance benchmarked — experiment is feasible overnight
- [x] Preliminary equivalence numbers recorded
- [x] Team has discussed and agreed on the interpretation framework

---

## 🔨 Step-by-Step Implementation

### Task 6.1 — Write the Analysis Plan

Create `docs/analysis_plan.md`:

```markdown
# Pre-Registered Analysis Plan
## GP-109: Can an AI Explanation Be Misused?

**Date of Registration**: [INSERT DATE — must be BEFORE any 2017-2018 data analysis]
**Registration Platform**: [OSF / arXiv / GitHub release with DOI]
**Team**: Akash Singh, Aditya Jha, Akshat Kumar (DTU, CSE)
**Guide**: Miss. Anjali Bansal

---

### 1. Research Question

Does a SHAP or LIME explanation help an adversary reduce their diabetes risk score 
more effectively than the same number of blind queries to the model?

### 2. Primary Hypothesis

We hypothesize that a single explanation provides an advantage equivalent to 
**fewer than 10 blind attempts** — i.e., explanations give only a small additional 
advantage over simple trial-and-error.

### 3. Datasets

- **Development data** (already analyzed): NHANES 2015–2016
- **Held-out test data** (NOT yet analyzed): NHANES 2017–2018
- Final results will be computed ONLY on the 2017–2018 data.

### 4. Models

- Logistic Regression (scikit-learn, random_state=42)
- Random Forest (scikit-learn, n_estimators=100, max_depth=10, random_state=123)
- Each algorithm is trained in two variants:
  - Model A: Uses nurse-measured height/weight
  - Model B: Uses self-reported height/weight
- Train/test split: 80/20, random_state=789, stratified

### 5. Explanation Methods

- SHAP (TreeSHAP for RF, LinearSHAP for LR)
- LIME (500 samples, 5 features, random_state=42)

### 6. Adversarial Strategies

**Strategy 1 (Explanation-Guided)**:
- Adversary receives one explanation (SHAP or LIME feature attributions)
- Changes the most influential gameable feature in the risk-reducing direction
- Change magnitude bounded by the 95th percentile of the observed 
  reported-vs-measured gap from NHANES

**Strategy 2 (Blind Trial-and-Error)**:
- Adversary receives N queries to the model (no explanation)
- Each query randomly perturbs a gameable feature within the lie budget
- The best (lowest) score found is kept
- N tested: [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]

### 7. Primary Metric

**Explanation Equivalence (N*)**:
The number of blind attempts that achieves the same average score reduction
as one explanation. Computed via linear interpolation on the 
N → mean_blind_reduction curve.

### 8. Secondary Metrics

- Mean and median score reduction for each strategy
- Percentage of patients where the explanation provides zero benefit
- Model A vs Model B gap (measured vs self-reported inputs)
- SHAP vs LIME agreement

### 9. Statistical Tests

- Paired Wilcoxon signed-rank test for strategy comparison
- Bootstrap 95% confidence intervals for equivalence estimates
- Effect size (Cohen's d) for practical significance

### 10. Expected Result

Based on our trial run on 2015-2016 data, we expect:
- Median equivalence: approximately [X] blind attempts (from trial)
- This would suggest that explanations provide [small/moderate/large] advantage

### 11. Interpretation Guide

| Result | Conclusion |
|--------|-----------|
| N* < 5 | Explanations not a meaningful threat; show openly |
| 5 ≤ N* ≤ 20 | Explanations provide moderate advantage; worth further study |
| N* > 20 | Explanations are powerful attack vectors; hiding may be warranted |

### 12. What We Will NOT Do

- We will NOT adjust the model hyperparameters after seeing 2017-2018 results
- We will NOT add or remove features after seeing 2017-2018 results
- We will NOT change the lie budget calculation method
- We will NOT change the primary metric definition
- We will NOT selectively report only favorable results
```

### Task 6.2 — Where to Publish

| Platform | Method | Timestamp Type |
|----------|--------|----------------|
| **OSF** (osf.io) | Create a pre-registration | Immutable, DOI assigned |
| **arXiv** | Upload as a short paper / tech report | Timestamp on submission |
| **GitHub Release** + **Zenodo** | Create a tagged release; Zenodo assigns DOI | DOI + Git hash |
| **Institutional repository** | Upload to DTU repository if available | Institutional timestamp |

> [!TIP]
> **Recommended approach**: Create a **GitHub release** tagged `v0.1-preregistration` with the analysis plan, then link it to **Zenodo** for a DOI. This is the fastest and most accessible option.

```bash
# Create the pre-registration release
git tag -a v0.1-preregistration -m "Pre-registered analysis plan (before 2017-2018 analysis)"
git push origin v0.1-preregistration
```

### Task 6.3 — Finalize Experiment Parameters

Based on trial results, confirm or adjust these settings (any changes from Stage 5 MUST be documented):

```python
"""
src/config.py — FINAL experiment parameters
Locked after pre-registration. Do NOT modify after publishing the plan.
"""

# These are FINAL — do not change after pre-registration
FINAL_CONFIG = {
    # Data
    'development_cycle': '2015_2016',    # Already analyzed
    'holdout_cycle': '2017_2018',        # NOT YET TOUCHED
    
    # Models
    'model_types': ['rf', 'lr'],
    'rf_n_estimators': 100,
    'rf_max_depth': 10,
    'rf_random_state': 123,
    'lr_random_state': 42,
    'lr_max_iter': 1000,
    
    # Split
    'test_size': 0.2,
    'split_random_state': 789,
    
    # Adversary
    'lie_budget_method': 'percentile',
    'lie_budget_percentile': 95,
    'blind_attempts': [1, 2, 3, 5, 10, 15, 20, 30, 50, 100],
    
    # Explanations
    'shap_method': 'tree',          # TreeSHAP for RF
    'lime_n_samples': 500,
    'lime_n_features': 5,
    'lime_random_state': 42,
    
    # Statistics
    'bootstrap_n_iterations': 1000,
    'confidence_level': 0.95,
    'adversarial_seed': 456,
}
```

---

## 📤 Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Analysis plan document** | `docs/analysis_plan.md` — full methodology specification |
| 2 | **Public timestamp** | Git tag + Zenodo DOI (or OSF registration) |
| 3 | **Locked config** | `FINAL_CONFIG` in `src/config.py` — immutable after this point |
| 4 | **Changelog** | Document any changes made between trial (Stage 5) and final settings |

---

## ✅ Completion Checklist

- [ ] Analysis plan document written and reviewed by all team members
- [ ] Guide (Miss. Anjali Bansal) has reviewed and approved the plan
- [ ] All experiment parameters finalized and documented in `FINAL_CONFIG`
- [ ] Changes from trial run settings explicitly listed
- [ ] Expected results stated with specific numbers
- [ ] Git tag `v0.1-preregistration` created
- [ ] Plan published on at least one timestamped platform
- [ ] **URL/DOI of published plan recorded**: ____________
- [ ] **Publication date recorded**: ____________
- [ ] All team members confirm: 2017–2018 data has NOT been examined
- [ ] Config file locked — no further changes allowed
