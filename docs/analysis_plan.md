# Pre-Registered Analysis Plan
## GP-109: Can an AI Explanation Be Misused? Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model

---

**Date of Registration:** 28 October 2026
**Git Tag:** `v0.1-preregistration`
**Project:** B.Tech Project (BTP), Delhi Technological University (DTU)
**Department:** Computer Science & Engineering
**Group:** GP-109
**Team Members:** Akash Singh, Aditya Jha, Akshat Kumar
**Guide:** Miss. Anjali Bansal

> **⚠️ COMMITMENT STATEMENT**
> This document is written and published BEFORE any analysis of the held-out 2017–2018 NHANES dataset. Once published, the methodology, primary metric, expected outcomes, and statistical tests described here are locked. No post-hoc changes will be made. This prevents p-hacking and HARKing (Hypothesizing After Results are Known).

---

## 1. Research Question

Does a SHAP or LIME explanation help an adversary reduce their predicted diabetes risk score more effectively than the same number of blind (random) queries to the model?

More specifically: **Is 1 explanation equivalent to more than N blind attempts at gaming the model?**

---

## 2. Primary Hypothesis

We hypothesize that a single SHAP or LIME explanation provides an advantage equivalent to **fewer than 10 blind attempts** — i.e., explanations give only a small additional advantage over blind trial-and-error. Formally:

> H₀: `mean_explanation_reduction ≤ mean_blind_reduction(N=10)`
> H₁: `mean_explanation_reduction > mean_blind_reduction(N=10)`

We expect to **fail to reject H₀** for most configurations, consistent with explanations being a limited but non-zero threat.

---

## 3. Motivation

Clinical AI models increasingly display feature attribution explanations (SHAP, LIME) to patients and clinicians for transparency. A patient who is denied a treatment or flagged for high risk may try to "game" the model by adjusting what they self-report. If explanations tell a patient exactly which self-reported feature to change, they may become a powerful cheating tool.

We test this concern empirically using real NHANES survey data, where we can directly compare nurse-measured values against patient self-reports.

---

## 3. Datasets

| Dataset | Role | Status at pre-registration |
|---------|------|---------------------------|
| NHANES 2015–2016 (9,971 participants) | **Development set** — used for trial run, gap analysis, model tuning | ✅ Already analyzed |
| NHANES 2017–2018 (9,254 participants) | **Held-out test set** — final results ONLY | ❌ NOT YET TOUCHED |

- **Source:** CDC National Health and Nutrition Examination Survey (public domain)
- **Target variable:** HbA1c ≥ 6.5% → Diabetes = 1 (ADA clinical threshold)
- **Lie budget source:** 95th percentile of |self_reported − nurse_measured| from **2015–2016 only**
  - Height: **±7.29 cm** (people tend to overestimate by +1.61 cm on average)
  - Weight: **±8.96 kg** (people tend to underestimate by −0.87 kg on average)

---

## 4. Feature Sets

Both models use the **same 7 features**, differing only in height/weight source:

| Feature | Model A (Measured) | Model B (Reported) | Gameable? |
|---------|-------------------|-------------------|-----------|
| `RIDAGEYR` | Age (years) | Age (years) | NO |
| `RIAGENDR` | Gender (1/2) | Gender (1/2) | NO |
| `BPXSY1` | Systolic BP (mmHg) | Systolic BP (mmHg) | NO |
| `SMQ020` | Smoker (yes/no) | Smoker (yes/no) | MAYBE |
| `PAQ650` | Vigorous activity | Vigorous activity | MAYBE |
| `measured_height_cm` | **Nurse-measured** | — | NO |
| `measured_weight_kg` | **Nurse-measured** | — | NO |
| `self_height_cm` | — | **Self-reported** | **YES** |
| `self_weight_kg` | — | **Self-reported** | **YES** |

---

## 5. Models

| Model | Algorithm | Key Hyperparameters |
|-------|-----------|---------------------|
| `measured_rf` | Random Forest | n_estimators=100, max_depth=10, random_state=123 |
| `measured_lr` | Logistic Regression | max_iter=1000, random_state=42 |
| `reported_rf` | Random Forest | n_estimators=100, max_depth=10, random_state=123 |
| `reported_lr` | Logistic Regression | max_iter=1000, random_state=42 |

- **Train/test split:** 80/20, stratified, random_state=789
- **Complete case analysis:** Only patients with no missing values in all 7 features are used

---

## 6. Explanation Methods

| Method | Config |
|--------|--------|
| **SHAP** | TreeSHAP for RF; LinearSHAP (KernelExplainer) for LR |
| **LIME** | 500 neighbourhood samples, top 5 features, random_state=42 |

---

## 7. Adversarial Strategies

### Strategy 1: Explanation-Guided Attack (1 call)
1. Request one SHAP or LIME explanation for the patient's feature vector
2. Identify the most influential **gameable** feature (highest absolute attribution)
3. Change it in the direction that reduces the diabetes risk score
4. Bound the change by the pre-computed lie budget (±7.29 cm height, ±8.96 kg weight)
5. Report the new (gamed) risk score

### Strategy 2: Blind Trial-and-Error (N calls, blind_attack_multi_n)
1. Make N sequential queries to the model, each with a random perturbation
2. Each query randomly selects a gameable feature and applies a uniform random change within the lie budget
3. Keep the **best (lowest) score found so far** — the adversary is rational and accumulates attempts
4. Record the best score reduction at each milestone N ∈ {1, 2, 3, 5, 10, 15, 20, 30, 50, 100}

> **Statistical correctness note:** The blind simulation is run ONCE up to max(N)=100, recording the running best at each milestone. This is equivalent to the adversary accumulating attempts and reflects the true sequential nature of blind trial-and-error. Running independent simulations per N would incorrectly over-estimate blind performance.

---

## 8. Primary Metric: Explanation Equivalence (N\*)

**Definition:** The number of blind attempts N* such that the blind adversary's mean score reduction equals the explanation-guided adversary's mean score reduction.

**Computation:**
1. For each patient: compute `explanation_reduction` (Strategy 1) and `blind_reduction[N]` for each N (Strategy 2)
2. Average across all patients: get `mean_explanation_reduction` and `mean_blind_reduction(N)` curve
3. Interpolate linearly to find N* where `mean_blind_reduction(N*) = mean_explanation_reduction`
4. If explanation is worse than even 1 blind attempt: N* = 0
5. If explanation exceeds even 100 blind attempts: N* = 101 (censored)

---

## 9. Secondary Metrics

- **Median N*** — robust to outliers
- **95% bootstrap CI** on mean and median N* (1,000 bootstrap iterations)
- **% patients where explanation is useless** (N* ≤ 1)
- **% patients where explanation is powerful** (N* ≥ 10)
- **Model A vs Model B comparison** — does measured vs self-reported input affect vulnerability?
- **SHAP vs LIME agreement** — do the two explanation methods agree on N*?

---

## 10. Statistical Tests (Pre-Registered)

All tests are **one-sided** (testing whether explanation > blind):

| Test | Comparison | α level |
|------|-----------|---------|
| Paired Wilcoxon signed-rank | explanation_reduction vs blind_reduction(N=10) per patient | 0.05 |
| Bootstrap 95% CI | mean N* and median N* | 95% |
| Cohen's d | effect size of explanation advantage | Reported regardless |

> Hypothesis: `explanation_reduction > blind_reduction(N=10)` — i.e., 1 explanation is worth more than 10 blind attempts.

---

## 11. Expected Results (from Trial Run on 2015–2016)

From our preliminary trial run on the development set (200 patients, `reported_rf_shap`):

| Metric | Trial Value |
|--------|------------|
| Mean N* | **9.5** blind attempts |
| 95% CI (mean) | [6.11, 13.17] |
| Median N* | **0.0** (explanation useless for majority) |
| % explanation useless | **63%** of patients |
| % explanation worth 10+ | **14%** of patients |

**Pre-registered interpretation of final results:**

| Final N* | Conclusion |
|----------|-----------|
| N* < 5 | Explanations offer minimal advantage; transparency is safe |
| 5 ≤ N* ≤ 20 | Explanations offer moderate advantage; caution recommended |
| N* > 20 | Explanations are powerful attack vectors; design countermeasures |

---

## 12. Exact Configurations to be Tested (Final Run)

All 8 combinations:

| # | Input Type | Model | Explanation | N patients (2017-2018 test set) |
|---|-----------|-------|-------------|--------------------------------|
| 1 | reported | RF | SHAP | All available |
| 2 | reported | RF | LIME | All available |
| 3 | reported | LR | SHAP | All available |
| 4 | reported | LR | LIME | All available |
| 5 | measured | RF | SHAP | All available |
| 6 | measured | RF | LIME | All available |
| 7 | measured | LR | SHAP | All available |
| 8 | measured | LR | LIME | All available |

---

## 13. What We Will NOT Do (Locked Rules)

- ❌ We will NOT examine 2017–2018 data before publishing this plan
- ❌ We will NOT adjust model hyperparameters after seeing final results
- ❌ We will NOT add or remove features after seeing final results
- ❌ We will NOT change the lie budget calculation method
- ❌ We will NOT change the primary metric (N*) definition
- ❌ We will NOT selectively report only favorable result configurations
- ❌ We will NOT change the statistical test or significance threshold
- ❌ We will NOT change the N values tested for blind attempts

---

## 14. Locked Configuration (FINAL_CONFIG in src/config.py)

```python
FINAL_CONFIG = {
    # Data
    'development_cycle': '2015_2016',
    'holdout_cycle': '2017_2018',

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
    'shap_method': 'tree',
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

## 15. Registration & Timestamping

| Method | Details |
|--------|---------|
| **GitHub tag** | `v0.1-preregistration` on branch `master` |
| **Git commit hash** | To be recorded here after tagging |
| **OSF (recommended)** | Upload this document to osf.io before final run |
| **Publication date** | 28 October 2026 |

---

## 16. Team Sign-Off

By proceeding with the final experiment (Stage 7), all team members confirm:

- [ ] We have read and understood this analysis plan
- [ ] We confirm the 2017–2018 NHANES data has NOT been examined
- [ ] We agree to report ALL 8 configurations regardless of outcome
- [ ] We agree that this plan is locked and cannot be changed after this date

**Akash Singh:** _________________________ Date: ___________

**Aditya Jha:** _________________________ Date: ___________

**Akshat Kumar:** _________________________ Date: ___________

**Guide (Miss. Anjali Bansal):** _________________________ Date: ___________

---

*This document is version-controlled. Git hash of this file at registration: see `git log docs/analysis_plan.md`*
