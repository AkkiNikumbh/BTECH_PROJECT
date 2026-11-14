# Pre-Registered Confirmatory Analysis Plan (v5)
## GP-109: Can an AI Explanation Be Misused? Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model

**Git tag:** `v0.2-preregistration` (pushed to GitHub **before** the
confirmatory data is processed; the tag's commit holds this file and the
exact code that will be run)
**Group:** GP-109, DTU CSE. **Guide:** Miss. Anjali Bansal

> **Why a second pre-registration?** The first plan (`docs/analysis_plan.md`,
> tag `v0.1-preregistration`) was meant to be tested on NHANES 2017–2018, but
> that data was analysed with the pre-registered code, and the code was then
> corrected (fixes 1–13). See `docs/preregistration_disclosure.md`. The
> 2017–2018 results are therefore **exploratory**. This plan fixes the
> corrected (v5) analysis in advance and tests it on a cycle **no one on the
> team has analysed**: NHANES August 2021 – August 2023.

---

## 1. Research question and hypothesis (unchanged from v0.1)

Does one SHAP or LIME explanation help a patient lower their predicted
diabetes risk more than the same number of blind attempts at the model?

* H₀: mean explanation reduction ≤ mean blind reduction at N = 10
* H₁: mean explanation reduction > mean blind reduction at N = 10
* Primary metric: **N\***, the number of blind attempts whose mean
  best-so-far reduction equals the mean reduction of one explanation-guided
  attempt (linear interpolation on the mean curve; 0 if the explanation is no
  better than one blind attempt; 101 if better than 100).

## 2. Data

| Role | Data | Status at this tag |
|---|---|---|
| Lie budget | NHANES 2015–2016 | analysed (development) |
| Exploratory test | NHANES 2017–2018 | analysed (see disclosure) |
| **Confirmatory test** | **NHANES Aug 2021 – Aug 2023 (files `*_L`)** | **not analysed** |

Before this tag, for 2021–2023 only the column names of the downloaded files
and the public CDC codebooks were read, to define the mapping in §3. No
values, distributions, prevalence or outcomes were looked at.

## 3. Variables (2021–2023 mapping)

Same 7 features as before; target HbA1c (`LBXGH`) ≥ 6.5%; adults ≥ 18;
complete cases; same NHANES missing codes.

| Feature | 2015–18 | 2021–2023 | Note |
|---|---|---|---|
| Age, sex | `RIDAGEYR`, `RIAGENDR` | same | |
| Systolic BP | `BPXSY1` (auscultatory) | `BPXOSY1` (oscillometric, 1st reading) | `BPX_L` does not exist |
| Smoking | `SMQ020` | same | |
| Vigorous activity | `PAQ650` (yes/no) | from `PAD810Q`: 0 → no (2), 1–200 → yes (1), 7777/9999 → missing | `PAQ650` removed |
| Self-reported height/weight | `WHD010`, `WHD020` | same | |
| Measured height/weight | `BMXHT`, `BMXWT` | same | |

## 4. Fixed analysis (code at this tag, `FINAL_CONFIG` in `src/config.py`)

* Models retrained on an 80/20 stratified split of 2021–2023 with the locked
  hyperparameters (RF: 100 trees, depth 10; LR: max_iter 1000; balanced class
  weights; seeds as in `src/config.py`).
* Lie budget from **2015–2016 only**: height ±7.28 cm (over-reported),
  weight ±8.96 kg (under-reported); realistic direction only; lie size
  uniform on (0, budget] for both attackers.
* Explanation-guided attacker: one SHAP or LIME explanation; most important
  self-reported feature whose risk-lowering direction is realistic; SHAP sign
  read relative to the average patient (fix 13); LIME read from its bins
  (fix 2); a lie that raises the score is undone (fix 12).
* Blind attacker: N ∈ {1, 2, 3, 5, 10, 15, 20, 30, 50, 100} random realistic
  lies, best-so-far.
* Only self-reported height/weight are gameable; Model A (measured) cannot
  be attacked.
* All 8 configurations (reported/measured × RF/LR × SHAP/LIME), all test
  patients.
* Run with `python scripts/run_confirmatory.py`, library versions from
  `requirements-lock.txt`.

## 5. Primary result and interpretation (unchanged thresholds)

**Primary configuration: Model B (self-reported), Random Forest, SHAP.**

| N\* | Conclusion |
|---|---|
| < 5 | explanations offer minimal advantage over trying the model |
| 5–20 | moderate advantage |
| > 20 | explanations are a powerful attack aid |

## 6. Statistics

* Bootstrap 95% CI of N\* (1,000 resamples of patients).
* One-sided paired Wilcoxon signed-rank test, explanation vs. blind N = 10,
  α = 0.05, per configuration; Holm-adjusted p-values across the 8
  configurations reported as secondary.
* Cohen's d of the paired difference.
* Secondary: per-patient N\* (mean, median, % ≤ 1, % ≥ 10), SHAP vs. LIME,
  Model A vs. B (AUC difference on the same test patients with a paired
  bootstrap CI, and how far Model B can be gamed).

## 7. Expected result (from the exploratory 2017–2018 run with this code)

NHANES 2017–2018, v5 code, N = 887 test patients (`results/tables/final_results.csv`):

| Config (Model B) | N\* | 95% CI | 1 explanation | 1 blind | 10 blind |
|---|---|---|---|---|---|
| RF + SHAP (primary) | 0.0 | [0.0, 1.22] | 0.0233 | 0.0240 | 0.0638 |
| RF + LIME | 0.0 | [0.0, 0.0] | 0.0181 | 0.0240 | 0.0638 |
| LR + SHAP | 0.0 | [0.0, 1.09] | 0.0242 | 0.0248 | 0.0457 |
| LR + LIME | 0.0 | [0.0, 0.0] | 0.0223 | 0.0248 | 0.0457 |

Model A (measured): 0 everywhere, by design.

**Prediction for 2021–2023:** primary N\* < 5 (low threat), with the upper
95% CI bound below 5; one explanation worth at most about one to two blind
attempts in every Model B configuration; the one-sided Wilcoxon test not
significant (explanation not better than 10 blind attempts) in any
configuration. A primary N\* ≥ 5 would contradict the expectation and will
be reported as such.

## 8. Locked rules

* No change to code, settings, features, metric, tests or thresholds after
  this tag. If the confirmatory run fails for a technical reason, the fix and
  its reason are committed and reported as a deviation, and results are not
  inspected before that commit.
* All 8 configurations are reported, whatever the result.
* The result is written into `results/tables/confirmatory_*` and reported
  unchanged in the paper and thesis.
