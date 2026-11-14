# Implementation Fixes (v2)

This repository is a fixed copy of
[`Akshat-stack47/BTP-XAI-Gaming`](https://github.com/Akshat-stack47/BTP-XAI-Gaming).
The original repo is left unchanged as the **comparison baseline**. The first commit
here is an unmodified snapshot of it, so `git diff <baseline-commit> HEAD` shows
exactly what changed.

> The files in `results/` were regenerated with the v4 code (fixes 1–12) in
> the library versions pinned in `requirements-lock.txt`.
>
> ⚠️ **Read [`preregistration_disclosure.md`](preregistration_disclosure.md)**:
> a first final run on the held-out 2017–2018 data (code as pre-registered)
> was seen before these fixes were made.

---

## Fix 1 — The measured model was attacked through nurse-measured values

**Files:** `src/config.py`, `scripts/run_final.py`, `scripts/run_trial.py`, `src/adversarial.py`

**Bug:** The list of features the patient may lie about came out empty for
Model A (measured inputs), and the code then *fell back* to
`measured_height_cm` / `measured_weight_kg`. So Model A was attacked
exactly like Model B, even though `docs/column_mapping.md` and
`docs/analysis_plan.md` §4 mark nurse-measured values as **not gameable**.
This made the Model A vs Model B comparison meaningless (both got nearly
identical N\*).

**Fix:** One shared definition, `GAMEABLE_FEATURES = ['self_height_cm', 'self_weight_kg']`
and `get_gameable_features()` in `src/config.py`, used by both scripts. Model A
now has **no** gameable features, so both attacks leave its score unchanged
(score reduction 0, N\* = 0). The blind attack handles an empty list instead of
crashing.

**Effect:** Model A vs Model B now measures what the plan intends: how much
vulnerability comes from the input being **self-reported** instead of measured.

## Fix 2 — LIME-guided attack sometimes moved in the wrong direction

**Files:** `src/explanations.py` (`get_top_gameable_feature`, new `lime_direction`), `src/adversarial.py`

**Bug:** LIME (with its default `discretize_continuous=True`) explains with
**bins**, for example `self_weight_kg <= 62.3` with weight +0.2, which means
"being in this LOW bin raises the risk". The old code always *decreased* the
feature when the weight was positive. That is right for `>` bins but
**wrong for `<=` bins**, and meaningless for range bins (`a < x <= b`).

**Fix:** The direction is now read from the bin and the sign:

| LIME condition | weight > 0 (bin raises risk) | weight < 0 (bin lowers risk) |
|---|---|---|
| `x > a` | decrease | increase |
| `x <= a` | **increase** | decrease |
| `a < x <= b` | leave the bin by the nearer edge | no direction, so try the next gameable feature |

SHAP is unchanged: positive contribution → decrease, negative → increase.
LIME settings (500 samples, 5 features, seed 42, discretised) are unchanged.

**Effect:** LIME configurations were under-estimated before. Expect LIME N\* to change.

## Fix 3 — Valid self-reported heights/weights were deleted as "refused"

**File:** `src/data_loader.py` (`NHANES_MISSING_CODES`, `replace_missing_codes`)

**Bug:** One shared list `[7, 9, 77, 99, 777, 999, 7777, 9999, ...]` was
applied to every questionnaire column. In the NHANES codebook (WHQ_I),
WHD010 (height) has valid values **47–84 inches** and WHD020 (weight)
**75–600 lb**, with only **7777 = Refused, 9999 = Don't know**. So valid answers such as
**77 in (195.6 cm)** and **99 lb** were set to missing, which removed some of the
tallest and lightest participants.

**Fix:** Codes are now set per column: WHD010/WHD020 → {7777, 9999};
SMQ020, PAQ605, PAQ650, DMDEDUC2 → {7, 9}.

**Effect:** Slightly more participants are kept. The reporting gap and lie
budget may shift slightly. Re-run `prepare_data.py`.

## Fix 5 — The final run used the wrong lie budget

**File:** `scripts/run_final.py`

**Bug:** `docs/analysis_plan.md` §3 pre-registers the lie budget as the 95th
percentile gap **from 2015–2016 only** (±7.29 cm / ±8.96 kg), and the changelog
lists it as UNCHANGED. But the final run used the **2017–2018** budget
(±6.87 cm / ±9.31 kg). That budget was also computed from the same held-out
patients who were then attacked.

**Fix:** The attack uses `lie_budget_used = lie_budget_dev` (2015–2016). The
2017–2018 gap is still computed and saved (`lie_budget` in the holdout pickle,
`lie_budget_observed_2017_2018` in the JSON) for the cross-cycle comparison table only.

## Other changes made along the way

* **Crash fix in `run_final.py`, step 8:** writing the master JSON called
  `float()` on the string `'positive'` / `'negative'` (`mean_direction`), which raises
  `ValueError`. This is why `final_results.json` is missing from the baseline repo.
* **One results cache folder for all scripts:** `config.FINAL_RUN_CACHE`, used by
  `run_final.py`, `run_analysis.py` and the review scripts. A new folder name is used
  for each version of the experiment code (now `final_run_cache_v3`), so results from
  older code are never silently re-loaded on resume.
* **Windows fixes:** console output and all result files are written as UTF-8
  (previously `UnicodeEncodeError` on symbols like ≈ ✓ ✗).
* **New tests:** `tests/test_fixes.py` (fixes 1–5) and `tests/test_fixes_v3.py` (fixes 4, 6–11).

---

# v3 — remaining implementation fixes (4, 6, 7, 8, 9, 10, 11)

All new settings are named switches in `src/config.py` (and recorded in
`FINAL_CONFIG`), so every choice is visible and can be reverted or used for a
sensitivity analysis.

## Fix 4 — Primary N\* now computed as pre-registered

**Files:** `src/metrics.py`, plus every table/figure/summary that shows the headline number

**Bug:** `analysis_plan.md` §8 defines N\* as: average the explanation reductions and
the blind curve over ALL patients, then interpolate ONE N\*. The code instead
computed N\* per patient and reported the **mean** of those. Patients where the
explanation beats 100 blind attempts are capped at 101, so a few of them pulled the mean
up (in the original results the median was 0 but the mean was 12.43).

**Fix:** `population_nstar()` implements §8 exactly. It is now the primary metric
(`summary['nstar']`, `Nstar` column), with a patient-bootstrap 95% CI (`nstar_ci`).
The per-patient numbers are kept as **secondary** metrics (§9): `PerPatient_Mean_Nstar`,
median, % useless, % worth 10+, and the new `Pct_censored`. The pre-registered
LOW/MODERATE/HIGH verdict now uses the primary N\*.

## Fix 6 — Same lie-size rule for both attackers (`LIE_MAGNITUDE = 'uniform'`)

**Bug:** The explanation-guided attacker always lied by the full maximum, while each
blind attempt drew a random size. Part of the explanation's "advantage" was just the
bigger lie, not the information in the explanation.

**Fix:** Both attackers draw every lie's size from the same rule. Default
`'uniform'`: size ~ U(0, max_lie]. The explanation only tells the attacker which
feature to change and in which direction. Alternative `'max'`: every lie, explanation
and blind, uses the full max_lie.

## Fix 7 — Only realistic lies (`LIE_DIRECTION = 'realistic'`)

**Bug:** Lies could go either way, e.g. claiming to be shorter, which people
almost never do. The plan says to use the real reported-vs-measured gap to define a
*realistic* lie.

**Fix:** Lies go only in the direction people actually misreport, read from the
2015–16 data (`mean_direction`): **height up, weight down**. The size limit is still
the pre-registered 95th percentile of |gap|. If the explanation says the realistic
lie on its top feature would *raise* the risk, the attacker tries the next gameable
feature. If none helps, it doesn't lie (reduction 0). Alternative `'both'` = old behaviour.

## Fix 8 — Per-patient random seeds

**Bug:** One random stream was shared across all patients and both explanation
methods. So SHAP and LIME were compared against *different* blind attempts for the
same patient, and LIME's sampling depended on which patients came before.

**Fix:** Patient *i* gets its own seeds (blind attempts, explanation lie size, LIME
sampling). SHAP vs LIME, and Model A vs B, now face identical blind attempts per patient.

## Fix 9 — LIME shows all features (`LIME_N_FEATURES = None`)

**Bug:** LIME returned only 5 of 7 features. When neither height nor weight was in
the top 5, the attack did nothing and the patient counted as "explanation useless".

**Fix:** LIME returns all 7 features. Its other settings (500 samples, seed 42, binned features) are unchanged.

## Fix 10 — Class imbalance (`CLASS_WEIGHT = 'balanced'`)

**Bug:** With ~13% diabetes and no class weighting, both models almost always
predicted "no diabetes" (recall ≈ 3%).

**Fix:** `class_weight='balanced'` for RF and LR. On the real 2017–18 data, recall
is now ≈ 69–78%. AUC is essentially unchanged; risk scores are now on a balanced scale.

## Fix 11 — SHAP background for logistic regression

**Bug:** `LinearExplainer` used the first 100 training rows as background.

**Fix:** A seeded random sample of 100 training rows (`SHAP_BACKGROUND_N`, `SEED_SHAP_BACKGROUND`).

## Fix 12 — The explanation-guided patient may undo a bad lie (`EXPLANATION_KEEP_BEST = True`)

**Files:** `src/adversarial.py`, `src/config.py`, `scripts/run_final.py`, `scripts/run_analysis.py`

**Bug:** The blind patient keeps the best score seen so far, *starting from
their true score*, so a blind lie that raises the risk is never counted
(reduction ≥ 0). The explanation-guided patient's single lie was always
counted, even when it **raised** the risk (23–31% of patients for the random
forest). The plan compares the two "with the same number of attempts", and
both patients see the score after each attempt, so this was an unequal rule
that biased the result against explanations.

**Fix:** If the explanation-guided lie raises the score, the patient keeps
their true answers (reduction 0), exactly like the blind patient. The raw
effect of the suggested lie is still stored (`explanation_raw_reduction`) and
reported in Table 7 (column "lie not undone") and in `final_results.csv`
(`Mean_exp_raw_reduction`, `Pct_lie_undone`). This is a **deviation** from the
pre-registered plan (step 5 of Strategy 1 reports the new score as is).

**Tests:** `tests/test_fixes_v4.py`.

## Supporting datasets from the research plan (§6.4)

* **UCI Heart Disease (UCI 45)** — `scripts/check_uci_heart.py` (+ `tests/test_uci_hand_check.py`):
  research-plan phase P1, "test the code on a small dataset where results can
  be checked by hand". Runs the real pipeline classes with a logistic
  regression and checks every output (risk score, SHAP values, explanation
  lie, best blind lie, N\*) against a closed-form hand calculation, and prints
  the numbers for the first patients so they can be redone with a calculator.
* **CDC Diabetes Health Indicators (UCI 891)** — `scripts/run_uci_cdc_comparison.py`:
  the comparison dataset. BRFSS 2015, 253,680 respondents, BMI from
  self-reported height/weight (the only gameable feature; no nurse
  measurement, so Model B only). BMI lie budget from the NHANES 2015–16 gap.
  Output: `results/tables/table8_uci891_comparison.*`.

Download both into `data/raw/` (see the README).

## Speed-up (no effect on results)

The random forest is trained on all cores but now predicts on one
(`n_jobs = 1` after training). The attack makes ~100 single-patient predictions
per patient, and starting a thread pool for each one was very slow. The full
`run_final.py` now takes **~35 minutes** instead of ~2 hours. Predictions are identical.

## ⚠️ Deviations from the pre-registered analysis plan

Fixes 4, 1, 3 and 5 bring the code **back in line** with `analysis_plan.md`. The
following are **deviations** from it and must be reported as such in the
paper/thesis (they were made before the final experiment was re-run):

| Setting | Pre-registered | Now | Why |
|---|---|---|---|
| Lie direction | either direction (implicit) | realistic only (height ↑, weight ↓) | plan: "realistic lie" |
| Lie size | explanation = max, blind = uniform | same rule for both (uniform) | fair comparison |
| LIME features shown | 5 | all 7 | height/weight were cut off |
| Class weight | none | balanced | models flagged almost no one |
| SHAP LR background | first 100 rows | random 100 rows | unbiased background |
| Random seeds | one stream | one per patient | identical blind baselines |
| Explanation lie that raised the score | counted (negative) | undone (0), like the blind attacker | fair comparison (fix 12) |

Reverting any of them is a one-line change in `src/config.py`, which is useful for a
sensitivity table.

## Not changed

The timeline. The pre-registration problem (held-out data seen before the
fixes) cannot be undone; it is disclosed in `preregistration_disclosure.md`.

## How to regenerate results

```bash
pip install -r requirements-lock.txt   # exact versions used for results/
python scripts/download_nhanes.py
python scripts/prepare_data.py       # 2015-2016 (seconds)
python scripts/run_trial.py          # optional: trial on 2015-2016 (~5 min)
python scripts/run_final.py          # 2017-2018 (~35 min)
python scripts/run_analysis.py       # figures + tables (~1 min)
python scripts/check_uci_heart.py    # P1 hand check on UCI 45 (seconds)
python scripts/run_uci_cdc_comparison.py   # UCI 891 comparison (~25 min)
python -m pytest tests/ -v
```
