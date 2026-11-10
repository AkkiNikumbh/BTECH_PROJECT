# Implementation Fixes (v2)

This repository is a fixed copy of
[`Akshat-stack47/BTP-XAI-Gaming`](https://github.com/Akshat-stack47/BTP-XAI-Gaming).
The original repo is left unchanged as the **comparison baseline**. The first commit
here is an unmodified snapshot of it, so `git diff <baseline-commit> HEAD` shows
exactly what changed.

> ⚠️ **The files in `results/` are still the OLD outputs from the baseline code.**
> They have not been regenerated. Re-run the pipeline (below) to get results
> from the fixed code.

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
* **New resume cache folder:** `run_final.py` now caches to
  `data/processed/final_run_cache_v2_fixed/`. The old `final_run_cache/` holds
  results from the buggy code and would otherwise be silently re-loaded and skipped on resume.
* **New tests:** `tests/test_fixes.py` covers each fix.

## Not changed (out of scope for this pass)

The N\* aggregation (per-patient mean vs the pre-registered mean-curve, issue 4),
the attacker magnitude asymmetry (issue 6), realistic lie direction (7),
blind-baseline seeding (8), LIME `num_features=5` (9), class imbalance (10),
the LR SHAP background (11), and the timeline / pre-registration process issues.

## How to regenerate results

```bash
pip install -r requirements.txt
python scripts/download_nhanes.py
python scripts/prepare_data.py       # 2015-2016 (fix 3 changes the cleaned data)
python scripts/run_trial.py          # optional: trial on 2015-2016
python scripts/run_final.py          # 2017-2018, several hours
python scripts/run_analysis.py       # figures + tables
python -m pytest tests/ -v
```
