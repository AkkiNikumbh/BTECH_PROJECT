# Pre-registration disclosure: what happened to the held-out data, and when

The research plan (§3, §7.1, phase P7) promises that the analysis plan is
published **before** the 2017–2018 data is examined, "so the conclusion
cannot be adjusted afterwards". This file records exactly how the project
departed from that, so a reader can judge the result. It is referenced from
the README, `docs/fixes_v2.md`, the paper (§ Pre-Registration and
Deviations) and the thesis (§ Corrections and Deviations).

## Timeline (from the git history)

| Date | Event |
|---|---|
| 28 Oct 2026 | Analysis plan `docs/analysis_plan.md` written and tagged `v0.1-preregistration` in `Akshat-stack47/BTP-XAI-Gaming`. |
| by 29 Oct 2026 | **First final run on the held-out 2017–2018 data**, with the code as pre-registered. Its results (`results/tables/final_results.csv`) are part of the baseline snapshot (commit `26292cc`) and were seen by the team. |
| 10–11 Nov 2026 | Code review found implementation bugs. Fixes 1–5 bring the code back to the plan; fixes 6–11 **change pre-registered settings** (see below). The final run was repeated on the same held-out data (v3). |
| 14 Nov 2026 | Fix 12 (explanation attacker may undo a lie that raised its score, as the blind attacker can) → v4 run. Fix 13 (SHAP sign read relative to the average patient), found by the new UCI 45 hand check → v5 run. All on the same held-out data. |
| 14 Nov 2026 | **New confirmatory plan** `docs/analysis_plan_v5_confirmatory.md` tagged `v0.2-preregistration` and pushed to GitHub **before** NHANES 2021–2023 (never analysed by the team) is processed. |

So the settings used for the reported result were chosen **after** a result
on the held-out data had been seen. The 2017–2018 data is therefore no longer
"unseen" in the sense of the research plan, and the reported result should be
read as a corrected analysis, not as a clean confirmatory test.

## What the first (pre-registered-settings) run reported

From the baseline snapshot, `results/tables/final_results.csv` (code as
pre-registered, including the bugs listed below). Note: this code reported the
**per-patient mean** N\* as its headline, not the pre-registered mean-curve
N\* (bug 4), so the numbers are not directly comparable with the v4 table.

| Config | Per-patient mean N\* | 95% CI | Median | % useless | % worth 10+ |
|---|---|---|---|---|---|
| reported RF SHAP | 12.43 | [10.54, 14.53] | 0.0 | 63.8 | 17.2 |
| reported RF LIME | 9.41 | [7.66, 11.27] | 0.0 | 67.9 | 12.6 |
| reported LR SHAP | 22.77 | [20.36, 25.28] | 0.0 | 59.0 | 32.5 |
| reported LR LIME | 22.71 | [20.28, 25.26] | 0.0 | 55.1 | 33.4 |
| measured RF SHAP | 10.27 | [8.42, 12.15] | 0.0 | 69.2 | 13.5 |
| measured RF LIME | 9.01 | [7.32, 10.77] | 0.0 | 69.8 | 11.8 |
| measured LR SHAP | 23.05 | [20.58, 25.61] | 0.0 | 57.1 | 33.3 |
| measured LR LIME | 21.27 | [18.99, 23.71] | 0.0 | 56.7 | 30.9 |

Read with the pre-registered thresholds, these would have been a
**moderate** (5–20) to **high** (> 20) threat. The corrected analysis
reports N\* = 0 (**low** threat). Most of that change is due to genuine bugs:

* the measured model was attacked through nurse-measured values (bug 1) —
  the "measured" rows above are not meaningful;
* the explanation attacker always used the **maximum** lie while each blind
  attempt drew a random smaller lie (deviation 6), so part of the
  explanation's advantage was simply a bigger lie;
* lies in either direction were allowed (deviation 7), including
  unrealistic ones (claiming to be shorter / heavier).

These are the reasons the settings were changed. They were nevertheless
changed after seeing a held-out result, and the direction of the change
(towards the expected "low threat" outcome) is the direction a reader
should be most sceptical of.

## Every change made after the pre-registration

Bug fixes that bring the code **back to** the plan (1–5) and changes that
**deviate** from it (6–12). Each deviation is a named switch in
`src/config.py`; setting it back reproduces the pre-registered behaviour.

| # | Change | Kind | Switch |
|---|---|---|---|
| 1 | Only self-reported height/weight are gameable | bug fix | `GAMEABLE_FEATURES` |
| 2 | LIME direction read from the bin condition | bug fix | — |
| 3 | Valid self-reports no longer deleted as missing codes | bug fix | — |
| 4 | Headline N\* from the mean curve (as pre-registered) | bug fix | `nstar_method` |
| 5 | 2015–2016 lie budget (as pre-registered) | bug fix | — |
| 6 | Same lie-size rule for both attackers (uniform) | **deviation** | `LIE_MAGNITUDE` |
| 7 | Realistic lie direction only (height ↑, weight ↓) | **deviation** | `LIE_DIRECTION` |
| 8 | One random seed per patient | **deviation** | — |
| 9 | LIME shows all 7 features (was 5) | **deviation** | `LIME_N_FEATURES` |
| 10 | Balanced class weights | **deviation** | `CLASS_WEIGHT` |
| 11 | Random SHAP background for LR | **deviation** | `SHAP_BACKGROUND_N` |
| 12 | Explanation attacker undoes a lie that raised its score (like the blind attacker) | **deviation** | `EXPLANATION_KEEP_BEST` |
| 13 | SHAP sign read relative to the average patient (towards the mean if positive) | bug fix / **deviation** | `SHAP_DIRECTION` |

## Fix 13: how the SHAP-guided patient reads the sign

Found by the P1 hand check on UCI 45 (`scripts/check_uci_heart.py`); fixed in
v5 (`SHAP_DIRECTION = 'toward_mean'`). The SHAP attacker reads a positive SHAP value as
"decrease this feature" (`get_top_gameable_feature`). But SHAP measures a
value *relative to the average*: for a linear model it is
w·(x − mean). A patient who is shorter than average gets a positive SHAP
value for height ("your height raises your risk"), and the useful move is to
claim to be **taller** (towards the average), not shorter. Under the current
rule that lie is unrealistic and is skipped, so the patient often gets no
suggestion at all. (LIME already handles this through its bins, fix 2.)

Effect, NHANES 2017–2018, Model B, SHAP, v4 rules otherwise
(rule changed to "positive SHAP → move towards the training mean,
negative → move away from it"):

| Config | Rule | Mean reduction, 1 explanation | No lie suggested | N\* |
|---|---|---|---|---|
| LR + SHAP | current | 0.0176 | 31.1% | 0.0 |
| LR + SHAP | towards the mean | 0.0240 | 0.3% | 0.0 |
| RF + SHAP | current | 0.0183 | 19.3% | 0.0 |
| RF + SHAP | towards the mean | 0.0233 | 1.8% | 0.0 |

Mean blind reduction for comparison: 1 attempt 0.0248 (LR) / 0.0240 (RF).
With the corrected reading one explanation is worth **about one** blind
attempt instead of less than one; N\* is 0 under both rules, so the
headline conclusion does not depend on this choice.

## Step-by-step effect on the primary configuration (Model B, RF + SHAP, 2017–2018)

| Code | Headline reported | 1 explanation | 1 blind | Source |
|---|---|---|---|---|
| original (pre-registered settings, bugs) | per-patient mean N\* 12.43 | — | — | baseline commit `26292cc` |
| v3 (fixes 1–11) | N\* 0.0 [0.0, 0.0] | 0.0129 | 0.0240 | git history |
| v4 (+ fix 12) | N\* 0.0 [0.0, 0.0] | 0.0183 | 0.0240 | `results/tables/sensitivity_v4_fix12_only.csv` |
| v5 (+ fix 13) | N\* 0.0 [0.0, 1.22] | 0.0233 | 0.0240 | `results/tables/final_results.csv` |

## The clean test: NHANES 2021–2023

Because 2017–2018 can no longer serve as an untouched test set, the v5 analysis
was pre-registered again (`docs/analysis_plan_v5_confirmatory.md`, tag
`v0.2-preregistration`, pushed before the data was processed) and run once on
NHANES August 2021 – August 2023 (`scripts/run_confirmatory.py`, results in
`results/tables/confirmatory_*`). That run is the confirmatory result; the
2017–2018 numbers are exploratory.

**Result (run once, unchanged, at commit `5b2f240`):** primary (Model B,
RF + SHAP) N\* = **1.42**, 95% CI [1.21, 1.62] → **low threat**, as
pre-registered. Model B N\* ranges 0–1.42; no Wilcoxon test significant
(p = 1.0, also after Holm). Every pre-registered prediction in §7 of the
plan held. Full table: `results/tables/confirmatory_results.csv`.

## How to report this

* State in the paper and thesis that a first final run on 2017–2018 was seen
  before fixes 1–12, and give its headline numbers (table above).
* Present the corrected result as the main result **and** the effect of each
  deviation (one switch at a time) as a sensitivity table, so the reader can
  see which choices drive N\*.
* Do not describe the 2017–2018 run as a clean, untouched confirmatory test.
