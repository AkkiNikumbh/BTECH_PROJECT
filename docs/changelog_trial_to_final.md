# Changelog: Trial Run → Pre-Registration
## GP-109 BTP Project

This document records every decision and change made between
the Stage 5 trial run and the Stage 6 pre-registration lock.

---

## Changes Made Based on Trial Run Findings

### 1. Critical Bug Fix: Blind Attack Simulation (IMPACT: HIGH)

**What changed:** `adversarial.py` — `blind_attack` rewritten to `blind_attack_multi_n`

**Old behaviour (incorrect):**
- For each N in [1, 2, 3, 5, 10, 15, 20, 30, 50, 100], ran a SEPARATE independent simulation
- Each fresh simulation got a new random draw from scratch
- This over-estimated blind performance at higher N, making N* artificially inflated

**New behaviour (correct):**
- Runs ONE continuous simulation up to max N = 100
- At each attempt, keeps the best score found SO FAR
- Records best reduction at each milestone N
- This correctly models sequential trial-and-error where the adversary
  accumulates attempts and keeps the best result seen

**Effect on results:** N* will be LOWER (explanations look LESS valuable)
because blind performance at high N was previously over-estimated.

**Why acceptable:** This is the scientifically correct model. The previous
version was a bug, not a design choice. The change was made BEFORE the
held-out dataset was analyzed.

---

### 2. Bug Fix: Metrics Interpolation Monotonicity (IMPACT: MEDIUM)

**What changed:** `metrics.py` — `compute_equivalence` function

**Old behaviour:** Used raw blind reduction values as x-axis for `interp1d`.
These were not guaranteed monotonic (could cause scipy errors or wrong results).

**New behaviour:** Enforces monotonicity via cumulative max (running best),
then deduplicates for stable interpolation. Uses `bounds_error=False` to
handle edge cases cleanly.

**Effect on results:** More stable, fewer NaN equivalence values.

---

### 3. Bug Fix: LIME Feature Name Extraction (IMPACT: LOW)

**What changed:** `explanations.py` — `get_top_gameable_feature`

**Old behaviour:** `feat.split(' ')[0]` — worked for simple conditions like
`self_height_cm > 1.23` but failed for range conditions like
`0.5 < self_weight_kg <= 2.1`

**New behaviour:** Checks if any gameable feature name appears as a SUBSTRING
of the LIME condition string. Handles all LIME output formats.

**Effect on results:** LIME configurations may now correctly identify the
target feature in more cases, potentially changing LIME N* values.

---

### 4. Bug Fix: Matplotlib Backend (IMPACT: NONE on results)

**What changed:** `visualization.py` — added `matplotlib.use('Agg')`

**Reason:** Background subprocess on Windows caused tkinter threading crash
when matplotlib tried to use GUI backend. Agg is non-interactive and
safe for subprocesses.

**Effect on results:** No effect on numeric results. Only affects whether
figures can be saved successfully.

---

## Settings NOT Changed (Confirmed Same as Trial)

| Setting | Trial Value | Final Value | Status |
|---------|------------|------------|--------|
| GLOBAL_SEED | 42 | 42 | UNCHANGED |
| RF n_estimators | 100 | 100 | UNCHANGED |
| RF max_depth | 10 | 10 | UNCHANGED |
| RF random_state | 123 | 123 | UNCHANGED |
| LR random_state | 42 | 42 | UNCHANGED |
| Test size | 0.2 | 0.2 | UNCHANGED |
| Split seed | 789 | 789 | UNCHANGED |
| Lie budget method | 95th pctile | 95th pctile | UNCHANGED |
| Height budget | ±7.29 cm | ±7.29 cm | UNCHANGED |
| Weight budget | ±8.96 kg | ±8.96 kg | UNCHANGED |
| Blind N list | [1,2,3,5,10,15,20,30,50,100] | [1,2,3,5,10,15,20,30,50,100] | UNCHANGED |
| LIME n_samples | 500 | 500 | UNCHANGED |
| LIME n_features | 5 | 5 | UNCHANGED |
| LIME seed | 42 | 42 | UNCHANGED |
| Bootstrap iters | 1000 | 1000 | UNCHANGED |
| Confidence level | 0.95 | 0.95 | UNCHANGED |
| Adversarial seed | 456 | 456 | UNCHANGED |

---

## Preliminary Trial Results (2015-2016, 200 patients, reported_rf_shap)

| Metric | Value |
|--------|-------|
| Mean N* | 9.5 blind attempts |
| 95% CI (mean) | [6.11, 13.17] |
| Median N* | 0.0 |
| % explanation useless | 63% |
| % explanation worth 10+ | 14% |

*Note: These numbers were from the UNFIXED version of blind_attack. The fixed
version will yield different numbers for the final trial summary.*

---

*Document created: 28 October 2026*
*Author: GP-109 team*
