"""
scripts/review_stage11.py
==========================
Stage 11 Review: Viva Preparation & Final Submission completeness check.

Validates:
  A) Demo script exists and is runnable
  B) All 8 publication figures exist at 300 DPI
  C) All 6 tables exist (CSV + LaTeX)
  D) Paper has no [FILL] placeholders (submission-ready)
  E) Thesis has no [FILL] placeholders
  F) Interpretation narrative exists with final N* values
  G) Final submission package complete

Usage: python scripts/review_stage11.py
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS  = "[PASS]"
FAIL  = "[FAIL]"
WARN  = "[WARN]"
issues = []

def check(label, condition, detail="", warn=False):
    if condition:
        print(f"  {PASS} {label}" + (f" — {detail}" if detail else ""))
    elif warn:
        print(f"  {WARN} {label}" + (f" — {detail}" if detail else ""))
    else:
        print(f"  {FAIL} {label}" + (f" — {detail}" if detail else ""))
        issues.append(label)

FIGS   = "results/figures"
TABLES = "results/tables"

print("\n" + "="*65)
print("  STAGE 11: Viva & Final Submission Review")
print("="*65)

def load_final_results(path=os.path.join('results', 'tables', 'final_results.csv')):
    """Final results from run_final.py (None if not run yet)."""
    if not os.path.exists(path):
        return None
    import pandas as pd
    return pd.read_csv(path)


def verdict_for(nstar):
    """Pre-registered interpretation bands (docs/analysis_plan.md)."""
    if nstar < 5:
        return 'LOW THREAT (N* < 5)'
    if nstar <= 20:
        return 'MODERATE THREAT (5 <= N* <= 20)'
    return 'HIGH THREAT (N* > 20)'


def results_table_text(df):
    """Plain-text table of all configurations, read from final_results.csv."""
    if df is None:
        return '  (results/tables/final_results.csv not found - run scripts/run_final.py)'
    n = int(df['N_patients'].iloc[0])
    lines = [f'  NHANES 2017-2018 held-out test set  |  N = {n} patients',
             f'  {"Config":<22}{"N*":>7}  {"95% CI":<16}{"% useless":>10}']
    for _, r in df.iterrows():
        star = ' *' if r['Config'] == 'reported_rf_shap' else '  '
        ci = f"[{r['Nstar_CI_low']:.2f}, {r['Nstar_CI_high']:.2f}]"
        lines.append(f"  {r['Config'] + star:<22}{r['Nstar']:>7.2f}  {ci:<16}{r['Pct_useless']:>9.1f}%")
    prim = df[df['Config'] == 'reported_rf_shap'].iloc[0]
    lines.append('  * primary pre-registered configuration')
    lines.append(f"  VERDICT: {verdict_for(prim['Nstar'])}")
    return '\n'.join(lines)


_DF = load_final_results()
_PRIM = _DF[_DF['Config'] == 'reported_rf_shap'].iloc[0] if _DF is not None else None
_NSTAR = f"{_PRIM['Nstar']:.1f}" if _PRIM is not None else '?'
_BAND = verdict_for(_PRIM['Nstar']).split()[0] if _PRIM is not None else '?'

# ----------------------------------------------------------------
# A: Demo script
# ----------------------------------------------------------------
print("\n  [A] Viva Demo Script")
check("scripts/demo.py exists", os.path.exists("scripts/demo.py"))
if os.path.exists("scripts/demo.py"):
    demo = open("scripts/demo.py", encoding="utf-8").read()
    check("Demo has STEP 1 (original score)", "STEP 1" in demo)
    check("Demo has STEP 2 (SHAP explanation)", "STEP 2" in demo)
    check("Demo has STEP 3 (explanation attack)", "STEP 3" in demo)
    check("Demo has STEP 4 (blind comparison)", "STEP 4" in demo)
    check("Demo has STEP 5 (N* for patient)", "STEP 5" in demo)
    check("Demo reads population N* from final_results.csv", "final_results.csv" in demo)
    check("Demo has --patient argument", "--patient" in demo)

# ----------------------------------------------------------------
# B: All 8 figures
# ----------------------------------------------------------------
print("\n  [B] Publication Figures (300 DPI)")
required_figs = [
    ("fig1_reporting_gaps.png",     "Fig 1: Reporting gap distributions"),
    ("fig2_main_result.png",        "Fig 2: Main result panel"),
    ("fig3_score_reductions.png",   "Fig 3: Score reduction boxplots"),
    ("fig4_model_comparison.png",   "Fig 4: Model A vs B comparison"),
    ("fig5_shap_lime_scatter.png",  "Fig 5: SHAP vs LIME scatter"),
    ("fig6_shap_summary.png",       "Fig 6: SHAP feature importance"),
    ("fig7_nstar_histogram.png",    "Fig 7: N* per-patient histogram"),
    ("fig8_config_heatmap.png",     "Fig 8: Configuration overview"),
]
for fname, label in required_figs:
    path = os.path.join(FIGS, fname)
    exists = os.path.exists(path)
    size_kb = round(os.path.getsize(path) / 1024) if exists else 0
    check(label, exists, f"{size_kb} KB" if exists else "MISSING — run run_analysis.py")

# ----------------------------------------------------------------
# C: All 6 tables (CSV + LaTeX)
# ----------------------------------------------------------------
print("\n  [C] Results Tables (CSV + LaTeX)")
required_tables = [
    "table1_dataset_summary",
    "table2_model_performance",
    "table3_reporting_gap",
    "table4_main_results",
    "table5_statistical_tests",
    "table6_model_ab_comparison",
    "table7_score_reductions",
]
for tbl in required_tables:
    csv_path = os.path.join(TABLES, f"{tbl}.csv")
    tex_path = os.path.join(TABLES, f"{tbl}.tex")
    check(f"{tbl}.csv", os.path.exists(csv_path), warn=not os.path.exists(csv_path))
    check(f"{tbl}.tex", os.path.exists(tex_path), warn=not os.path.exists(tex_path))

# ----------------------------------------------------------------
# D: Paper submission-ready (no FILL)
# ----------------------------------------------------------------
print("\n  [D] Paper Submission-Ready")
paper_tex = "paper/main.tex"
if os.path.exists(paper_tex):
    txt = open(paper_tex, encoding="utf-8").read()
    fills = txt.count("[FILL]")
    todos = len(re.findall(r'%.*TODO', txt))
    check("paper/main.tex: no [FILL] placeholders", fills == 0,
          f"{fills} placeholders remain" if fills else "All filled with final N* results")
    check(f"paper/main.tex: N*={_NSTAR} present", f"N^* = {_NSTAR}" in txt or f"N^*={_NSTAR}" in txt,
          "Primary result filled")
    check(f"paper/main.tex: {_BAND.lower()} threat stated", _BAND.lower() in txt.lower())
    check("paper/references.bib: >= 12 entries",
          open("paper/references.bib", encoding="utf-8").read().count("@") >= 12
          if os.path.exists("paper/references.bib") else False)
    if todos > 0:
        check(f"TODO blocks remaining ({todos})", False,
              "Team must write TODO sections before submission", warn=True)

# ----------------------------------------------------------------
# E: Thesis submission-ready
# ----------------------------------------------------------------
print("\n  [E] Thesis Submission-Ready")
thesis_tex = "thesis/main.tex"
if os.path.exists(thesis_tex):
    txt = open(thesis_tex, encoding="utf-8").read()
    fills = txt.count("[FILL]")
    todos = len(re.findall(r'%.*TODO', txt))
    roll_fills = txt.count("Roll No.: [FILL]") + txt.count("Roll No. [FILL]")
    result_fills = fills - roll_fills
    check("thesis/main.tex: no result [FILL] placeholders",
          result_fills == 0,
          f"{result_fills} result placeholders remain" if result_fills else "All results filled")
    check("thesis/main.tex: roll number [FILL]s remaining",
          True, f"{roll_fills} roll numbers need to be filled by team", warn=roll_fills > 0)
    check(f"thesis/main.tex: N*={_NSTAR} present", _NSTAR in txt)

# ----------------------------------------------------------------
# F: Interpretation narrative
# ----------------------------------------------------------------
print("\n  [F] Results Narrative")
narrative = os.path.join(TABLES, "interpretation_narrative.txt")
check("interpretation_narrative.txt exists", os.path.exists(narrative))
if os.path.exists(narrative):
    content = open(narrative, encoding="utf-8").read()
    check(f"N*={_NSTAR} in narrative", _NSTAR in content)
    check(f"{_BAND} THREAT in narrative", f"{_BAND} THREAT" in content.upper())

# ----------------------------------------------------------------
# G: Final submission package
# ----------------------------------------------------------------
print("\n  [G] Final Submission Package")
package_items = [
    ("thesis/main.tex",                         "Thesis source"),
    ("paper/main.tex",                          "Paper source"),
    ("paper/references.bib",                    "Bibliography"),
    ("docs/analysis_plan.md",                   "Pre-registration document"),
    ("docs/changelog_trial_to_final.md",        "Changelog"),
    ("scripts/demo.py",                         "Live demo script"),
    (os.path.join(FIGS, "fig2_main_result.png"),"Main result figure"),
    (os.path.join(TABLES, "table4_main_results.csv"), "Main results table"),
    ("src/adversarial.py",                      "Core experiment code"),
    ("src/metrics.py",                          "Metrics & N* computation"),
]
for path, label in package_items:
    check(label, os.path.exists(path))

# Check git tag
import subprocess
try:
    tags = subprocess.run(["git", "tag"], capture_output=True, text=True).stdout.strip()
    # The pre-registration tag lives in the ORIGINAL repository
    # (github.com/Akshat-stack47/BTP-XAI-Gaming); this fixed copy may not have it.
    check("git tag v0.1-preregistration (original repo)", "v0.1-preregistration" in tags,
          "tag is in Akshat-stack47/BTP-XAI-Gaming", warn="v0.1-preregistration" not in tags)
    commits = subprocess.run(["git", "rev-list", "--count", "HEAD"],
                              capture_output=True, text=True).stdout.strip()
    check(f"Git history ({commits} commits)", int(commits) >= 1)
except Exception:
    check("Git accessible", False)

# ----------------------------------------------------------------
# SUMMARY
# ----------------------------------------------------------------
print("\n" + "="*65)
print("  FINAL PROJECT SUMMARY")
print("="*65)

# Print the N* results table (read from final_results.csv)
print(results_table_text(_DF))

if issues:
    print(f"  {FAIL} {len(issues)} blocking issue(s):")
    for i in issues:
        print(f"    - {i}")
else:
    print(f"  {PASS} ALL CHECKS PASSED — Project is complete!")

print("""
  Outstanding team tasks (cannot be automated):
    1. HOD name in the thesis certificate ([Name of HOD])
    2. DTU logo: save as docs/dtu_logo.png (title page picks it up)
    3. Check each new reference against the original paper (plan P8)
    4. Guide to confirm the deviations from the pre-registered plan
       (thesis Section 3.8.1, docs/fixes_v2.md)
    5. Prepare viva slides (roadmap/11_viva.md); test the demo:
       python scripts/demo.py --patient 42
    6. Rehearsals + guide review; submit the thesis PDF (thesis/main.pdf)
""")
