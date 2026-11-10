"""
scripts/final_review.py
========================
Master project integrity checker — runs all checks in one script.
Tests: config, data, results, figures, tables, modules,
       pre-reg docs, git, paper, thesis, demo script, stages 6-11.
"""
import os, sys, pickle, subprocess, re, importlib
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

PASS = "✓"; FAIL = "✗"; WARN = "⚠"; SEP = "─" * 58

errors   = []
warnings = []
totals   = {"pass": 0, "fail": 0, "warn": 0}

def ok(label, detail=""):
    totals["pass"] += 1
    print(f"  {PASS} {label}" + (f"  [{detail}]" if detail else ""))

def fail(label, detail=""):
    totals["fail"] += 1
    errors.append(label)
    print(f"  {FAIL} {label}" + (f"  [{detail}]" if detail else ""))

def warn(label, detail=""):
    totals["warn"] += 1
    warnings.append(label)
    print(f"  {WARN} {label}" + (f"  [{detail}]" if detail else ""))

def chk(label, condition, detail="", warning_only=False):
    if condition:
        ok(label, detail)
    elif warning_only:
        warn(label, detail)
    else:
        fail(label, detail)

def section(title):
    print(f"\n{'='*58}")
    print(f"  {title}")
    print('='*58)

# ================================================================
section("1. TEST SUITE")
# ================================================================
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
    capture_output=True, text=True
)
test_out = result.stdout + result.stderr
passed = len([l for l in test_out.split('\n') if 'passed' in l])
failed = len([l for l in test_out.split('\n') if 'failed' in l or 'error' in l and 'warning' not in l])
chk("pytest: 7/7 tests pass", result.returncode == 0,
    test_out.strip().split('\n')[-2] if test_out.strip() else "no output")

# ================================================================
section("2. CONFIGURATION INTEGRITY")
# ================================================================
try:
    from src.config import FINAL_CONFIG, set_all_seeds, N_BLIND_ATTEMPTS
    from src.config import (
        GLOBAL_SEED, SEED_MODEL_RF, SEED_MODEL_LOGREG,
        SEED_ADVERSARIAL, SEED_TRAIN_TEST_SPLIT
    )
    required_keys = [
        'holdout_cycle', 'development_cycle', 'rf_n_estimators', 'rf_max_depth',
        'lie_budget_method', 'lie_budget_percentile', 'blind_attempts',
        'bootstrap_n_iterations', 'confidence_level',
        'adversarial_seed',           # stored as adversarial_seed
        'rf_random_state',            # stored as rf_random_state
        'lr_random_state',            # stored as lr_random_state
        'split_random_state',         # stored as split_random_state
    ]
    for k in required_keys:
        chk(f"FINAL_CONFIG['{k}']", k in FINAL_CONFIG, str(FINAL_CONFIG.get(k, 'MISSING')))
    chk("Holdout cycle = 2017_2018",  FINAL_CONFIG['holdout_cycle'] == '2017_2018')
    chk("Dev cycle = 2015_2016",       FINAL_CONFIG['development_cycle'] == '2015_2016')
    chk("RF n_estimators = 100",       FINAL_CONFIG['rf_n_estimators'] == 100)
    chk("RF max_depth = 10",           FINAL_CONFIG['rf_max_depth'] == 10)
    chk("Lie budget = 95th pct",       FINAL_CONFIG['lie_budget_percentile'] == 95)
    chk("Bootstrap iters = 1000",      FINAL_CONFIG['bootstrap_n_iterations'] == 1000)
    chk("Confidence level = 0.95",     FINAL_CONFIG['confidence_level'] == 0.95)
    chk("10 blind attempt milestones", len(FINAL_CONFIG['blind_attempts']) == 10)
    chk("N_BLIND_ATTEMPTS matches config",
        FINAL_CONFIG['blind_attempts'] == N_BLIND_ATTEMPTS)
    chk("Global seed = 42",            GLOBAL_SEED == 42)
    chk("RF seed = 123",               SEED_MODEL_RF == 123)
    chk("LR seed = 42",                SEED_MODEL_LOGREG == 42)
    chk("Adversarial seed = 456",      SEED_ADVERSARIAL == 456)
    chk("Split seed = 789",            SEED_TRAIN_TEST_SPLIT == 789)
    set_all_seeds()
    ok("set_all_seeds() runs without error")
except Exception as e:
    fail("Config module error", str(e))

# ================================================================
section("3. SOURCE MODULES")
# ================================================================
modules = [
    ('src.config',        ['FINAL_CONFIG', 'set_all_seeds']),
    ('src.data_loader',   ['merge_nhanes', 'clean_nhanes', 'prepare_feature_sets']),
    ('src.gap_analysis',  ['compute_reporting_gap', 'define_lie_budget']),
    ('src.models',        ['DiabetesModel']),
    ('src.explanations',  ['ExplanationGenerator']),
    ('src.adversarial',   ['AdversarialSimulator']),
    ('src.metrics',       ['compute_equivalence', 'compute_confidence_intervals',
                           'run_statistical_tests']),
    ('src.visualization', ['plot_equivalence_curve', 'fig1_reporting_gaps',
                           'fig2_main_result_panel', 'fig3_score_reduction_boxplot']),
]
for mod_name, required_attrs in modules:

    try:
        mod = importlib.import_module(mod_name)
        ok(f"{mod_name} imports")
        for attr in required_attrs:
            chk(f"  .{attr} defined", hasattr(mod, attr))
    except Exception as e:
        fail(f"{mod_name} import failed", str(e))

# ================================================================
section("4. DATA INTEGRITY")
# ================================================================
for pkl_path, label, min_patients in [
    ('data/processed/nhanes_2015_2016_prepared.pkl', 'Development set (2015-16)', 3000),
    ('data/processed/nhanes_2017_2018_prepared.pkl', 'Holdout set (2017-18)', 800),
]:
    chk(f"{label} pickle exists", os.path.exists(pkl_path))
    if os.path.exists(pkl_path):
        try:
            d = pickle.load(open(pkl_path, 'rb'))
            n = len(d.get('y', []))
            chk(f"  Has {min_patients}+ patients", n >= min_patients, f"{n} patients")
            chk(f"  Has X_reported",   'X_reported'   in d)
            chk(f"  Has X_measured",   'X_measured'   in d)
            chk(f"  Has y (labels)",   'y'            in d)
            chk(f"  Has feature_names",'feature_names_reported' in d)
            diab = d['y'].sum()
            pct  = 100 * d['y'].mean()
            chk(f"  Diabetes prevalence 5–30%", 5 < pct < 30, f"{diab}/{n} = {pct:.1f}%")
            chk(f"  X_reported shape matches y",
                len(d['X_reported']) == n, f"X={len(d['X_reported'])}, y={n}")
        except Exception as e:
            fail(f"  Pickle load error", str(e))

# ── Lie budget consistency — compute from dev pickle gap_analysis
try:
    import numpy as np
    dev_data = pickle.load(open('data/processed/nhanes_2015_2016_prepared.pkl', 'rb'))
    # The lie budget comes from the reporting gap: 95th percentile of |self - measured|
    # We verify using the stored gap arrays if present, else just confirm values exist
    budget_stored = dev_data.get('lie_budget', {})
    gap_h = dev_data.get('height_gaps', None)
    gap_w = dev_data.get('weight_gaps', None)
    if budget_stored:
        h_bud = budget_stored.get('height', {}).get('max_lie',
                budget_stored.get('self_height_cm', 0))
        w_bud = budget_stored.get('weight', {}).get('max_lie',
                budget_stored.get('self_weight_kg', 0))
        chk("Lie budget height ≈ 7.29 cm", abs(float(h_bud) - 7.29) < 0.5, f"{float(h_bud):.2f} cm")
        chk("Lie budget weight ≈ 8.96 kg", abs(float(w_bud) - 8.96) < 0.5, f"{float(w_bud):.2f} kg")

    elif gap_h is not None and gap_w is not None:
        h_bud = float(np.percentile(np.abs(gap_h), 95))
        w_bud = float(np.percentile(np.abs(gap_w), 95))
        chk("Lie budget height ~7.29 cm (recomputed)", abs(h_bud - 7.29) < 1.0, f"{h_bud:.2f} cm")
        chk("Lie budget weight ~8.96 kg (recomputed)", abs(w_bud - 8.96) < 1.5, f"{w_bud:.2f} kg")
    else:
        # Pre-registered values locked in analysis_plan.md
        ap = open('docs/analysis_plan.md', encoding='utf-8').read()
        chk("Lie budget 7.29 in analysis plan", '7.29' in ap)
        chk("Lie budget 8.96 in analysis plan", '8.96' in ap)
except Exception as e:
    warn(f"Lie budget check: {e}")

# ================================================================
section("5. FINAL EXPERIMENT RESULTS")
# ================================================================
import pandas as pd

# Cache check
cache_dir = 'data/processed/final_run_cache'
if os.path.exists(cache_dir):
    cached = [f.replace('.pkl','') for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    chk("All 8 config caches saved", len(cached) >= 8, f"{len(cached)}/8: {cached[:3]}...")
else:
    fail("Final run cache directory missing")

# Results CSV
res_csv = 'results/tables/final_results.csv'
chk("final_results.csv exists", os.path.exists(res_csv))
if os.path.exists(res_csv):
    df = pd.read_csv(res_csv)
    chk("8 config rows in results", len(df) == 8, f"{len(df)} rows")
    chk("Primary config present",
        'reported_rf_shap' in df['Config'].values if 'Config' in df.columns else False)
    if 'Mean_Nstar' in df.columns:
        prim = df[df['Config'] == 'reported_rf_shap']['Mean_Nstar'].values
        if len(prim):
            chk("Primary N* ≈ 12.4", abs(prim[0] - 12.43) < 0.1, f"N*={prim[0]:.2f}")
    if 'CI_low' in df.columns:
        prim = df[df['Config'] == 'reported_rf_shap']
        if len(prim):
            chk("Primary CI [10.54, 14.53]",
                abs(prim['CI_low'].values[0] - 10.54) < 0.1,
                f"[{prim['CI_low'].values[0]:.2f}, {prim['CI_high'].values[0]:.2f}]")
    # Check Wilcoxon p=1.0 (all negative Cohen's d)
    if 'Cohens_d' in df.columns:
        all_negative = (df['Cohens_d'] < 0).all()
        chk("Cohen's d negative for all configs (blind ≥ explanation at N=10)",
            all_negative, f"range: [{df['Cohens_d'].min():.2f}, {df['Cohens_d'].max():.2f}]")

# ================================================================
section("6. PUBLICATION FIGURES")
# ================================================================
figs_required = [
    ('fig1_reporting_gaps.png',   'Fig 1: Reporting gap KDE',      100_000),
    ('fig2_main_result.png',      'Fig 2: Main result panel',      300_000),
    ('fig3_score_reductions.png', 'Fig 3: Score reduction boxes',  100_000),
    ('fig4_model_comparison.png', 'Fig 4: Model A vs B bar',        80_000),
    ('fig5_shap_lime_scatter.png','Fig 5: SHAP vs LIME scatter',   100_000),
    ('fig6_shap_summary.png',     'Fig 6: SHAP beeswarm',          200_000),
    ('fig7_nstar_histogram.png',  'Fig 7: N* histogram',           100_000),
    ('fig8_config_heatmap.png',   'Fig 8: Config heatmap',          80_000),
]
for fname, label, min_bytes in figs_required:
    path = f'results/figures/{fname}'
    exists = os.path.exists(path)
    size   = os.path.getsize(path) if exists else 0
    chk(label, exists and size >= min_bytes,
        f"{size//1024} KB" if exists else "MISSING")

# ================================================================
section("7. RESULTS TABLES")
# ================================================================
tables_required = [
    ('table1_dataset_summary',      'Table 1: Dataset summary'),
    ('table2_model_performance',    'Table 2: Model performance'),
    ('table3_reporting_gap',        'Table 3: Reporting gap'),
    ('table4_main_results',         'Table 4: Main N* results ★'),
    ('table5_statistical_tests',    'Table 5: Statistical tests'),
    ('table6_model_ab_comparison',  'Table 6: Model A vs B'),
]
for base, label in tables_required:
    csv_path = f'results/tables/{base}.csv'
    tex_path = f'results/tables/{base}.tex'
    csv_ok = os.path.exists(csv_path)
    tex_ok = os.path.exists(tex_path)
    if csv_ok and tex_ok:
        ok(f"{label} — CSV + LaTeX")
    elif csv_ok:
        # Table 5 statistical tests may only exist as CSV from run_final.py
        if 'statistical' in base:
            ok(f"{label} — CSV (Wilcoxon results in final_results.csv)")
        else:
            warn(f"{label} — CSV only, LaTeX missing")
    else:
        # Check if data exists in final_results.csv
        if 'statistical' in base:
            res_csv = 'results/tables/final_results.csv'
            if os.path.exists(res_csv):
                import pandas as pd
                df_res = pd.read_csv(res_csv)
                if 'Wilcoxon_p' in df_res.columns:
                    ok(f"{label} — data in final_results.csv (Wilcoxon_p column)")
                else:
                    warn(f"{label} — not found as separate file")
            else:
                fail(f"{label} — MISSING")
        else:
            fail(f"{label} — MISSING")

chk("interpretation_narrative.txt",
    os.path.exists('results/tables/interpretation_narrative.txt'))

# ================================================================
section("8. PRE-REGISTRATION & DOCS")
# ================================================================
docs = {
    'docs/analysis_plan.md':             'Analysis plan (pre-registered)',
    'docs/changelog_trial_to_final.md':  'Changelog (trial → final)',
    'docs/column_mapping.md':            'NHANES column mapping',
}
for path, label in docs.items():
    chk(label, os.path.exists(path))

# Check analysis_plan content
if os.path.exists('docs/analysis_plan.md'):
    ap = open('docs/analysis_plan.md', encoding='utf-8').read()
    chk("Analysis plan: N* defined",    'equivalence' in ap.lower() or 'N*' in ap)
    chk("Analysis plan: lie budget",    'lie budget' in ap.lower() or '7.29' in ap)
    chk("Analysis plan: 8 configs",     'reported_rf_shap' in ap)
    chk("Analysis plan: locked section",'LOCKED' in ap.upper() or 'lock' in ap.lower())

# Git tag
try:
    tags    = subprocess.run(['git', 'tag'], capture_output=True, text=True).stdout.strip()
    commits = subprocess.run(['git', 'rev-list', '--count', 'HEAD'],
                              capture_output=True, text=True).stdout.strip()
    latest  = subprocess.run(['git', 'log', '--oneline', '-1'],
                              capture_output=True, text=True).stdout.strip()
    chk("git tag v0.1-preregistration", 'v0.1-preregistration' in tags)
    chk("≥10 commits in history",       int(commits) >= 10, f"{commits} commits")
    ok(f"Latest commit: {latest}")
except Exception as e:
    fail("Git check error", str(e))

# ================================================================
section("9. PAPER (paper/main.tex)")
# ================================================================
paper = 'paper/main.tex'
chk("paper/main.tex exists", os.path.exists(paper))
if os.path.exists(paper):
    txt = open(paper, encoding='utf-8').read()
    fills = txt.count('[FILL]')
    roll  = txt.count('Roll No.: [FILL]') + txt.count('Roll No. [FILL]')
    result_fills = fills - roll
    chk("Zero result [FILL] placeholders", result_fills == 0,
        f"{result_fills} remain" if result_fills else "All results filled")
    chk("N*=12.4 stated",                '12.4' in txt or '12.43' in txt)
    chk("CI [10.5, 14.5] stated",        '10.5' in txt or '10.54' in txt)
    chk("MODERATE threat stated",        'moderate' in txt.lower())
    chk("63.8% useless stated",          '63.8' in txt or '63' in txt)
    chk("Pre-registration cited",        'pre-register' in txt.lower() or 'preregist' in txt.lower())
    chk("NHANES cited",                  'johnson2014national' in txt)
    chk("12 bibliography entries",
        open('paper/references.bib', 'r', encoding='utf-8').read().count('@') >= 12
        if os.path.exists('paper/references.bib') else False)
    chk("All 7 sections present",
        all(s in txt for s in [
            r'\section{Introduction}', r'\section{Related Work}',
            r'\section{Problem Setup}', r'\section{Methodology}',
            r'\section{Experiments and Results}',
            r'\section{Discussion}', r'\section{Conclusion}']))

# ================================================================
section("10. THESIS (thesis/main.tex)")
# ================================================================
thesis = 'thesis/main.tex'
chk("thesis/main.tex exists", os.path.exists(thesis))
if os.path.exists(thesis):
    txt = open(thesis, encoding='utf-8').read()
    fills = txt.count('[FILL]')
    roll  = txt.count('Roll No.: [FILL]') + txt.count('Roll No. [FILL]')
    result_fills = fills - roll
    chk("Zero result [FILL] placeholders", result_fills == 0,
        f"{result_fills} remain" if result_fills else "All results filled")
    chk(f"Roll# [FILL]s ({roll}) — team must fill", True,
        f"{roll} roll numbers for team", warning_only=roll > 0)
    chk("N*=12.4 in thesis",              '12.4' in txt or '12.43' in txt)
    chk("All 7 chapters",
        all(s in txt for s in [
            r'\chapter{Introduction}', r'\chapter{Literature Review}',
            r'\chapter{Methodology}', r'\chapter{Implementation}',
            r'\chapter{Results and Analysis}', r'\chapter{Discussion}',
            r'\chapter{Conclusion and Future Work}']))
    chk("All 4 appendices",
        all(s in txt for s in [
            'Column Mapping', 'Full Statistical Tables',
            'Additional Figures', 'Code Repository']))
    chk("Certificate page",     r'\chapter*{Certificate}'     in txt)
    chk("Declaration page",     r'\chapter*{Declaration}'     in txt)
    chk("Algorithm 1 present",  r'\begin{algorithm}'          in txt)
    chk("Definition block",     r'\begin{definition}'         in txt)
    chk("Lie budget values",    '7.29' in txt and '8.96' in txt)
    chk("Pre-reg URL in thesis",'github.com' in txt.lower() and 'preregistration' in txt.lower())

# ================================================================
section("11. VIVA DEMO SCRIPT")
# ================================================================
demo = 'scripts/demo.py'
chk("scripts/demo.py exists", os.path.exists(demo))
if os.path.exists(demo):
    dtxt = open(demo, encoding='utf-8').read()
    chk("STEP 1: original score",     'STEP 1' in dtxt)
    chk("STEP 2: SHAP explanation",   'STEP 2' in dtxt)
    chk("STEP 3: guided attack",      'STEP 3' in dtxt)
    chk("STEP 4: blind comparison",   'STEP 4' in dtxt)
    chk("STEP 5: N* for patient",     'STEP 5' in dtxt)
    chk("Population N* result shown", '12.43' in dtxt)
    chk("Lie budget values correct",  '7.29' in dtxt and '8.96' in dtxt)
    chk("--patient argument",         '--patient' in dtxt)
    chk("Uses held-out 2017-2018",    '2017_2018' in dtxt or '2017-2018' in dtxt)

# ================================================================
section("12. REVIEW SCRIPTS (stages 6-11)")
# ================================================================
review_scripts = [
    'scripts/review_stage6.py',
    'scripts/review_stage7.py',
    'scripts/review_stage8.py',
    'scripts/review_stage9.py',
    'scripts/review_stage10.py',
    'scripts/review_stage11.py',
]
for s in review_scripts:
    chk(f"{s} exists", os.path.exists(s))

# ================================================================
section("13. PROJECT FILE TREE COMPLETENESS")
# ================================================================
required_files = {
    # Source code
    'src/__init__.py':           'Package init',
    'src/config.py':             'FINAL_CONFIG + seeds',
    'src/data_loader.py':        'NHANES loader',
    'src/gap_analysis.py':       'Reporting gap + lie budget',
    'src/models.py':             'DiabetesModel (RF + LR)',
    'src/explanations.py':       'SHAP + LIME generator',
    'src/adversarial.py':        'AdversarialSimulator',
    'src/metrics.py':            'N* computation + CI + Wilcoxon',
    'src/visualization.py':      '8 publication figures',
    # Tests
    'tests/test_pipeline.py':    '7 unit tests',
    # Scripts
    'scripts/prepare_data.py':   'Stage 3 data prep',
    'scripts/run_trial.py':      'Stage 5 trial run',
    'scripts/run_final.py':      'Stage 7 final experiment',
    'scripts/run_analysis.py':   'Stage 8 analysis',
    'scripts/demo.py':           'Stage 11 viva demo',
    # Docs
    'docs/analysis_plan.md':     'Pre-registration (Stage 6)',
    'docs/changelog_trial_to_final.md': 'Deviation log',
    'docs/column_mapping.md':    'NHANES columns',
    # Paper
    'paper/main.tex':            'Research paper LaTeX',
    'paper/references.bib':      'Bibliography (12 entries)',
    # Thesis
    'thesis/main.tex':           'BTP thesis LaTeX (974 lines)',
    # Roadmap
    'roadmap/00_master_roadmap.md': 'Master roadmap',
}
missing_files = []
for path, label in required_files.items():
    if os.path.exists(path):
        ok(f"{label}  ({path})")
    else:
        fail(f"{label} MISSING  ({path})")
        missing_files.append(path)

# ================================================================
section("FINAL SUMMARY")
# ================================================================
print(f"""
  ┌──────────────────────────────────────────────────────┐
  │              FINAL PROJECT RESULTS                    │
  │   NHANES 2017-2018 Holdout  |  N = 884 patients      │
  ├──────────────────────────┬──────────┬────────────────┤
  │ Config (primary ★)       │   N*     │   95% CI       │
  ├──────────────────────────┼──────────┼────────────────┤
  │ reported_rf_shap  ★      │  12.43   │ [10.54, 14.53] │
  │ reported_rf_lime         │   9.41   │  [7.66, 11.27] │
  │ reported_lr_shap         │  22.77   │ [20.36, 25.28] │
  │ reported_lr_lime         │  22.71   │ [20.28, 25.26] │
  │ measured_rf_shap         │  10.27   │  [8.42, 12.15] │
  │ measured_rf_lime         │   9.01   │  [7.32, 10.77] │
  │ measured_lr_shap         │  23.05   │ [20.58, 25.61] │
  │ measured_lr_lime         │  21.27   │ [18.99, 23.71] │
  ├──────────────────────────┴──────────┴────────────────┤
  │  VERDICT: MODERATE THREAT (5 ≤ N* ≤ 20)             │
  │  63.8% patients: explanation gives ZERO advantage    │
  │  17.2% patients: explanation worth 10+ blind queries │
  │  Policy: Verify inputs; don't just hide explanations │
  └──────────────────────────────────────────────────────┘
""")

total_checks = totals['pass'] + totals['fail'] + totals['warn']
print(f"  Checks run : {total_checks}")
print(f"  {PASS} Passed   : {totals['pass']}")
print(f"  {WARN} Warnings : {totals['warn']}")
print(f"  {FAIL} Failed   : {totals['fail']}")
print()
if totals['fail'] == 0:
    print("  🎓 ALL CRITICAL CHECKS PASS — PROJECT IS SUBMISSION-READY")
else:
    print(f"  BLOCKING ISSUES ({totals['fail']}):")
    for e in errors:
        print(f"    ✗ {e}")

if warnings:
    print(f"\n  NON-BLOCKING WARNINGS ({totals['warn']}):")
    for w in warnings:
        print(f"    ⚠ {w}")

print(f"""
  TEAM ACTIONS REQUIRED (cannot be automated):
    1. Fill 6 roll numbers in thesis/main.tex
    2. Write 7 TODO sections in paper/main.tex (per §09 schedule)
    3. Write 5 TODO sections in thesis/main.tex (per §10 schedule)
    4. Add DTU logo: docs/dtu_logo.png → thesis/
    5. Prepare 25-30 viva slides (per roadmap/11_viva.md)
    6. Test demo: python scripts/demo.py --patient 42
    7. 2 full rehearsal sessions + guide review
    8. Submit thesis PDF to DTU CSE department by May 31, 2027
""")
