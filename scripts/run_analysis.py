"""
scripts/run_analysis.py
========================
Stage 8: Analysis, Figures & Tables

Reads the completed final experiment results and produces:
  - All 8 publication-quality figures (300 DPI)
  - All 6 results tables (CSV + LaTeX format)
  - Full narrative interpretation text
  - Statistical verification log

Run AFTER scripts/run_final.py has completed.
Usage: python scripts/run_analysis.py
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.config import (
    DATA_PROCESSED, RESULTS_FIGURES, RESULTS_TABLES, FINAL_CONFIG
)
from src.visualization import (
    fig1_reporting_gaps,
    fig2_main_result_panel,
    fig3_score_reduction_boxplot,
    fig4_model_ab_comparison,
    fig5_shap_lime_scatter,
    fig7_equivalence_histogram,
    fig8_heatmap_configurations,
)
from src.metrics import compute_equivalence, compute_confidence_intervals

BLIND_ATTEMPTS = FINAL_CONFIG['blind_attempts']

print("\n" + "=" * 70)
print("  STAGE 8: Analysis, Figures & Tables")
print("  Reading results from completed final experiment")
print("=" * 70)

# ============================================================
# LOAD RESULTS
# ============================================================

# Load holdout data (for Fig 1)
holdout_pkl = os.path.join(DATA_PROCESSED, 'nhanes_2017_2018_prepared.pkl')
dev_pkl     = os.path.join(DATA_PROCESSED, 'nhanes_2015_2016_prepared.pkl')

if not os.path.exists(holdout_pkl):
    print("\n[ERROR] Final experiment not yet complete.")
    print(f"        Missing: {holdout_pkl}")
    print("        Run scripts/run_final.py first.")
    sys.exit(1)

with open(holdout_pkl, 'rb') as f:
    holdout_data = pickle.load(f)
with open(dev_pkl, 'rb') as f:
    dev_data = pickle.load(f)

print(f"\n  Holdout data: {len(holdout_data['y']):,} total patients")
print(f"  Diabetes prevalence: {holdout_data['y'].mean()*100:.1f}%")

# Load per-config cached results
cache_dir = os.path.join(DATA_PROCESSED, 'final_run_cache')
all_results = {}
configs_8 = [
    'reported_rf_shap', 'reported_rf_lime',
    'reported_lr_shap', 'reported_lr_lime',
    'measured_rf_shap', 'measured_rf_lime',
    'measured_lr_shap', 'measured_lr_lime',
]

print("\n  Loading per-config results:")
for key in configs_8:
    cache_path = os.path.join(cache_dir, f'{key}.pkl')
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            all_results[key] = pickle.load(f)
        s = all_results[key]['summary']
        print(f"    {key:30s} — N*={s['mean_equivalence']:.1f} "
              f"[{all_results[key]['confidence_interval']['mean_ci'][0]:.1f}, "
              f"{all_results[key]['confidence_interval']['mean_ci'][1]:.1f}]")
    else:
        print(f"    {key:30s} — MISSING (not yet computed)")

if not all_results:
    print("\n[ERROR] No config results found. Run scripts/run_final.py first.")
    sys.exit(1)

# Load model metrics
metrics_path = os.path.join(RESULTS_TABLES, 'final_model_metrics.json')
model_metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        model_metrics = json.load(f)

# ============================================================
# GENERATE ALL 8 FIGURES
# ============================================================
print("\n" + "=" * 70)
print("  Generating Publication-Quality Figures")
print("=" * 70)

# Fig 1: Reporting gap distributions
print("\n  [Fig 1] Reporting gap distributions")
df_holdout = holdout_data['df_complete']
fig1_reporting_gaps(
    df_holdout,
    save_path=os.path.join(RESULTS_FIGURES, 'fig1_reporting_gaps.png')
)

# Fig 2: Main result panel (2×2 SHAP equivalence curves)
print("\n  [Fig 2] Main result panel (SHAP configs)")
fig2_main_result_panel(
    all_results, BLIND_ATTEMPTS,
    save_path=os.path.join(RESULTS_FIGURES, 'fig2_main_result.png')
)

# Fig 3: Score reduction box plots
print("\n  [Fig 3] Score reduction box plots")
fig3_score_reduction_boxplot(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'fig3_score_reductions.png')
)

# Fig 4: Model A vs Model B grouped bar
print("\n  [Fig 4] Model A vs B comparison")
fig4_model_ab_comparison(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'fig4_model_comparison.png')
)

# Fig 5: SHAP vs LIME scatter
print("\n  [Fig 5] SHAP vs LIME agreement scatter")
fig5_shap_lime_scatter(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'fig5_shap_lime_scatter.png')
)

# Fig 6: SHAP beeswarm (feature importance) — uses shap library directly
print("\n  [Fig 6] SHAP feature importance summary")
try:
    import shap, matplotlib
    main_key = 'reported_rf_shap'
    if main_key in all_results:
        from src.data_loader import merge_nhanes, clean_nhanes, create_diabetes_target, prepare_feature_sets
        from src.gap_analysis import compute_reporting_gap
        from src.models import train_all_models
        from src.config import set_all_seeds
        set_all_seeds()

        df_tmp = merge_nhanes(cycle='2017_2018')
        df_tmp = clean_nhanes(df_tmp)
        df_tmp = create_diabetes_target(df_tmp)
        df_tmp = compute_reporting_gap(df_tmp)
        X_m, X_r, y, fn_m, fn_r, _ = prepare_feature_sets(df_tmp)

        from sklearn.model_selection import train_test_split
        import numpy as np
        from src.config import SEED_TRAIN_TEST_SPLIT, TEST_SIZE
        indices = np.arange(len(y))
        idx_tr, idx_te = train_test_split(
            indices, test_size=TEST_SIZE,
            random_state=SEED_TRAIN_TEST_SPLIT, stratify=y
        )
        from src.models import DiabetesModel
        m = DiabetesModel(model_type='rf', input_type='reported')
        m.train(X_r[idx_tr], y[idx_tr], feature_names=fn_r)

        explainer = shap.TreeExplainer(m.model)
        shap_vals = explainer(X_r[idx_te])
        if hasattr(shap_vals, 'values'):
            sv = shap_vals.values
        else:
            sv = shap_vals
        if sv.ndim == 3:
            sv = sv[:, :, 1]

        import matplotlib.pyplot as plt
        shap.summary_plot(
            sv, X_r[idx_te],
            feature_names=fn_r,
            show=False,
            plot_type='dot',
            max_display=7,
        )
        fig6_path = os.path.join(RESULTS_FIGURES, 'fig6_shap_summary.png')
        plt.title('Fig 6: SHAP Feature Importance Summary\n'
                  '(Model B — Self-Reported, Random Forest)', fontweight='bold')
        plt.savefig(fig6_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {fig6_path}")
except Exception as e:
    print(f"  [WARN] SHAP summary plot skipped: {e}")

# Fig 7: Per-patient N* histogram
print("\n  [Fig 7] Per-patient N* histogram")
fig7_equivalence_histogram(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'fig7_nstar_histogram.png')
)

# Fig 8: Configuration heatmap
print("\n  [Fig 8] Configuration heatmap")
fig8_heatmap_configurations(
    all_results,
    save_path=os.path.join(RESULTS_FIGURES, 'fig8_config_heatmap.png')
)

print(f"\n  All figures saved to: {RESULTS_FIGURES}")

# ============================================================
# GENERATE ALL 6 TABLES (CSV + LaTeX)
# ============================================================
print("\n" + "=" * 70)
print("  Generating Results Tables (CSV + LaTeX)")
print("=" * 70)


def df_to_latex(df, caption, label, col_format=None):
    """Convert DataFrame to a LaTeX table string."""
    n_cols = len(df.columns)
    if col_format is None:
        col_format = 'l' + 'r' * (n_cols - 1)
    lines = [
        r'\begin{table}[htbp]',
        r'  \centering',
        f'  \\caption{{{caption}}}',
        f'  \\label{{{label}}}',
        r'  \small',
        f'  \\begin{{tabular}}{{{col_format}}}',
        r'    \hline',
        '    ' + ' & '.join(str(c) for c in df.columns) + r' \\',
        r'    \hline',
    ]
    for _, row in df.iterrows():
        lines.append('    ' + ' & '.join(str(v) for v in row.values) + r' \\')
    lines += [
        r'    \hline',
        r'  \end{tabular}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


def save_table(df, name, caption, label):
    csv_path = os.path.join(RESULTS_TABLES, f'{name}.csv')
    tex_path = os.path.join(RESULTS_TABLES, f'{name}.tex')
    df.to_csv(csv_path, index=False)
    with open(tex_path, 'w') as f:
        f.write(df_to_latex(df, caption, label))
    print(f"  Saved: {csv_path}  +  {tex_path}")


# --- Table 1: Dataset Summary ---
print("\n  [Table 1] Dataset summary")
n_total_dev     = int(len(dev_data['y']))
n_diab_dev      = int(dev_data['y'].sum())
n_total_hold    = int(len(holdout_data['y']))
n_diab_hold     = int(holdout_data['y'].sum())

t1 = pd.DataFrame([
    ['NHANES 2015–2016', 'Development',  n_total_dev,  n_diab_dev,
     f"{n_diab_dev/n_total_dev*100:.1f}%",
     round(float(dev_data['lie_budget']['height']['max_lie']), 2),
     round(float(dev_data['lie_budget']['weight']['max_lie']), 2)],
    ['NHANES 2017–2018', 'Held-out test', n_total_hold, n_diab_hold,
     f"{n_diab_hold/n_total_hold*100:.1f}%",
     round(float(holdout_data['lie_budget']['height']['max_lie']), 2),
     round(float(holdout_data['lie_budget']['weight']['max_lie']), 2)],
], columns=['Cycle', 'Role', 'N', 'N Diabetes', 'Prevalence',
            'Height budget (cm)', 'Weight budget (kg)'])
save_table(t1, 'table1_dataset_summary', 'Dataset Summary', 'tab:dataset')

# --- Table 2: Model Performance ---
print("\n  [Table 2] Model performance")
if model_metrics:
    rows2 = []
    for key, m in model_metrics.items():
        parts = key.split('_')
        rows2.append({
            'Config':    key,
            'Input':     parts[0],
            'Model':     parts[1].upper(),
            'Accuracy':  round(m.get('accuracy',  0), 4),
            'Precision': round(m.get('precision', 0), 4),
            'Recall':    round(m.get('recall',    0), 4),
            'F1':        round(m.get('f1',        0), 4),
            'AUC-ROC':   round(m.get('auc_roc',   0), 4),
        })
    t2 = pd.DataFrame(rows2)
    save_table(t2, 'table2_model_performance',
               'Model Performance on NHANES 2017–2018 Test Set', 'tab:model_perf')

# --- Table 3: Reporting Gap Comparison ---
print("\n  [Table 3] Reporting gap comparison")
gap_rows = []
for feat in ['height', 'weight']:
    dev_b  = dev_data['lie_budget'][feat]
    hold_b = holdout_data['lie_budget'][feat]
    unit   = 'cm' if feat == 'height' else 'kg'
    gap_rows.append({
        'Feature':             feat.capitalize(),
        'Dev mean direction':  dev_b.get('mean_direction', 'N/A'),
        'Dev 95th pct budget': f"{dev_b['max_lie']:.2f} {unit}",
        'Holdout 95th pct':    f"{hold_b['max_lie']:.2f} {unit}",
        'Absolute change':     f"{abs(hold_b['max_lie']-dev_b['max_lie']):.2f} {unit}",
    })
t3 = pd.DataFrame(gap_rows)
save_table(t3, 'table3_reporting_gap',
           'Reporting Gap Comparison: 2015--2016 vs 2017--2018', 'tab:gap')

# --- Table 4: Main Results (primary) ---
print("\n  [Table 4] Main results — equivalence metric")
rows4 = []
for key in configs_8:
    if key not in all_results:
        continue
    d   = all_results[key]
    s   = d['summary']
    ci  = d['confidence_interval']
    parts = key.split('_')
    exp_m = parts[-1].upper()
    mdl   = parts[-2].upper()
    inp   = '_'.join(parts[:-2]).capitalize()

    rows4.append({
        'Config':         key,
        'Input':          inp,
        'Model':          mdl,
        'Expl.':          exp_m,
        'N patients':     s['n_patients'],
        'Mean N*':        round(s['mean_equivalence'], 2),
        '95% CI':         f"[{ci['mean_ci'][0]:.2f}, {ci['mean_ci'][1]:.2f}]",
        'Median N*':      round(s['median_equivalence'], 2),
        'Std':            round(s['std_equivalence'], 2),
        '% Useless':      f"{s['pct_explanation_useless']:.1f}%",
        '% Worth 10+':    f"{s['pct_explanation_worth_10plus']:.1f}%",
    })
t4 = pd.DataFrame(rows4)
save_table(t4, 'table4_main_results',
           'Main Results: Explanation Equivalence N* Across All 8 Configurations',
           'tab:main_results')

# --- Table 5: Statistical Tests ---
print("\n  [Table 5] Statistical tests")
stat_path = os.path.join(RESULTS_TABLES, 'final_results.json')
if os.path.exists(stat_path):
    with open(stat_path) as f:
        master = json.load(f)
    rows5 = []
    for key, res in master.get('results', {}).items():
        rows5.append({
            'Config':      key,
            'Wilcoxon p':  f"{res.get('wilcoxon_p', 1.0):.6f}",
            "Cohen's d":   f"{res.get('cohens_d', 0.0):.3f}",
            'Significant': '✓' if res.get('significant', False) else '✗',
        })
    if rows5:
        t5 = pd.DataFrame(rows5)
        save_table(t5, 'table5_statistical_tests',
                   'Statistical Tests: Explanation vs Blind (N=10), Wilcoxon Signed-Rank',
                   'tab:stats')

# --- Table 6: Model A vs B Comparison ---
print("\n  [Table 6] Model A vs B comparison")
rows6 = []
for suffix in ['rf_shap', 'rf_lime', 'lr_shap', 'lr_lime']:
    m_key = f'measured_{suffix}'
    r_key = f'reported_{suffix}'
    if m_key in all_results and r_key in all_results:
        m_n = all_results[m_key]['summary']['mean_equivalence']
        r_n = all_results[r_key]['summary']['mean_equivalence']
        rows6.append({
            'Config':                suffix.upper().replace('_', '+'),
            'Model A (Measured) N*': round(m_n, 2),
            'Model B (Reported) N*': round(r_n, 2),
            'Difference (B-A)':      round(r_n - m_n, 2),
            'B is more vulnerable':  '✓' if r_n > m_n else '✗',
        })
if rows6:
    t6 = pd.DataFrame(rows6)
    save_table(t6, 'table6_model_ab_comparison',
               'Model A (Measured) vs Model B (Self-Reported): Vulnerability Comparison',
               'tab:model_ab')

print(f"\n  All tables saved to: {RESULTS_TABLES}")

# ============================================================
# INTERPRETATION NARRATIVE
# ============================================================
print("\n" + "=" * 70)
print("  RESULTS INTERPRETATION")
print("=" * 70)

main_key = 'reported_rf_shap'
if main_key in all_results:
    s  = all_results[main_key]['summary']
    ci = all_results[main_key]['confidence_interval']
    n  = s['n_patients']

    mean_n = s['mean_equivalence']
    med_n  = s['median_equivalence']

    if mean_n < 5:
        verdict = "LOW THREAT — Explanations offer minimal advantage. Transparency is SAFE."
    elif mean_n <= 20:
        verdict = "MODERATE THREAT — Explanations offer some advantage. Caution RECOMMENDED."
    else:
        verdict = "HIGH THREAT — Explanations are powerful attack vectors. Countermeasures WARRANTED."

    narrative = f"""
======================================================================
  MAIN FINDING (reported_rf_shap, N={n} patients)
======================================================================

  1 SHAP explanation ≈ {mean_n:.1f} blind attempts (mean)
  1 SHAP explanation ≈ {med_n:.1f} blind attempts (median)
  95% CI (mean): [{ci['mean_ci'][0]:.2f}, {ci['mean_ci'][1]:.2f}]

  {s['pct_explanation_useless']:.0f}% of patients: explanation provides ZERO benefit
  {s['pct_explanation_worth_10plus']:.0f}% of patients: explanation worth 10+ blind attempts

  Pre-registered interpretation: {verdict}
======================================================================
"""
    print(narrative)

    # Write narrative to file
    narr_path = os.path.join(RESULTS_TABLES, 'interpretation_narrative.txt')
    with open(narr_path, 'w', encoding='utf-8') as f:
        f.write(narrative)
        f.write("\n\nAll Configurations:\n")
        f.write("-" * 50 + "\n")
        for key in configs_8:
            if key not in all_results:
                continue
            ks = all_results[key]['summary']
            f.write(f"  {key:30s}  N*={ks['mean_equivalence']:.2f}  "
                    f"(median={ks['median_equivalence']:.2f})\n")
    print(f"  Narrative saved: {narr_path}")

# ============================================================
# SUMMARY COUNT
# ============================================================
figs_saved = [f for f in os.listdir(RESULTS_FIGURES) if f.startswith('fig') and f.endswith('.png')]
tbls_saved = [f for f in os.listdir(RESULTS_TABLES) if f.endswith('.csv') or f.endswith('.tex')]

print("\n" + "=" * 70)
print(f"  STAGE 8 COMPLETE")
print(f"  Figures: {len(figs_saved)} PNG files in {RESULTS_FIGURES}")
print(f"  Tables:  {len(tbls_saved)} CSV/LaTeX files in {RESULTS_TABLES}")
print("  Proceed to Stage 9: Paper Writing")
print("=" * 70)
