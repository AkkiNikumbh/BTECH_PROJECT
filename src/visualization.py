"""
Visualization functions for the experiment.
All plots use publication-quality settings (300 DPI).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend — no tkinter, safe for subprocesses
import matplotlib.pyplot as plt
import seaborn as sns

# Publication style settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'explanation': '#E74C3C',
    'blind':       '#3498DB',
    'measured':    '#27AE60',
    'reported':    '#F39C12',
    'shap':        '#9B59B6',
    'lime':        '#1ABC9C',
}


# ============================================================
# CORE EXPERIMENT FIGURES (used by run_trial.py & run_final.py)
# ============================================================

def plot_equivalence_curve(results, blind_attempts_list, title='', save_path=None):
    """
    THE KEY VISUALIZATION.
    Shows blind attempts vs. score reduction with explanation line.
    The crossing point is the equivalence metric N*.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_reductions = [r['explanation_reduction'] for r in results]
    mean_exp = np.mean(exp_reductions)

    blind_means, blind_stds = [], []
    for n in blind_attempts_list:
        reductions = [r['blind_reductions'][n] for r in results]
        blind_means.append(np.mean(reductions))
        blind_stds.append(np.std(reductions))

    ax.plot(blind_attempts_list, blind_means, 'o-', linewidth=2,
            markersize=6, label='Blind Trial-and-Error', zorder=3,
            color=COLORS['blind'])
    ax.fill_between(blind_attempts_list,
                    np.array(blind_means) - np.array(blind_stds),
                    np.array(blind_means) + np.array(blind_stds),
                    alpha=0.2, color=COLORS['blind'])

    ax.axhline(y=mean_exp, color=COLORS['explanation'], linestyle='--',
               linewidth=2, label=f'Explanation (mean = {mean_exp:.4f})')

    for i in range(len(blind_means) - 1):
        if blind_means[i] <= mean_exp <= blind_means[i + 1]:
            frac = (mean_exp - blind_means[i]) / (blind_means[i + 1] - blind_means[i])
            crossing_n = blind_attempts_list[i] + frac * (
                blind_attempts_list[i + 1] - blind_attempts_list[i]
            )
            ax.axvline(x=crossing_n, color='green', linestyle=':', linewidth=2,
                       label=f'Equivalence: N*≈{crossing_n:.1f}')
            ax.plot(crossing_n, mean_exp, 'g*', markersize=15, zorder=5)
            break

    ax.set_xlabel('Number of Blind Attempts')
    ax.set_ylabel('Mean Score Reduction')
    ax.set_title(title or 'Explanation vs. Blind: Equivalence Curve')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def plot_score_reductions(results, save_path=None):
    """Box plots comparing score reductions across strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))

    exp_reds = [r['explanation_reduction'] for r in results]
    data_to_plot = [exp_reds]
    labels = ['Explanation']

    for n in [1, 5, 10, 20, 50, 100]:
        if n in results[0]['blind_reductions']:
            data_to_plot.append([r['blind_reductions'][n] for r in results])
            labels.append(f'Blind (N={n})')

    try:
        bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
    except TypeError:
        bp = ax.boxplot(data_to_plot, patch_artist=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    colors = [COLORS['explanation']] + [COLORS['blind']] * (len(labels) - 1)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel('Score Reduction')
    ax.set_title('Score Reduction: Explanation vs. Blind Attempts')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def plot_model_comparison(all_results, save_path=None):
    """Bar chart comparing measured vs self-reported model vulnerability."""
    fig, ax = plt.subplots(figsize=(10, 6))

    configs, measured_vals, reported_vals = [], [], []
    for suffix in ['rf_shap', 'rf_lime', 'lr_shap', 'lr_lime']:
        m_key = f'measured_{suffix}'
        r_key = f'reported_{suffix}'
        if m_key in all_results and r_key in all_results:
            configs.append(suffix.upper().replace('_', '+'))
            measured_vals.append(all_results[m_key]['summary']['median_equivalence'])
            reported_vals.append(all_results[r_key]['summary']['median_equivalence'])

    if not configs:
        return

    x = np.arange(len(configs))
    width = 0.35
    bars1 = ax.bar(x - width / 2, measured_vals, width,
                   label='Model A (Measured)', color=COLORS['measured'], alpha=0.85)
    bars2 = ax.bar(x + width / 2, reported_vals, width,
                   label='Model B (Self-Reported)', color=COLORS['reported'], alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., h + 0.1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Configuration')
    ax.set_ylabel('Median Equivalence (N*)')
    ax.set_title('Model A vs Model B: Does Input Verification Matter?', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


# ============================================================
# PUBLICATION FIGURES — Stage 8 (8 total)
# ============================================================

def fig1_reporting_gaps(df, save_path=None):
    """
    Fig 1 — Reporting Gap Distributions.
    Height and weight gap histograms with KDE overlay.
    Shows how patients systematically misreport their measurements.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    gap_cols = [
        ('height_gap_cm', 'Height Gap (self − measured, cm)',
         COLORS['shap'], 'Overestimate ▶'),
        ('weight_gap_kg', 'Weight Gap (self − measured, kg)',
         COLORS['lime'], '◀ Underestimate'),
    ]

    for ax, (col, xlabel, color, note) in zip(axes, gap_cols):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        data = df[col].dropna()
        ax.hist(data, bins=50, color=color, alpha=0.6,
                edgecolor='white', linewidth=0.4, density=True)
        data.plot.kde(ax=ax, color=color, linewidth=2.5)

        mean_val = float(data.mean())
        ax.axvline(0,        color='gray',  linestyle='--', linewidth=1.5, label='Zero gap')
        ax.axvline(mean_val, color='black', linestyle='-',  linewidth=2,
                   label=f'Mean = {mean_val:+.2f}')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.2)
        ax.text(0.03, 0.95, note, transform=ax.transAxes,
                fontsize=10, va='top', color='gray', style='italic')

    fig.suptitle('Fig 1: Reporting Gap Distributions (NHANES 2017–2018)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def fig2_main_result_panel(all_results, blind_attempts_list, save_path=None):
    """
    Fig 2 — THE MAIN RESULT PANEL.
    2×2 equivalence curves for the 4 SHAP configs.
    """
    configs = [
        ('reported_rf_shap',  'Model B + RF + SHAP\n(Primary Config)'),
        ('reported_lr_shap',  'Model B + LR + SHAP'),
        ('measured_rf_shap',  'Model A + RF + SHAP'),
        ('measured_lr_shap',  'Model A + LR + SHAP'),
    ]
    configs = [(k, t) for k, t in configs if k in all_results]
    if not configs:
        return

    n_plots = min(len(configs), 4)
    rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    axes_flat = np.array(axes).flatten()

    for ax, (key, title) in zip(axes_flat[:n_plots], configs):
        data    = all_results[key]
        results = data['results']

        exp_reds  = np.array([r['explanation_reduction'] for r in results])
        mean_exp  = float(np.mean(exp_reds))
        ci_lo     = float(np.percentile(exp_reds, 2.5))
        ci_hi     = float(np.percentile(exp_reds, 97.5))

        b_means, b_lo, b_hi = [], [], []
        for n in blind_attempts_list:
            reds = np.array([r['blind_reductions'][n] for r in results])
            b_means.append(float(np.mean(reds)))
            b_lo.append(float(np.percentile(reds, 2.5)))
            b_hi.append(float(np.percentile(reds, 97.5)))

        ax.plot(blind_attempts_list, b_means, 'o-',
                color=COLORS['blind'], linewidth=2, markersize=5,
                label='Blind trial-and-error', zorder=3)
        ax.fill_between(blind_attempts_list, b_lo, b_hi,
                        alpha=0.15, color=COLORS['blind'])
        ax.axhline(y=mean_exp, color=COLORS['explanation'],
                   linestyle='--', linewidth=2,
                   label=f'1 explanation ({mean_exp:.4f})')
        ax.axhspan(ci_lo, ci_hi, alpha=0.08, color=COLORS['explanation'])

        n_star = data['summary']['mean_equivalence']
        ax.axvline(x=n_star, color='#2C3E50', linestyle=':', linewidth=1.5, alpha=0.8)
        ax.annotate(f'N*≈{n_star:.1f}', xy=(n_star, mean_exp),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, fontweight='bold', color='#2C3E50')

        ax.set_xlabel('Blind Attempts (log scale)')
        ax.set_ylabel('Mean Score Reduction')
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[n_plots:]:
        ax.set_visible(False)

    fig.suptitle('Fig 2: Explanation Equivalence — 1 SHAP explanation ≈ N* blind attempts',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def fig3_score_reduction_boxplot(all_results, save_path=None):
    """
    Fig 3 — Score Reduction Comparison (Box plots).
    Explanation vs blind at key N values.
    """
    main_key = 'reported_rf_shap'
    if main_key not in all_results:
        main_key = list(all_results.keys())[0]

    results  = all_results[main_key]['results']
    ns_show  = [n for n in [1, 5, 10, 20, 50, 100]
                if n in results[0]['blind_reductions']]

    data_all    = [[r['explanation_reduction'] for r in results]]
    labels      = ['Explanation\n(SHAP)']
    box_colors  = [COLORS['explanation']]

    for n in ns_show:
        data_all.append([r['blind_reductions'][n] for r in results])
        labels.append(f'Blind\n(N={n})')
        box_colors.append(COLORS['blind'])

    fig, ax = plt.subplots(figsize=(12, 6))
    try:
        bp = ax.boxplot(data_all, tick_labels=labels, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2},
                        whiskerprops={'linewidth': 1.5},
                        capprops={'linewidth': 1.5})
    except TypeError:
        bp = ax.boxplot(data_all, patch_artist=True,
                        medianprops={'color': 'black', 'linewidth': 2},
                        whiskerprops={'linewidth': 1.5},
                        capprops={'linewidth': 1.5})
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel('Score Reduction (Probability units)', fontsize=12)
    ax.set_title('Fig 3: Score Reduction — Explanation vs Blind Attempts '
                 f'({main_key.upper().replace("_"," ")})',
                 fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    ax.axhline(y=0, color='gray', linewidth=0.8, linestyle='--')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def fig4_model_ab_comparison(all_results, save_path=None):
    """
    Fig 4 — Model A (measured) vs Model B (self-reported) vulnerability.
    Grouped bar chart.
    """
    configs, measured_vals, reported_vals = [], [], []
    for suffix in ['rf_shap', 'rf_lime', 'lr_shap', 'lr_lime']:
        m_key = f'measured_{suffix}'
        r_key = f'reported_{suffix}'
        if m_key in all_results and r_key in all_results:
            configs.append(suffix.upper().replace('_', '+'))
            measured_vals.append(all_results[m_key]['summary']['mean_equivalence'])
            reported_vals.append(all_results[r_key]['summary']['mean_equivalence'])

    if not configs:
        return

    x = np.arange(len(configs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, measured_vals, width,
                   label='Model A (Measured inputs)', color=COLORS['measured'], alpha=0.85)
    bars2 = ax.bar(x + width/2, reported_vals, width,
                   label='Model B (Self-reported inputs)', color=COLORS['reported'], alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                f'{h:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xlabel('Configuration', fontsize=12)
    ax.set_ylabel('Mean Equivalence (N*)', fontsize=12)
    ax.set_title('Fig 4: Model A vs Model B — Does Input Verification Reduce Gaming Risk?',
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.2, axis='y')
    ax.axhline(y=10, color='gray', linewidth=1, linestyle=':', alpha=0.6,
               label='N*=10 reference')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def fig5_shap_lime_scatter(all_results, save_path=None):
    """
    Fig 5 — SHAP vs LIME Agreement Scatter.
    Each point = one model+input config.
    """
    pairs = []
    for prefix in ['reported_rf', 'reported_lr', 'measured_rf', 'measured_lr']:
        k_shap = f'{prefix}_shap'
        k_lime = f'{prefix}_lime'
        if k_shap in all_results and k_lime in all_results:
            pairs.append((
                all_results[k_shap]['summary']['mean_equivalence'],
                all_results[k_lime]['summary']['mean_equivalence'],
                prefix,
            ))

    if not pairs:
        return

    shap_vals = [p[0] for p in pairs]
    lime_vals = [p[1] for p in pairs]
    labels    = [p[2].replace('_', '\n') for p in pairs]

    fig, ax = plt.subplots(figsize=(7, 7))
    colors_pt = [COLORS['measured'] if 'measured' in p[2] else COLORS['reported']
                 for p in pairs]

    for x, y, lbl, c in zip(shap_vals, lime_vals, labels, colors_pt):
        ax.scatter(x, y, color=c, s=220, zorder=5, edgecolors='white', linewidth=1.5)
        ax.annotate(lbl, (x, y), textcoords='offset points',
                    xytext=(8, 4), fontsize=9)

    max_val = max(max(shap_vals, default=1), max(lime_vals, default=1)) * 1.2 or 2
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1, alpha=0.4)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_xlabel('SHAP Equivalence (Mean N*)', fontsize=12)
    ax.set_ylabel('LIME Equivalence (Mean N*)', fontsize=12)
    ax.set_title('Fig 5: SHAP vs LIME Agreement\n(diagonal = identical)', fontweight='bold')
    ax.grid(True, alpha=0.2)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=COLORS['reported'], label='Model B (Self-Reported)'),
        Patch(facecolor=COLORS['measured'], label='Model A (Measured)'),
    ], fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()


def fig7_equivalence_histogram(all_results, save_path=None):
    """
    Fig 7 — Per-Patient N* Histogram.
    Histogram of individual equivalence values for the main config.
    """
    main_key = 'reported_rf_shap'
    if main_key not in all_results:
        main_key = list(all_results.keys())[0]

    equivalences = all_results[main_key]['equivalences']
    valid = equivalences[~np.isnan(equivalences)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid, bins=30, color=COLORS['shap'], alpha=0.75,
            edgecolor='white', linewidth=0.5)

    mean_val   = float(np.mean(valid))
    median_val = float(np.median(valid))
    ax.axvline(mean_val,   color='red',    linestyle='--', linewidth=2,
               label=f'Mean = {mean_val:.1f}')
    ax.axvline(median_val, color='orange', linestyle='-',  linewidth=2,
               label=f'Median = {median_val:.1f}')
    ax.axvline(10, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label='N*=10 threshold')

    pct_useless = float((valid <= 1).mean() * 100)
    pct_10plus  = float((valid >= 10).mean() * 100)
    ax.text(0.98, 0.95,
            f'{pct_useless:.0f}% useless (N*≤1)\n{pct_10plus:.0f}% worth ≥10 attempts',
            transform=ax.transAxes, ha='right', va='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Per-Patient N* (blind attempts)', fontsize=12)
    ax.set_ylabel('Number of Patients', fontsize=12)
    ax.set_title(f'Fig 7: Distribution of N* — {main_key.upper().replace("_"," ")}\n'
                 f'(N={len(valid)} patients)', fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"  Saved: {save_path}")
    plt.close()


def fig8_heatmap_configurations(all_results, save_path=None):
    """
    Fig 8 — Heatmap of all 8 configurations.
    Cell = Mean N*. Quick visual scan of all results.
    """
    row_keys   = ['reported_rf', 'reported_lr', 'measured_rf', 'measured_lr']
    row_labels = ['Reported + RF', 'Reported + LR', 'Measured + RF', 'Measured + LR']
    col_keys   = ['shap', 'lime']
    col_labels = ['SHAP', 'LIME']

    matrix = np.full((len(row_keys), len(col_keys)), np.nan)
    for i, rk in enumerate(row_keys):
        for j, ck in enumerate(col_keys):
            key = f'{rk}_{ck}'
            if key in all_results:
                matrix[i, j] = all_results[key]['summary']['mean_equivalence']

    if np.all(np.isnan(matrix)):
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    valid_vals = matrix[~np.isnan(matrix)]
    im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                   vmin=0, vmax=valid_vals.max() if len(valid_vals) else 1)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=13)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    mid = valid_vals.mean() if len(valid_vals) else 0.5
    for i in range(len(row_keys)):
        for j in range(len(col_keys)):
            val = matrix[i, j]
            if not np.isnan(val):
                txt_color = 'white' if val > mid else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=14, fontweight='bold', color=txt_color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Mean N* (blind attempt equivalents)', fontsize=10)
    ax.set_title('Fig 8: N* Heatmap — All 8 Configurations\n'
                 '(higher = explanation more valuable to adversary)',
                 fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"  Saved: {save_path}")
    plt.close()
