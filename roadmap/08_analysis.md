# 📈 Stage 8: Analysis, Graphs & Statistical Testing
## Period: 1 March – 31 March 2027 (4 weeks)

> **← Previous**: [Stage 7 — Final Experiments](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/07_final_experiments.md)
> **← Back to** [Master Roadmap](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/00_master_roadmap.md)
> **→ Next Stage**: [Stage 9 — Research Paper](file:///c:/Users/ashua/OneDrive/ドキュメント/Desktop/BTP/roadmap/09_paper_writing.md)

---

## 🎯 Objectives

1. Perform in-depth analysis of all final experiment results
2. Create **publication-quality** figures and tables
3. Verify all statistical tests and confidence intervals
4. Cross-check every reference cited in the literature review against the original paper
5. Write up the complete results section with interpretation

---

## 📋 Prerequisites (from Stage 7)

- [x] All 8 experiment configurations complete with raw results
- [x] Equivalence metrics, CIs, and statistical tests computed
- [x] No deviations from pre-registered plan

---

## 🔨 Step-by-Step Implementation

### Task 8.1 — Publication-Quality Figures to Produce

| Figure # | Title | Description | Type |
|----------|-------|-------------|------|
| Fig 1 | Reporting Gap Distributions | Height and weight gap histograms with KDE | Histogram |
| Fig 2 | Equivalence Curve (Main Result) | Blind attempts vs. score reduction with explanation line | Line + shaded CI |
| Fig 3 | Score Reduction Comparison | Explanation vs. blind strategies box plot | Box plot |
| Fig 4 | Model A vs. Model B | Measured vs. self-reported vulnerability | Grouped bar |
| Fig 5 | SHAP vs. LIME Agreement | Scatter of equivalences from both methods | Scatter |
| Fig 6 | SHAP Summary Plot | Feature importance across all patients | Beeswarm |
| Fig 7 | Per-Patient Equivalence Distribution | Histogram of individual N* values | Histogram |
| Fig 8 | Heatmap of Configurations | All 8 configs in one view | Heatmap |

### Task 8.2 — Publication-Quality Figure Code

```python
"""
src/visualization.py — Publication-quality figures
"""
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

# Publication style settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {
    'explanation': '#E74C3C',
    'blind': '#3498DB',
    'measured': '#27AE60',
    'reported': '#F39C12',
    'shap': '#9B59B6',
    'lime': '#1ABC9C',
}


def fig2_main_result(all_results, blind_attempts, save_path):
    """
    THE MAIN RESULT FIGURE.
    Shows the equivalence curve with confidence bands.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    configs = [
        ('measured_rf_shap', 'Measured + RF + SHAP'),
        ('measured_lr_shap', 'Measured + LR + SHAP'),
        ('reported_rf_shap', 'Reported + RF + SHAP'),
        ('reported_lr_shap', 'Reported + LR + SHAP'),
    ]
    
    for ax, (key, title) in zip(axes.flatten(), configs):
        data = all_results[key]
        results = data['results']
        
        # Explanation line
        exp_reds = np.array([r['explanation_reduction'] for r in results])
        mean_exp = np.mean(exp_reds)
        ci_exp = (np.percentile(exp_reds, 2.5), np.percentile(exp_reds, 97.5))
        
        # Blind curve
        blind_means = []
        blind_cis = []
        for n in blind_attempts:
            reds = np.array([r['blind_reductions'][n] for r in results])
            blind_means.append(np.mean(reds))
            blind_cis.append((np.percentile(reds, 2.5), np.percentile(reds, 97.5)))
        
        blind_lower = [c[0] for c in blind_cis]
        blind_upper = [c[1] for c in blind_cis]
        
        # Plot
        ax.plot(blind_attempts, blind_means, 'o-', color=COLORS['blind'],
                linewidth=2, markersize=5, label='Blind', zorder=3)
        ax.fill_between(blind_attempts, blind_lower, blind_upper,
                        alpha=0.15, color=COLORS['blind'])
        
        ax.axhline(y=mean_exp, color=COLORS['explanation'], linestyle='--',
                   linewidth=2, label=f'Explanation ({mean_exp:.4f})')
        ax.axhspan(ci_exp[0], ci_exp[1], alpha=0.1, color=COLORS['explanation'])
        
        # Find and mark crossing
        equiv = data['summary']['median_equivalence']
        ax.axvline(x=equiv, color='#2C3E50', linestyle=':', alpha=0.7)
        ax.annotate(f'N*≈{equiv:.0f}', xy=(equiv, mean_exp),
                   fontsize=9, fontweight='bold', color='#2C3E50',
                   xytext=(10, 10), textcoords='offset points')
        
        ax.set_xlabel('Number of Blind Attempts')
        ax.set_ylabel('Mean Score Reduction')
        ax.set_title(title, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.set_xscale('log')
        ax.grid(True, alpha=0.2)
    
    plt.suptitle('Main Result: Explanation Equivalence Across Configurations',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def fig4_model_comparison(all_results, save_path):
    """
    Bar chart comparing Model A (measured) vs Model B (reported).
    Shows whether the input source affects vulnerability.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['RF+SHAP', 'RF+LIME', 'LR+SHAP', 'LR+LIME']
    measured_equivs = []
    reported_equivs = []
    
    for cat_key in ['rf_shap', 'rf_lime', 'lr_shap', 'lr_lime']:
        m_key = f'measured_{cat_key}'
        r_key = f'reported_{cat_key}'
        measured_equivs.append(all_results[m_key]['summary']['median_equivalence'])
        reported_equivs.append(all_results[r_key]['summary']['median_equivalence'])
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, measured_equivs, width, 
                   label='Model A (Measured)', color=COLORS['measured'], alpha=0.85)
    bars2 = ax.bar(x + width/2, reported_equivs, width,
                   label='Model B (Self-Reported)', color=COLORS['reported'], alpha=0.85)
    
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('Median Equivalence (N*)')
    ax.set_title('Model A vs Model B: Does Input Verification Matter?', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
```

### Task 8.3 — Results Tables to Produce

| Table # | Content | Format |
|---------|---------|--------|
| Table 1 | Dataset summary (N, prevalence, demographics) | LaTeX / CSV |
| Table 2 | Model performance (accuracy, AUC, F1 for all 4 models) | LaTeX / CSV |
| Table 3 | Reporting gap summary (height and weight, 2015–16 vs 2017–18) | LaTeX / CSV |
| Table 4 | **Main results: Equivalence metric across all 8 configs** | LaTeX / CSV |
| Table 5 | Statistical tests (Wilcoxon p-values, Cohen's d) | LaTeX / CSV |
| Table 6 | Model A vs Model B comparison | LaTeX / CSV |

### Task 8.4 — Reference Verification

For each of the 5+4 cited papers, verify:

| Check | Action |
|-------|--------|
| Author names correct? | Compare with actual paper |
| Year correct? | Check publication date |
| Venue correct? | Verify conference/journal name |
| Page numbers correct? | Check against proceedings |
| Claims we make about the paper accurate? | Re-read abstract and relevant sections |
| DOI / URL working? | Click and verify |

---

## ✅ Completion Checklist

- [ ] All 8 publication-quality figures generated
- [ ] All 6 results tables produced in both CSV and LaTeX formats
- [ ] Statistical tests verified and documented
- [ ] Confidence intervals correct and included in figures
- [ ] Model A vs. Model B comparison analyzed and interpreted
- [ ] SHAP vs. LIME agreement quantified
- [ ] All 9 references verified against original papers
- [ ] No discrepancies found between pre-registered plan and execution
- [ ] Results narrative written for the paper (Stage 9 input)
- [ ] All figures and tables committed to Git
