"""
Reporting gap analysis — measures how much people's self-reported
height/weight differs from the nurse's measurement.
This defines the 'realistic lie budget' for the adversary.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_reporting_gap(df):
    """
    Compute the gap between self-reported and nurse-measured values.

    Returns the DataFrame with added gap columns.
    """
    df = df.copy()

    height_mask = df['self_height_cm'].notna() & df['measured_height_cm'].notna()
    weight_mask = df['self_weight_kg'].notna() & df['measured_weight_kg'].notna()

    df.loc[height_mask, 'height_gap_cm'] = (
        df.loc[height_mask, 'self_height_cm'] -
        df.loc[height_mask, 'measured_height_cm']
    )

    df.loc[weight_mask, 'weight_gap_kg'] = (
        df.loc[weight_mask, 'self_weight_kg'] -
        df.loc[weight_mask, 'measured_weight_kg']
    )

    for gap_col, unit in [('height_gap_cm', 'cm'), ('weight_gap_kg', 'kg')]:
        gap = df[gap_col].dropna()
        print(f"\n{'='*50}")
        print(f"  REPORTING GAP: {gap_col}")
        print(f"{'='*50}")
        print(f"  N samples:   {len(gap)}")
        print(f"  Mean gap:    {gap.mean():+.2f} {unit}")
        print(f"  Median gap:  {gap.median():+.2f} {unit}")
        print(f"  Std dev:     {gap.std():.2f} {unit}")
        print(f"  Range:       [{gap.min():+.2f}, {gap.max():+.2f}]")
        print(f"  IQR:         [{gap.quantile(0.25):+.2f}, {gap.quantile(0.75):+.2f}]")
        print(f"  95th pctile: {gap.abs().quantile(0.95):.2f} {unit}")

    return df


def define_lie_budget(df, method='percentile', pct=95):
    """
    Define the maximum lie magnitude based on observed gaps.

    Parameters
    ----------
    df : DataFrame with gap columns from compute_reporting_gap()
    method : str — 'percentile' or 'std'
    pct : float — percentile (for 'percentile') or number of stds

    Returns
    -------
    dict with lie budget for each feature
    """
    budget = {}

    for gap_col, feature_name in [
        ('height_gap_cm', 'height'),
        ('weight_gap_kg', 'weight')
    ]:
        gap = df[gap_col].dropna()
        abs_gap = gap.abs()

        if method == 'percentile':
            max_lie = abs_gap.quantile(pct / 100)
        elif method == 'std':
            max_lie = abs_gap.mean() + pct * abs_gap.std()
        else:
            raise ValueError(f"Unknown method: {method}")

        budget[feature_name] = {
            'max_lie': max_lie,
            'mean_direction': 'positive' if gap.mean() > 0 else 'negative',
            'mean_magnitude': abs_gap.mean(),
        }

        direction = 'overestimate' if gap.mean() > 0 else 'underestimate'
        print(f"\n  LIE BUDGET for {feature_name}:")
        print(f"    Max permissible lie: ±{max_lie:.2f}")
        print(f"    People tend to {direction}")
        print(f"    Average misreport: {abs_gap.mean():.2f}")

    return budget


def plot_reporting_gaps(df, save_dir='results/figures'):
    """
    Generate publication-quality plots of the reporting gaps.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Height gap
    ax = axes[0]
    gap_h = df['height_gap_cm'].dropna()
    sns.histplot(gap_h, bins=50, kde=True, ax=ax, color='#3498db', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect accuracy')
    ax.axvline(x=gap_h.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Mean = {gap_h.mean():+.1f} cm')
    ax.set_xlabel('Self-Reported − Measured Height (cm)')
    ax.set_ylabel('Count')
    ax.set_title('Height Reporting Gap Distribution')
    ax.legend()

    # Weight gap
    ax = axes[1]
    gap_w = df['weight_gap_kg'].dropna()
    sns.histplot(gap_w, bins=50, kde=True, ax=ax, color='#e74c3c', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Perfect accuracy')
    ax.axvline(x=gap_w.mean(), color='orange', linestyle='-', linewidth=2,
               label=f'Mean = {gap_w.mean():+.1f} kg')
    ax.set_xlabel('Self-Reported − Measured Weight (kg)')
    ax.set_ylabel('Count')
    ax.set_title('Weight Reporting Gap Distribution')
    ax.legend()

    plt.tight_layout()
    filepath = os.path.join(save_dir, 'reporting_gaps.png')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved gap plot to: {filepath}")
