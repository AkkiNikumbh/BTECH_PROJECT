"""
Data loading and preprocessing for NHANES datasets.
Handles merging survey + examination files, cleaning, and preparing
the two feature sets (measured vs self-reported).
"""
import pandas as pd
import numpy as np
import os
from src.config import DATA_RAW, DATA_PROCESSED


def load_xpt(filepath):
    """Load a SAS XPT transport file into a pandas DataFrame."""
    return pd.read_sas(filepath, format='xport')


def merge_nhanes(cycle='2015_2016'):
    """
    Merge all NHANES component files for a given survey cycle.

    Parameters
    ----------
    cycle : str
        '2015_2016' or '2017_2018'

    Returns
    -------
    pd.DataFrame
        Merged dataset with all selected columns
    """
    suffix = 'I' if cycle == '2015_2016' else 'J'
    raw_dir = os.path.join(DATA_RAW, f'nhanes_{cycle}')

    # Load each component
    components = {
        'DEMO': (f'DEMO_{suffix}.XPT', ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR']),
        'BMX':  (f'BMX_{suffix}.XPT',  ['SEQN', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST']),
        'WHQ':  (f'WHQ_{suffix}.XPT',  ['SEQN', 'WHD010', 'WHD020']),
        'GHB':  (f'GHB_{suffix}.XPT',  ['SEQN', 'LBXGH']),
        'BPX':  (f'BPX_{suffix}.XPT',  ['SEQN', 'BPXSY1', 'BPXDI1']),
        'SMQ':  (f'SMQ_{suffix}.XPT',  ['SEQN', 'SMQ020']),
        'PAQ':  (f'PAQ_{suffix}.XPT',  ['SEQN', 'PAQ605', 'PAQ650']),
    }

    # Start with demographics
    demo_file, demo_cols = components.pop('DEMO')
    merged = load_xpt(os.path.join(raw_dir, demo_file))
    available = [c for c in demo_cols if c in merged.columns]
    merged = merged[available].copy()
    print(f"[DATA] Loaded DEMO ({cycle}): {merged.shape}")

    # Merge remaining components
    for name, (filename, cols) in components.items():
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"[DATA] WARNING: {filename} not found, skipping")
            continue
        df = load_xpt(filepath)
        available = [c for c in cols if c in df.columns]
        merged = merged.merge(df[available], on='SEQN', how='left')
        print(f"[DATA] Merged {name}: shape now {merged.shape}")

    print(f"[DATA] Final merged shape: {merged.shape}")
    return merged


def clean_nhanes(df):
    """
    Clean the merged NHANES dataset.

    Steps:
    1. Filter adults only (age >= 18)
    2. Handle NHANES special codes (7=Refused, 9=Don't Know)
    3. Convert units for self-reported height/weight
    4. Remove rows with missing critical columns
    """
    df = df.copy()

    # Step 1: Adults only
    df = df[df['RIDAGEYR'] >= 18].copy()
    print(f"[CLEAN] After filtering adults (age>=18): {len(df)} rows")

    # Step 2: Replace NHANES special codes with NaN
    refuse_codes = [7, 9, 77, 99, 777, 999, 7777, 9999, 77777, 99999]
    questionnaire_cols = ['WHD010', 'WHD020', 'SMQ020', 'PAQ605', 'PAQ650', 'DMDEDUC2']
    for col in questionnaire_cols:
        if col in df.columns:
            df[col] = df[col].replace(refuse_codes, np.nan)

    # Step 3: Unit conversion for self-reported height/weight
    # WHD010 = self-reported height in INCHES
    # WHD020 = self-reported weight in POUNDS
    # BMXHT  = measured height in CM
    # BMXWT  = measured weight in KG
    if 'WHD010' in df.columns:
        df['self_height_cm'] = df['WHD010'] * 2.54           # inches → cm
    if 'WHD020' in df.columns:
        df['self_weight_kg'] = df['WHD020'] * 0.453592        # pounds → kg

    # Rename measured columns for clarity
    if 'BMXHT' in df.columns:
        df['measured_height_cm'] = df['BMXHT']
    if 'BMXWT' in df.columns:
        df['measured_weight_kg'] = df['BMXWT']

    # Step 4: Remove rows missing critical columns
    critical_cols = ['LBXGH', 'RIDAGEYR', 'RIAGENDR']
    existing_critical = [c for c in critical_cols if c in df.columns]
    df = df.dropna(subset=existing_critical)
    print(f"[CLEAN] After dropping missing critical cols: {len(df)} rows")

    return df


def create_diabetes_target(df):
    """
    Create binary diabetes target from HbA1c.

    ADA Thresholds:
    - Normal:       HbA1c < 5.7%
    - Pre-diabetes: 5.7% <= HbA1c < 6.5%
    - Diabetes:     HbA1c >= 6.5%

    We use binary: 0 = No diabetes (HbA1c < 6.5%), 1 = Diabetes (HbA1c >= 6.5%)
    This is an OBJECTIVE measurement — not self-reported.
    """
    df = df.copy()
    df['diabetes'] = (df['LBXGH'] >= 6.5).astype(float)
    df = df.dropna(subset=['diabetes'])
    print(f"[TARGET] Diabetes prevalence: {df['diabetes'].mean()*100:.1f}% "
          f"({int(df['diabetes'].sum())}/{len(df)})")
    return df


def prepare_feature_sets(df):
    """
    Create two feature matrices from the same patients:
    - Model A features: Use MEASURED height/weight (nurse data)
    - Model B features: Use SELF-REPORTED height/weight (patient data)
    - All other features are identical between the two

    Returns
    -------
    X_measured, X_reported, y, feature_names_measured, feature_names_reported, df_complete
    """
    # Common features (same in both models)
    common_features = ['RIDAGEYR', 'RIAGENDR', 'BPXSY1', 'SMQ020', 'PAQ650']

    # Model A: Nurse-measured height/weight
    measured_specific = ['measured_height_cm', 'measured_weight_kg']

    # Model B: Self-reported height/weight
    reported_specific = ['self_height_cm', 'self_weight_kg']

    # Require all features to be non-null
    all_cols = common_features + measured_specific + reported_specific + ['diabetes']
    existing_cols = [c for c in all_cols if c in df.columns]
    df_complete = df.dropna(subset=existing_cols).copy()

    print(f"[FEATURES] Rows with complete data for both models: {len(df_complete)}")

    feature_names_measured = common_features + measured_specific
    feature_names_reported = common_features + reported_specific

    X_measured = df_complete[feature_names_measured].values
    X_reported = df_complete[feature_names_reported].values
    y = df_complete['diabetes'].values

    print(f"[FEATURES] Feature count (each model): {X_measured.shape[1]}")
    print(f"[FEATURES] Diabetes: {int(y.sum())} ({y.mean()*100:.1f}%) | "
          f"No diabetes: {int(len(y) - y.sum())} ({(1-y.mean())*100:.1f}%)")

    return X_measured, X_reported, y, feature_names_measured, feature_names_reported, df_complete
