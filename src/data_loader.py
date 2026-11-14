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
        '2015_2016', '2017_2018' or '2021_2023' (confirmatory cycle,
        pre-registered in docs/analysis_plan_v5_confirmatory.md)

    Returns
    -------
    pd.DataFrame
        Merged dataset with all selected columns
    """
    suffix = {'2015_2016': 'I', '2017_2018': 'J', '2021_2023': 'L'}[cycle]
    raw_dir = os.path.join(DATA_RAW, f'nhanes_{cycle}')

    if cycle == '2021_2023':
        return _merge_nhanes_2021_2023(raw_dir)

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


def pad810q_to_paq650(pad810q):
    """2021-23 PAD810Q (times of vigorous leisure activity) -> PAQ650 coding
    (1 = yes, 2 = no). 0 -> 2, 1..200 -> 1, 7777 / 9999 / missing -> NaN."""
    q = pd.Series(pad810q, dtype=float)
    q = q.where(~q.isin([7777, 9999]))
    return np.where(q.isna(), np.nan, np.where(q > 0, 1.0, 2.0))


def _merge_nhanes_2021_2023(raw_dir):
    """NHANES August 2021 - August 2023 (suffix L), mapped to the column
    names of the earlier cycles. Two variables changed (pre-registered in
    docs/analysis_plan_v5_confirmatory.md §3):

    * BPXSY1 (auscultatory, 1st reading) -> BPXOSY1 (oscillometric, 1st
      reading); BPX_L does not exist, only BPXO_L.
    * PAQ650 "any vigorous recreational activity (1 yes / 2 no)" no longer
      exists. From PAD810Q "how often vigorous leisure-time activity":
      0 -> 2 (no), 1..200 -> 1 (yes), 7777/9999/missing -> missing.
    """
    def load(name, cols):
        df = load_xpt(os.path.join(raw_dir, f'{name}_L.XPT'))
        return df[[c for c in cols if c in df.columns]]

    merged = load('DEMO', ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR'])
    print(f"[DATA] Loaded DEMO (2021_2023): {merged.shape}")
    parts = [
        ('BMX', load('BMX', ['SEQN', 'BMXHT', 'BMXWT', 'BMXBMI', 'BMXWAIST'])),
        ('WHQ', load('WHQ', ['SEQN', 'WHD010', 'WHD020'])),
        ('GHB', load('GHB', ['SEQN', 'LBXGH'])),
        ('BPXO', load('BPXO', ['SEQN', 'BPXOSY1', 'BPXODI1']).rename(
            columns={'BPXOSY1': 'BPXSY1', 'BPXODI1': 'BPXDI1'})),
        ('SMQ', load('SMQ', ['SEQN', 'SMQ020'])),
    ]
    paq = load('PAQ', ['SEQN', 'PAD810Q'])
    paq = pd.DataFrame({'SEQN': paq['SEQN'],
                        'PAQ650': pad810q_to_paq650(paq['PAD810Q'])})
    parts.append(('PAQ (from PAD810Q)', paq))
    for name, df in parts:
        merged = merged.merge(df, on='SEQN', how='left')
        print(f"[DATA] Merged {name}: shape now {merged.shape}")
    return merged


# NHANES "Refused" / "Don't know" codes, per variable (from the codebooks).
# WHD010 (height, inches) and WHD020 (weight, pounds) use 7777 / 9999;
# yes/no and category questions use 7 / 9.
NHANES_MISSING_CODES = {
    'WHD010':   [7777, 9999],
    'WHD020':   [7777, 9999],
    'SMQ020':   [7, 9],
    'PAQ605':   [7, 9],
    'PAQ650':   [7, 9],
    'DMDEDUC2': [7, 9],
}


def replace_missing_codes(df):
    """Replace NHANES refused/don't-know codes with NaN, per column."""
    df = df.copy()
    for col, codes in NHANES_MISSING_CODES.items():
        if col in df.columns:
            df[col] = df[col].replace(codes, np.nan)
    return df


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

    # Step 2: Replace NHANES special codes with NaN — PER COLUMN.
    # FIX 3: the old code applied one shared list [7, 9, 77, 99, 777, ...]
    # to every column, which deleted VALID self-reports such as a height
    # of 77 inches (195.6 cm) or a weight of 99 lb. Codes below are from
    # the NHANES codebooks for these variables.
    df = replace_missing_codes(df)

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
