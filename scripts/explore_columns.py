"""
Stage 2: Explore NHANES columns and produce the column mapping document.
Usage: python scripts/explore_columns.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')

CYCLE_SUFFIX = {'2015_2016': 'I', '2017_2018': 'J'}

# Files and the columns we care about (from research plan)
TARGET_COLS = {
    'DEMO': ['SEQN','RIDAGEYR','RIAGENDR','RIDRETH3','DMDEDUC2','INDFMPIR'],
    'BMX' : ['SEQN','BMXHT','BMXWT','BMXBMI','BMXWAIST'],
    'WHQ' : ['SEQN','WHD010','WHD020'],
    'GHB' : ['SEQN','LBXGH'],
    'BPX' : ['SEQN','BPXSY1','BPXDI1'],
    'SMQ' : ['SEQN','SMQ020'],
    'PAQ' : ['SEQN','PAQ605','PAQ650'],
}

COLUMN_DESCRIPTIONS = {
    'SEQN'    : 'Respondent sequence number (unique patient ID)',
    'RIDAGEYR': 'Age in years at screening',
    'RIAGENDR': 'Gender (1=Male, 2=Female)',
    'RIDRETH3': 'Race/Hispanic origin (6 categories)',
    'DMDEDUC2': 'Education level (adults 20+)',
    'INDFMPIR': 'Ratio of family income to poverty',
    'BMXHT'   : '[MEASURED] Standing height (cm) — nurse measured',
    'BMXWT'   : '[MEASURED] Body weight (kg) — nurse measured',
    'BMXBMI'  : '[MEASURED] Body Mass Index (kg/m^2) — calculated from measured',
    'BMXWAIST': '[MEASURED] Waist circumference (cm) — nurse measured',
    'WHD010'  : '[SELF-REPORTED] Current self-reported height (inches)',
    'WHD020'  : '[SELF-REPORTED] Current self-reported weight (pounds)',
    'LBXGH'   : '[LAB/OBJECTIVE] Glycohemoglobin HbA1c (%) — DIABETES TARGET',
    'BPXSY1'  : '[MEASURED] Systolic blood pressure 1st reading (mmHg)',
    'BPXDI1'  : '[MEASURED] Diastolic blood pressure 1st reading (mmHg)',
    'SMQ020'  : '[SELF-REPORTED] Smoked at least 100 cigarettes in life (1=Yes,2=No)',
    'PAQ605'  : '[SELF-REPORTED] Vigorous work activity (1=Yes,2=No)',
    'PAQ650'  : '[SELF-REPORTED] Vigorous recreational activities (1=Yes,2=No)',
}

GAMEABILITY = {
    'WHD010': ('YES', 'Patient verbally reports — bounded by reported-vs-measured gap'),
    'WHD020': ('YES', 'Patient verbally reports — bounded by reported-vs-measured gap'),
    'SMQ020': ('MAYBE', 'Binary self-report, can flip yes/no'),
    'PAQ605': ('MAYBE', 'Binary self-report, can claim more activity'),
    'PAQ650': ('MAYBE', 'Binary self-report, can claim more activity'),
    'BMXHT' : ('NO', 'Nurse measures directly — not manipulable'),
    'BMXWT' : ('NO', 'Nurse measures directly — not manipulable'),
    'BPXSY1': ('NO', 'Nurse measures directly — not manipulable'),
    'BPXDI1': ('NO', 'Nurse measures directly — not manipulable'),
    'LBXGH' : ('NO', 'Lab blood test — CANNOT be faked by patient'),
    'RIDAGEYR':('NO', 'Verified by ID — not manipulable'),
    'RIAGENDR':('NO', 'Recorded by staff — not manipulable'),
}

def explore_cycle(cycle):
    suffix = CYCLE_SUFFIX[cycle]
    raw_dir = os.path.join(RAW_DIR, f'nhanes_{cycle}')
    print(f"\n{'='*60}")
    print(f"  NHANES {cycle.replace('_','-')} (suffix _{suffix})")
    print(f"{'='*60}")

    summary = {}
    for component, cols in TARGET_COLS.items():
        filename = f"{component}_{suffix}.XPT"
        filepath = os.path.join(raw_dir, filename)
        if not os.path.exists(filepath):
            print(f"\n  [{component}] FILE NOT FOUND: {filename}")
            continue
        df = pd.read_sas(filepath, format='xport')
        available = [c for c in cols if c in df.columns]
        missing   = [c for c in cols if c not in df.columns]
        print(f"\n  [{component}] {filename} — {df.shape[0]:,} rows, {df.shape[1]} total cols")
        for col in available:
            n_valid = int(df[col].notna().sum())
            pct     = n_valid / len(df) * 100
            desc    = COLUMN_DESCRIPTIONS.get(col, '')
            game    = GAMEABILITY.get(col, ('—',''))[0]
            print(f"    {col:12s} | {n_valid:6,} valid ({pct:5.1f}%) | gameable={game:<6} | {desc}")
        if missing:
            print(f"    MISSING cols: {missing}")
        summary[component] = df[available].copy()
    return summary

if __name__ == '__main__':
    for cycle in ['2015_2016']:          # Only explore dev set (Stage 2 rule)
        explore_cycle(cycle)

    # Write column mapping doc
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'docs')
    os.makedirs(docs_dir, exist_ok=True)
    doc_path = os.path.join(docs_dir, 'column_mapping.md')
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write("# NHANES Column Mapping Document\n")
        f.write("## GP-109 BTP Project\n\n")
        f.write("| Column | Description | Gameable? | Reason |\n")
        f.write("|--------|-------------|-----------|--------|\n")
        for col, desc in COLUMN_DESCRIPTIONS.items():
            g, reason = GAMEABILITY.get(col, ('—',''))
            f.write(f"| `{col}` | {desc} | {g} | {reason} |\n")
        f.write("\n## Critical Pairs (lie budget columns)\n")
        f.write("| Self-Reported | Measured | Unit Conversion |\n")
        f.write("|--------------|----------|-----------------|\n")
        f.write("| `WHD010` (inches) | `BMXHT` (cm) | × 2.54 |\n")
        f.write("| `WHD020` (pounds) | `BMXWT` (kg) | × 0.453592 |\n")
        f.write("\n## Target Variable\n")
        f.write("`LBXGH >= 6.5%` → diabetes = 1 (ADA threshold)\n")
    print(f"\nColumn mapping saved: {doc_path}")
