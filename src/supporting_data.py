"""
Supporting datasets from the research plan (§6.4):

* UCI Heart Disease (UCI id 45) — a small dataset used to TEST THE CODE,
  where results can be checked by hand (work-plan phase P1).
  File: data/raw/uci_45_heart_disease/heart_disease.zip
        (processed.cleveland.data inside, 303 patients)

* CDC Diabetes Health Indicators (UCI id 891) — BRFSS 2015 telephone survey,
  253,680 respondents, used for COMPARISON. Its BMI is computed from
  SELF-REPORTED height and weight and there is no nurse measurement, so it
  only supports a "Model B" (self-reported) comparison.
  File: data/raw/uci_891_cdc_diabetes/data.csv

Both are downloaded from archive.ics.uci.edu and live under data/raw/
(git-ignored), like the NHANES files.
"""
import os
import zipfile
import numpy as np
import pandas as pd

from src.config import DATA_RAW

UCI45_DIR = os.path.join(DATA_RAW, 'uci_45_heart_disease')
UCI45_ZIP = os.path.join(UCI45_DIR, 'heart_disease.zip')
UCI891_CSV = os.path.join(DATA_RAW, 'uci_891_cdc_diabetes', 'data.csv')

# Column names from heart-disease.names (the 14 attributes everyone uses)
UCI45_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']


def load_uci_heart(zip_path=UCI45_ZIP):
    """Cleveland subset of UCI Heart Disease. Target: num > 0 (disease).

    Rows with '?' (missing ca / thal) are dropped -> 297 patients.
    """
    with zipfile.ZipFile(zip_path) as z:
        with z.open('processed.cleveland.data') as f:
            df = pd.read_csv(f, header=None, names=UCI45_COLUMNS,
                             na_values='?')
    df = df.dropna().reset_index(drop=True)
    df['target'] = (df['num'] > 0).astype(float)
    return df


# UCI 891 features chosen to mirror the NHANES feature set as closely as the
# BRFSS questions allow: age, sex, blood pressure, smoking, physical activity
# and (self-reported) body size.
UCI891_FEATURES = ['Age', 'Sex', 'HighBP', 'Smoker', 'PhysActivity', 'BMI']
UCI891_GAMEABLE = ['BMI']


def load_cdc_diabetes(csv_path=UCI891_CSV):
    """CDC Diabetes Health Indicators. Target: Diabetes_binary (0/1).

    Note: in BRFSS the diabetes label is ALSO self-reported (unlike the
    NHANES HbA1c blood test) — a limitation of this comparison.
    """
    df = pd.read_csv(csv_path)
    X = df[UCI891_FEATURES].to_numpy(dtype=float)
    y = df['Diabetes_binary'].to_numpy(dtype=float)
    return X, y, list(UCI891_FEATURES)


def nhanes_bmi_lie_budget(df_complete, pct=95):
    """Realistic BMI lie, from the NHANES 2015-2016 (development) gap.

    BMI from self-reported height/weight minus BMI from the nurse's
    measurement; same rule as the height/weight budget (95th percentile of
    the absolute gap, direction = sign of the mean gap).
    """
    bmi_self = df_complete['self_weight_kg'] / (df_complete['self_height_cm'] / 100) ** 2
    bmi_meas = df_complete['measured_weight_kg'] / (df_complete['measured_height_cm'] / 100) ** 2
    gap = (bmi_self - bmi_meas).dropna()
    return {
        'BMI': {
            'max_lie': float(gap.abs().quantile(pct / 100)),
            'mean_direction': 'positive' if gap.mean() > 0 else 'negative',
            'mean_magnitude': float(gap.abs().mean()),
            'mean_gap': float(gap.mean()),
            'n': int(len(gap)),
        }
    }
