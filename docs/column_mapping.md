# NHANES Column Mapping Document
## GP-109 BTP Project

| Column | Description | Gameable? | Reason |
|--------|-------------|-----------|--------|
| `SEQN` | Respondent sequence number (unique patient ID) | — |  |
| `RIDAGEYR` | Age in years at screening | NO | Verified by ID — not manipulable |
| `RIAGENDR` | Gender (1=Male, 2=Female) | NO | Recorded by staff — not manipulable |
| `RIDRETH3` | Race/Hispanic origin (6 categories) | — |  |
| `DMDEDUC2` | Education level (adults 20+) | — |  |
| `INDFMPIR` | Ratio of family income to poverty | — |  |
| `BMXHT` | [MEASURED] Standing height (cm) — nurse measured | NO | Nurse measures directly — not manipulable |
| `BMXWT` | [MEASURED] Body weight (kg) — nurse measured | NO | Nurse measures directly — not manipulable |
| `BMXBMI` | [MEASURED] Body Mass Index (kg/m^2) — calculated from measured | — |  |
| `BMXWAIST` | [MEASURED] Waist circumference (cm) — nurse measured | — |  |
| `WHD010` | [SELF-REPORTED] Current self-reported height (inches) | YES | Patient verbally reports — bounded by reported-vs-measured gap |
| `WHD020` | [SELF-REPORTED] Current self-reported weight (pounds) | YES | Patient verbally reports — bounded by reported-vs-measured gap |
| `LBXGH` | [LAB/OBJECTIVE] Glycohemoglobin HbA1c (%) — DIABETES TARGET | NO | Lab blood test — CANNOT be faked by patient |
| `BPXSY1` | [MEASURED] Systolic blood pressure 1st reading (mmHg) | NO | Nurse measures directly — not manipulable |
| `BPXDI1` | [MEASURED] Diastolic blood pressure 1st reading (mmHg) | NO | Nurse measures directly — not manipulable |
| `SMQ020` | [SELF-REPORTED] Smoked at least 100 cigarettes in life (1=Yes,2=No) | MAYBE | Binary self-report, can flip yes/no |
| `PAQ605` | [SELF-REPORTED] Vigorous work activity (1=Yes,2=No) | MAYBE | Binary self-report, can claim more activity |
| `PAQ650` | [SELF-REPORTED] Vigorous recreational activities (1=Yes,2=No) | MAYBE | Binary self-report, can claim more activity |

## Critical Pairs (lie budget columns)
| Self-Reported | Measured | Unit Conversion |
|--------------|----------|-----------------|
| `WHD010` (inches) | `BMXHT` (cm) | × 2.54 |
| `WHD020` (pounds) | `BMXWT` (kg) | × 0.453592 |

## Target Variable
`LBXGH >= 6.5%` → diabetes = 1 (ADA threshold)
