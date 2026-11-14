# Can an AI Explanation Be Misused?
### Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model

**B.Tech Project (BTP) | Group GP-109 | Delhi Technological University, CSE | 2026–27**

[![Pre-registration](https://img.shields.io/badge/Pre--registered-v0.1--preregistration-blue)](https://github.com/Akshat-stack47/BTP-XAI-Gaming/releases/tag/v0.1-preregistration)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-40%2F40%20passing-brightgreen)]()
[![Review](https://img.shields.io/badge/Integrity%20checks-163%2F164%20(1%20warning)-brightgreen)]()

---

> ### 🔧 v2 — implementation fixes
> This is a **fixed copy** of [`Akshat-stack47/BTP-XAI-Gaming`](https://github.com/Akshat-stack47/BTP-XAI-Gaming),
> which stays unchanged as the comparison baseline. Four implementation bugs are fixed:
> (1) the measured model was attacked through nurse-measured values,
> (2) the LIME-guided attack sometimes moved in the wrong direction,
> (3) valid self-reported heights/weights were deleted as "refused" codes,
> (5) the final run used the 2017–18 lie budget instead of the pre-registered 2015–16 one.
>
> **v3** also fixes the remaining implementation issues:
> (4) the headline N\* is now computed as pre-registered (from the mean curve),
> (6) both attackers use the same lie-size rule, (7) only realistic lies (height ↑, weight ↓),
> (8) per-patient random seeds, (9) LIME shows all features, (10) balanced class weights,
> (11) a random SHAP background. Some of these deviate from the pre-registered plan;
> they are listed in the deviations table in the details file.
> Details: [`docs/fixes_v2.md`](docs/fixes_v2.md).
>
> ✅ **The results, figures and tables in `results/` were regenerated with the v3 code**
> on the real NHANES data (2015–16 trial, 2017–18 held-out final run).

---

## 📋 Overview

This project investigates whether **XAI explanations (SHAP/LIME) create a gaming vulnerability** in clinical AI systems.

We introduce the **Explanation Equivalence metric N\***: the number of blind trial-and-error attempts whose best outcome matches one explanation-guided attempt. A high N\* means the explanation is very useful to a potential adversary.

**Research Question:** Does having a SHAP/LIME explanation help a patient game a diabetes risk model?

---

## 🔬 Key Result

Tested on **NHANES 2017–2018 held-out data (N = 887 test patients)**, v3 code:

| Configuration | **N\*** (primary, pre-registered) | 95% CI | Per-patient mean N\* | % Explanation Useless |
|---|---|---|---|---|
| **RF + SHAP, Reported ★** | **0.0** | **[0.0, 0.0]** | 7.69 | **76.0%** |
| RF + LIME, Reported | 0.0 | [0.0, 0.0] | 7.07 | 73.8% |
| LR + SHAP, Reported | 0.0 | [0.0, 0.0] | 2.58 | 66.4% |
| LR + LIME, Reported | 0.0 | [0.0, 0.0] | 2.99 | 56.3% |
| RF + SHAP, Measured | 0.0 | [0.0, 0.0] | 0.00 | 100% |
| RF + LIME, Measured | 0.0 | [0.0, 0.0] | 0.00 | 100% |
| LR + SHAP, Measured | 0.0 | [0.0, 0.0] | 0.00 | 100% |
| LR + LIME, Measured | 0.0 | [0.0, 0.0] | 0.00 | 100% |

★ Primary pre-registered configuration. Full table: [`results/tables/final_results.csv`](results/tables/final_results.csv).

> **Verdict: LOW THREAT** (N\* < 5)
> On average, one explanation-guided lie lowers the risk score **less** than one random realistic lie
> (RF+SHAP: 0.013 vs 0.024). The explanation leads the patient not to lie (3–31% of patients), or, for the
> random forest, to a lie that **raises** their risk (23–31%).
> Model A (nurse-measured inputs) cannot be gamed at all; Model B can be lowered by ~0.06 with 10 blind attempts.
> **Policy implication: verify self-reported inputs rather than withhold explanations.**

---

## 🗂️ Project Structure

```
BTP/
├── src/                    # Core source code
│   ├── config.py           # FINAL_CONFIG + random seeds
│   ├── data_loader.py      # NHANES XPT data loading
│   ├── gap_analysis.py     # Reporting gap + lie budget
│   ├── models.py           # DiabetesModel (RF + LR)
│   ├── explanations.py     # SHAP + LIME generator
│   ├── adversarial.py      # AdversarialSimulator (core experiment)
│   ├── metrics.py          # N* computation + bootstrap CI + Wilcoxon
│   └── visualization.py   # All 8 publication figures
│
├── scripts/
│   ├── prepare_data.py     # Stage 3: data preparation
│   ├── run_trial.py        # Stage 5: trial run on dev set
│   ├── run_final.py        # Stage 7: final experiment on held-out data
│   ├── run_analysis.py     # Stage 8: figures + tables
│   ├── demo.py             # Live viva demo (patient walkthrough)
│   └── final_review.py     # 164-check project integrity script
│
├── results/
│   ├── figures/            # 8 publication-quality PNG figures (300 DPI)
│   └── tables/             # 6 results tables (CSV + LaTeX)
│
├── docs/
│   ├── analysis_plan.md    # Pre-registration document (locked at v0.1-preregistration)
│   ├── changelog_trial_to_final.md
│   └── column_mapping.md
│
├── paper/                  # Research paper (LaTeX)
├── thesis/                 # BTP thesis (LaTeX, 7 chapters + 4 appendices)
├── tests/                  # 40 unit tests
└── roadmap/                # Stage-by-stage project roadmap (11 stages)
```

---

## ⚙️ Setup & Reproduction

### 1. Clone and install dependencies

```bash
git clone https://github.com/AkkiNikumbh/BTECH_PROJECT.git
cd BTECH_PROJECT
pip install numpy pandas scikit-learn shap lime scipy matplotlib seaborn tqdm pytest
```

### 2. Download NHANES data

Download the following XPT files from [CDC NHANES](https://www.cdc.gov/nchs/nhanes/) for cycles **2015–2016** and **2017–2018**:
- Demographics (DEMO)
- Examination: Body Measures (BMX)
- Questionnaire: Blood Pressure & Cholesterol (BPQ), Physical Activity (PAQ), Smoking (SMQ)
- Laboratory: Glycohemoglobin (GHB)

Place in `data/raw/nhanes_2015_2016/` and `data/raw/nhanes_2017_2018/`.

### 3. Prepare data

```bash
python scripts/prepare_data.py
```

### 4. Run the final experiment

```bash
python scripts/run_final.py
```

### 5. Generate all figures and tables

```bash
python scripts/run_analysis.py
```

### 6. Run the viva demo

```bash
python scripts/demo.py --patient 42
```

### 7. Run tests

```bash
python -m pytest tests/ -v          # 40 unit tests
python scripts/final_review.py      # 164-check integrity
```

---

## 📊 Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Reporting gap KDE — height & weight self-reporting bias |
| Fig 2 ★ | Main result: N* equivalence curve across all 8 configs |
| Fig 3 | Score reduction boxplots (explanation vs. blind) |
| Fig 4 | Model A vs Model B: how much the risk score can be lowered |
| Fig 5 | SHAP vs. LIME agreement (per-patient score reductions) |
| Fig 6 | SHAP feature importance beeswarm |
| Fig 7 | Per-patient N* histogram (population distribution) |
| Fig 8 | All 8 configs: one explanation vs one blind attempt |

---

## 🔒 Pre-Registration

This study was **pre-registered before any analysis of the held-out 2017–2018 data**.

- Pre-registration document: [`docs/analysis_plan.md`](docs/analysis_plan.md)
- Git tag: [`v0.1-preregistration`](https://github.com/Akshat-stack47/BTP-XAI-Gaming/releases/tag/v0.1-preregistration)
- All deviations logged: [`docs/changelog_trial_to_final.md`](docs/changelog_trial_to_final.md)

---

## 👥 Team

| Name | Role |
|------|------|
| Akshat Kumar | Lead developer, experimental design |
| Akash Singh | Data pipeline, analysis, paper writing |
| Aditya Jha | Visualization, thesis writing |

**Guide:** Miss. Anjali Bansal, DTU CSE

---

## 📄 License

This project is for academic purposes (DTU BTP 2026–27). Data from [NHANES](https://www.cdc.gov/nchs/nhanes/) (public domain).
