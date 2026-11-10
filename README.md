# Can an AI Explanation Be Misused?
### Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model

**B.Tech Project (BTP) | Group GP-109 | Delhi Technological University, CSE | 2026–27**

[![Pre-registration](https://img.shields.io/badge/Pre--registered-v0.1--preregistration-blue)](https://github.com/Akshat-stack47/BTP-XAI-Gaming/releases/tag/v0.1-preregistration)
[![Python](https://img.shields.io/badge/Python-3.12-green)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-7%2F7%20passing-brightgreen)]()
[![Review](https://img.shields.io/badge/Integrity%20checks-164%2F164-brightgreen)]()

---

## 📋 Overview

This project investigates whether **XAI explanations (SHAP/LIME) create a gaming vulnerability** in clinical AI systems.

We introduce the **Explanation Equivalence metric N\***: the number of blind trial-and-error attempts whose best outcome matches one explanation-guided attempt. A high N\* means the explanation is very useful to a potential adversary.

**Research Question:** Does having a SHAP/LIME explanation help a patient game a diabetes risk model?

---

## 🔬 Key Result

Tested on **NHANES 2017–2018 held-out data (N = 884 patients)**:

| Configuration | N* (mean) | 95% CI | % Explanation Useless |
|---|---|---|---|
| **RF + SHAP, Reported ★** | **12.43** | **[10.54, 14.53]** | **63.8%** |
| RF + LIME, Reported | 9.41 | [7.66, 11.27] | 67.9% |
| LR + SHAP, Reported | 22.77 | [20.36, 25.28] | 59.0% |
| LR + LIME, Reported | 22.71 | [20.28, 25.26] | 55.1% |
| RF + SHAP, Measured | 10.27 | [8.42, 12.15] | 69.2% |
| RF + LIME, Measured | 9.01 | [7.32, 10.77] | 69.8% |
| LR + SHAP, Measured | 23.05 | [20.58, 25.61] | 57.1% |
| LR + LIME, Measured | 21.27 | [18.99, 23.71] | 56.7% |

★ Primary pre-registered configuration

> **Verdict: MODERATE THREAT** (5 ≤ N* ≤ 20)
> For 63.8% of patients, the explanation provides **zero advantage** over a single random guess.
> **Policy implication: Verify inputs rather than withhold explanations.**

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
├── tests/                  # 7 unit tests
└── roadmap/                # Stage-by-stage project roadmap (11 stages)
```

---

## ⚙️ Setup & Reproduction

### 1. Clone and install dependencies

```bash
git clone https://github.com/Akshat-stack47/BTP-XAI-Gaming.git
cd BTP-XAI-Gaming
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
python -m pytest tests/ -v          # 7 unit tests
python scripts/final_review.py      # 164-check integrity
```

---

## 📊 Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Reporting gap KDE — height & weight self-reporting bias |
| Fig 2 ★ | Main result: N* equivalence curve across all 8 configs |
| Fig 3 | Score reduction boxplots (explanation vs. blind) |
| Fig 4 | Model A (measured) vs Model B (reported) comparison |
| Fig 5 | SHAP vs. LIME per-patient N* scatter |
| Fig 6 | SHAP feature importance beeswarm |
| Fig 7 | Per-patient N* histogram (population distribution) |
| Fig 8 | Configuration heatmap (all 8 configs) |

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
