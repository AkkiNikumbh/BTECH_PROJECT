# GP-109 Antigravity Prompt Package

**Project:** *Can an AI Explanation Be Misused? Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model*  
**Institution/group:** Delhi Technological University (DTU), Department of Computer Science & Engineering; Group GP-109  
**Package purpose:** Generate a repository containing 14 separate Markdown documents, `AGENTS.md`, and a clearly stubbed Python code skeleton. This is a prompt package, not an implementation or a report of experimental results.

## How to use this package

1. Open Google Antigravity in the intended project workspace/repository.
2. Paste **Prompt 1: Master Prompt** once. It defines the whole project, guardrails, repository layout, and acceptance standard.
3. Then paste **Stage 1**, let Antigravity finish and report, review its files, and only then paste **Stage 2**. Repeat for **Stage 3**.
4. Do not allow the agent to inspect, load, profile, summarize, or otherwise use NHANES 2017–2018 data until the public pre-registration record exists and its timestamp/permalink have been recorded. This is a hard gate, not a reminder.
5. Treat all items labelled **NEEDS GUIDE APPROVAL** as proposed defaults or unresolved questions, not as approved methodology.

---

# Part I — Ambiguities and inconsistencies to resolve explicitly

The items below were identified from the six-page PDF, the accompanying project requirements, and a preliminary literature/source check. The prompt instructions preserve original wording where required, distinguish facts from proposals, and prohibit an agent from silently inventing a resolution.

| ID | Ambiguity / inconsistency | Handling in the prompt package | Status |
|---|---|---|---|
| A-01 | **The original schedule violates the pre-registration hard gate.** Original P7 begins 8 Jan 2027, before P6's 14 Jan 2027 end. | Preserve the original schedule as supplied; show the proposed schedule separately with P7 on 15 Jan–20 Feb 2027; document the dependency and block all real 2017–2018 data access until the timestamped public pre-registration record exists. | Corrected schedule is proposed; **NEEDS GUIDE APPROVAL**. Hard gate is mandatory regardless. |
| A-02 | **Measured-input model conflicts with permitted edits.** The two-model design swaps reported and measured height/weight, but the attacker may edit only self-reported height/weight; nurse-measured values are expressly immutable. | Do not make measurements editable. Implement interfaces that can represent an unavailable/not-applicable attack result. Mark the measured-model OAE comparison as not interpretable under the current action set. Request guide approval for the scientific comparison before claiming an OAE difference. | Critical; **NEEDS GUIDE APPROVAL**. |
| A-03 | **“95th percentile of the gap” does not define signedness or feature-specific tails.** Weight is intended to move down and height slightly up, but a single signed gap percentile does not necessarily define both bounds. Units, sentinel codes and outliers also matter. | In P2, verify variable meanings/units from official NHANES codebooks. Document directional, feature-specific gap calculation using only development data. Keep lower-weight/higher-height as the stated intended directions. Do not silently switch between signed and absolute gaps. | **NEEDS GUIDE APPROVAL** for the exact calculation. |
| A-04 | **OAE “matches” is not operationally exact.** The empirical success-rate curves may not be identical at any listed budget; Arm Q might never reach Arm E by k=100. | Proposed definition: smallest grid value k where Arm Q's success rate is at least Arm E's; if none exists, report OAE as right-censored (`>100`) rather than inventing a value. Preserve the exact definition and make this operational interpretation explicit for guide approval. | **NEEDS GUIDE APPROVAL**. |
| A-05 | **The one budget for Holm correction is missing.** B4 says one pre-fixed budget but does not name it. | Propose `k=10` because it is on the required budget grid and lies near its middle. Register and freeze it before P7. Do not describe it as already approved. | **NEEDS GUIDE APPROVAL**. |
| A-06 | **Model family is underspecified.** The stack lists logistic regression and random forest, but the primary algorithm and role of each are not locked. B4's primary correction family only names SHAP/LIME × reported/measured representation, not an additional algorithm dimension. | Document a proposal: random forest as primary (consistent with the stated fast TreeSHAP path) and logistic regression as a separately labelled secondary/robustness analysis; do not broaden or change the confirmatory comparison family without approval. | **NEEDS GUIDE APPROVAL**. |
| A-07 | **Train/validation/trial/final split protocol is not fully specified.** P5 trials on 2015–2016; P7 evaluates on 2017–2018. It is unclear exactly which 2015–2016 rows train the model, set the threshold, and supply trial estimates. | Propose fixed, documented train/validation/development-evaluation partitions inside 2015–2016. Fit preprocessing/model on train only, choose the 90%-sensitivity threshold on validation, and freeze them before the 2017–2018 evaluation. Never use holdout performance for tuning. The exact split proportions/strategy need guide approval and registration. | **NEEDS GUIDE APPROVAL**. |
| A-08 | **The primary success denominator needs clarification.** The plan says the model must initially flag high risk, then a flip is counted on participants whose true label is diabetic. It does not spell out whether the denominator is all diabetic adults or initially high-risk diabetic adults. | Proposed interpretation: define the eligible primary cohort before attack as true-label-positive participants who are also initially above the model-specific high-risk threshold; report cohort counts and the denominator explicitly. This is not a clinical diagnostic protocol. | **NEEDS GUIDE APPROVAL**. |
| A-09 | **Attribution sign does not uniquely determine a feature movement.** SHAP/LIME sign describes a feature's contribution under a chosen output and baseline/local surrogate; it is not automatically a derivative or causal effect. | Define a reproducible, model-output-specific mapping from attribution sign to candidate direction, subject it to hand-checkable tests, and describe it as a heuristic. Do not claim causal recourse. Keep it within the bounded feature actions. | **NEEDS GUIDE APPROVAL**. |
| A-10 | **Explanation request versus attempt accounting needs to be explicit.** B2 says one explanation request counts as one attempt, while both arms receive the same attempt budget; SHAP and LIME create separate comparison cells. | For each patient/seed/model representation, define separate matched SHAP and LIME E-arms against a Q control with the same permitted score-query budget. Count an E-arm explanation request as one attempt in the main analysis. Include an “explanation is free” sensitivity analysis. State precisely what one attempt returns and log the counts. | Operational details to register; **NEEDS GUIDE APPROVAL** where unresolved. |
| A-11 | **NHANES variables are expected, not confirmed.** WHD010/WHD020 and BMXHT/BMXWT are candidate names, but names, units, availability, skip patterns and sentinel values must be checked in each cycle's codebook. | Do not hard-code these candidates as verified variables. P2 must inventory the 2015–2016 and 2017–2018 codebooks and write a versioned mapping/data dictionary. Only verified mapped variables may be selected. | P2 evidence required; no silent assumptions. |
| A-12 | **Diabetes label and robustness target need data-level definitions.** HbA1c ≥ 6.5% is primary; the robustness label adds fasting plasma glucose ≥ 126 mg/dL in the fasting subsample. Sample membership, lab names, units, and fasting subsample restrictions are not mapped in the PDF. | Verify official lab variables/units and missingness/fasting rules in P2. Implement primary and robustness targets separately; never substitute a self-reported diagnosis for the blood-test target. | P2 verification plus **NEEDS GUIDE APPROVAL** for any gap. |
| A-13 | **Pregnancy exclusion variable is not named.** The population is adults 20+ with pregnant participants excluded. | Find the appropriate cycle-specific official indicator and limitations in P2; document who can be classified as pregnant and who has missing/unknown status. Do not invent a variable name or treat missing status as automatically non-pregnant. | P2 verification required. |
| A-14 | **Feature leakage rules and “identical except one column group” can conflict with engineered features.** BMI is derived from each height/weight representation, and the two representation columns must not accidentally enter the same model. | Define a shared non-height/weight feature set plus exactly one representation-specific group. Recompute the appropriate BMI from the corresponding pair where BMI is included. Keep label/label proxies (glucose, HbA1c, diagnosis, diabetes medication) out of features. Document feature parity and test it. | Locked principle; details need codebook and guide review. |
| A-15 | **Supporting-dataset roles are not detailed.** UCI Diabetes Health Indicators (dataset 891) is for comparison and UCI Heart Disease (dataset 45) is for code testing, but no exact target/features/analysis are specified. | Keep NHANES as the only primary confirmatory study. Use Heart Disease only for pipeline/hand-checkable QA; use Diabetes Health Indicators only for a labelled comparison after its data dictionary and label/features are audited. Do not pool results or imply it uses the same HbA1c label. | **NEEDS GUIDE APPROVAL** for comparison scope. |
| A-16 | **Reference currency wording conflicts with the bibliography.** The PDF says the five references are all from the last five years but includes 2021 publications; whether those qualify depends on the exact publication dates and the proposal's date. | Preserve the five chosen references; list the “last five years” criterion as an item to reconcile with the guide. Do not replace references silently. Verify exact dates and cite the original paper/publisher metadata. | **NEEDS GUIDE APPROVAL** if the criterion is strict. |
| A-17 | **The Milli et al. (2019) attribution appears potentially mismatched.** The PDF attributes the “ten explanation queries can match a thousand ordinary ones” statement to Milli et al. and describes that work as FAT* 2019. This must be checked in the original paper before retaining. | Treat the statement as unverified; locate the exact passage or remove/rewrite the claim after supervisor review. Do not move the result to another author/paper without evidence. | Manual verification required. |
| A-18 | **The novelty claim may be too broad.** The PDF says no study measures whether explanations help people improve scores compared with the same number of ordinary attempts. A potentially related paper, Arora et al., “Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations” (AAAI 2022), studies people editing inputs to reduce model confidence and includes no-explanation comparisons. Its exact overlap with this clinical/tabular/OAE study must be checked. | Add it to the related-work gap check as a candidate adjacent study, not as an unapproved replacement for the required bibliography. Narrow the novelty claim if its task or comparison overlaps; explicitly contrast population/task, data modality, explainer/query budget, model output and OAE metric. Guide approval is required for changes to the research-gap statement. | Manual review and **NEEDS GUIDE APPROVAL**. |
| A-19 | **“Transparency is usually better” attribution to Hikri & Rosenfeld is not yet adequately supported by the PDF.** The original document says only the abstract had been seen. | Keep the user-requested citation format for ref. [2], but explicitly flag it for manual verification of author spelling, version, conference metadata, and the precise conclusion supported by the full paper. Do not state this claim as established until verified. | Manual verification required. |
| A-20 | **SHAP/LIME relationship is overgeneralized in the prose.** The PDF says a common version of SHAP is mathematically very close to LIME. This must be scoped to the specific variants and weighted linear-regression construction, not generalized to every SHAP method. | Explain the variant-level relationship accurately and cite an original/authoritative source. Keep the experiment configuration explicit: TreeSHAP for tree models where applicable and LIME settings fully recorded. | Verify before final literature write-up. |
| A-21 | **The broad legal statement (“laws often require explanations”) has no jurisdiction or legal citation specified.** | Avoid universal/legal advice wording. If retained in the paper, identify the jurisdictions/rules and verify the legal claim. Otherwise describe regulation as motivation without claiming all laws require SHAP/LIME. | Research-writing review. |
| A-22 | **Research ethics statement is written as a conclusion.** The PDF says ethics approval is not required because data are public and anonymous; actual institutional review requirements are institution-specific. | Present this as the plan's proposed ethics assessment and ask the guide/DTU process to confirm it. Do not claim institutional approval/exemption was granted unless documented. | **NEEDS GUIDE APPROVAL**. |
| A-23 | **Timeline dates and Gantt are not fully consistent/detail-complete.** Several PDF phase rows omit the year, phases overlap, and the original Gantt displays only its own limited spare-time cell(s). | Expand years from context and label the expansion as normalization. Preserve the original dates and original Gantt buffer separately from the proposed schedule; show dependencies and use the proposed dates only as a proposal. Do not invent extra spare time. | Schedule proposal needs approval. |
| A-24 | **Compute-time estimates mix work categories.** Data/model preparation and explanation generation are each “a few minutes,” main experiments are 6–11 hours, and total computer time is 30–50 hours. | Preserve all estimates, label them approximate CPU compute-time estimates, and track actual timing later; do not promise these estimates are guaranteed. Keep overnight execution/re-run buffer. | Retain as estimates. |

### External verification leads relevant to the ambiguity register

These are leads for the literature agent, not a substitute for opening and checking the original paper in full:

- Arora et al., *Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations*, AAAI 2022: https://ojs.aaai.org/index.php/AAAI/article/view/20464
- Hikri and Rosenfeld, *Ambiguous Strategic Classification*, arXiv:2606.10137: https://arxiv.org/abs/2606.10137; official PMLR listing to re-check against the full paper and current proceedings metadata: https://proceedings.mlr.press/v306/hikri26a.html
- Bilodeau et al., *Impossibility theorems for feature attribution*, PNAS 2024: https://doi.org/10.1073/pnas.2304406120
- König et al., *Improvement-Focused Causal Recourse (ICR)*, AAAI 2023: https://doi.org/10.1609/aaai.v37i10.26398
- Shokri et al., *On the Privacy Risks of Model Explanations*, AIES 2021: https://doi.org/10.1145/3461702.3462533
- Ghalme et al., *Strategic Classification in the Dark*, ICML 2021 / PMLR 139: https://proceedings.mlr.press/v139/ghalme21a.html
- LIME original KDD page: https://www.kdd.org/kdd2016/subtopic/view/why-should-i-trust-you-explaining-the-predictions-of-any-classifier
- Adebayo et al., *Sanity Checks for Saliency Maps*, NeurIPS 2018: https://proceedings.neurips.cc/paper/2018/hash/294a8ed24b1ad22ec2e7efea049b8737-Abstract.html
- Bordt et al., *Post-Hoc Explanations Fail to Achieve their Purpose in Adversarial Contexts*, FAccT 2022: https://facctconference.org/static/pdfs_2022/facct22-3533153.pdf
- Official NHANES 2015–2016 questionnaire variable list: https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Component=Questionnaire&Cycle=2015-2016
- Official NHANES 2017–2018 variable list: https://wwwn.cdc.gov/nchs/nhanes/search/variablelist.aspx?Cycle=2017-2018

---

# Part II — Prompt 1: Master Prompt for Google Antigravity

Copy everything in the prompt below into Antigravity as the master instruction.

## MASTER PROMPT START

### Role

Act as a senior research-engineering lead, an experimental ML engineer, a scientific reproducibility reviewer, a data-stewardship lead, and a technical writer supporting a B.Tech CSE final-year research project. Produce a coherent, reviewable repository consisting of separate Markdown documents, an `AGENTS.md` rules file, and a Python code skeleton with stubs only. Do not fabricate implementation results, statistics, data, paper claims, citations, page numbers, approvals, or experimental output.

### Context and objective

Project title: **“Can an AI Explanation Be Misused? Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model”**. This extends the approved synopsis **“Evaluating the Trustworthiness of Explainable AI Methods in Clinical Risk Prediction”**, which identified adversarial robustness of explanations as future work.

Team: Group GP-109, Department of Computer Science & Engineering, Delhi Technological University.

- Final-year members: Akash Singh (23/CS/036), Aditya Jha (23/CS/025), Akshat Kumar (23/CS/039).
- Junior member: Ishan Singh (25/CS/168), second year.
- Guide: Miss Anjali Bansal, Department of CSE.
- Co-guide: none specified.

Research question: In a paired simulated experiment on the same eligible patients, does access to a SHAP or LIME explanation improve an agent's ability to move a model's risk score below a fixed high-risk threshold more than the same number of ordinary score-only attempts? Report the result as an ordinary-attempt equivalent (OAE). This is a measurement study on public, anonymous data, not a real-world deception tool and not a medical recommendation system.

The comparison arms are:

- **Arm Q (query-only control):** receives the returned risk score after each attempt and no explanation.
- **Arm E (explanation arm):** receives the returned score and the selected SHAP or LIME attribution for the current input, using the attribution-guided movement rule documented in the experiment specification.

Use the same patient cases, matching seed structure, permitted actions, attempt-budget grid, and comparable scoring-query accounting across arms. An explanation request counts as one attempt in the main analysis; “explanation is free” is a sensitivity analysis.

### Source authority and conflict handling

Use these source priorities:

1. The supplied six-page `GP109_Detailed_Research_Plan_updated.pdf` for the original plan, team, original phase dates, stated references, compute estimates, ethics position and expected outputs.
2. The accompanying user requirements (Sections A–D) for expanded project facts, locked design decisions, deliverables, and mandatory output format.
3. The ambiguity register in this prompt package for issues that must be exposed instead of silently resolved.

Treat Sections B1–B6 as the intended defaults throughout the documents, but repeat every Section B decision in `docs/13_Assumptions_and_Open_Questions.md` with the exact tag **NEEDS GUIDE APPROVAL**. That tag does not mean the decision is rejected; it means it must be brought to the project guide and not misrepresented as approved.

Where a source fact is clear, preserve it. Where it is underspecified or conflicting, (a) state the conflict, (b) provide a proposed operational default, (c) record it in document 13 as **NEEDS GUIDE APPROVAL**, and (d) ensure no downstream document contradicts the register. Do not change the scientific question or silently delete a difficult requirement to make implementation easier.

### Constraints

1. Build a repository of **separate files**. Do not collapse the deliverables into one long document. Create exactly these project documents under `/docs`:
   - `00_README_and_Index.md`
   - `01_Project_Charter.md`
   - `02_Roadmap.md`
   - `03_Phase_Breakdown.md`
   - `04_Implementation_Plan.md`
   - `05_Data_Specification.md`
   - `06_Experiment_Design.md`
   - `07_Statistical_Analysis_Plan.md`
   - `08_Preregistration_Template.md`
   - `09_Testing_and_QA.md`
   - `10_Risk_Register.md`
   - `11_Literature_and_References.md`
   - `12_Paper_and_Thesis_Outline.md`
   - `13_Assumptions_and_Open_Questions.md`
2. Put `AGENTS.md` at repository root. Include all agent-level restrictions stated below.
3. Make a code skeleton only: signatures, dataclasses/config shape where useful, interfaces, TODOs, and tests. No fake results, fabricated plots, populated result tables, invented outcome values, or pseudo-results presented as real. A minimal clearly labelled toy fixture for tests is allowed; never present it as research data.
4. Do not download, open, load, profile, sample, list records from, derive statistics from, or otherwise inspect actual NHANES 2017–2018 data before a published, timestamped pre-registration exists. Do not create a hidden shortcut that bypasses this gate. Do not put the holdout dataset in the repository. The rule applies even if the file is already on disk or a user asks you to inspect it early. Tests must use toy data or mocks, not the real holdout.
5. The holdout hard gate must be enforced by code as well as documented in prose. Final-run code should check a pre-registration manifest containing at least the public permalink, timestamp (timezone-aware), version/hash of the registered plan, and the required immutable configuration/version references. If the record is absent, invalid, or earlier/later in an inconsistent way, refuse to load final holdout files. Do not assume the mere existence of `docs/08_Preregistration_Template.md` proves that a public pre-registration occurred.
6. Do not invent citations or state that a citation is verified merely because a search snippet exists. Verify each work against the original paper and authoritative publisher/proceedings metadata. Record verification status and evidence. Specifically preserve the required reference-[2] citation format: **“I. Hikri and N. Rosenfeld, Ambiguous Strategic Classification, ICML 2026, arXiv:2606.10137”** and flag it for manual verification of author spelling against the arXiv PDF, conference venue/volume/pages metadata, and whether the paper supports the plan's transparency claim. Do not treat the claim “transparency is usually better” as confirmed until its full-paper argument is checked.
7. Explicitly verify all five numbered references in the PDF and review all foundational / other works named in the literature review. The attribution of “ten explanation queries can match a thousand ordinary ones” to Milli et al. (2019) is unverified: find the exact support in the original paper or report that it cannot be substantiated. Investigate Arora et al. (AAAI 2022), “Explain, Edit, and Understand: Rethinking User Study Design for Evaluating Model Explanations,” as potentially related work; do not claim novelty without a documented comparison. Do not silently replace the bibliography or overstate the gap.
8. All claims, recommendations and conclusions must distinguish: plan-supplied facts; verified external literature; proposed defaults; hypotheses; and measured results. There are no measured results at skeleton stage.
9. Preserve these scope and ethics limitations: public, anonymous data only; no collection from people; simulated agent for measurement only; no user-facing tool for deceiving insurers/clinicians; no deployment; no medical recommendations. Phrase the plan's “no ethics approval needed” statement as the current plan's assessment subject to DTU/guide confirmation, not as proof of an institutional exemption.
10. Python 3.9 or later; target CPU-only laptops with 8–16 GB RAM and less than 5 GB total storage. No GPU requirement. Use the requested stack: pandas, numpy, scikit-learn (logistic regression and random forest), shap, lime, scipy, statsmodels, matplotlib, seaborn, and Git. Use fixed seeds and compatibility-aware dependency versions; do not rely on GPUs.
11. Keep raw downloaded datasets outside Git and document download source, cycle, file, retrieval date, checksum, license/conditions and variable mapping. Do not commit personal or other restricted data. All NHANES data in this project are public and anonymous, but reproducibility metadata still matters.
12. Never generate synthetic results and label them as actual findings. If code is not implemented, say “stub / TODO”; if a decision is unresolved, say “NEEDS GUIDE APPROVAL”; if a source could not be confirmed, say “unverified”.
13. Update the traceability matrix in document 00 after every task/stage. Each A-section, B-section and PDF detail must point to at least one document, code module/test, or task.
14. Use clear, specific English and imperative instructions. Avoid generic filler. Make deliverables and acceptance criteria testable.

### Locked decisions — carry into every relevant document

#### B1. Attacker model

- Editable features: only patient-plausibly misreported **self-reported height and weight**; BMI, if included, is derived from the currently editable pair.
- Immutable features: age, sex, nurse-measured values, lab values, and all other features unless explicitly approved otherwise.
- Derive lie bounds from the empirical NHANES distribution of self-reported minus measured gaps using development data only. Main setting: 95th percentile; intended movement direction: lower reported weight and slightly higher reported height. Sensitivity settings: 50th, 90th and 99th percentile.
- Same optimizer in both arms: coordinate hill-climbing with random restarts over editable features inside bounds.
- Arm Q sees only the returned score per attempt.
- Arm E also sees SHAP or LIME attribution for the current input and uses its sign under a documented rule to choose a feature/direction greedily.
- Critical unresolved issue: if the measured-input model uses only immutable nurse-measured height/weight, its OAE is not estimable under the current attacker actions. Do not make those fields editable. Put the contradiction and options in document 13 and require guide approval before reporting a two-model OAE difference.

#### B2. Attempts, success, outcomes

- Budget sweep exactly `k in {1, 2, 3, 5, 10, 20, 50, 100}`; same patients for the paired comparison.
- One explanation request counts as one attempt in the main analysis; conduct an “explanation is free” sensitivity analysis.
- Primary success: a baseline high-risk case is moved below the threshold, producing a risk-class flip. The threshold is the operating point achieving 90% sensitivity on the validation set.
- Count “cheat success” on true diabetic cases; record and resolve the exact primary denominator before registration. The proposed denominator is the cohort of true-label-positive patients initially flagged high risk, because only these cases can experience the specified high-to-low flip.
- Secondary outcome: mean risk-score reduction.
- Headline OAE: smallest k at which Arm Q matches Arm E success rate. Because exact matching may not occur, propose “Arm Q reaches or exceeds Arm E” and right-censor OAE at `>100` if no grid point qualifies; document as **NEEDS GUIDE APPROVAL**.

#### B3. Label/population/leakage

- Primary label: diabetes where HbA1c >= 6.5% (blood test).
- Robustness label: HbA1c >= 6.5% OR fasting plasma glucose >= 126 mg/dL, using the eligible fasting subsample only.
- Population: adults age 20 and above; exclude pregnant participants.
- Never include glucose, HbA1c, diabetes diagnosis, or diabetes medication as model features.
- Candidate reported variables: WHD010/WHD020. Candidate measured variables: BMXHT/BMXWT. These are **unverified candidates** until P2 confirms the variable names, codebook descriptions, units, cycle availability, skip patterns and special/missing codes for both NHANES cycles. Never hard-code them as verified before P2.

#### B4. Statistics

- Paired design using the same patients in E and Q.
- McNemar test for paired flip rates at each budget.
- Wilcoxon signed-rank test for paired risk-score reduction.
- Patient-level paired bootstrap with 2,000 resamples for confidence intervals on OAE and risk difference.
- 20 attacker-randomness seeds.
- Effect sizes: risk difference, paired odds ratio, OAE.
- Pre-registered “small advantage” definition: OAE <= 2.
- Holm correction over the primary family `{SHAP, LIME} x {self-reported model, measured model}` at one fixed budget. The budget is not specified; propose k=10 as a candidate, mark **NEEDS GUIDE APPROVAL**, and freeze before P7. If the measured-model OAE is not estimable due to B1, do not fabricate an entry for that family cell.
- Two-model contrast: difference in OAE between self-reported and measured model, with bootstrap CI, only if the guide-approved operational definitions make both quantities estimable and scientifically comparable.

#### B5. Timeline

The original timeline must appear in document 02 alongside the proposed correction. Keep the original Gantt chart's spare/buffer time rather than reallocating it without approval.

| Phase | Original schedule | Proposed schedule | Notes |
|---|---|---|---|
| P1 | 8–21 Oct 2026 | 8–21 Oct 2026; close out by 31 Oct 2026 | Overdue as of 28 Oct 2026; close out in parallel with P2 |
| P2 | 15 Oct–14 Nov 2026 | 15 Oct–14 Nov 2026 | Verify NHANES codebooks / column mapping |
| P3 | 1–30 Nov 2026 | 1–30 Nov 2026 | Prepare data and reported-vs-measured gaps |
| P4 | 8 Nov–7 Dec 2026 | 8 Nov–7 Dec 2026 | Build experiment code/train models; overlaps P3 and P5 |
| P5 | 1–14 Dec 2026 | 1–14 Dec 2026 | Trial on 2015–2016 data |
| P6 | 1–14 Jan 2027 | 1–14 Jan 2027 | Publish timestamped analysis plan before final run |
| P7 | 8 Jan–7 Mar 2027 | 15 Jan–20 Feb 2027 | Original schedule violates gate; proposal creates rerun/analysis buffer |
| P8 | 1–31 Mar 2027 | 21 Feb–20 Mar 2027 | Analyses, figures/tables, reference verification |
| P9 | 15 Mar–30 Apr 2027 | 15 Mar–30 Apr 2027 | Write/submit paper to a conference selected later |
| P10 | 15 Apr–31 May 2027 | 15 Apr–31 May 2027 | Thesis/final report |
| P11 | 22 May–14 Jun 2027 | 22 May–14 Jun 2027 | Viva and final submission |

In the PDF, some dates do not repeat the year; expand years from the phase sequence and identify that as formatting normalization. Keep the original Gantt's gray buffer cell(s) visible in the original-version rendering/transcription. Do not silently extend the buffer beyond what the PDF shows. Proposed dates are not approved dates until the guide agrees.

#### B6. References

In P8, verify the five numbered references in the PDF against the original papers/publisher pages. Never invent citations, page numbers, venue names or results. Reference [2] must be written as: **I. Hikri and N. Rosenfeld, Ambiguous Strategic Classification, ICML 2026, arXiv:2606.10137** and flagged for manual verification of author spelling in the arXiv PDF, venue pages/volume metadata and the plan's claim that the paper suggests transparency is usually better. The plan says only its abstract has been reviewed; do not overstate the claim.

### Required documents and acceptance criteria

#### `docs/00_README_and_Index.md` — overview, index, traceability

Include the project purpose, repository map, team/guide, safety/ethics scope, “what can be run now vs. later,” prerequisites, no-results warning, doc navigation, stage workflow, and hard-gate summary. Add a traceability matrix mapping **every PDF section/detail and every item in Sections A and B** to one or more documents, code modules, tests or tasks. At minimum include separate rows for every individual item and sub-item in Section A and for each of B1–B6; also map every source PDF section, original timeline date, Gantt/buffer, hardware/tool/compute item, output, and ethics note. Mark items unresolved if they require approval. Add a short status table after each stage; agents must update it every time they complete a task.

**Acceptance:** no plan requirement is untraceable; all 14 docs are linked; critical holdout gate and the measured-model inconsistency are obvious; no claim that experiment results already exist.

#### `docs/01_Project_Charter.md` — charter

Include exact title and approved-synopsis continuity; team names, roll numbers, guide/no co-guide, problem statement, research gap stated carefully, research question, objectives, hypotheses, in-scope/out-of-scope items, expected contributions, target users (academic reviewers only), ethics boundaries, compute/storage assumptions, and success criteria. State the expected finding (small advantage) as a pre-run expectation, not a result, and explicitly state both outcomes are reportable. Explain that the aim is to measure disclosure effects, not weaken transparency obligations.

**Acceptance:** all team and project facts match the sources; no unverified novelty or legal assertion is stated as fact; “no ethics approval needed” is qualified as a plan assessment pending institutional/guide confirmation.

#### `docs/02_Roadmap.md` — master timeline

Include both original and proposed schedules exactly as above; identify what changed and why; milestones, dependencies, critical path, Mermaid Gantt charts for both versions, P1 overdue-closeout action, original Gantt buffer representation, proposed rerun buffer, decision gates, risks, and the hard gate between P6 and P7. Explain overlaps rather than silently shifting phases. Show approval status of proposed changes.

**Acceptance:** all 11 phase date ranges appear for both schedules; P7 cannot precede pre-registration; critical path and buffer are visible; Mermaid syntax is valid; no invented spare-time intervals.

#### `docs/03_Phase_Breakdown.md` — P1 through P11

Create a separate detailed section for every phase, in order. Each phase must include: objective; inputs; tasks and subtasks; owners as placeholders (`Member 1`, `Member 2`, `Member 3`, `Junior`); deliverables; acceptance criteria; definition of done; dependencies; risks/mitigations; exit checklist; evidence/artifacts to commit; and gate conditions. Use the phase descriptions in the PDF: P1 environment and hand-checkable small dataset; P2 NHANES documentation and column classification; P3 merges and reported-vs-measured gap analysis; P4 experiment code/models; P5 2015–2016 trial and first graphs; P6 freeze settings and timestamp/publish analysis plan; P7 final 2017–2018 runs; P8 analysis, graphs/tables, verify five references; P9 paper; P10 thesis/report; P11 viva/final submission. Include the proposed closeout of P1 by 31 Oct in parallel with P2.

**Acceptance:** exactly P1–P11, each with all required subheadings and a testable completion checklist; P7's entry checklist checks the timestamped registration manifest and refuses all holdout access if missing.

#### `docs/04_Implementation_Plan.md` — architecture and build order

Define repository/file tree; modular package boundaries; class/function responsibilities; data-flow diagram (Mermaid); configurations; seeds; logging; experiment/run manifests; environment setup for CPU laptops; Git strategy/branching/commit hygiene; code style/type hints/docstrings; error handling; data-version/checksum strategy; experiment artifact naming; and incremental build order. Plan for Python 3.9+, pandas, numpy, scikit-learn logistic regression/random forest, shap, lime, scipy, statsmodels, matplotlib and seaborn. Use TreeSHAP for tree estimators where suitable. Include a single central configuration pattern, no hard-coded unverified column names, and a pre-registration holdout guard.

**Acceptance:** every module has a clear single purpose; data flow distinguishes development from holdout; seeds/configuration and logs are centrally defined; code structure supports stubs without fake results; gate is enforceable and testable.

#### `docs/05_Data_Specification.md` — datasets and variable provenance

Specify an inventory procedure for NHANES 2015–2016 and 2017–2018 questionnaire, examination and laboratory files; identify SEQN as the candidate join key but verify the official role; produce a codebook-backed table with cycle, component/file, variable, description, units, eligibility/skip pattern, special/missing values, mapping decision, and source URL/version. Check the candidates WHD010/WHD020/BMXHT/BMXWT against each cycle before use. Document merging, duplicate handling, adult 20+ rule, pregnancy exclusion and missing status, the HbA1c label, fasting-glucose robustness label, missing-data/cleaning rules, label leakage exclusions, representation-specific height/weight/BMI, sample-flow counts to be filled only when actually computed, splits, survey weights/design limitations, and dataset separation. NHANES 2015–2016 is development/trial data; NHANES 2017–2018 is untouched final holdout. Document UCI Diabetes Health Indicators (UCI 891) as comparison-only and UCI Heart Disease (UCI 45) as code-testing-only unless the guide approves another role. Never pool their scores with NHANES confirmatory findings.

**Acceptance:** every chosen variable has official codebook evidence; unverified values remain TODOs; all leakage exclusions are asserted in tests; no real 2017–2018 record is accessed before the gate.

#### `docs/06_Experiment_Design.md` — formal E/Q experiments

Specify Arm E and Arm Q mathematically and procedurally, paired patient/seed identity, baseline eligibility, risk score definition, threshold locking, feature editing, immutable columns, BMI derivation, feature-specific gap distribution/bounds, restarts, coordinate order, tie-breaking, stopping rules, failed/invalid moves, score-query logging and budget accounting, explanation-request accounting, SHAP output selection/baseline, LIME configuration, attribution-sign/direction mapping, all k values, primary and secondary outcomes, OAE calculation, censoring when no budget qualifies, the “explanation is free” sensitivity, 50th/90th/95th/99th percentile sensitivity settings, fasting-label robustness, and the two-model comparison. Include a formal subsection called **Measured-model actionability conflict**. Keep it unresolved until guide approval; if the only permitted editable variables are absent from the measured model, do not force an OAE estimate.

**Acceptance:** every B1/B2 item is operationally specified or explicitly flagged; identical patients and budget accounting can be audited; no feature outside the approved action set can change; no real-world deception feature/tool is built.

#### `docs/07_Statistical_Analysis_Plan.md` — hypotheses and inference

Prespecify primary/secondary hypotheses; estimands; population/denominators; risk difference; paired odds ratio; OAE and censoring; McNemar tests at every budget; Wilcoxon signed-rank for paired risk-score reductions; patient-level paired bootstrap with exactly 2,000 resamples for OAE/risk-difference CIs; 20 attacker seeds; uncertainty aggregation over patients and seeds; Holm correction for the primary family `{SHAP,LIME} × {self-reported, measured}` at one frozen budget; proposed `k=10` visibly tagged **NEEDS GUIDE APPROVAL** until signed off; effect-size and CI reporting; analysis of both model representations; robustness/sensitivity analyses; missing/unestimable comparisons; and reporting templates for every outcome. Distinguish statistical significance from practical size; define small advantage OAE <= 2 before P7. Do not cherry-pick thresholds or budgets after holdout review.

**Acceptance:** formula/definition for each metric and test; sample pairing identified; correction family and budget fixed before holdout; all outcomes including null, large, censored and unestimable results have a reporting path; no fake power or results claims.

#### `docs/08_Preregistration_Template.md` — timestamped plan for P6

Produce a publication-ready preregistration template intended to be published online with an immutable timestamp before the final experiment. Include project/title/team; research question; hypotheses and stated expected finding; data cycles/splits and holdout rules; target label and population; exact feature inventory and leakage guard; model/threshold procedure; E/Q definitions; permitted edits/bounds; SHAP/LIME settings; budgets; attempt accounting; outcomes and OAE; exclusions/missingness; statistical tests/effect sizes/bootstrap/seeds/Holm family/fixed budget; small-advantage threshold; sensitivity analyses; stopping and rerun policy; compute environment/seeds/code commit hash/config hashes; ethics statement; deviations policy; publication permalink/timestamp; and a blank results section clearly reserved for post-run outcomes. Include an immutable preregistration manifest schema for gate checks. Expected result: **we expect explanations to confer only a small advantage, operationally OAE <= 2; this is a hypothesis, not a finding**. State explicitly that a large advantage is also reportable.

**Acceptance:** can be published unchanged after decisions are approved; timestamp/permalink/hash fields are explicit; the plan contains no results from the untouched holdout; a deviation must be documented rather than silently edited.

#### `docs/09_Testing_and_QA.md` — tests and quality gates

Define unit tests for transforms, joins, BMI calculation, missing-value/sentinel handling, label construction, target-leakage rejection, feature parity, risk threshold, budget accounting, deterministic seeds, explainer wrappers, action bounds, coordinate search interfaces, OAE edge cases/censoring, paired statistics, bootstrap shape, plotting interfaces, and preregistration gate. Include hand-checkable toy datasets and UCI Heart Disease (dataset 45) strictly for pipeline/code testing. Test reproducibility across same seed/config, test that changing a seed changes stochastic paths where expected, and check that the real holdout path is never opened before the gate. Provide commands and expected assertions, not fabricated model metrics.

**Acceptance:** tests fail closed for leakage and holdout access; unit tests are deterministic; toy fixtures are marked synthetic; no expected performance score is hard-coded.

#### `docs/10_Risk_Register.md` — risks

Create a table with ID, category, cause, likelihood, impact, warning signal, prevention, contingency, owner placeholder, buffer/dependency, and status. Cover: holdout contamination/pre-registration delay; measured-model actionability contradiction; codebook variable mismatch; label leakage; missing/fasting labs and population exclusions; model/explainer incompatibility; long overnight runs and CPU/RAM/storage; stochastic non-reproducibility; statistical non-estimability; citation mismatch/novelty risk; schedule overlap/P1 closeout; teamwork/ownership; ethical misuse/misleading framing; and conference submission uncertainty. Include the 6–11 hour main run and 30–50 computer-hour project estimate, overnight run plan, and rerun buffer. Keep probabilities/impact ratings qualitative until the team sets them; do not invent numerical risk scores.

**Acceptance:** each high-impact risk has an actionable mitigation, owner placeholder and trigger; holdout gate and citation checks are not buried; mitigation does not assume extra compute hardware.

#### `docs/11_Literature_and_References.md` — literature matrix and citation audit

Summarize each named work, its actual research question/method/result, its direct relevance and limits for this project, and a verified canonical citation/DOI/proceedings/arXiv URL. Cover all work named in the PDF: Ribeiro et al. (LIME, KDD 2016); Lundberg & Lee (SHAP, NeurIPS 2017); Adebayo et al. (NeurIPS 2018); Bilodeau et al. (PNAS 2024); Milli et al. (2019); Shokri et al. (AIES 2021); Ghalme et al. (ICML 2021); Hikri & Rosenfeld (ICML 2026/arXiv:2606.10137); König et al. (AAAI 2023); and Bordt et al. (FAccT 2022). Also examine Arora et al. (AAAI 2022) as a potentially close prior study for the research-gap review; if adding it to the formal cited bibliography, flag that decision to the guide. Do not make up missing bibliographic fields, page ranges or paper findings.

Verification checklist for the five numbered references printed in the PDF:

1. B. Bilodeau, N. Jaques, P. W. Koh, and B. Kim, “Impossibility theorems for feature attribution,” *Proceedings of the National Academy of Sciences*, vol. 121, no. 2, e2304406120, 2024.
2. I. Hikri and N. Rosenfeld, “Ambiguous Strategic Classification,” ICML 2026, arXiv:2606.10137. **MANUAL VERIFICATION REQUIRED:** check author spelling against the arXiv PDF; verify proceedings title, volume/pages/year from an authoritative proceedings record; inspect the full paper before repeating the assertion that it implies transparency is usually better.
3. G. König, T. Freiesleben, and M. Grosse-Wentrup, “Improvement-Focused Causal Recourse (ICR),” *Proceedings of the AAAI Conference on Artificial Intelligence*, vol. 37, no. 10, pp. 11847–11855, 2023.
4. R. Shokri, M. Strobel, and Y. Zick, “On the Privacy Risks of Model Explanations,” *Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society (AIES)*, pp. 231–241, 2021.
5. G. Ghalme, V. Nair, I. Eilat, I. Talgam-Cohen, and N. Rosenfeld, “Strategic Classification in the Dark,” *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR 139, pp. 3672–3681, 2021.

Treat the five-year recency statement as a separate check because the PDF's 2021 items may or may not satisfy a strict date-based five-year window. Reconcile rather than silently replace them. Check the attribution of “ten explanation queries can match a thousand ordinary ones” to Milli et al. (2019) directly in the original. Compare the project's claimed novelty with Arora et al. (2022) and other directly relevant work found during the literature search. Scope the SHAP/LIME mathematical relationship to the specific variants actually supported by the literature.

**Acceptance:** every named work has status `verified`, `partially verified`, or `unverified` with evidence; reference [2] stays flagged until manual checks close; the “ten vs. thousand” statement has verified support or is marked unsupported for supervisor review; research gap is appropriately bounded to the study's precise task/setting/metric.

#### `docs/12_Paper_and_Thesis_Outline.md` — writing and viva

Provide section-by-section outlines for the research paper (P9), B.Tech thesis/final project report (P10), and viva/final submission (P11). Include tentative section headings, key research questions, required figures/tables, where each claim will be sourced, responsible owner placeholders, and a rule that results sections remain placeholders until the analysis is executed. Include plausible viva questions and answer-planning prompts on: problem/gap; SHAP vs LIME; attack model; reported vs measured values; label/leakage; paired design; threshold/sensitivity; OAE interpretation; statistical tests and CIs; multiple comparisons; pre-registration; holdout protocol; ethical implications; limitations and generalizability; why public survey data; why no claim of real-world deception; and what happens if the measured model has no editable inputs.

**Acceptance:** outlines align with docs 05–08; no results are invented; both possible result patterns and a not-estimable measured-model outcome have reporting plans.

#### `docs/13_Assumptions_and_Open_Questions.md` — approval register

List every B1–B6 decision verbatim or faithfully, each tagged **NEEDS GUIDE APPROVAL**, plus any additional missing detail or conflict identified in the ambiguity register. For each item include: source; why it matters; proposed default; alternatives; consequences if unresolved; who should decide (guide/team placeholder); decision/status/date fields left blank. Must include, at minimum: measured-model attacker contradiction; directional percentile method; OAE “matches” meaning and censoring; exact fixed Holm budget (candidate k=10); model family hierarchy; data split/threshold procedure; target denominator; explanation accounting and attribution sign mapping; cycle-specific codebook and pregnancy variable; fasting subsample rules; supporting datasets' roles; recency rule for five references; Milli attribution verification; Arora et al. related-work overlap; reference [2] metadata/transparency claim; legal and ethics qualification; and schedule proposal approval. Never mark a question approved without explicit evidence.

**Acceptance:** every section-B decision is present, visibly tagged, and consistent with the other docs; no implicit “approved” language; critical questions are highlighted for the guide.

### `AGENTS.md` rules file

Create the root `AGENTS.md` with enforceable instructions for all agents:

- Always use fixed, recorded random seeds; expose seed/configuration in each run manifest.
- NEVER read or load actual NHANES 2017–2018 data before a valid, public pre-registration timestamp/permalink/plan hash exists. Never profile or preview it early. Tests use toy fixtures/mocks only.
- Do not fabricate numbers, model scores, sample counts, plots, literature claims, citations, page numbers, approvals, or run output.
- Cite only verified sources; use original papers and authoritative publisher/proceedings records; leave uncertain claims marked unverified.
- Preserve the user's exact research question and scope; do not construct a deployable deception or medical decision product.
- Do not make nurse-measured values editable; the measured-model conflict must remain visible pending approval.
- Keep raw data out of Git; do not place holdout files in the repository.
- Update `docs/00_README_and_Index.md` traceability matrix after every task/stage.
- Keep `docs/13_Assumptions_and_Open_Questions.md` in sync after any proposed methodology or plan change.
- Keep code skeleton as stubs with TODOs; no fake results or simulated “final” artifacts.
- Use Python 3.9+, CPU-first dependencies, type hints, clear naming, no global unseeded randomness, no hidden data access, and tests for data leakage/holdout gate.
- Stage rule: after each staged prompt, stop and report exactly what was created/changed, tests/checks performed, unresolved assumptions, and the next stage's prerequisites. Do not continue to the next stage automatically.

### Code skeleton — stubs only, no fake results

Create the following structure (minor additions are allowed only when justified and traced in doc 00):

```text
gp109-explanation-misuse/
├── README.md
├── AGENTS.md
├── requirements.txt
├── .gitignore
├── pyproject.toml                 # optional; if used, keep Python >=3.9 compatible
├── configs/
│   ├── default.yaml
│   ├── experiment.yaml
│   └── variable_mapping.example.yaml  # candidates marked UNVERIFIED until P2
├── docs/
│   ├── 00_README_and_Index.md
│   ├── 01_Project_Charter.md
│   ├── 02_Roadmap.md
│   ├── 03_Phase_Breakdown.md
│   ├── 04_Implementation_Plan.md
│   ├── 05_Data_Specification.md
│   ├── 06_Experiment_Design.md
│   ├── 07_Statistical_Analysis_Plan.md
│   ├── 08_Preregistration_Template.md
│   ├── 09_Testing_and_QA.md
│   ├── 10_Risk_Register.md
│   ├── 11_Literature_and_References.md
│   ├── 12_Paper_and_Thesis_Outline.md
│   └── 13_Assumptions_and_Open_Questions.md
├── data/
│   ├── README.md                  # download, provenance, checksums; no raw datasets committed
│   ├── raw/.gitkeep
│   └── processed/.gitkeep
├── preregistration/
│   ├── README.md
│   └── manifest.example.yaml      # example schema only; not a real registration record
├── src/gp109/
│   ├── __init__.py
│   ├── config.py
│   ├── reproducibility.py
│   ├── logging_utils.py
│   ├── tracking.py
│   ├── holdout_gate.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── inventory.py
│   │   ├── nhanes.py
│   │   ├── merge.py
│   │   ├── cleaning.py
│   │   ├── labels.py
│   │   └── splits.py
│   ├── features.py
│   ├── models.py
│   ├── explainers.py
│   ├── attack/
│   │   ├── __init__.py
│   │   ├── bounds.py
│   │   ├── search.py
│   │   └── arms.py
│   ├── metrics.py
│   ├── statistics.py
│   ├── plots.py
│   └── experiment.py
├── scripts/
│   ├── check_environment.py
│   ├── run_unit_tests.py
│   ├── run_trial.py
│   └── run_final.py                 # must check holdout gate before opening data
├── tests/
│   ├── conftest.py
│   ├── fixtures/                    # toy hand-checkable data only
│   ├── test_labels.py
│   ├── test_feature_leakage.py
│   ├── test_feature_parity.py
│   ├── test_bounds.py
│   ├── test_attempt_budget.py
│   ├── test_oae.py
│   ├── test_statistics.py
│   ├── test_reproducibility.py
│   └── test_holdout_gate.py
└── reports/
    └── figures/.gitkeep
```

Stub requirements:

- `requirements.txt`: list compatible dependencies for pandas, numpy, scikit-learn, shap, lime, scipy, statsmodels, matplotlib, seaborn, plus a test runner if selected. Avoid pins that drop Python 3.9 support. Keep dependencies modest for CPU-only laptops.
- `configs/default.yaml`: random seed(s), data root, log level, output paths, and run metadata placeholders. `configs/experiment.yaml`: attempt grid, 20 attack seeds, 2,000 bootstrap resamples, 90%-sensitivity threshold target, percentile sensitivity grid, candidate primary Holm budget `k=10` labelled pending approval, and label/specification placeholders. Do not list unverified variables as confirmed.
- `src/gp109/data/nhanes.py`: stubs for locating documented input files, reading only approved cycle data, verifying columns/schema and returning typed data structures. The holdout gate must execute before any final-cycle file is opened.
- `merge.py`, `cleaning.py`, `labels.py`, `features.py`: stubs for joins, cycle-aware cleaning, HbA1c label, fasting glucose robustness label, adult/pregnancy eligibility, reported/measured feature sets, derived BMI, and leakage checks.
- `models.py`: train/predict interfaces for logistic regression and random forest; no fitted result artifacts shipped.
- `explainers.py`: SHAP and LIME wrappers with explicit model-output, explainer configuration and random-state arguments; TreeSHAP where suitable. No claims that specific settings are approved until recorded.
- `attack/bounds.py`: interface for cycle/data-source-labelled empirical gap bounds and percentile sensitivity. Do not compute bounds from final holdout data.
- `attack/search.py` and `attack/arms.py`: interfaces for coordinate hill-climbing, random restarts, Arm Q, Arm E-SHAP and Arm E-LIME. Explicitly reject edits to immutable features. Handle the measured-model conflict without making measurements editable.
- `metrics.py`: stubs for baseline risk, risk-class flip, risk reduction and OAE/censoring.
- `statistics.py`: stubs for paired McNemar, Wilcoxon, risk difference, paired odds ratio, patient-level paired bootstrap, and Holm correction. Accept seeds/resamples/config; no result values hard-coded.
- `plots.py`: plotting function signatures and TODOs for attempt curves, paired effects, OAE estimates/CI, and reported-versus-measured gap distributions. No fabricated charts.
- `tracking.py`/`logging_utils.py`: record config hash, data manifest hash, code revision, model/explainer versions, seed, budgets, timestamps and timing. Do not log actual sensitive raw rows into general logs.
- `holdout_gate.py`: validate a published registration manifest (permalink, timezone-aware timestamp, plan hash/version, and required settings). A template file is not enough. Fail closed if absent/invalid. Test with toy manifests; don't touch real holdout data in tests.
- `run_trial.py`: intended for permitted 2015–2016 development/trial workflow only.
- `run_final.py`: first validate preregistration gate, then load the final holdout; if not registered, exit before any attempt to open the holdout path. This is a stub until the research team implements it.
- `tests/`: small hand-checkable deterministic fixtures, leakage guards, bounds/immutability, paired alignment, attempt counts and OAE edge cases. No tests rely on real NHANES 2017–2018 files.
- `.gitignore`: raw/processed data, run outputs, environment files, caches, logs with row-level material, generated non-approved result artifacts, `.venv`, etc. Retain only placeholder files and documentation in Git.

Every stub must contain a concise docstring, type hints where practical, and a `TODO` or `NotImplementedError` explaining the implementation expected in its phase. Do not fill in fake numbers to make the code appear complete.

### Quality bar

- Cross-document consistency: names, dates, sample definitions, variables, budget grid, statistics, roles, scope and gate must agree everywhere.
- Traceability: every project fact/design choice maps into doc 00's matrix; all B decisions appear in doc 13 with **NEEDS GUIDE APPROVAL**.
- Methodological honesty: do not resolve the measured-model actionability contradiction by stealth; do not make broad novelty claims before comparing related work; do not represent a stub or hypothesis as a result.
- Reproducibility: fixed/recorded random seeds, config/data/code hashes, deterministic unit tests and explicit software versions.
- No data leakage: label proxies excluded; final holdout unavailable before registration; only codebook-verified variables used.
- Ethics: no real-world evasion/deception deployment, no medical advice or participant recruitment.
- Documentation quality: precise imperatives, acceptance criteria, clear Mermaid diagrams, explicit uncertainty, no generic filler.
- Resource fit: CPU-only, 8–16 GB RAM, under 5 GB storage; expected prep/training few minutes; explanation generation a few minutes with fast TreeSHAP where applicable; main runs 6–11 hours; total computer time 30–50 hours. Treat these as planning estimates.
- The repository should be useful immediately for review but should not mislead anyone into believing the experiments have already run.

### Working order

1. Inspect and inventory the supplied project plan and this master prompt. Do not access NHANES 2017–2018 data.
2. Generate Stage 1 deliverables (docs 00–03) and stop. Report files made, acceptance checks, unresolved questions and traceability coverage.
3. After the user reviews Stage 1, generate Stage 2 deliverables (docs 04–07) and stop. Check that methods remain consistent with docs 00–03.
4. After the user reviews Stage 2, generate Stage 3 deliverables (docs 08–13), `AGENTS.md`, and the code skeleton; stop and report.
5. Run documentation/code-structure checks and tests that use toy fixtures only. Confirm that no real final-holdout file was read. Update the traceability matrix after every stage.
6. Do not move into full experimental implementation, download the holdout, train final models, run final attacks, or report results as part of these prompt-generation stages.

### Final response format after each stage

Return: (a) created/updated paths; (b) a short summary per deliverable; (c) acceptance checklist with pass/fail/not-yet; (d) assumptions still needing guide approval; (e) traceability rows added/updated; (f) tests/checks run and their output; (g) an explicit statement that NHANES 2017–2018 has not been accessed. Stop there.

## MASTER PROMPT END

---

# Part III — Prompt 2: Stage 1 (documents 00–03)

Paste only after Antigravity has received the Master Prompt. Stage 1 must be self-contained enough to execute without being truncated.

## STAGE 1 PROMPT START

Using the Master Prompt and `AGENTS.md` rules (if already present), create only the following files:

- `docs/00_README_and_Index.md`
- `docs/01_Project_Charter.md`
- `docs/02_Roadmap.md`
- `docs/03_Phase_Breakdown.md`

Do not start Stage 2 or generate code yet. Do not access any NHANES 2017–2018 data.

### Stage 1 task list

1. Create a useful README/index with the exact project/team details, repo index placeholder, document navigation, scope/ethics guardrails, phase overview and a full traceability matrix. Map each PDF section and every individual user requirement in Sections A and B to a document/task. Include a status column (`covered`, `partially covered`, `NEEDS GUIDE APPROVAL`, `not yet implemented`). Include the original-vs-proposed timeline change and measured-model actionability issue.
2. Write the charter with exact title, approved synopsis continuity, project group/team/guide, problem, research question, objectives, scope and exclusions, ethical constraints, compute assumptions and success criteria. State expected result only as an expectation; say a large effect or a null/unestimable result must also be reported honestly. Keep any claims about the novelty or legal requirements appropriately qualified.
3. Write the roadmap with both the original PDF schedule and proposed correction dates exactly as in the Master Prompt. Include a valid Mermaid Gantt for each schedule, dependencies, critical path, overlapping phases, overdue P1 closeout, original Gantt buffer, proposed rerun buffer, risks and the pre-registration hard gate. Do not delete the original schedule to hide its conflict.
4. Write P1–P11 breakdown. For every phase include objective, inputs, tasks/subtasks, owner placeholders (`Member 1`, `Member 2`, `Member 3`, `Junior`), outputs, acceptance criteria, definition of done, dependency, risk/mitigation and exit checklist. Capture the phase-specific tasks exactly from the Master Prompt/PDF. P1 should explicitly close out by 31 Oct 2026 in parallel with P2 under the proposed plan. P6 must publish the timestamped pre-registration before any P7 holdout access.
5. Make the traceability matrix explicit enough that an independent reviewer can follow each requirement to its document or task. Do not write “all items covered” without row-level evidence.

### Stage 1 acceptance gate

- All four files exist and are readable Markdown.
- Team, guide, title, synopsis, problem, expected outcomes, compute requirements and ethics limitations are accurate.
- All 11 original phase date ranges and all 11 proposed ranges appear.
- The original P7 conflict is visible and the hard gate is not negotiable.
- Both schedules have valid Mermaid Gantts; phases/buffer are not silently rearranged.
- Each phase P1–P11 has all required fields.
- Every PDF/Section A/Section B item has a traceability row, or is explicitly listed as requiring a later-stage document/task.
- The guide-approval state of proposed dates and unresolved measured-model design is clear.
- No results are fabricated; no final holdout data was accessed.

After these files are created, **stop**. Report exactly which files were created, the checks performed, coverage gaps, traceability changes, any unresolved conflicts, and confirm that no NHANES 2017–2018 data was accessed. Do not proceed to Stage 2.

## STAGE 1 PROMPT END

---

# Part IV — Prompt 3: Stage 2 (documents 04–07)

Paste only after the Stage 1 files have been reviewed and accepted.

## STAGE 2 PROMPT START

Continue the GP-109 repository according to the Master Prompt and accepted Stage 1 documents. Create only:

- `docs/04_Implementation_Plan.md`
- `docs/05_Data_Specification.md`
- `docs/06_Experiment_Design.md`
- `docs/07_Statistical_Analysis_Plan.md`

Do not create docs 08–13 or the code skeleton yet. Do not access any NHANES 2017–2018 data. Update `docs/00_README_and_Index.md` traceability after completing the stage.

### Stage 2 task list

1. Design the repository/module architecture and documented build order; define configuration, reproducibility, logging/experiment tracking, Git conventions, dependency handling, CPU-only resource budget, and flow diagram. Keep code interfaces and details compatible with Python 3.9+.
2. Write a codebook-first NHANES data specification. WHD010/WHD020/BMXHT/BMXWT are only candidates until verified from the official codebooks for each cycle. Specify source file inventory, join key verification, units, missing/special codes, adult 20+ eligibility, pregnancy exclusion, primary HbA1c target and fasting-subsample robustness target, leakage exclusions, merges, feature construction, trial-vs-holdout separation and supporting dataset roles. Use no actual 2017–2018 records.
3. Formalize Arm E and Arm Q, paired units, 20 seeds, budgets `{1,2,3,5,10,20,50,100}`, attempt and explanation accounting, coordinate hill-climbing/random restarts, feature-specific lie bounds at 50th/90th/95th/99th percentiles, lower-weight/slightly-higher-height intention, primary/secondary outcomes, threshold at 90% validation sensitivity, SHAP/LIME setup and OAE. Explicitly discuss the measured-model actionability conflict: measured features are immutable under B1, so do not claim a comparable measured-model OAE unless the guide approves a scientifically coherent action set.
4. Specify statistical hypotheses/tests/effect sizes, McNemar at each budget, Wilcoxon for paired score reductions, 2,000 patient-level paired bootstrap resamples for OAE and risk difference, 20 seeds, Holm correction for `{SHAP,LIME} × {reported,measured}` at one fixed budget, proposed k=10 pending approval, small advantage OAE <=2, risk-difference and paired odds-ratio reporting, two-model OAE difference/CI when estimable, and every sensitivity analysis. Mark any underspecified denominator/split/model family/metric convention for guide approval.
5. Ensure docs 04–07 match the accepted title, scope, schedule, facts and constraints from docs 00–03. Use diagrams and tables where they improve clarity, not decorative filler.
6. Make no performance result, sample count, variable mapping, bound value, citation verification, or final-holdout statistic up. A field that depends on later data inspection must be documented as a planned computation, with no value supplied.

### Stage 2 acceptance gate

- All four files exist and are linked from doc 00.
- The stated tech stack, Python 3.9+, fixed seeds, CPU/memory/storage limits and compute estimates appear.
- Variable candidates are not labelled as verified until P2 codebook review.
- Labels, leakage exclusions, splits, pregnancy and fasting-subsample rules are explicit or flagged.
- Experiment setup includes every locked B1/B2 choice, every budget and sensitivity setting, E/Q fairness and query accounting.
- Statistics include every B4 test, seed/resample count, effect size and correction family.
- The measured-model contradiction is conspicuous and is not “solved” by permitting edits to measured values.
- The primary k=10 Holm budget and other proposals are tagged **NEEDS GUIDE APPROVAL**.
- Docs 00–03 remain consistent and the traceability matrix is updated.
- No results are fabricated; no NHANES 2017–2018 data was accessed.

After creation, **stop** and report paths, summaries, acceptance status, documentation changes, unresolved decisions and traceability updates. Confirm that no NHANES 2017–2018 data was accessed. Do not proceed to Stage 3.

## STAGE 2 PROMPT END

---

# Part V — Prompt 4: Stage 3 (documents 08–13, AGENTS.md, code skeleton)

Paste only after Stage 2 has been reviewed and accepted.

## STAGE 3 PROMPT START

Continue the GP-109 repository using the Master Prompt and accepted docs 00–07. Create:

- `docs/08_Preregistration_Template.md`
- `docs/09_Testing_and_QA.md`
- `docs/10_Risk_Register.md`
- `docs/11_Literature_and_References.md`
- `docs/12_Paper_and_Thesis_Outline.md`
- `docs/13_Assumptions_and_Open_Questions.md`
- root `AGENTS.md`
- the stub-only code skeleton described in the Master Prompt, including requirements and configuration files

Do not access any NHANES 2017–2018 data. Do not run a final experiment or create result figures. Update `docs/00_README_and_Index.md` traceability after all files are generated.

### Stage 3 task list

1. Create an online-publication-ready preregistration template that fixes every confirmatory choice before final holdout access: question/hypotheses/expected result, data and splits, labels, population, features/leakage, model/threshold, E/Q arms, feature bounds, explanation settings, k grid, outcomes/OAE, tests/effect sizes/bootstrap/seeds/Holm family/fixed budget, sensitivities, stopping/deviation policy, reproducibility hashes and the public URL/timestamp. Include a separate manifest schema for enforcement. Clearly distinguish an empty template from an actual published record.
2. Write QA and tests using hand-checkable toy datasets; include UCI Heart Disease (dataset 45) as a planned code-test source only. Cover label/leakage/feature parity, data/merge sanity, immutable fields, bounds, threshold and attempt accounting, deterministic seeds, OAE no-crossing/censoring, paired tests, and the preregistration gate. The tests must never need real NHANES 2017–2018 files.
3. Produce a qualitative risk register with mitigation, warning signs, owner placeholders and buffer actions. Cover project, data, methods, references/novelty, schedule, compute, team, statistics, ethics and holdout contamination. Preserve the stated 6–11 hour main run, 30–50 total computer hours, overnight-run expectation and original Gantt spare time.
4. Audit literature against original papers/authoritative proceedings records. Cover LIME 2016, SHAP 2017, Adebayo 2018, Bilodeau 2024, Milli 2019, Shokri et al. 2021, Ghalme et al. 2021, Hikri & Rosenfeld 2026, König et al. 2023, and Bordt et al. 2022. Verify the five numbered bibliography items carefully. Keep ref. [2] in the exact required citation form and flag manual author/venue/pages/transparency-claim checks. Do not assert the “ten explanation queries vs. one thousand ordinary queries” claim unless the original Milli paper supports it. Compare the gap claim with Arora et al. (AAAI 2022) as potentially close prior work. If source evidence does not support the broad gap, state that and propose a narrower gap for guide approval, not a silent rewrite.
5. Create a paper/thesis/viva outline whose results sections are placeholders. Include likely viva questions, especially the measured-input model's lack of editable height/weight under the locked attacker model and how the guide resolves it.
6. Create document 13's approval register. Every B1–B6 item must carry **NEEDS GUIDE APPROVAL**. Add every additional ambiguity from the ambiguity register, show proposed defaults and alternatives, leave approval dates/decisions blank, and label critical questions.
7. Create `AGENTS.md` with fixed seeds, holdout hard gate, no fabricated results/citations, verified-source requirement, traceability updates, scope/ethics restrictions and stage-stop behavior.
8. Build only a stub code skeleton. Use type hints/docstrings/TODOs/`NotImplementedError` as appropriate. No real attack implementation results, fake datasets, fitted model results, or plots. Enforce holdout gate before final data-file access; the gate must fail closed if the real timestamped public registration record is missing or invalid. A manifest example is not registration.
9. Update doc 00's traceability table to account for all documents, all code stubs, config files and tests. Check cross-document consistency without accessing the final data.

### Stage 3 acceptance gate

- All docs 08–13 and root `AGENTS.md` exist.
- All five numbered references have a verification checklist; ref [2], Milli claim and research-gap overlap are visibly flagged until supported.
- Preregistration template includes the expected result (OAE <=2) as a hypothesis and says both large and small results are reportable.
- Ethics and scope prohibit real-world deception deployment and medical recommendations.
- All Section B items appear in doc 13 with **NEEDS GUIDE APPROVAL**.
- Code skeleton has all requested stubs: NHANES loading/merge; labels/features; both model interfaces; SHAP/LIME wrappers; Arm Q/Arm E attacker/search; OAE/statistics; plotting; configs/logging/tracking; tests and hard gate.
- No fake results, values, output figures, sample counts or verified-source claims are introduced.
- Holdout access fails closed before any final data file is opened if registration metadata is unavailable.
- Traceability matrix maps every fact/decision/artifact to a doc/module/test/task and has been updated.
- No NHANES 2017–2018 data was accessed.

After creation, **stop**. Report paths and summaries, check results, unresolved guide approvals, reference-verification statuses, traceability updates, and confirm that no NHANES 2017–2018 data was accessed. Do not proceed to full implementation or experimental execution.

## STAGE 3 PROMPT END

---

# Part VI — Final Antigravity self-check checklist

Use this checklist after Stage 3. Every item must be marked **PASS**, **FAIL**, or **NOT YET / NEEDS GUIDE APPROVAL**, with a path or evidence pointer. An unsupported “PASS” is a failure.

## Project identity and team

- [ ] Exact title: “Can an AI Explanation Be Misused? Testing Whether SHAP and LIME Explanations Help Someone Cheat a Clinical Risk Model”.
- [ ] Approved synopsis “Evaluating the Trustworthiness of Explainable AI Methods in Clinical Risk Prediction” and planned future work on adversarial robustness are captured.
- [ ] Group GP-109; DTU CSE; Akash Singh 23/CS/036, Aditya Jha 23/CS/025, Akshat Kumar 23/CS/039; junior Ishan Singh 25/CS/168, second year; guide Miss Anjali Bansal; no co-guide.
- [ ] Research question compares explanation-enabled and score-only agents on the same patients, same attempt budgets; OAE is central outcome.
- [ ] Disclosure measurement only; not a real-world cheating/deception tool; no medical recommendations; no participant data collection.

## Original/proposed schedule and gates

- [ ] P1 original/proposed 8–21 Oct 2026 and proposed closeout by 31 Oct 2026 in parallel with P2.
- [ ] P2 15 Oct–14 Nov 2026.
- [ ] P3 1–30 Nov 2026.
- [ ] P4 8 Nov–7 Dec 2026.
- [ ] P5 1–14 Dec 2026.
- [ ] P6 1–14 Jan 2027.
- [ ] P7 original 8 Jan–7 Mar 2027; proposed 15 Jan–20 Feb 2027; original conflict is explicitly documented.
- [ ] P8 original 1–31 Mar 2027; proposed 21 Feb–20 Mar 2027.
- [ ] P9 15 Mar–30 Apr 2027.
- [ ] P10 15 Apr–31 May 2027.
- [ ] P11 22 May–14 Jun 2027.
- [ ] Original and proposed Gantt charts, phase overlaps, critical path, dependencies and original spare/buffer time appear.
- [ ] The hard gate appears in documents, `AGENTS.md`, the final-run entry point and tests: no real NHANES 2017–2018 data is touched before a valid public pre-registration timestamp, permalink and plan hash exist.

## Datasets and data handling

- [ ] NHANES 2015–2016 is the trial/development cycle.
- [ ] NHANES 2017–2018 is the untouched final holdout, not opened at prompt/skeleton stage.
- [ ] Candidate WHD010, WHD020, BMXHT, BMXWT are explicitly marked UNVERIFIED until P2 checks both cycles' official codebooks.
- [ ] Merge key, files, units, missing/special codes, adult 20+ filter, pregnancy exclusion and fasting-subset logic are documented from evidence, not guessed.
- [ ] Primary diabetes label is HbA1c >= 6.5%; robustness label is HbA1c >= 6.5% OR fasting plasma glucose >= 126 mg/dL in fasting subsample only.
- [ ] Glucose, HbA1c, diagnosis and diabetes medication are excluded from model features.
- [ ] Shared non-height/weight feature set plus exactly one representation-specific height/weight/BMI group is documented/tested.
- [ ] UCI Diabetes Health Indicators (dataset 891) is comparison-only unless approved; UCI Heart Disease (dataset 45) is code-testing-only.
- [ ] No raw dataset committed to Git; file hashes/provenance and retrieval metadata documented.

## Experiment decisions

- [ ] Arm Q sees score only; Arm E sees score plus SHAP or LIME attribution; paired cases/seed structure is documented.
- [ ] Only self-reported height/weight are editable; age, sex, measured values, lab values and other features remain immutable.
- [ ] Critical measured-model actionability contradiction is explicit; measured values have not been made editable to produce an OAE.
- [ ] Empirical gap-based bounds and 50th/90th/95th/99th percentile sensitivity settings are present; main setting is 95th percentile; intended direction is lower weight/slightly higher height; exact bound algorithm is approved or marked pending.
- [ ] Coordinate hill-climbing/random restarts and attribution-guided direction rule are documented.
- [ ] Budgets exactly `{1, 2, 3, 5, 10, 20, 50, 100}`; one explanation request counts as one attempt for main analysis; “explanation is free” sensitivity included.
- [ ] Threshold achieves 90% sensitivity on a validation set and is frozen before final holdout evaluation.
- [ ] Primary risk-class flip, diabetic-case restriction, denominator and secondary mean risk-score reduction are explicit.
- [ ] OAE has an explicit match/censoring rule; no OAE is fabricated when no grid budget qualifies.
- [ ] Expected small advantage is defined as OAE <= 2; it is clearly an expectation/hypothesis, not a result.

## Statistics

- [ ] Paired design and same-patient E/Q comparison documented.
- [ ] McNemar for flip rates at each budget.
- [ ] Wilcoxon signed-rank for paired score reductions.
- [ ] 2,000 patient-level paired bootstrap resamples for OAE and risk-difference CIs.
- [ ] 20 attacker-randomness seeds.
- [ ] Effect sizes include risk difference, paired odds ratio and OAE.
- [ ] Holm family is `{SHAP, LIME} × {self-reported model, measured model}` at one fixed budget; candidate `k=10` is labelled **NEEDS GUIDE APPROVAL** until approved/frozen.
- [ ] Two-model OAE difference and bootstrap CI are attempted only if both OAE values are meaningful under the approved action model.
- [ ] Null, small, large, right-censored and unestimable outcomes have prespecified reporting approaches.

## Tech stack, runtime and artifacts

- [ ] Python 3.9+; pandas, numpy, scikit-learn logistic regression/random forest, shap, lime, scipy, statsmodels, matplotlib, seaborn, Git.
- [ ] Fixed/recorded seeds, configuration, run manifests, version/hash and reproducibility guidance present.
- [ ] CPU-only laptops, 8–16 GB RAM, under 5 GB storage; no GPU assumed.
- [ ] Data prep/training few minutes; explanations few minutes with fast TreeSHAP when suitable; main runs 6–11 hours overnight; total compute time 30–50 hours are listed as planning estimates.
- [ ] Pipeline, documented prepared-data workflow, reported-vs-measured study, timestamped plan, plots/results placeholders, two-model comparison, paper, thesis/report and viva are all traceable outputs.
- [ ] Skeleton contains data loading/merge, labels/features, both model interfaces, SHAP/LIME wrappers, attacker/search/arms, OAE/statistics, plotting, configs, logs/tracking, gate and test stubs.
- [ ] Code remains stubs/TODOs; no fake results, fake sample counts, fake plots, trained models or claims of successful experiment execution.

## Literature and citation quality

- [ ] LIME 2016, SHAP 2017, Adebayo 2018, Bilodeau 2024, Milli 2019, Shokri et al. 2021, Ghalme et al. 2021, Hikri & Rosenfeld 2026, König et al. 2023, and Bordt et al. 2022 are addressed.
- [ ] All five numbered references from the PDF are checked against originals/authoritative metadata.
- [ ] Reference [2] uses the required citation form with arXiv:2606.10137 and remains flagged for author spelling, proceedings details, and the transparency-claim audit.
- [ ] Milli et al. “ten explanation queries vs. thousand ordinary queries” statement is verified from the original or clearly flagged as unsupported; no substitute attribution is invented.
- [ ] Arora et al. (AAAI 2022) is checked for overlap with the claimed research gap; novelty statement is appropriately scoped and changes require guide approval.
- [ ] “Five references from the last five years” is reconciled against publication dates rather than silently assumed.
- [ ] Citations are traceable; no invented page ranges, venues, results or quotes.

## Required files and traceability

- [ ] All 14 `docs/00`–`docs/13` Markdown files exist separately.
- [ ] Root `AGENTS.md` exists and includes mandatory rules.
- [ ] `docs/13_Assumptions_and_Open_Questions.md` lists every B1–B6 decision, each labelled **NEEDS GUIDE APPROVAL**, plus all unresolved issues.
- [ ] `docs/00_README_and_Index.md` traceability matrix maps every source fact/decision to at least one document/code module/test/task.
- [ ] Traceability was updated after every stage.
- [ ] Each stage stopped and reported artifacts/checks before the next stage was invoked.
- [ ] No NHANES 2017–2018 data were accessed during prompt generation or code-skeleton stages.

---

## Completion rule

If any checkbox is `FAIL`, fix the artifact or report the blocker explicitly. If any checkbox is `NOT YET / NEEDS GUIDE APPROVAL`, preserve the label and list the decision owner; do not turn it into a pass by assumption. Do not proceed to the final experimental run until the guide has resolved the critical methodological questions and the pre-registration is publicly timestamped.
