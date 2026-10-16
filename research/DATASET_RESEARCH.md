# Dataset Research (Phase 1)

**Date of research:** 2026-10-16
**Goal:** find a legally reusable dataset with (a) student academic records, (b) a genuine,
observed outcome that can define "falling behind", and ideally (c) enough longitudinal records
to justify an LSTM.

## How to read this document

Every factual claim carries a verification level:

| Level | Meaning |
|-------|---------|
| **V1 — fetched** | Read directly from the source page or file during this session |
| **V2 — official-page summary** | Seen in a search-engine summary of the official page; the page itself could not be opened from this environment |
| **V3 — secondary** | Reported by third-party papers, mirrors or search summaries; not yet confirmed at the official source |
| **TODO-DL** | Will be checked automatically when the file is downloaded (row counts, columns, class counts, missing values) |

**Environment constraint:** the cloud environment's network policy blocks
`archive.ics.uci.edu`, `analyse.kmi.open.ac.uk`, `openml.org`, `zenodo.org`, `kaggle.com`,
`huggingface.co`, `nature.com` and `mdpi.com`. Official pages could therefore only be seen
through web-search summaries (V2/V3). **No dataset has been downloaded.** Nothing below should be
read as verified file content until the TODO-DL checks have run.

Search queries used: public university student performance datasets; student dropout and
academic success datasets; semester-wise SGPA / internal assessment / attendance datasets
(India); OULAD documentation; UCI student-performance datasets; Kaggle student-risk datasets.

---

## Candidate 1 — UCI: Predict Students' Dropout and Academic Success (UCI id 697)

| Item | Finding | Level |
|------|---------|-------|
| Official source | <https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success> (DOI 10.24432/C5MC89) | V2 |
| Paper | Realinho, V.; Machado, J.; Baptista, L.; Martins, M.V. *Predicting Student Dropout and Academic Success.* Data 2022, 7(11), 146. <https://www.mdpi.com/2306-5729/7/11/146> | V3 |
| Archive copy | Zenodo, DOI 10.5281/zenodo.5777339 | V3 |
| Size | 4,424 records, 36 features + target | V2 (TODO-DL) |
| Unit of observation | **One student per row** (snapshot at enrolment + end of semesters 1 and 2) | V2 |
| Institution | Polytechnic Institute of Portalegre, Portugal; cohorts reported as 2008–2018 | V3 |
| Features (groups) | Enrolment/academic path (application mode, course, previous qualification and grade, admission grade); demographics (age, gender, nationality, marital status, displaced, international, special needs); socio-economic (parents' qualifications/occupations, scholarship, debtor, tuition fees up to date); macro-economic (unemployment, inflation, GDP); **curricular units for semester 1 and semester 2** (credited, enrolled, evaluations, approved, grade, without evaluations) | V2 groups; exact column list TODO-DL |
| Outcome | **Observed** 3-class status at the end of the normal course duration: Dropout / Enrolled / Graduate | V2 |
| Class balance | Graduate 2,209 (49.9 %), Dropout 1,421 (32.1 %), Enrolled 794 (17.9 %) | V3 (TODO-DL) |
| Missing values | Reported as none | V2/V3 (TODO-DL) |
| Licence | CC BY 4.0 (attribution required) | V3 — two secondary papers; official page not opened |
| Programmatic download | UCI static zip or the `ucimlrepo` package; both use `archive.ics.uci.edu` (**currently blocked here**) | V2 |

**Suitability**

- *MLP:* **Good.** Medium-sized, tabular, tertiary education, genuine outcome.
- *LSTM:* **Weak.** Only two time steps (semester 1, semester 2) of six curricular-unit
  features. An LSTM can be run, but it has little temporal structure to exploit; we would
  report it as a limited experiment rather than as evidence for sequence modelling.

**Leakage and validity concerns**

- Students who drop out during semester 2 will show near-zero semester-2 activity, so a model
  that uses semester-2 features partly "sees" the outcome. A semester-1-only model is the more
  honest early-warning setting.
- The timing of `Debtor` and `Tuition fees up to date` is not clear from summaries; if they are
  recorded late they leak financial dropout signals. To be checked in the paper.
- Demographic and socio-economic fields (gender, nationality, marital status, parents'
  occupations) raise fairness concerns; they are excluded from model inputs by default and
  considered only for a fairness audit.
- One institution, one country, a specific period: results do **not** transfer to an Indian
  engineering college without local validation.

---

## Candidate 2 — OULAD: Open University Learning Analytics Dataset

| Item | Finding | Level |
|------|---------|-------|
| Official source | <https://analyse.kmi.open.ac.uk/open-dataset> | V2 |
| Paper | Kuzilek, J., Hlosta, M. & Zdrahal, Z. *Open University Learning Analytics dataset.* Scientific Data 4, 170171 (2017). <https://www.nature.com/articles/sdata2017171> | V2 |
| Mirror README (read directly) | <https://github.com/shshen-closer/Open-University-Learning-Analytics-dataset> — confirms 22 courses, 32,593 students, 10,655,280 daily click summaries, CC-BY 4.0 | V1 (mirror text, not the data) |
| Licence | **CC BY 4.0** | V1 (mirror README) + V2 (official page and paper summaries) |
| Size | 22 module presentations, 32,593 students, 10,655,280 daily VLE-interaction rows; seven CSV tables linked by identifiers | V1/V2 |
| Unit of observation | Student × module-presentation registration; plus **dated** daily activity and assessment submissions | V2 |
| Assessments | Types TMA (tutor-marked), CMA (computer-marked), Exam; `date` = days since presentation start | V2 |
| Outcome | Final result per registration; commonly used categories Distinction / Pass / Fail / Withdrawn | V3 (TODO-DL) |
| Sensitive fields | Gender, region, age band, deprivation (IMD) band, disability | V3 (TODO-DL) |
| Programmatic download | Official zip from the OU site (**currently blocked here**); also listed on UCI | V2 |

**Suitability**

- *MLP:* Good on aggregated features (e.g. activity and scores up to week *k*).
- *LSTM:* **Strong.** Genuine weekly sequences of activity and assessment scores with an
  observed final outcome — the canonical public dataset for early-warning sequence models.
- *Schema match with the college format:* **Poor.** No attendance percentage, no SGPA, no
  CIE/ETE/termwork; distance learning rather than a campus engineering programme.
- *Cost:* the activity table is large (≈10.7 M rows); manageable on this machine but adds
  pipeline complexity for a second-year project.

---

## Candidate 3 — UCI: Student Performance (UCI id 320)

| Item | Finding | Level |
|------|---------|-------|
| Official source | <https://archive.ics.uci.edu/dataset/320/student+performance> (DOI 10.24432/C5TG7T) | V2 |
| Paper | Cortez, P. & Silva, A. *Using Data Mining to Predict Secondary School Student Performance.* FUBUTEC 2008 | V2 |
| Licence | CC BY 4.0 | V2 (third-party mirrors disagree — CC0, CC BY-SA; rely on UCI only) |
| Size | 649 students (Portuguese) and 395 students (Mathematics); 30 features + G1, G2, G3 | V2 |
| Unit | One student per row, **secondary school** (Portugal) | V2 |
| Outcome | G3 final grade (0–20); G1, G2 are earlier period grades. Pass/fail can be derived (G3 ≥ 10) | V2 |
| Concerns | Small; school-level, not university; includes alcohol consumption, family relationships and romantic status — inappropriate as risk-score inputs | V2 |

**Suitability:** MLP possible but small; LSTM **not justified** (three points of one grade
variable). Useful only as a teaching example.

---

## Candidate 4 — UCI: Student Academics Performance (UCI id 467)

| Item | Finding | Level |
|------|---------|-------|
| Official source | <https://archive.ics.uci.edu/dataset/467/student+academics+performance> | V2 |
| Authors | Hussain, S.; Dahan, N. A.; Ba-Alwi, F.; Ribata, N. (2018) | V2 |
| Licence | CC BY 4.0 | V2 |
| Size | 300 instances, 22 features | V2 |
| Context | Indian colleges (Assam) according to secondary literature; not stated in the UCI summary | V3 |
| Structure | Internal-assessment percentage and end-semester percentage — the **closest structural match** to CIE/ETE among public datasets | V3 (TODO-DL) |
| Concerns | Very small; single time point; target is a *concurrent* end-semester grade, not a future outcome; attribute list reportedly includes caste and family-income fields — sensitive | V3 (TODO-DL) |

**Suitability:** weak for MLP (300 rows), unsuitable for LSTM. Could be used to sanity-check
CIE→ETE relationships when designing the synthetic generator, not as a training source.

---

## Candidate 5 — UCI: Higher Education Students Performance Evaluation (UCI id 856)

| Item | Finding | Level |
|------|---------|-------|
| Official source | <https://archive.ics.uci.edu/dataset/856/higher+education+students+performance+evaluation> | V2 |
| Licence | CC BY 4.0 | V2 |
| Size | 145 instances, 31 features (a later paper reports 147 after preprocessing) | V2/V3 |
| Context | Faculty of Engineering and Faculty of Educational Sciences students, collected 2019 | V2 |
| Concerns | Questionnaire self-reports; tiny; single time point | V2 |

**Suitability:** not suitable for either model beyond a toy example.

---

## Candidate 6 — PredictGrad (GitHub: ShailKPatel/PredictGrad)

| Item | Finding | Level |
|------|---------|-------|
| Source | <https://github.com/ShailKPatel/PredictGrad> | V1 (repository page) |
| Size | 905 students, 56 columns, semesters 1–3; theory and practical marks 0–100; attendance for semesters 1–2 only | V1 |
| Provenance | "A local engineering college" (unnamed); "digitized academic records, de-identified" | V1 |
| Licence | Repository is MIT; **no separate data licence**, no consent or ethics statement | V1 |
| Outcome | Not observed: the risk flag is derived from *predicted* semester-3 percentile drops | V1 |
| Concerns | Gender and religion were "inferred algorithmically from names" — an ethical red flag | V1 |

**Decision: rejected as a training source.** It is the closest schema match (Indian
engineering college, multi-semester marks and attendance), but provenance, consent and data
licence are unclear, and its target is model-derived rather than observed. Not downloaded.

---

## Candidate 7 — Kaggle "student risk / dropout" datasets

Examples found: *Student Academic Performance & Dropout Risk*
(<https://www.kaggle.com/datasets/mohankrishnathalla/student-academic-performance-and-dropout-risk>),
*Student Academic Risk Monitoring Dataset*
(<https://www.kaggle.com/datasets/freshersstaff/student-academic-risk-monitoring-dataset>),
*Students Performance in Exams*.

- Kaggle is blocked from this environment, so descriptions could not be read (V3 at best).
- Several listings describe calibrated, generated or "fictitious institution" data, and
  provenance is generally undocumented.

**Decision: rejected** unless a specific dataset documents its origin, collection method and
licence. Using an undocumented synthetic dataset would be worse than our own documented
synthetic generator, because we could not explain its label.

---

## Leads noted but not yet examined

- KU Leuven de-identified clickstream dataset for two first-year bachelor courses over three
  academic years (paper titled *Open data, private learners…*). Only an abstract on an
  aggregator site was found; venue, licence and access not verified.
- Mendeley Data: *Students_Academic_Performance_Evaluation_Dataset*
  (<https://data.mendeley.com/datasets/dc3797vf3t/1>) — appeared in search results; not opened.
- A 2026 survey of open learning-analytics datasets (Švábenský et al., ACM TKDD, *Open Datasets
  in Learning Analytics: Trends, Challenges, and Best Practice*) reports 172 datasets — a good
  index for future work. URL not verified.

---

## Conclusions

1. **No public dataset matches the college format with an observed future outcome.** None
   combines semester SGPA, subject-wise CIE/ETE/termwork, attendance and an observed
   next-period result.
2. **Best real-outcome tabular dataset:** UCI Dropout & Academic Success (Candidate 1) —
   tertiary education, observed outcome, CC BY 4.0, adequate size. Suited to the MLP; weak for
   the LSTM.
3. **Best genuinely longitudinal dataset:** OULAD (Candidate 2) — observed outcome, CC BY 4.0,
   real weekly sequences. Suited to the LSTM; poor schema match.
4. **Therefore** the college-format application must be demonstrated on a documented synthetic
   dataset (Phase 4). The recommended plan adds Candidate 1 as a real-outcome benchmark
   (see `docs/PROJECT_PLAN.md` §5), with OULAD as an optional stretch goal.
5. All V2/V3 facts must be re-checked automatically when a dataset is downloaded. Until then,
   none of these datasets is treated as verified.

See `DATASET_COMPARISON.md` for the side-by-side scoring.
