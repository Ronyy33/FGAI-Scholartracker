# Synthetic Data Design

**What this is:** the reasoning behind ScholarTrack's synthetic dataset — what it copies from
your reference file, how records are generated, how labels stay separate from features, and
what the data can and cannot show.

**Where the numbers are:** every parameter lives in `configs/synthetic.yaml` (the single source
of truth). Every statistic of the *generated* data is in `reports/synthetic_data_profile.md`,
which is computed by code. This document deliberately avoids restating those numbers so the
three files cannot drift apart.

> **Honesty statement.** This dataset is invented. Any model trained on it learns to recover
> *our* generating process. Its metrics say nothing about how well ScholarTrack would predict
> real students' outcomes.

---

## 1. Why synthetic data, and why build our own

- No public dataset matches the college format with an observed next-semester outcome
  (`research/DATASET_RESEARCH.md`).
- Your reference file supplies the **format** but covers one semester (no history for the
  LSTM, no future outcome for labels) and its current marks are inconsistent with its own prior
  SGPA (`reports/data_quality_report.md` §10).
- Undocumented synthetic datasets (e.g. many on Kaggle) cannot be explained in a viva. Ours is
  fully specified: every assumption is written down and every file can be regenerated exactly.

## 2. What is copied from the reference file — and what is not

| Copied | Not copied |
|--------|------------|
| Column names and order (semester 5 export is column-for-column identical, minus the two stray columns) | Any PRN, name or individual record |
| Subject codes, maximum marks and assessment schemes of semester 5 | The planted invalid values (generated data stays in range) |
| Credits recovered from the file (3,3,3,3,3,1,1,2,1,2,1) | The very wide spread of semester-5 marks |
| Value formats (1-decimal attendance, whole-number marks, 2-decimal SGPA) | The demographic mix (gender/age are generated independently and have no effect) |
| Approximate levels of prior SGPA, attendance and study hours, and the *direction and rough strength* of their relationships with marks | |

Other semesters (1–4, 6–8) use an **illustrative** curriculum with the same structure, so every
student can have a multi-semester history.

## 3. How a student's history is generated

Every student has hidden traits, drawn once:

| Trait | Meaning | Effect |
|-------|---------|--------|
| ability | general academic strength | raises marks every semester |
| engagement trait | tendency to attend, study and submit work | raises attendance, study hours, termwork; correlated with ability |
| trajectory + trend | improving, stable or declining; a per-semester drift | moves performance (and part of engagement) up or down over time |
| job propensity | how likely to work part-time | part-time work slightly lowers engagement in that semester |

Then, **semester by semester, in time order**:

1. **Performance state** = ability + trend × (semester − 1) + carry-over noise + shock.
   *Carry-over noise* (an AR(1) process) means a good or bad semester partly persists. A
   *shock* is an occasional temporary setback (illness, family event) that may half-persist
   into the next semester and then fades.
2. **Engagement state** = engagement trait + part of the trend + its own carry-over noise
   + part of any shock − a small part-time-job effect.
3. **Attendance** per subject from engagement plus student-semester and subject noise,
   bounded to [20, 100] %.
4. **Subject score** = mostly performance, partly engagement, a small direct attendance effect,
   minus a fixed subject difficulty, plus noise. CIE, ETE and termwork percentages are drawn
   from this score with their own noise (ETE noisiest: exam-day variation), then converted to
   marks using each subject's maxima. A small, performance-dependent chance of missing the
   ETE (recorded as 0) models absence.
5. **Outcomes** (pass/fail, grade points, SGPA) are computed with `scholartrack.targets` — the
   same code that labels any other dataset. This semester's SGPA becomes next semester's
   `prior_sgpa`.
6. **Attrition:** after each semester, a student may leave; the chance rises sharply after poor
   results and failures. Leavers have no later records.
7. Study hours (self-reported) depend on engagement and ability; gender, age and commute time
   are generated **independently of everything** — any model importance assigned to them later
   would be spurious, which makes them a useful built-in sanity check.

Cohorts admitted 2019–2024 are generated; cohort 2024 stops at semester 5 in 2026-27 (like the
reference file) and cohort 2023 at semester 7. Their latest semesters are exactly the
"score now, outcome unknown" situation the app will face.

## 4. Keeping the label separate from the features

- The generator writes **marks, never labels**. The label for semester *t* is derived later
  from semester *t + 1*'s marks by `targets.next_semester_labels`.
- Semester *t + 1* marks contain fresh randomness (carry-over noise, subject noise, exam noise,
  possible shocks) that is **not** visible at time *t*, so the label cannot be computed from
  the model's inputs. The profile (§4.1) shows that students with the same current SGPA can end
  up with either outcome.
- The hidden traits are saved in `latent_truth.csv` for teaching and analysis. They are
  **never** model inputs; a test fails if any model-facing table contains a latent or label
  column.

## 5. Missing values

| Mechanism | Where | Why |
|-----------|-------|-----|
| Missing completely at random | small per-cell rates on attendance, CIE, termwork, survey fields, derived columns | data-entry gaps |
| Rarer gaps in official results | ETE and totals | published results are seldom missing |
| **Missing not at random** | study hours: disengaged students skip it more often | realistic survey behaviour; teaches why "missing" can itself carry information |
| Structurally absent | `prior_sgpa` in semester 1; CIE for ETE-only subjects; electives in semesters without them | not data gaps — no value exists |

Missing cells are blanked **after** outcomes are computed: an official SGPA exists even if one
cell is missing from an export, just as in a real college system.

## 6. Edge cases deliberately included

Attendance of exactly 100 %; ETE absence (0 marks); students failing several subjects;
temporary shocks with recovery; students who leave; improving, stable and declining students;
semester-1 records with no prior SGPA; the current cohort whose future is unobserved.
`tests/test_synthetic.py` checks that all of these occur except recovery after a shock, which
follows directly from the shock model (§3) rather than being tested separately.

## 7. How the parameters were chosen

Design targets were fixed **before** looking at results: SGPA level and spread close to the
reference's prior SGPA, a next-semester difficulty rate of roughly 25–35 %, attrition below
10 %, and correlations with marks similar in direction and size to the reference's cleaned
data. Tuning used a separate **development seed (1)**; the official seed was used only after
tuning ended, so the official dataset was never used to choose parameters.

Five configurations were tried (all on the development seed):

| # | Change from the previous | SGPA mean / sd | Next-semester label rate | Left programme | Attendance–marks r (sem 5) |
|---|--------------------------|----------------|--------------------------|----------------|----------------------------|
| A | first guess | 6.14 / 1.91 | 41.7 % | 17.3 % | 0.60 |
| B | higher mean marks, smaller spreads, gentler trends, lower attrition | 6.53 / 1.70 | 32.2 % | 5.8 % | 0.59 |
| C | marks a little higher and less spread | 6.79 / 1.49 | 25.2 % | 5.2 % | 0.59 |
| D | marks higher again, less subject/exam noise, weaker attendance–engagement link | 6.94 / 1.43 | 21.9 % | 4.8 % | 0.50 |
| **E (chosen)** | marks +1 point, attendance link weaker still | **7.06 / 1.40** | **19.6 %** | **4.6 %** | **0.46** |

The chosen setting puts the SGPA mean nearest the reference (≈ 7.2) and the attendance
correlation nearest its cleaned value, at the cost of a difficulty rate (≈ 20 %) below the
initial 25–35 % guess. The reference's SGPA level is real information about the format; the
prevalence guess was not, so the reference won. A ~20 % positive rate is also a realistic
imbalanced setting in which precision, recall and PR-AUC matter more than accuracy.

## 8. Limitations

- **Invented relationships.** Effect sizes are assumptions. Real colleges may show weaker,
  stronger or different patterns (e.g. attendance rules, re-examinations, grace marks).
- **No re-examinations or backlog clearance.** A failed subject is not re-attempted later.
- **Simplified curriculum.** One illustrative syllabus for all cohorts; no subject choice beyond
  one elective name per semester.
- **No real fairness signal.** Demographics have no effect by construction, so the data cannot
  be used to study bias in real populations.
- **Informative censoring** is present by design (leavers lack a next semester); its size is
  reported in the profile.

## 9. Regenerating the data

```bash
uv run python scripts/generate_synthetic_data.py                  # official dataset + profile
uv run python scripts/generate_synthetic_data.py --seed 7 --out-dir /tmp/syn7 --no-report
```

The first command rewrites `data/synthetic/` (Git-ignored) and `reports/synthetic_data_profile.md`.
`data/synthetic/manifest.json` records the seed, configuration hashes and a SHA-256 for every
file; re-running produces byte-identical files (verified, and enforced by
`tests/test_synthetic.py`).
