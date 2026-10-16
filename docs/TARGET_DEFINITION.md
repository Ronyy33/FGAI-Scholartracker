# Target Definition — What "Falling Behind" Means in ScholarTrack

**Status:** proposed definition, implemented in `src/scholartrack/targets.py` and
`configs/target.yaml`. The *structure* follows the plan you approved; the *thresholds* are
illustrative and await your confirmation or your college's official rules (§9).

---

## 1. The question the model answers

> **At the end of semester *t*, once results are declared, what is the probability that this
> student will have an *academic-difficulty semester* in semester *t + 1*?**

The answer becomes the Academic Risk Score: `round(100 × calibrated probability)`.

## 2. Definition

| Element | Definition |
|---------|------------|
| Prediction unit | One student at the end of one semester *t* (one row of the `semesters` table) |
| Prediction time | After semester-*t* results are published, before semester *t + 1* begins |
| Horizon | One semester (`horizon_semesters: 1`) |
| Positive class (label = 1) | Semester *t + 1* is an **academic-difficulty semester**: the student **fails at least one subject** OR obtains an **SGPA below 6.0** |
| Negative class (label = 0) | Semester *t + 1* has no failed subject **and** SGPA ≥ 6.0, with every subject outcome known |
| Undetermined (label = NaN) | See §6 — never counted as "not at risk" |

**Subject pass rule** (`configs/curriculum.yaml → grading`): total ≥ 40 % of the subject's
maximum **and**, for subjects with an end-term exam, ETE ≥ 40 % of the ETE maximum.

**SGPA** = Σ(credits × grade points) / Σ credits, using the 10-point letter-grade scale
(O = 10, A+ = 9, A = 8, B+ = 7, B = 6, C = 5, P = 4, F = 0). That letter scale is widely
attributed to the UGC's Choice Based Credit System guidelines
([Vikaspedia, *Choice Based Credit System*](https://education.vikaspedia.in/viewcontent/education/policies-and-schemes/choice-based-credit-system-cbcs?lgn=en));
the **percentage cut-offs differ between universities**, and one hosted copy of the guidelines
shows a different scale
([IARE copy](https://www.iare.ac.in/sites/default/files/UGC_GUIDELINES_ON_ADOPTION_OF_CHOICE_BASED_CREDIT_SYSTEM.pdf)).
Our cut-offs (90/80/70/60/55/50/40) are therefore **illustrative**.

## 3. Why this definition

1. **It is concrete and observable.** A failed subject (a "backlog") and a low SGPA are events
   colleges already record. No interpretation of motivation or effort is involved.
2. **It looks forward.** Predicting semester *t + 1* from semester *t* gives mentors a full
   semester to act, and keeps a clean line between what is known (the past) and what is
   predicted (the future).
3. **It suits both models.** The MLP uses a fixed set of features about the latest semester and
   short history; the LSTM reads the student's whole semester-by-semester sequence up to *t*.
4. **It works with your file format.** A college-format CSV of semester *t* (like the reference
   file) contains everything needed to *score* students for semester *t + 1*.

Combining the two conditions matters: a student can pass every subject narrowly (SGPA 5.2, no
failure) or have one failure with a decent SGPA; both are signs of falling behind.

### Alternatives considered

| Alternative | Why not the primary target |
|-------------|----------------------------|
| Dropout / leaving the programme | Not recorded in the college format; rare and often non-academic |
| Predict SGPA itself (regression) | Useful as a secondary output later, but colleges act on thresholds; a classifier with calibrated probability maps directly to a risk score |
| In-semester early warning (predict semester *t*'s own outcome from attendance + CIE mid-semester) | Attractive and supported by the reference format, but attendance in the file is end-of-semester, so using it mid-semester would leak later information. Kept as a possible extension |
| "Failed ≥ 1 subject" alone | On the reference file's marks 66 % of students would be positive; combining with SGPA and a coherent generator gives a more informative target (§8) |

## 4. Features allowed at prediction time

Only information recorded **up to the end of semester *t***:

| Allowed | Excluded |
|---------|----------|
| SGPA history (semesters ≤ *t*), trends | Anything from semester *t + 1* (that *is* the label) |
| Semester-*t* attendance, CIE, ETE, termwork, totals, failed-subject counts | `PRN`, `student_name` (identifiers) |
| Credit-weighted marks of semester *t* | `gender`, `age` (sensitive; excluded by default, fairness review) |
| Self-reported study hours / part-time job — **subject to review in Phase 5** | Any pre-computed "risk" column (none exists in the reference file) |

The feature pipeline (Phase 5) will enforce this with an explicit allow-list and a test that
fails if any semester-*t + 1* value reaches the model.

## 5. How labels are generated (provenance)

- Labels are **computed by rule** from the marks of semester *t + 1* by
  `targets.semester_outcomes` → `targets.next_semester_labels`. There is no hidden label column.
- **Synthetic data (Phase 4):** the generator produces *marks*; it never writes a label. The
  label is derived afterwards with the same code as for any other dataset. Because the marks
  of semester *t + 1* depend on random variation that is not visible at time *t*, the label is
  **not** a deterministic function of the model's inputs.
- **Reference CSV:** it holds one semester only, so it has no *t + 1* and **no labels**. It is
  used for validation, schema design and as an example input to score.
- **Important:** every metric computed on synthetic labels measures how well a model recovers
  *our documented generating process*, not real-world predictive ability.

## 6. Incomplete records

| Situation | `label_status` | Used for training/evaluation? |
|-----------|----------------|-------------------------------|
| Semester *t + 1* not observed (latest semester; student left) | `future_semester_not_observed` | No — but can be **scored** (this is exactly the deployment case) |
| Semester *t + 1* observed but outcome undeterminable (e.g. an ETE mark missing and no failure known) | `future_outcome_undetermined` | No |
| A known failure in *t + 1* even if other marks are missing | `labelled` (label = 1) | Yes |
| Missing values among semester-*t* features | `labelled` | Yes — handled by imputation fitted on training data only (Phase 5) |

**Known bias:** students who leave the programme are more likely to be struggling, and they have
no *t + 1* record. Excluding them makes the training population look slightly healthier than
reality (*informative censoring*). In synthetic data we can measure this effect; it will be
reported in Phase 4/7.

## 7. What the model cannot establish

- **Causes.** A pattern such as "low attendance → higher risk" is an association, not a cause.
- **Motivation, ability, effort or personal circumstances.** None of these is measured.
- **Real-world accuracy.** With synthetic labels, no claim can be made about real students.
- **Fairness on real populations.** That requires real outcome data and a dedicated audit.

## 8. Expected prevalence

On the reference file's own semester-5 marks, this rule marks **66 %** of students as having a
difficulty semester (`reports/data_quality_report.md` §10) — mainly because those marks are far
more spread out than the file's `prior_sgpa` implies. The Phase 4 generator is designed around a
coherent multi-semester process instead; the resulting positive rate will be measured and
reported, not assumed.

## 9. What needs your input

| Setting | Current (illustrative) value | Where |
|---------|------------------------------|-------|
| SGPA threshold | 6.0 | `configs/target.yaml` |
| Pass rule | total ≥ 40 % and ETE ≥ 40 % | `configs/curriculum.yaml` |
| Grade cut-offs | 90/80/70/60/55/50/40 → 10…4 | `configs/curriculum.yaml` |

If your college publishes an academic ordinance with these rules, sharing it lets us replace the
illustrative values; only the YAML files change, no code.

## 10. Leakage checklist (enforced from Phase 5 onward)

- [ ] Feature builder takes only semesters ≤ *t* for a row at time *t* (unit test).
- [ ] No column derived from semester *t + 1* marks in the feature matrix (unit test).
- [ ] Students are split by `student_key`: no student appears in two partitions (unit test).
- [ ] Imputers/scalers fitted on the training partition only (unit test).
- [ ] Decision threshold and calibration chosen on validation, never on test.
