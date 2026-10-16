# Viva Preparation

Short, honest answers to the questions an examiner is likely to ask, with where to point in
the repository. Practise saying them in your own words — the goal is to understand, not to
memorise.

## 1. The 60-second summary

"ScholarTrack predicts, at the end of a semester, the probability that a student's next
semester will contain a failed subject or an SGPA below 6. It turns that probability into a
0–100 risk score with a band, shows which groups of factors drove it, and suggests supportive
actions. Because no public dataset matched our college's CSV format with future outcomes, I
generated a documented synthetic dataset from the college file's schema. I compared a
baseline, logistic regression, an MLP and an LSTM, made every choice on validation data,
committed the choice to Git, and then scored the test set once. The three learned models tied
at ROC-AUC about 0.88, so the rule picked the simplest, logistic regression. The probabilities
are calibrated. Everything is served by a FastAPI back end and a React dashboard, tested in CI
and packaged in Docker. The numbers describe synthetic data, not real students."

## 2. Problem and target

**Q: What exactly are you predicting?**
A binary label for student *s* at the end of semester *t*: 1 if semester *t + 1* has at least
one failed subject (total < 40 % or exam < 40 %) or SGPA < 6.0. `docs/TARGET_DEFINITION.md`.

**Q: Why next semester and not the current one?**
At the end of semester *t* all of *t*'s marks are known, so predicting *t* would be trivial or
leaky. Predicting *t + 1* gives mentors a whole semester to act.

**Q: What about students whose next semester is unknown?**
Their label is "unknown", not 0. They are scored but never trained on. Treating them as 0
would teach the model that the newest students are always fine.

**Q: Are the thresholds official?**
No — illustrative and configurable in `configs/target.yaml`. They need the college's rules.

## 3. Data

**Q: Why synthetic data? Isn't that cheating?**
It is a limitation, stated everywhere. The reference file has one semester and no outcome, so
it cannot train a forward-looking model. The synthetic generator reproduces the file's exact
format and plausible dynamics so that the *pipeline* can be built and tested honestly. The
results measure recovery of a known process, not real-world accuracy.

**Q: How do you know the synthetic data is not too easy?**
The generator writes marks only; labels come from the same rule as for real data. A diagnostic
gave an *oracle* the hidden generating variables: it reached only ROC-AUC 0.936, so the task
has irreducible noise. Parameters were tuned on a different seed from the official dataset.
`reports/signal_diagnostic.md`, `reports/synthetic_data_design.md`.

**Q: What problems did the reference file have?**
12 impossible values (e.g. attendance 145 %), an unknown elective, two stray columns, and
reported summaries inconsistent with the marks. Invalid values become missing — never guessed.
Credits were recovered exactly by least squares. `reports/data_quality_report.md`.

## 4. Method

**Q: Why split by student?**
Rows of one student are correlated. A row-level split would put a student's semester 4 in
training and semester 5 in test, inflating scores. Tests check that no student crosses
partitions.

**Q: How do you prevent data leakage?**
(1) Features use only semester ≤ *t* (tested by deleting later semesters). (2) Imputation and
scaling statistics come from training rows only. (3) Thresholds, calibration and model choice
use validation only. (4) The test set is guarded by a flag and a manifest and was scored once.

**Q: What are the features?**
15 per semester: current SGPA, credit-weighted marks, number failed, share below 50 %, weakest
subject, CIE / ETE / termwork averages, attendance, lowest subject attendance, share of subjects
below 75 % attendance, previous SGPA and change, study hours, semester — plus 3 missing flags.
Not used: names, PRNs, gender, age, commute, job. `docs/DATA_DICTIONARY.md`.

**Q: Explain the MLP.**
Input of 18 standardised numbers → a hidden layer of 32 neurons, each computing
ReLU(w·x + b) → dropout 0.1 → one output logit → sigmoid gives the probability. Trained with
AdamW on binary cross-entropy, early stopping on validation log loss.

**Q: Explain the LSTM and why you used it.**
It reads a student's semesters in order; at each step its gates decide what to keep from the
past (forget gate), what to add (input gate) and what to output. The hidden state after the
student's last semester goes to a linear layer. Sequences of different lengths are *packed*
so padding is ignored. It was used because the project brief asks whether history helps.

**Q: Why log loss for selection, not accuracy?**
Log loss rewards both correct ranking and honest probabilities; the risk score *is* a
probability. Accuracy is misleading with 20 % positives.

## 5. Results

**Q: Which model is best?**
None significantly. Test ROC-AUC 0.878 / 0.881 / 0.880 (LR / MLP / LSTM); paired bootstrap
differences all include 0. The pre-registered rule chose logistic regression: simplest,
best validation log loss, calibrated, easiest to explain.

**Q: So why did the LSTM not help?**
In this data the next semester depends on the student's current level, which the latest
semester already captures; history summaries added nothing in a diagnostic. A sequence model
cannot beat a snapshot model when the sequence has no extra information. On real data it
might — that is untested.

**Q: Validation AUC was 0.926 but test 0.878. Overfitting?**
No: all models dropped equally, including the barely tuned logistic regression, the 95 %
intervals overlap, and 7-fold cross-validation by student gives 0.896–0.934. Partitions of
~180 students vary that much by chance. `reports/split_variation.md`.

**Q: What does a risk score of 54 mean?**
About a 54 % estimated chance of a difficulty semester. Calibration on test: the Moderate band
(30–59) had 36 % observed difficulty, High (60–100) had 76 %.

**Q: Recall and precision at your threshold?**
Recall 69 %, precision 56 %, 25 % of students flagged (threshold 0.306 chosen on validation by
F1). Accuracy 82.8 % — but 79.6 % is free by predicting "fine" for everyone.

**Q: What is Top-K capture?**
If mentors can meet only the top 10 % (110 of 1,100), those 110 contain 89 of the 224 who later
struggled: 39.7 % captured, the maximum possible being 49.1 %; precision@K 81 %.

**Q: How did you get confidence intervals?**
Bootstrap resampling of *students* (not rows), 95 % percentile intervals.

## 6. Explanations and ethics

**Q: How are explanations computed?**
For each concept group (current results, previous semester and trend, attendance, study
time, stage), replace the student's values with a typical student's (training medians),
re-score, and report the change in points. Groups avoid misleading signs from correlated
single indicators.

**Q: Your model says low attendance lowers risk. Is the model wrong?**
It is a faithful association in this data: among students with the *same grades*, lower
attendance signals higher underlying ability (they reached those grades with less
attendance) — "explaining away". It is not causal and not advice; the app says so, and the
attendance suggestion follows the college's 75 % guideline, not the model.

**Q: How do you avoid harming students?**
Supportive suggestions only; a disclaimer on every prediction; no automatic decisions; names
never used; small groups suppressed in analytics; uploads not stored; no real data in a public
demo. Real use would need a fairness review and institutional approval.

**Q: Is there authentication?**
No, deliberately not a fake one. The public demo serves synthetic data only. A real
deployment would put the college's single sign-on in front of the app.

## 7. Engineering

**Q: How is the system structured?**
Offline scripts produce a frozen, hashed model bundle; the API loads only that bundle and
serves the built dashboard. `docs/ARCHITECTURE.md`.

**Q: How do you know it works?**
162 Python tests and 10 frontend tests; ruff, TypeScript and ESLint checks; GitHub Actions
runs them, an API smoke test, and builds the Docker image and checks its health endpoint.

**Q: How would you deploy it?**
One Docker container on Hugging Face Spaces or Render; `docs/DEPLOYMENT_GUIDE.md`.

**Q: What would you do with more time?**
Real multi-semester data with outcomes (under a data agreement); the UCI 697 benchmark; a
fairness analysis by group; threshold sign-off with the college; institutional authentication;
monitoring for data drift.

## 8. Definitions to have ready

| Term | One-line definition |
|------|---------------------|
| ROC-AUC | Probability that a random positive is ranked above a random negative |
| PR-AUC | Average precision across recall levels; baseline = positive rate (0.20) |
| Brier score | Mean squared error of the probabilities (lower is better) |
| Calibration | Predicted probabilities match observed frequencies |
| ECE | Average gap between predicted and observed rates across bins |
| Log loss | Penalty −log(p) for the probability given to what actually happened |
| Platt scaling | Logistic regression on the model's score to recalibrate probabilities |
| Early stopping | Stop training when validation loss stops improving |
| Dropout | Randomly zero neurons during training to reduce overfitting |
| Packed sequence | PyTorch format that lets an LSTM skip padding |
| Leakage | Using information at training time that would not be available at prediction time |
| Bootstrap | Resampling with replacement to estimate uncertainty |
