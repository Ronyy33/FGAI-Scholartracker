# Project Status

**Last updated:** 2026-10-16
**Completed:** Phases 0–13 and 15 — data, target, synthetic data, features, models, evaluation,
explanations, API, dashboard, tests, CI, Docker image, documentation.
**Remaining:** Phase 14 deployment needs *your* hosting account (steps in `docs/DEPLOYMENT_GUIDE.md`);
decisions listed under "Blockers and decisions needed".

## Summary

| Area | State |
|------|-------|
| Plan and architecture | `docs/PROJECT_PLAN.md`; two-track strategy approved |
| Dataset research | Written; official sources unreachable from this environment, facts tagged by verification level |
| Reference CSV | Received; validated; 12 impossible values, 1 unknown elective, 2 stray columns; credits recovered exactly (`reports/data_quality_report.md`) |
| Target | Defined and implemented: next-semester failure OR SGPA < 6.0 (illustrative thresholds) |
| Synthetic data (Track S) | Generated: 1,200 students, 8,561 student-semesters, 20.5 % positive label rate among 7,300 labelled rows (`reports/synthetic_data_profile.md`) |
| Real-outcome benchmark (Track R, UCI 697) | **Blocked**: `archive.ics.uci.edu` still denied by the network policy |
| Features | 15 snapshot features + 3 missing indicators; split 70/15/15 by student (5,112 / 1,088 / 1,100 labelled rows) |
| Models trained | Run `20261016T200932Z-8acaad7`: majority, logistic regression, MLP, LSTM. Validation ROC-AUC 0.926 / 0.923 / 0.923 (LR / MLP / LSTM); evaluated once on test: ROC-AUC 0.878 / 0.881 / 0.880, statistically tied; **logistic regression chosen** (`reports/evaluation_report.md`) |
| Explanations | Concept-group effects in risk points + 8 supportive rules; attendance explaining-away documented (`reports/explanations.md`, EXP-003) |
| Backend | FastAPI, frozen hash-checked bundle (`models/serving/`), single + batch prediction, demo + analytics endpoints, optional counts-only history (off) |
| Frontend | React + TypeScript dashboard, six pages, verified in Chromium (desktop + phone width), CSV download verified |
| Tests | Python: 162 passing locally (in CI 158 pass, 4 skip without the private CSV). Frontend: 10 Vitest tests, `tsc`, ESLint clean |
| Lint / format | ruff check + format: clean |
| Git | Branch `claude/modest-bell-box1op` pushed and up to date; nothing merged into `main` |
| CI | GitHub Actions: Python, frontend and Docker jobs — **first run green** (run 38969332620) |
| Deployment | **Ready, not deployed**: Docker image builds and passes its health check in CI; deploying needs your account |

## Environment facts (verified 2026-10-16)

- Linux, 4 CPUs, 15 GiB RAM, no GPU. Python 3.13.16, uv 0.11.32, Node 22.22.0. Docker CLI present
  but no daemon, so images are built and checked by GitHub Actions instead.
- Repository `Ronyy33/FGAI-Scholartracker` is **public**; your `main` holds the README placeholder
  and the reference CSV; all project work is on `claude/modest-bell-box1op`.
- Network: PyPI, npm, GitHub reachable. **Denied:** `archive.ics.uci.edu`,
  `analyse.kmi.open.ac.uk`, `download.pytorch.org` (needed for small CPU-only PyTorch wheels).

## Blockers and decisions needed

| Item | Needed from you | Unblocks |
|------|-----------------|----------|
| Network allow-list | Add `archive.ics.uci.edu` (optionally `analyse.kmi.open.ac.uk`) | Track R real-data benchmark (UCI 697) |
| Hosting account | Create a Hugging Face Spaces or Render account and follow `docs/DEPLOYMENT_GUIDE.md` | Public demo URL |
| Thresholds | Confirm SGPA < 6.0 and the 40 %/40 % pass rule, or share your college's rules | Final target sign-off |
| Reference CSV on `main` | Decide whether to delete it from `main` (it stays in history) | Privacy hygiene of the public repo |
| Merge to `main` | Ask for a pull request when you want the work on `main` | Visibility on the default branch |

## Decisions taken (ordinary implementation choices)

- Invalid values become missing, never "corrected"; rows with errors are dropped and listed.
- ETE-only subjects: CIE is *not applicable* (not zero). Stray columns ignored.
- The synthetic generator copies the reference *schema*, not its inconsistent marks spread;
  parameters were tuned on a development seed, never on the official dataset.
- Using the existing repository rather than creating `scholartrack-academic-risk` (renamable later).
