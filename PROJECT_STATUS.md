# Project Status

**Last updated:** 2026-10-16
**Current milestone:** M1 — workspace inspection, plan, dataset research, scaffold → **complete (local)**
**Next milestone:** M2 — reference CSV inspection and validation module → **blocked on the reference CSV**

## Summary

| Area | State |
|------|-------|
| Plan and architecture | Written (`docs/PROJECT_PLAN.md`); dataset strategy awaiting approval |
| Dataset research | Written; official sources could not be opened (network policy), so facts carry verification levels |
| Reference CSV | **Not found** in workspace, filesystem, GitHub remote or Google Drive |
| Data downloaded | None |
| Models trained | None |
| Backend / frontend | Not started |
| Tests | 11 passing (repository hygiene) |
| Lint / format | ruff check and ruff format: clean |
| Git | Committed locally on `claude/modest-bell-box1op`; **not pushed** (awaiting your approval) |
| CI / deployment | Not started |

## Environment facts (verified 2026-10-16)

- Linux, 4 CPUs, 15 GiB RAM, no GPU. Python 3.13.16, uv 0.11.32, Node 22.22.0, npm 10.9.4, Docker present.
- GitHub repository `Ronyy33/FGAI-Scholartracker` is **public**; default branch `main`; no branches existed before this work.
- Network policy allows PyPI, npm and raw.githubusercontent.com; **denies** `archive.ics.uci.edu`,
  `analyse.kmi.open.ac.uk`, `download.pytorch.org`, openml.org, zenodo.org, kaggle.com, huggingface.co.

## Blockers

| Blocker | Needed from you | Unblocks |
|---------|-----------------|----------|
| Reference CSV missing | Provide the file (see options in the latest chat message) — **not** via the public repo | Phase 2 onward for the college format |
| `archive.ics.uci.edu` blocked | Allow the host in the environment's network settings, or supply the UCI zip | Track R (real-outcome benchmark) |
| Dataset strategy | Choose two-track (recommended), synthetic-only, or two-track + OULAD | Phases 3–6 scope |
| Push approval | Confirm pushing to `claude/modest-bell-box1op` | Remote backup of the work |

## Decisions pending your approval

1. Dataset strategy (`docs/PROJECT_PLAN.md` §5).
2. Target definition — will be finalised in Phase 3 after the CSV is inspected.
3. Whether your college has documented academic rules (pass marks, attendance minimum, SGPA
   thresholds). Without them, thresholds stay illustrative and configurable.

## Decisions taken (ordinary implementation choices)

- Python packaging with uv + hatchling; ruff for lint/format; pytest.
- `data/` fully ignored by Git except README and placeholders; tabular formats ignored repo-wide
  except `tests/fixtures/`.
- Using the existing repository `Ronyy33/FGAI-Scholartracker` rather than creating
  `scholartrack-academic-risk`. GitHub lets you rename it later, and the old URLs keep redirecting.
