"""Synthetic multi-semester student records in the college CSV format.

How the data is made (read top to bottom)
-----------------------------------------
1. **Students.** Each synthetic student gets hidden traits: *ability*, *engagement* (correlated
   with ability), a *trajectory* (improving / stable / declining) with a per-semester *trend*,
   and a propensity to take a part-time job. Gender, age and commute time are also drawn, but
   **they have no effect on anything** — they exist only so the files match the reference
   schema (and so we can check later that no model "finds" an effect that is not there).
2. **Semesters, in time order.** Each semester, a student's *performance* and *engagement*
   states are updated: trait + trend × time + carry-over noise + occasional temporary shocks.
   From those states we draw attendance, CIE, ETE and termwork marks for every subject of that
   semester (subject codes, credits and maximum marks come from ``configs/curriculum.yaml``).
3. **Outcomes.** Pass/fail, grade points and SGPA are computed with the *same* code used for
   real files (``scholartrack.targets``). The SGPA of semester *s* becomes ``prior_sgpa`` of
   semester *s + 1*, so the history is internally consistent.
4. **Attrition.** After each semester, struggling students have a small chance of leaving.
5. **Missing values** are blanked out cell by cell, mostly in survey-like fields.

The generator writes **marks, not labels**. The "academic difficulty" label is derived later by
``scholartrack.targets`` from the next semester's marks, exactly as for any other dataset.
The hidden traits are saved separately (``latent``) for teaching and analysis only; they must
never be used as model inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from scholartrack.data.adapter import SEMESTER_COLUMNS, SUBJECT_TABLE_COLUMNS
from scholartrack.data.curriculum import Curriculum, SubjectSpec
from scholartrack.paths import CONFIG_DIR
from scholartrack.targets import TargetConfig, semester_outcomes, subject_outcomes

YEAR_LABELS = {1: "F.Y.", 2: "S.Y.", 3: "T.Y.", 4: "L.Y."}  # first / second / third / last year


def load_synthetic_config(path: Path | str | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else CONFIG_DIR / "synthetic.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@dataclass
class SyntheticDataset:
    students: pd.DataFrame  # one row per student: key, synthetic PRN/name, static fields
    semesters: pd.DataFrame  # internal representation (same columns as the adapter's)
    subjects: pd.DataFrame  # internal representation (same columns as the adapter's)
    latent: pd.DataFrame  # hidden generating variables - NEVER a model input
    config: dict[str, Any]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def _make_students(cfg: dict[str, Any], rng: np.random.Generator) -> pd.DataFrame:
    pop, lat, ne = cfg["population"], cfg["latent"], cfg["no_effect_fields"]
    per_cohort = pop["students_per_cohort"]
    cohort = np.repeat(pop["cohorts"], per_cohort)
    number = np.tile(np.arange(1, per_cohort + 1), len(pop["cohorts"]))
    n = len(cohort)

    ability = rng.standard_normal(n)
    rho = lat["engagement_ability_correlation"]
    engagement = rho * ability + np.sqrt(1 - rho**2) * rng.standard_normal(n)

    mix = lat["trajectory_mix"]
    trajectory = rng.choice(list(mix), size=n, p=list(mix.values()))
    trends = lat["trend_per_semester"]
    improving = rng.uniform(*trends["improving"], size=n)
    declining = rng.uniform(*trends["declining"], size=n)
    stable = rng.normal(0, trends["stable_sd"], size=n)
    trend = np.select(
        [trajectory == "improving", trajectory == "declining"], [improving, declining], stable
    )

    gender_p = ne["gender"]
    gender = rng.choice(list(gender_p), size=n, p=list(gender_p.values()))
    age_p = ne["entry_age"]
    entry_age = rng.choice([int(a) for a in age_p], size=n, p=list(age_p.values()))
    commute_cfg = ne["commute"]
    hostel = rng.random(n) < commute_cfg["hostel_share"]
    commute = np.clip(
        np.round(rng.gamma(2.0, commute_cfg["mean_mins"] / 2, size=n)), 1, commute_cfg["max_mins"]
    )
    commute = np.where(hostel, 0.0, commute)
    job_propensity = rng.normal(0, cfg["part_time_job"]["propensity_sd"], size=n)

    current = pop["current_academic_year_start"]
    offset = 1 if pop["current_semester_parity"] == "odd" else 2
    last_semester = np.minimum(8, 2 * (current - cohort) + offset)

    divisions = pop["divisions_per_cohort"]
    prefix = pop["id_prefix"]
    return pd.DataFrame(
        {
            "student_key": [f"S{i:06d}" for i in range(1, n + 1)],
            "prn": [f"{prefix}{c % 100:02d}CE{k:04d}" for c, k in zip(cohort, number, strict=True)],
            "student_name": [
                f"Synthetic Student {c % 100:02d}{k:04d}"
                for c, k in zip(cohort, number, strict=True)
            ],
            "cohort": cohort,
            "division": (number - 1) * divisions // per_cohort + 1,
            "last_semester": last_semester,
            "gender": gender,
            "entry_age": entry_age,
            "commute_time_mins": commute,
            "ability": ability,
            "engagement_trait": engagement,
            "trajectory": trajectory,
            "trend": trend,
            "job_propensity": job_propensity,
        }
    )


def _subject_marks(
    spec: SubjectSpec,
    score: np.ndarray,
    engagement: np.ndarray,
    performance: np.ndarray,
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Draw marks for one subject for all students. Returns cie / ete / termwork / total."""
    n = len(score)
    m = cfg["marks"]
    nan = np.full(n, np.nan)

    def pct(part: dict[str, Any], latent: np.ndarray) -> np.ndarray:
        noisy = latent + rng.normal(0, part["noise_sd"], size=n)
        return np.clip(part["mean_pct"] + part["scale_pct"] * noisy, 0, 100)

    def ete_marks(maximum: float) -> np.ndarray:
        marks = np.round(pct(m["ete"], score) / 100 * maximum)
        absent_p = _sigmoid(
            m["ete_absent"]["base_logit"] + m["ete_absent"]["performance_coef"] * performance
        )
        return np.where(rng.random(n) < absent_p, 0.0, marks)

    if spec.scheme == "theory":
        cie = np.round(pct(m["cie"], score) / 100 * spec.cie_max)
        ete = ete_marks(spec.ete_max)
        return {"cie": cie, "ete": ete, "termwork": nan, "total": cie + ete}
    if spec.scheme == "ete_only":
        ete = ete_marks(spec.max_marks)
        return {"cie": nan, "ete": ete, "termwork": nan, "total": ete}
    tw_cfg = m["termwork"]
    share = tw_cfg["engagement_share"]
    termwork = np.round(
        pct(tw_cfg, (1 - share) * score + share * engagement) / 100 * spec.max_marks
    )
    return {"cie": nan, "ete": nan, "termwork": termwork, "total": termwork}


def generate(cfg: dict[str, Any], curriculum: Curriculum, target: TargetConfig) -> SyntheticDataset:
    """Generate the full synthetic dataset. Same config + same seed => identical output."""
    rng = np.random.default_rng(cfg["seed"])
    students = _make_students(cfg, rng)
    n = len(students)
    lat, att_cfg, sh_cfg = cfg["latent"], cfg["attendance"], cfg["study_hours"]
    score_cfg, job_cfg = cfg["marks"]["subject_score"], cfg["part_time_job"]
    difficulty = {
        code: rng.normal(0, score_cfg["subject_difficulty_sd"])
        for code in sorted(curriculum.subjects)
    }

    ability = students["ability"].to_numpy()
    trait = students["engagement_trait"].to_numpy()
    trend = students["trend"].to_numpy()
    last_semester = students["last_semester"].to_numpy()
    active = np.ones(n, dtype=bool)
    perf_noise, eng_noise, carried_shock = np.zeros(n), np.zeros(n), np.zeros(n)
    prev_sgpa = np.full(n, np.nan)

    semester_frames, subject_frames, latent_frames = [], [], []
    for s in range(1, 9):
        year = (s + 1) // 2
        # Every draw is made for all students, active or not, so one student's history never
        # changes the random numbers used for another.
        perf_noise = lat["carryover"] * perf_noise + rng.normal(0, lat["performance_noise_sd"], n)
        eng_noise = lat["carryover"] * eng_noise + rng.normal(0, lat["engagement_noise_sd"], n)
        shock_cfg = lat["shock"]
        new_shock = np.where(
            rng.random(n) < shock_cfg["probability"], -rng.uniform(*shock_cfg["size"], size=n), 0.0
        )
        shock = new_shock + carried_shock
        carried_shock = np.where(rng.random(n) < shock_cfg["persistence"], 0.5 * new_shock, 0.0)
        job_p = _sigmoid(
            _logit(job_cfg["year1_rate"])
            + job_cfg["logit_increase_per_year"] * (year - 1)
            + students["job_propensity"].to_numpy()
        )
        job = rng.random(n) < job_p

        performance = ability + trend * (s - 1) + perf_noise + shock
        engagement = (
            trait
            + lat["engagement_follows_trend"] * trend * (s - 1)
            + eng_noise
            + shock_cfg["engagement_share"] * shock
            + job_cfg["engagement_effect"] * job
        )

        semester_attendance = (
            att_cfg["mean_pct"]
            + att_cfg["engagement_coef"] * engagement
            + rng.normal(0, att_cfg["student_semester_sd"], n)
        )
        study_hours = np.clip(
            sh_cfg["mean"]
            + sh_cfg["engagement_coef"] * engagement
            + sh_cfg["ability_coef"] * ability
            + rng.normal(0, sh_cfg["noise_sd"], n),
            0,
            sh_cfg["max"],
        )

        here = active & (last_semester >= s)
        keys = students["student_key"].to_numpy()[here]
        blocks = []
        for spec in curriculum.semester_subjects(s):
            attendance = np.clip(
                np.round(semester_attendance + rng.normal(0, att_cfg["subject_sd"], n), 1),
                att_cfg["min_pct"],
                100,
            )
            score = (
                score_cfg["performance_weight"] * performance
                + score_cfg["engagement_weight"] * engagement
                + score_cfg["attendance_effect_per_10pct"] * (attendance - att_cfg["mean_pct"]) / 10
                - difficulty[spec.code]
                + rng.normal(0, score_cfg["noise_sd"], n)
            )
            marks = _subject_marks(spec, score, engagement, performance, cfg, rng)
            blocks.append(
                pd.DataFrame(
                    {
                        "student_key": keys,
                        "semester": s,
                        "subject_code": spec.code,
                        "scheme": spec.scheme,
                        "credits": spec.credits,
                        "max_marks": spec.max_marks,
                        "cie_max": np.nan
                        if spec.component_max("cie") is None
                        else spec.component_max("cie"),
                        "ete_max": np.nan
                        if spec.component_max("ete") is None
                        else spec.component_max("ete"),
                        "attendance_pct": attendance[here],
                        **{k: v[here] for k, v in marks.items()},
                    }
                )
            )
        subjects = pd.concat(blocks, ignore_index=True)
        subjects["total_pct"] = subjects["total"] / subjects["max_marks"] * 100
        subjects = subjects[SUBJECT_TABLE_COLUMNS]
        subject_frames.append(subjects)

        outcomes = semester_outcomes(subject_outcomes(subjects, curriculum.grading), target)
        outcomes = outcomes.set_index("student_key").loc[keys]
        weights = subjects["credits"]
        cw = subjects.assign(
            w_att=subjects["attendance_pct"] * weights, w_pct=subjects["total_pct"] * weights
        )
        cw = cw.groupby("student_key")[["w_att", "w_pct", "credits"]].sum().loc[keys]

        options = curriculum.elective_options.get(s, ())
        elective = rng.choice(options, size=n) if options else np.full(n, None, dtype=object)
        cohort = students["cohort"].to_numpy()
        semester_frames.append(
            pd.DataFrame(
                {
                    "student_key": keys,
                    "class_division": [
                        f"{YEAR_LABELS[year]}{d}" for d in students["division"].to_numpy()[here]
                    ],
                    "academic_year": [
                        f"{c + year - 1}-{(c + year) % 100:02d}" for c in cohort[here]
                    ],
                    "semester": s,
                    "elective_subject": elective[here],
                    "age": students["entry_age"].to_numpy()[here] + year - 1,
                    "gender": students["gender"].to_numpy()[here],
                    "prior_sgpa": np.round(prev_sgpa[here], 2),
                    "study_hours_per_week": np.round(study_hours[here], 1),
                    "commute_time_mins": students["commute_time_mins"].to_numpy()[here],
                    "part_time_job": np.where(job[here], "Yes", "No"),
                    "overall_attendance_pct": np.round((cw["w_att"] / cw["credits"]).to_numpy(), 1),
                    "credit_weighted_marks_pct": np.round(
                        (cw["w_pct"] / cw["credits"]).to_numpy(), 1
                    ),
                }
            )
        )

        n_failed = np.zeros(n)
        n_failed[here] = outcomes["n_failed"].to_numpy()
        prev_sgpa = np.full(n, np.nan)
        prev_sgpa[here] = outcomes["sgpa"].to_numpy()

        attr = cfg["attrition"]
        leave_p = _sigmoid(
            attr["base_logit"]
            + attr["performance_coef"] * performance
            + attr["failures_coef"] * np.minimum(n_failed, 3)
        )
        leaves = (rng.random(n) < leave_p) & here & (s < last_semester)
        latent_frames.append(
            pd.DataFrame(
                {
                    "student_key": keys,
                    "semester": s,
                    "performance": performance[here],
                    "engagement": engagement[here],
                    "shock": shock[here],
                    "sgpa_complete_data": outcomes["sgpa"].to_numpy(),
                    "n_failed_complete_data": outcomes["n_failed"].to_numpy(),
                    "left_after_semester": leaves[here],
                }
            )
        )
        active &= ~leaves

    semesters = pd.concat(semester_frames, ignore_index=True)
    subjects = pd.concat(subject_frames, ignore_index=True)
    latent = pd.concat(latent_frames, ignore_index=True)
    semesters, subjects = _inject_missingness(semesters, subjects, latent, cfg["missingness"], rng)
    traits = ["student_key", "cohort", "ability", "engagement_trait", "trajectory", "trend"]
    latent = latent.merge(students[traits], on="student_key", how="left")

    order = ["student_key", "semester"]
    return SyntheticDataset(
        students=students,
        semesters=semesters[["student_key", *SEMESTER_COLUMNS]].sort_values(
            order, ignore_index=True
        ),
        subjects=subjects.sort_values([*order, "subject_code"], ignore_index=True),
        latent=latent.sort_values(order, ignore_index=True),
        config=cfg,
    )


SEMESTER_MISSING_FIELDS = (
    "prior_sgpa",
    "commute_time_mins",
    "part_time_job",
    "elective_subject",
    "gender",
    "overall_attendance_pct",
    "credit_weighted_marks_pct",
)
SUBJECT_MISSING_FIELDS = ("attendance_pct", "cie", "termwork", "ete", "total")


def _inject_missingness(
    semesters: pd.DataFrame,
    subjects: pd.DataFrame,
    latent: pd.DataFrame,
    rates: dict[str, float],
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blank out cells at the configured rates. Values are removed, never altered."""
    sem, subj = semesters.copy(), subjects.copy()
    # latent and semesters were built row by row in the same order.
    assert (latent["student_key"].to_numpy() == sem["student_key"].to_numpy()).all()

    for column in SEMESTER_MISSING_FIELDS:
        blank = rng.random(len(sem)) < rates[column]
        sem[column] = sem[column].where(~blank, None if sem[column].dtype == object else np.nan)
    # Missing-not-at-random: disengaged students skip the study-hours question more often.
    p = rates["study_hours_per_week"] + rates["study_hours_extra_if_disengaged"] * (
        latent["engagement"].to_numpy() < -1
    )
    sem.loc[rng.random(len(sem)) < p, "study_hours_per_week"] = np.nan

    for column in SUBJECT_MISSING_FIELDS:
        blank = rng.random(len(subj)) < rates[column]
        subj.loc[blank, column] = np.nan
    subj["total_pct"] = subj["total"] / subj["max_marks"] * 100
    return sem, subj


STUDENT_EXPORT_COLUMNS = [
    "PRN",
    "student_name",
    "class_division",
    "academic_year",
    "semester",
    "elective_subject",
    "age",
    "gender",
    "prior_sgpa",
    "study_hours_per_week",
    "commute_time_mins",
    "part_time_job",
]


def to_college_format(
    dataset: SyntheticDataset, curriculum: Curriculum
) -> dict[tuple[str, int], pd.DataFrame]:
    """Export one wide CSV table per (academic year, semester), in the reference file's layout.

    Column order per subject follows the reference: theory / ETE-only subjects
    ``attendance_pct, CIE_marks, ETE_marks, total_marks, max_marks``; termwork subjects
    ``attendance_pct, termwork_marks, max_marks, total_marks``. ETE-only subjects carry a CIE
    column of zeros, as in the reference file.
    """
    students = dataset.students.set_index("student_key")
    tables: dict[tuple[str, int], pd.DataFrame] = {}
    for (year, semester), sem in dataset.semesters.groupby(["academic_year", "semester"]):
        keys = sem["student_key"].to_numpy()
        frame = pd.DataFrame(
            {
                "PRN": students.loc[keys, "prn"].to_numpy(),
                "student_name": students.loc[keys, "student_name"].to_numpy(),
            }
        )
        for column in STUDENT_EXPORT_COLUMNS[2:]:
            frame[column] = sem[column].to_numpy()
        subj = dataset.subjects[dataset.subjects["semester"] == semester]
        subj = subj.set_index(["subject_code", "student_key"])
        for spec in curriculum.semester_subjects(int(semester)):
            rows = subj.loc[spec.code].reindex(keys)
            c = spec.code
            frame[f"{c}_attendance_pct"] = rows["attendance_pct"].to_numpy()
            if spec.scheme == "termwork":
                frame[f"{c}_termwork_marks"] = _int(rows["termwork"])
                frame[f"{c}_max_marks"] = _int(rows["max_marks"])
                frame[f"{c}_total_marks"] = _int(rows["total"])
            else:
                cie = rows["cie"] if spec.scheme == "theory" else pd.Series(0.0, index=rows.index)
                frame[f"{c}_CIE_marks"] = _int(cie)
                frame[f"{c}_ETE_marks"] = _int(rows["ete"])
                frame[f"{c}_total_marks"] = _int(rows["total"])
                frame[f"{c}_max_marks"] = _int(rows["max_marks"])
        frame["overall_attendance_pct"] = sem["overall_attendance_pct"].to_numpy()
        frame["credit_weighted_marks_pct"] = sem["credit_weighted_marks_pct"].to_numpy()
        frame["age"] = _int(frame["age"])
        tables[(str(year), int(semester))] = frame
    return tables


def _int(values: pd.Series) -> pd.api.extensions.ExtensionArray:
    """Whole-number marks with blanks preserved (prints 23, not 23.0).

    Returned as a nullable-integer array: converting it to a plain NumPy array would turn the
    column back into floats whenever it contains a blank.
    """
    return pd.array(np.round(np.asarray(values, dtype=float)), dtype="Int64")
