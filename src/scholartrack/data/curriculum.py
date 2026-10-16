"""Curriculum and grading rules loaded from ``configs/curriculum.yaml``.

A *curriculum* says, for each semester, which subjects exist, how many credits each carries,
how it is assessed (its *scheme*) and its maximum marks. The grading rules turn marks into
pass/fail decisions and grade points, which in turn give an SGPA.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from scholartrack.paths import CONFIG_DIR

SCHEMES = ("theory", "ete_only", "termwork")

# Which mark components each scheme is made of.
SCHEME_COMPONENTS: dict[str, tuple[str, ...]] = {
    "theory": ("cie", "ete"),
    "ete_only": ("ete",),
    "termwork": ("termwork",),
}


@dataclass(frozen=True)
class SubjectSpec:
    code: str
    semester: int
    credits: float
    scheme: str
    max_marks: float
    cie_max: float | None = None
    ete_max: float | None = None

    def component_max(self, component: str) -> float | None:
        """Maximum marks for one component, or None if the scheme has no such component."""
        if component not in SCHEME_COMPONENTS[self.scheme]:
            return None
        if component == "cie":
            return self.cie_max
        if component == "ete":
            return self.ete_max if self.scheme == "theory" else self.max_marks
        return self.max_marks  # termwork

    @property
    def has_ete(self) -> bool:
        return "ete" in SCHEME_COMPONENTS[self.scheme]


@dataclass(frozen=True)
class GradingRules:
    pass_total_pct: float
    pass_ete_pct: float
    grade_points: tuple[tuple[float, float], ...]  # (min %, points), descending

    def points_for(self, total_pct: np.ndarray, passed: np.ndarray) -> np.ndarray:
        """Grade points for arrays of total percentages; failed subjects get 0."""
        total_pct = np.asarray(total_pct, dtype=float)
        points = np.zeros_like(total_pct)
        assigned = np.zeros(total_pct.shape, dtype=bool)
        for minimum, value in self.grade_points:
            hit = (total_pct >= minimum) & ~assigned
            points[hit] = value
            assigned |= hit
        points[~np.asarray(passed, dtype=bool)] = 0.0
        points[np.isnan(total_pct)] = np.nan
        return points


@dataclass(frozen=True)
class Curriculum:
    subjects: dict[str, SubjectSpec]
    elective_options: dict[int, tuple[str, ...]]
    grading: GradingRules
    sources: dict[int, str]

    def semester_subjects(self, semester: int) -> list[SubjectSpec]:
        return [s for s in self.subjects.values() if s.semester == semester]

    def get(self, code: str) -> SubjectSpec | None:
        return self.subjects.get(code)

    @property
    def semesters(self) -> list[int]:
        return sorted({s.semester for s in self.subjects.values()})


def load_curriculum(path: Path | str | None = None) -> Curriculum:
    path = Path(path) if path is not None else CONFIG_DIR / "curriculum.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    grading_raw = raw["grading"]
    points = tuple(
        sorted((tuple(map(float, row)) for row in grading_raw["grade_points"]), reverse=True)
    )
    grading = GradingRules(
        pass_total_pct=float(grading_raw["pass_total_pct"]),
        pass_ete_pct=float(grading_raw["pass_ete_pct"]),
        grade_points=points,  # type: ignore[arg-type]
    )

    subjects: dict[str, SubjectSpec] = {}
    electives: dict[int, tuple[str, ...]] = {}
    sources: dict[int, str] = {}
    for sem_key, sem in raw["semesters"].items():
        semester = int(sem_key)
        sources[semester] = str(sem.get("source", "unspecified"))
        electives[semester] = tuple(sem.get("elective_options", ()))
        for code, spec in sem["subjects"].items():
            if spec["scheme"] not in SCHEMES:
                raise ValueError(f"{code}: unknown scheme {spec['scheme']!r}")
            if code in subjects:
                raise ValueError(f"{code} appears in more than one semester")
            subjects[code] = SubjectSpec(
                code=code,
                semester=semester,
                credits=float(spec["credits"]),
                scheme=spec["scheme"],
                max_marks=float(spec["max_marks"]),
                cie_max=float(spec["cie_max"]) if "cie_max" in spec else None,
                ete_max=float(spec["ete_max"]) if "ete_max" in spec else None,
            )
    return Curriculum(
        subjects=subjects, elective_options=electives, grading=grading, sources=sources
    )
