"""Train / validation / test split by student.

All rows of one student go to the same partition. Otherwise a model could memorise a student
from their semester-3 row and be "tested" on their semester-4 row, which inflates results.
Each admission cohort is split in the same proportions so every partition covers all cohorts
and semesters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PARTITIONS = ("train", "validation", "test")


def split_students(entry_years: pd.Series, fractions: dict[str, float], seed: int) -> pd.Series:
    """Assign each student (index of ``entry_years``) to a partition.

    Students are sorted by key, then shuffled with a seeded generator inside each cohort, so the
    assignment is reproducible and does not depend on row order.
    """
    rng = np.random.default_rng(seed)
    assignment = pd.Series(index=entry_years.index, dtype=object)
    for _, members in entry_years.groupby(entry_years, sort=True):
        keys = np.sort(members.index.to_numpy())
        rng.shuffle(keys)
        n = len(keys)
        n_train = round(n * fractions["train"])
        n_val = round(n * fractions["validation"])
        assignment[keys[:n_train]] = "train"
        assignment[keys[n_train : n_train + n_val]] = "validation"
        assignment[keys[n_train + n_val :]] = "test"
    return assignment
