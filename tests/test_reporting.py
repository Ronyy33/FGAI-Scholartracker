"""Markdown table helper."""

import pandas as pd

from scholartrack.reporting import fmt, md_table


def test_md_table_escapes_pipes_and_formats_numbers():
    table = md_table(pd.DataFrame({"a|b": ["x|y"], "n": [2.0], "f": [0.12345]}))
    lines = table.splitlines()
    assert lines[0] == "| a\\|b | n | f |"
    assert lines[2] == "| x\\|y | 2 | 0.12 |"


def test_fmt_missing_and_bool():
    assert fmt(float("nan")) == "—"
    assert fmt(True) == "yes"


def test_fmt_rounds_before_integer_check_and_keeps_text():
    assert fmt(0.28 * 100, 1) == "28"
    assert fmt(56.04, 1) == "56"  # rounds to a whole number, shown without ".0"
    assert fmt(56.06, 1) == "56.1"
    assert fmt("2.5") == "2.5"  # pre-formatted text is not re-parsed
