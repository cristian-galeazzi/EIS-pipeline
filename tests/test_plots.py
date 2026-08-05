"""Guard tests for pipeline/plots.py, what a figure says.

These tests touch no computed number. They pin what is printed.
"""

import pandas as pd

from pipeline.plots import _condition_suptitle

# --------------------------------------------------------------------------
# _condition_suptitle
#
# pipeline/plots.py has no set_title anywhere, so the suptitle is the only text
# on a figure that identifies the measurement. Two properties follow: it must
# keep printing exactly what it prints today when nothing is asked of it, and it
# must never collapse to nothing while a label is available.
# --------------------------------------------------------------------------

# far above any physical p(O2): pure O2 at ambient is about 1 bar, so a value
# like this can only be an idle probe's output
IDLE_PROBE_BAR = 4200.0


def _df(p: float | None = 2.1e-3) -> pd.DataFrame:
    return pd.DataFrame({"pO2_mean": [] if p is None else [p, p]})


def test_default_call_is_unchanged_from_today() -> None:
    assert _condition_suptitle(_df()) == r"$p$(O$_2$) = 2.1e-03 bar"


def test_label_precedes_the_pressure() -> None:
    assert _condition_suptitle(_df(), "Ar-100 O2-20 | 400-600C") == (
        r"Ar-100 O2-20 | 400-600C,  $p$(O$_2$) = 2.1e-03 bar")


def test_probe_off_shows_the_label_alone() -> None:
    assert _condition_suptitle(_df(IDLE_PROBE_BAR), "Air | 400-600C",
                               show_pO2=False) == "Air | 400-600C"


def test_probe_off_without_a_label_is_empty_rather_than_malformed() -> None:
    # the failure this guards against is "p(O2) =  bar", a title with a hole
    assert _condition_suptitle(_df(IDLE_PROBE_BAR), "", show_pO2=False) == ""


def test_no_pO2_column_falls_back_to_the_label() -> None:
    assert _condition_suptitle(pd.DataFrame({"x": [1]}),
                               "Air | 400-600C") == "Air | 400-600C"


def test_nonpositive_pressures_are_ignored_as_before() -> None:
    df = pd.DataFrame({"pO2_mean": [0.0, -1.0]})
    assert _condition_suptitle(df, "Air") == "Air"


def test_the_median_is_used_not_the_first_row() -> None:
    df = pd.DataFrame({"pO2_mean": [0.20, 0.21, 0.22]})
    assert _condition_suptitle(df) == r"$p$(O$_2$) = 0.21 bar"
