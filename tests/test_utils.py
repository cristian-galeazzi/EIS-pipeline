"""Guard tests for pipeline/utils.py, the helpers the stages must agree on.

Grouped by function, each group stating the defect it guards against,
because none of these rules is obvious from the code alone.
"""

from pathlib import Path

from pipeline.utils import condition_pO2_map

# --------------------------------------------------------------------------
# condition_pO2_map: the probe-off switch
#
# A run whose lambda probe was off still has a pO2_mean column, filled with the
# idle probe's drifting output. The switch is what tells the selector and the
# figures that those numbers are not measurements.
# --------------------------------------------------------------------------


def test_disabled_map_reports_no_pressure_for_any_condition(tmp_path: Path) -> None:
    out = condition_pO2_map(tmp_path, ["cond_A", "cond_B"], enabled=False)
    assert out == {"cond_A": None, "cond_B": None}


def test_enabled_is_the_default(tmp_path: Path) -> None:
    assert condition_pO2_map(tmp_path, ["cond_A"]) == {"cond_A": None}


def test_disabled_map_never_reads_a_file(tmp_path: Path) -> None:
    # a directory where reading would raise: the switch must short-circuit
    # before any xlsx is opened
    (tmp_path / "Results" / "cond_A").mkdir(parents=True)
    (tmp_path / "Results" / "cond_A" / "stage3_fit.xlsx").write_text("not an xlsx")
    assert condition_pO2_map(tmp_path, ["cond_A"], enabled=False) == {"cond_A": None}
