"""Guard tests for pipeline/utils.py, the helpers the stages must agree on.

Grouped by function, each group stating the defect it guards against,
because none of these rules is obvious from the code alone.
"""

from pathlib import Path

import pytest

from pipeline.utils import condition_label, condition_pO2_map

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


# --------------------------------------------------------------------------
# condition_label: one naming rule for every condition shown to the user
#
# The label identifies the atmosphere on every figure and in the stage-2 table,
# so the rule has to survive the shapes a condition folder actually takes:
#
#     {sample}_{bank}_{gas}_{T_hi}_{T_lo}_{step}     bank letter present
#     {sample}_{gas}_{T_hi}_{T_lo}_{step}            no bank letter
#     {sample}_{bank}_{gas}                          no ramp suffix
#
# and it has to keep working when the sample id carries a run suffix the folder
# does not have, which is how a repeated measurement of one pellet is named.
# --------------------------------------------------------------------------

# (sample_id, condition folder, expected label)
LABEL_SHAPES = [
    ("S1", "S1_B_O2_600_400_25",           "O2 | 400-600C"),
    ("S1", "S1_B_Ar_600_400_25",           "Ar | 400-600C"),
    ("S1", "S1_B_Ar-80_O2-20_600_400_25",  "Ar-80 O2-20 | 400-600C"),
    ("S1", "S1_B_Ar-100_O2-0,25_600_400_25", "Ar-100 O2-0,25 | 400-600C"),
    ("S2", "S2_O2_600_400_25",             "O2 | 400-600C"),
    ("S2", "S2_Ar-100_600_400_25",         "Ar-100 | 400-600C"),
    ("LONG_SAMPLE_ID", "LONG_SAMPLE_ID_B_N2_700_300_50", "N2 | 300-700C"),
]


@pytest.mark.parametrize("sample_id,condition,expected", LABEL_SHAPES)
def test_the_sample_prefix_and_bank_letter_are_stripped(
        sample_id: str, condition: str, expected: str) -> None:
    assert condition_label(condition, sample_id) == expected


@pytest.mark.parametrize("sample_id,condition,expected", [
    ("S1_Tvar", "S1_B_Air_600_400_25",   "Air | 400-600C"),
    ("S2_Tvar", "S2_Air_600_400_25",     "Air | 400-600C"),
    ("S1_run2", "S1_B_O2_600_400_25",    "O2 | 400-600C"),
])
def test_a_run_suffix_on_the_sample_id_still_yields_the_gas(
        sample_id: str, condition: str, expected: str) -> None:
    # the folder keeps the original prefix, the sample id gains a suffix: the
    # shared prefix is what has to be stripped, not the whole id
    assert condition_label(condition, sample_id) == expected


def test_air_is_a_gas_not_a_bank_letter() -> None:
    # the leading-token rule drops a bank or position letter such as B; Air is
    # three characters long and must survive it
    assert condition_label("S1_Air_600_400_25", "S1") == "Air | 400-600C"


def test_a_sample_id_sharing_no_prefix_leaves_the_name_alone() -> None:
    # documented limitation: an unrelated id strips nothing, so the folder's own
    # tokens survive into the label. docs/INPUT_FORMAT.md states the suffix
    # convention precisely to avoid this.
    assert condition_label("S1_B_O2_600_400_25", "Unrelated") == "B O2 | 400-600C"


def test_condition_without_a_temperature_range_is_returned_as_is() -> None:
    assert condition_label("S1_Ar-100", "S1") == "Ar-100"


def test_condition_equal_to_the_sample_id_yields_an_empty_label() -> None:
    assert condition_label("S1", "S1") == ""


def test_a_two_letter_gas_is_not_mistaken_for_a_bank_letter() -> None:
    assert condition_label("S1_H2_600_400_25", "S1") == "H2 | 400-600C"
