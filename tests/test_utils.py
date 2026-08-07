"""Guard tests for pipeline/utils.py, the helpers the stages must agree on.

Grouped by function, each group stating the defect it guards against,
because none of these rules is obvious from the code alone.
"""

from pathlib import Path

import pandas as pd
import pytest

from pipeline.utils import (
    condition_label,
    condition_pO2_map,
    format_pO2_value,
    format_sci,
    stage2_pool_names,
)

# --------------------------------------------------------------------------
# format_sci / format_pO2_value: the power of ten
#
# "1.0e-03" is a programming language's notation, not a printed one. A figure
# carrying it reads as unfinished output, and its "e" is already the base of
# the natural logarithm. IUPAC Green Book: a value is a mantissa times a power
# of ten.
# --------------------------------------------------------------------------


def test_the_mathtext_form_is_a_power_of_ten() -> None:
    assert format_sci(2.1e-3, mathtext=True) == r"2.1\times10^{-3}"


def test_the_plain_form_uses_unicode_superscripts() -> None:
    assert format_sci(2.1e-3) == "2.1×10⁻³"


def test_a_positive_exponent_carries_no_plus_sign() -> None:
    assert format_sci(4.2e5, mathtext=True) == r"4.2\times10^{5}"
    assert format_sci(4.2e5) == "4.2×10⁵"


def test_digits_controls_the_mantissa_only() -> None:
    assert format_sci(1.234e-7, digits=2) == "1.23×10⁻⁷"


def test_a_non_finite_value_formats_to_nothing() -> None:
    assert format_sci(float("nan")) == ""
    assert format_sci(float("inf"), mathtext=True) == ""


def test_zero_has_no_power_of_ten() -> None:
    # the value is finite, so the guard above lets it through: without a case
    # of its own it would print "0.0×10⁰"
    assert format_sci(0.0) == "0"
    assert format_sci(0.0, mathtext=True) == "0"


def test_the_decimal_range_of_pO2_is_unchanged() -> None:
    # the threshold is not what this change touches: 0.21 bar stays decimal
    assert format_pO2_value(0.21) == "0.21"
    assert format_pO2_value(0.21, mathtext=True) == "0.21"
    assert format_pO2_value(0.01) == "0.01"


def test_a_low_pressure_is_a_power_of_ten_in_both_renderings() -> None:
    assert format_pO2_value(2.1e-3) == "2.1×10⁻³"
    assert format_pO2_value(2.1e-3, mathtext=True) == r"2.1\times10^{-3}"


def test_a_reading_above_the_decimal_window_is_a_power_of_ten_too() -> None:
    # .2g switches to "1e+02" on its own at 100, and that string inside the
    # suptitle's math span sets an italic e and a spaced binary plus. No
    # physical p(O2) reaches 100 bar; an idle lambda probe's reading does.
    assert format_pO2_value(99.0) == "99"
    assert format_pO2_value(99.9, mathtext=True) == r"1.0\times10^{2}"
    assert format_pO2_value(8715.0, mathtext=True) == r"8.7\times10^{3}"
    assert format_pO2_value(8715.0) == "8.7×10³"


def test_an_absent_pressure_is_still_the_empty_string() -> None:
    # the guard the suptitle relies on to avoid printing "p(O2) =  bar"
    for bad in (None, float("nan"), 0.0, -1.0):
        assert format_pO2_value(bad) == ""
        assert format_pO2_value(bad, mathtext=True) == ""


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


# --------------------------------------------------------------------------
# stage2_pool_names: which filename stage 2 resolves
#
# Stage 1 copies each VALID spectrum into ISM validation/ under its auto_label,
# which differs from the raw filename whenever that name carried no temperature.
# Stage 2 resolves paths against that directory, so it needs the right column.
#
# The second property matters more: every VALID row is returned. Which spectra
# enter the analysis is settled by the furnace log, not by the filename, so a
# spectrum cannot be dropped for having been named sequentially.
# --------------------------------------------------------------------------


def test_acquisition_labeled_names_are_returned_unchanged() -> None:
    df = pd.DataFrame({
        "file":       ["s_400C.ism", "s_400C_2.ism", "s_425C.ism"],
        "auto_label": ["s_400C.ism", "s_400C_2.ism", "s_425C.ism"],
    })
    assert list(stage2_pool_names(df)) == ["s_400C.ism", "s_400C_2.ism", "s_425C.ism"]


def test_sequential_names_resolve_through_auto_label() -> None:
    df = pd.DataFrame({
        "file":       ["s_017.ism", "s_018.ism"],
        "auto_label": ["s_400C.ism", "s_400C_2.ism"],
    })
    assert list(stage2_pool_names(df)) == ["s_400C.ism", "s_400C_2.ism"]


def test_a_sequentially_named_spectrum_is_never_dropped() -> None:
    # the defect this replaces: a filename filter kept only the first row,
    # excluding a valid spectrum because of how the operator named it
    df = pd.DataFrame({
        "file":       ["s_400C.ism", "s_017.ism"],
        "auto_label": ["s_400C.ism", "s_425C.ism"],
    })
    assert list(stage2_pool_names(df)) == ["s_400C.ism", "s_425C.ism"]


def test_missing_auto_label_falls_back_to_the_raw_name() -> None:
    df = pd.DataFrame({"file": ["s_017.ism", "s_400C.ism"],
                       "auto_label": [None, None]})
    assert list(stage2_pool_names(df)) == ["s_017.ism", "s_400C.ism"]


def test_no_auto_label_column_at_all() -> None:
    df = pd.DataFrame({"file": ["s_400C.ism"]})
    assert list(stage2_pool_names(df)) == ["s_400C.ism"]


def test_index_is_preserved_so_the_caller_can_align_rows() -> None:
    df = pd.DataFrame({"file": ["s_017.ism", "s_018.ism"],
                       "auto_label": ["s_400C.ism", "s_400C_2.ism"]},
                      index=[7, 9])
    assert list(stage2_pool_names(df).index) == [7, 9]


def test_an_empty_valid_sheet_yields_an_empty_pool() -> None:
    df = pd.DataFrame({"file": [], "auto_label": []})
    assert list(stage2_pool_names(df)) == []
