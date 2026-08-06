"""Guard tests for interactive.select_sample: empty or unknown input must
re-prompt instead of being accepted (a past run saved session params under
an empty sample_id).

The cases at the bottom guard interactive.dialed, the other way a saved value
stops matching what the panel showed."""

from pathlib import Path

import pytest

from pipeline.interactive import dialed, select_sample


def _feed(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> None:
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda _prompt: next(it))


def _make_sample(base: Path, name: str) -> None:
    (base / name / "Raw data").mkdir(parents=True)


def test_number_selects_from_list(tmp_path, monkeypatch):
    _make_sample(tmp_path, "S01")
    _make_sample(tmp_path, "S02")
    _feed(monkeypatch, ["2"])
    assert select_sample(tmp_path) == "S02"


def test_empty_input_reprompts(tmp_path, monkeypatch):
    _make_sample(tmp_path, "S01")
    _feed(monkeypatch, ["", "   ", "1"])
    assert select_sample(tmp_path) == "S01"


def test_unknown_name_reprompts(tmp_path, monkeypatch):
    _make_sample(tmp_path, "S01")
    _feed(monkeypatch, ["TYPO", "S01"])
    assert select_sample(tmp_path) == "S01"


def test_csv_only_folder_accepted(tmp_path, monkeypatch):
    # No Raw data/, so never discovered, but the folder exists on disk.
    (tmp_path / "CSV_SAMPLE" / "input_spectra").mkdir(parents=True)
    _feed(monkeypatch, ["CSV_SAMPLE"])
    assert select_sample(tmp_path) == "CSV_SAMPLE"


def test_out_of_range_number_reprompts(tmp_path, monkeypatch):
    _make_sample(tmp_path, "S01")
    _feed(monkeypatch, ["5", "0", "1"])
    assert select_sample(tmp_path) == "S01"


def test_no_samples_found_requires_existing_folder(tmp_path, monkeypatch):
    (tmp_path / "CSV_SAMPLE").mkdir()
    _feed(monkeypatch, ["", "MISSING", "CSV_SAMPLE"])
    assert select_sample(tmp_path) == "CSV_SAMPLE"


# --------------------------------------------------------------------------
# interactive.dialed: what a person types must be saved as they typed it
#
# Widget steps land on binary noise (0.76 + 0.01 = 0.7700000000000001) which
# used to reach session.json and, through it, the exported metadata sheets.
# --------------------------------------------------------------------------


def test_the_noise_a_step_leaves_behind_is_dropped() -> None:
    assert dialed(0.76 + 0.01) == 0.77
    assert repr(dialed(2.9000000000000004)) == "2.9"


def test_rounding_follows_what_the_widget_shows() -> None:
    assert dialed(2.9000000000000004, 1) == 2.9
    assert dialed(31.622776601683793, 2) == 31.62


def test_a_small_value_keeps_its_magnitude() -> None:
    # a log slider reaches 1e-06: rounding to decimals here would flatten it
    assert dialed(1e-06) == 1e-06
    assert dialed(3.1622776601683795e-05) == 3.16227766017e-05


def test_a_typed_value_survives_untouched() -> None:
    # 12 significant digits is past anything a person types into these panels
    assert dialed(31.6) == 31.6
    assert dialed(0.76) == 0.76
    assert dialed(1200.0) == 1200.0
