"""Guard tests for interactive.select_sample: empty or unknown input must
re-prompt instead of being accepted (a past run saved session params under
an empty sample_id)."""

from pathlib import Path

import pytest

from pipeline.interactive import select_sample


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
