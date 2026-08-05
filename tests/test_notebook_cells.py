"""Guard tests for the logic that lives inside notebook cells.

Most of the pipeline is importable and tested through `pipeline/`, but a few
decisions are made in the notebooks themselves: which replica a button pins,
what a skipped condition brings back into memory. Both defects covered here
were found by running the pipeline on real samples, not by reading the code.

A notebook cell cannot be imported, so each test extracts the source it needs
and either executes it against stubs or asserts on its structure. Both are
weaker than a unit test and are used only where nothing else reaches.

**These tests address cells by index.** Inserting a cell above one of the
constants below silently moves the target, so the extraction helpers assert on
a marker in the source rather than trusting the index alone.
"""

import json
import textwrap
import types
from pathlib import Path

import numpy as np
import pytest

from pipeline.drt import clip_spectrum
from pipeline.ingest import load_csv_spectrum, load_ism

ROOT = Path(__file__).resolve().parents[1]
STAGE2_NOTEBOOK = ROOT / "stage2_kk.ipynb"
STAGE3_NOTEBOOK = ROOT / "stage3_drt.ipynb"

PANEL_CELL = 12        # stage 2: the Lin-KK tuning panel
BATCH_CELL = 8         # stage 3: Step 1, batch DRT
FIT_CELL = 12          # stage 3: Step 2, batch Zarc fit


def _cell(notebook: Path, index: int, must_contain: str) -> str:
    """Return the joined source of one cell, checked against a marker.

    The marker is what makes an index safe to hard-code: if a cell is inserted
    above the target, the assertion fails loudly instead of testing the wrong
    code.

    >>> "_drt_results" in _cell(STAGE3_NOTEBOOK, BATCH_CELL, "_drt_results")
    True
    """
    src = "".join(json.loads(notebook.read_text())["cells"][index]["source"])
    assert must_contain in src, (
        f"{notebook.name} cell {index} no longer contains {must_contain!r}; "
        f"the cell indices at the top of this file need updating")
    return src


# --------------------------------------------------------------------------
# stage 2: the button that pins a replica
#
# The button used to work as a toggle: choosing the replica the scoring already
# prefers cleared the stored override instead of writing one. That made the
# automatic choice impossible to pin, which is the one thing a pin is for, since
# a wider candidate pool can hand "automatic" to a different file on a later run.
#
# The assertions are about what is written, never about what is computed: the
# selection itself is settled by the batch cell, not here.
# --------------------------------------------------------------------------


def _callback_source() -> str:
    """Return the dedented source of _on_force_replica from the panel cell.

    >>> "def _on_force_replica" in _callback_source()
    True
    """
    src = _cell(STAGE2_NOTEBOOK, PANEL_CELL, "def _on_force_replica")
    start = src.index("        def _on_force_replica(_btn):")
    end = src.index("        _w_force.on_click(_on_force_replica)")
    return textwrap.dedent(src[start:end])


def _record(name: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(path=types.SimpleNamespace(name=name))


@pytest.fixture
def panel() -> tuple:
    """A loaded callback, its override store and the messages it emitted."""
    said: list[str] = []
    overrides: dict = {}
    # best_idx and kk_best_idx deliberately differ: an override is in effect,
    # so the replica in use is not the one the scoring would pick on its own
    data = {"records": [_record("a.ism"), _record("b.ism")],
            "best_idx": 1, "kk_best_idx": 0}
    scope = {
        "OVERRIDES": overrides,
        "all_kk_data": {"COND": {600: data}},
        "wc": types.SimpleNamespace(value="COND"),
        "wT": types.SimpleNamespace(value=600),
        "_w_replica": types.SimpleNamespace(value=None),
        "_refresh_replica": lambda *a: None,
        "_ov_source_html": lambda *a: None,
        "_on_retest": lambda *a: None,
        "_save_overrides": lambda *a: said.append("SAVED"),
        "_say": said.append,
    }
    exec(compile(_callback_source(), "<panel>", "exec"), scope)
    return scope, overrides, said


def test_pinning_the_automatic_pick_writes_an_override(panel) -> None:
    scope, overrides, said = panel
    scope["_w_replica"].value = "a.ism"       # kk_best_idx = 0
    scope["_on_force_replica"](None)
    assert overrides == {"COND": {600: "a.ism"}}
    assert "SAVED" in said


def test_pinning_the_automatic_pick_says_so(panel) -> None:
    scope, _, said = panel
    scope["_w_replica"].value = "a.ism"
    scope["_on_force_replica"](None)
    assert "same as the automatic pick" in said[-1]


def test_pinning_a_different_replica_writes_it(panel) -> None:
    scope, overrides, said = panel
    scope["_w_replica"].value = "b.ism"
    scope["_on_force_replica"](None)
    assert overrides == {"COND": {600: "b.ism"}}
    assert "SAVED" in said


def test_the_note_follows_the_scoring_choice_not_the_one_in_effect(panel) -> None:
    # b.ism is best_idx, the replica currently in effect; it is not the
    # automatic pick, so it must not be reported as one
    scope, _, said = panel
    scope["_w_replica"].value = "b.ism"
    scope["_on_force_replica"](None)
    assert "same as the automatic pick" not in said[-1]


def test_an_empty_dropdown_writes_nothing(panel) -> None:
    scope, overrides, said = panel
    scope["_on_force_replica"](None)
    assert overrides == {} and said == []


def test_an_unknown_group_writes_nothing(panel) -> None:
    scope, overrides, said = panel
    scope["_w_replica"].value = "a.ism"
    scope["wT"].value = 999
    scope["_on_force_replica"](None)
    assert overrides == {} and said == []


def test_the_callback_never_deletes_a_stored_override(panel) -> None:
    # removal has its own control; a button that both writes and clears is how
    # a pin silently disappeared when the chosen file matched the scoring
    assert "remove_override_entries" not in _callback_source()


# --------------------------------------------------------------------------
# stage 3: what SKIP_EXISTING brings back
#
# `SKIP_EXISTING = True` reloads a computed condition from `stage3_drt.xlsx`.
# It used to reload the peaks alone, so the three cells that consume them (the
# explorer, the batch Zarc fit and the live tuning panel) could not run, and the
# export wrote an empty `DRT_Spectra` sheet over the one it had just read.
# --------------------------------------------------------------------------


def _loader_source() -> str:
    """Return the source of _load_clipped, dedented.

    >>> _loader_source().startswith("def _load_clipped")
    True
    """
    src = _cell(STAGE3_NOTEBOOK, BATCH_CELL, "def _load_clipped(")
    start = src.index("def _load_clipped(")
    end = src.index("# Guard: replica overrides")
    return textwrap.dedent(src[start:end])


def _spectrum_file(path: Path) -> None:
    """Write a three-decade synthetic spectrum where the loader expects one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["freq,Z_re,Z_im"]
    for f in (1e4, 1e3, 1e2, 1e1, 1e0):
        rows.append(f"{f},{1000.0 + f},{500.0 / f}")
    path.write_text("\n".join(rows) + "\n")


@pytest.fixture
def loader(tmp_path):
    """_load_clipped bound to a synthetic sample directory."""
    scope = {
        "sample_dir": tmp_path,
        "np": np,
        "load_csv_spectrum": load_csv_spectrum,
        "load_ism": load_ism,
        "clip_spectrum": clip_spectrum,
    }
    exec(compile(_loader_source(), "<batch>", "exec"), scope)
    return scope["_load_clipped"], tmp_path


def test_a_missing_file_returns_none(loader) -> None:
    load, _ = loader
    assert load("COND", "absent_400C.csv", None, None) is None


def test_the_validation_folder_is_read(loader) -> None:
    load, root = loader
    _spectrum_file(root / "ISM validation" / "COND" / "s_400C.csv")
    freq, Z_re, Z_im = load("COND", "s_400C.csv", None, None)
    assert len(freq) == 5 and len(Z_re) == 5 and len(Z_im) == 5


def test_the_csv_entry_folder_is_the_fallback(loader) -> None:
    load, root = loader
    _spectrum_file(root / "input_spectra" / "COND" / "s_400C.csv")
    assert load("COND", "s_400C.csv", None, None) is not None


def test_the_validation_folder_wins_over_the_entry_folder(loader) -> None:
    # stage 1 copies the VALID spectra into ISM validation/, so a file present
    # in both places is the same measurement; the order must stay deterministic
    load, root = loader
    _spectrum_file(root / "ISM validation" / "COND" / "s_400C.csv")
    (root / "input_spectra" / "COND").mkdir(parents=True)
    (root / "input_spectra" / "COND" / "s_400C.csv").write_text(
        "freq,Z_re,Z_im\n1,1,1\n")
    freq, _, _ = load("COND", "s_400C.csv", None, None)
    assert len(freq) == 5


def test_the_stage2_frequency_cuts_are_applied(loader) -> None:
    # the fit must see the same window the stored DRT saw, or the seeds and the
    # data disagree at the ends of the spectrum
    load, root = loader
    _spectrum_file(root / "ISM validation" / "COND" / "s_400C.csv")
    freq, _, _ = load("COND", "s_400C.csv", 5.0, 5e3)
    assert freq.max() <= 5e3 and freq.min() >= 5.0
    assert len(freq) == 3


def test_the_skip_branch_stores_the_loaded_spectrum() -> None:
    src = _cell(STAGE3_NOTEBOOK, BATCH_CELL, "_drt_results")
    assert '"freq": None, "Z_re": None, "Z_im": None' not in src
    assert '"freq": freq, "Z_re": Z_re, "Z_im": Z_im' in src


def test_the_skip_branch_carries_the_stored_gamma_curve() -> None:
    # without it the export rewrites DRT_Spectra empty: merge_sheet_by_T is a
    # full overwrite whenever FOCUS_T is None
    src = _cell(STAGE3_NOTEBOOK, BATCH_CELL, "_drt_results")
    assert 'df_sp = _sheets.get("DRT_Spectra")' in src
    assert '"spectra"' in src
    assert 'drt.get("spectra")' in _cell(STAGE3_NOTEBOOK, FIT_CELL, "_drt_results")


def test_a_stale_stage2_selection_is_refused() -> None:
    src = _cell(STAGE3_NOTEBOOK, BATCH_CELL, "_drt_results")
    assert 'elif fname and _sel.iloc[0]["file"] != fname:' in src
