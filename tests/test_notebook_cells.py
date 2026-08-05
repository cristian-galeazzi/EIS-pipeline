"""Guard tests for the logic that lives inside notebook cells.

Most of the pipeline is importable and tested through `pipeline/`, but a few
decisions are made in the notebooks themselves, such as which replica a button
pins. The defects covered here were found by running the pipeline on real
samples, not by reading the code.

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

import pytest

ROOT = Path(__file__).resolve().parents[1]
STAGE2_NOTEBOOK = ROOT / "stage2_kk.ipynb"

PANEL_CELL = 12        # stage 2: the Lin-KK tuning panel


def _cell(notebook: Path, index: int, must_contain: str) -> str:
    """Return the joined source of one cell, checked against a marker.

    The marker is what makes an index safe to hard-code: if a cell is inserted
    above the target, the assertion fails loudly instead of testing the wrong
    code.

    >>> "_on_force_replica" in _cell(STAGE2_NOTEBOOK, PANEL_CELL, "_on_force")
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
