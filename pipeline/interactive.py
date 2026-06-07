"""
pipeline/interactive.py
=======================
UI-only helpers for the EIS notebooks (ipywidgets panels).

This module contains **no calculation logic** — it only builds reusable widget
layouts so the same controls (FOCUS selector, labelled sliders) can be dropped
into NB02 / NB03 / NB04 without duplicating boilerplate. Every helper degrades
gracefully when ipywidgets is not installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence


def discover_samples(base_dir: Path | str) -> list[str]:
    """Return sorted list of sample folder names under base_dir.

    A folder qualifies if it contains a ``Raw data/`` or ``Raw oven*/``
    subdirectory. ``sample_template/`` is always excluded.
    """
    base = Path(base_dir)
    return sorted(
        p.name for p in base.iterdir()
        if p.is_dir()
        and p.name != "sample_template"
        and (any(p.glob("Raw data")) or any(p.glob("Raw oven*")))
    )


def discover_conditions(sample_dir: Path | str ) -> list[str]:
    """
    List condition folders under ``{sample_dir}/Results``, sorted.

    Parameters
    ----------
    sample_dir : sample root folder (e.g. ``EIS program/SAMPLE_ID``).
    require    : if given, only folders containing this file are returned
                 (e.g. ``"stage2_kk.xlsx"`` for NB03, ``"stage3_fit.xlsx"`` for NB04).
                 Pass ``None`` to list every sub-folder.

    Returns
    -------
    list[str] : condition folder names (empty if Results/ does not exist yet).
    """
    base = Path(sample_dir) 
    for source in ["ISM validation", "Raw data", "input_spectra"]:
        folder = base / source
        if folder.exists():
            return sorted(d.name for d in folder.iterdir() if d.is_dir())
    for folder in sorted(base.glob("Raw oven*")):
        if folder.is_dir():
            return sorted(d.name for d in folder.iterdir() if d.is_dir())
    return []
    
def discover_conditions_from_session(cfg: dict) -> list[str]:
    return sorted(cfg.get("conditions", []))



def labeled(widget: Any, html_text: str) -> Any:
    """
    Place a short grey description caption to the right of a control (L4).

    Returns an ``HBox(widget, caption)`` so the extended meaning of the
    parameter is always visible next to the slider/button.
    """
    import ipywidgets as W
    cap = W.HTML(f"<span style='color:#666;font-size:11px'>{html_text}</span>")
    return W.HBox([widget, cap])


def make_focus_panel(
    conditions: Sequence[str],
    temps:      Sequence[int],
    set_focus:  Callable[[str | None, int | None], None],
    init_cond:  str | None = None,
    init_T:     int | None = None,
) -> None:
    """
    Build a FOCUS selector: ON/OFF toggle + condition dropdown + temperature
    dropdown, both auto-populated. Lets the user restrict processing to one
    condition and/or one temperature by clicking, instead of editing the
    config cell or typing a folder name.

    ``set_focus(condition_or_None, T_or_None)`` is invoked on every change so
    the calling notebook can write its own ``FOCUS_CONDITION`` / ``FOCUS_T``
    module globals. The panel is displayed immediately.

    Parameters
    ----------
    conditions : auto-discovered condition folder names (dropdown options).
    temps      : temperature options [°C].
    set_focus  : callback receiving (condition|None, T|None).
    init_cond  : pre-selected condition (None = all).
    init_T     : pre-selected temperature (None = all).
    """
    try:
        import ipywidgets as W
        from IPython.display import display
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[INFO] FOCUS panel needs ipywidgets ({exc}).")
        # Still apply the requested initial focus so behaviour is unchanged.
        set_focus(init_cond, init_T)
        return None

    ALL_C, ALL_T = "(all conditions)", "(all T)"
    conditions = list(conditions)
    temps      = list(temps)

    on = W.ToggleButton(
        value=(init_cond is not None or init_T is not None),
        description="FOCUS", icon="filter",
        layout=W.Layout(width="120px"),
        tooltip="Restrict processing to one condition and/or one temperature")
    cdd = W.Dropdown(
        options=[ALL_C] + conditions,
        value=(init_cond if init_cond in conditions else ALL_C),
        description="Condition:", layout=W.Layout(width="470px"),
        tooltip="FOCUS_CONDITION: process only this condition folder")
    tdd = W.Dropdown(
        options=[ALL_T] + temps,
        value=(init_T if init_T in temps else ALL_T),
        description="T [°C]:", layout=W.Layout(width="200px"),
        tooltip="FOCUS_T: process only this temperature")
    lbl = W.HTML()

    def _sync(*_):
        if on.value:
            cond = None if cdd.value == ALL_C else cdd.value
            T    = None if tdd.value == ALL_T else int(tdd.value)
            on.button_style = "warning"
            set_focus(cond, T)
            lbl.value = (
                "<b style='color:#b36b00'>FOCUS active</b> &nbsp;"
                f"condition: {cond or 'all'}, T: {T if T is not None else 'all'}. "
                "Re-run the batch cell to apply.")
        else:
            on.button_style = ""
            set_focus(None, None)
            lbl.value = ("<span style='color:#555'>FOCUS off. "
                         "All conditions and temperatures will be processed.</span>")

    for w in (on, cdd, tdd):
        w.observe(_sync, names="value")
    _sync()
    display(W.VBox([W.HBox([on, cdd, tdd]), lbl]))
    return None
