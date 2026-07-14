"""
pipeline/interactive.py
=======================
UI-only helpers for the EIS notebooks (ipywidgets panels).

This module contains **no calculation logic** - it only builds reusable widget
layouts so the same controls (condition selector with focus temperature,
banners) can be dropped into the stage notebooks without duplicating
boilerplate. Every helper degrades gracefully when ipywidgets is not installed.

Button style convention (all notebook panels follow it):
  ``primary``  recompute/replot in place (Retest KK, Re-fit, Replot ...)
  ``success``  export/persist to disk (Export PLOT_WINDOWS)
  ``warning``  apply overrides / affects the batch (Apply preset)
  neutral      pure selection (condition selector button)
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence


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


def select_sample(notebook_dir: Path | str, show_list: bool = False) -> str:
    """
    Prompt for the sample to work on and return its folder name.

    Accepts either the number shown in the discovered-samples list or a
    folder name typed directly. Falls back to a free-text prompt when no
    sample folders are found. One implementation for all five notebooks.
    """
    import sys

    found = discover_samples(Path(notebook_dir))
    if not found:
        return input("Sample folder name: ").strip()
    if show_list:
        print("Available samples:")
        for i, name in enumerate(found, 1):
            print(f"  {i}. {name}")
        sys.stdout.flush()
    sel = input("Sample number (or name): ").strip()
    return found[int(sel) - 1] if sel.isdigit() and 1 <= int(sel) <= len(found) else sel


def _param_source_message(from_saved: bool, stage: str) -> str:
    """Plain-text summary announcing where a config cell reads its parameters,
    plus how to switch mode (edit USE_SAVED_PARAMS and re-run the cell).

    >>> "LOCKED" in _param_source_message(True, "Stage 3")
    True
    >>> "BUILD MODE" in _param_source_message(False, "Stage 3")
    True
    """
    if from_saved:
        return (f"{stage}: LOCKED, reproducing the calibration saved in "
                f"session.json (config cell, widgets and Apply buttons do not "
                f"save). To edit: set USE_SAVED_PARAMS = False and re-run.")
    return (f"{stage}: BUILD MODE, starting from the notebook values; widget "
            f"and Apply edits are saved to session.json (merge, only what you "
            f"touch). To reproduce the saved calibration: set "
            f"USE_SAVED_PARAMS = True and re-run.")


def param_source_banner(from_saved: bool, stage: str) -> str:
    """Show a green/amber banner for the USE_SAVED_PARAMS write-protect switch
    and return the plain-text summary.

    ``from_saved=True`` renders green: reproduction mode, everything (scalars
    and per-condition overrides) loads from session.json and no widget or
    Apply button writes back. ``from_saved=False`` renders amber: build mode,
    the config cell is the base and every edit is merge-saved. Falls back to a
    plain print when IPython/HTML rendering is unavailable, so it never breaks
    a headless run.

    >>> param_source_banner(True, "Stage 3")  # doctest: +SKIP
    """
    msg = _param_source_message(from_saved, stage)
    fg, bg = ("#1a7f37", "#dafbe1") if from_saved else ("#9a6700", "#fff8c5")
    icon = "●" if from_saved else "⚠"
    html = (
        f"<div style='padding:7px 12px;border-radius:6px;margin:2px 0;"
        f"background:{bg};color:{fg};font-weight:600;"
        f"font-family:sans-serif;display:inline-block'>{icon}&nbsp; {msg}</div>"
    )
    try:
        from IPython.display import HTML, display
        display(HTML(html))
    except Exception:
        print(msg)
    return msg


def discover_conditions(sample_dir: Path | str,
                        require: str | None = None) -> list[str]:
    """
    List condition folder names for a sample, sorted.

    Sources are tried in order and the first existing one wins:
    ``ISM validation/``, ``Raw data/``, ``input_spectra/``, then any
    ``Raw oven*/`` folder. This favours the most processed source, so the
    list matches what the later stages can actually consume.

    Parameters
    ----------
    sample_dir : sample root folder (e.g. ``EIS program/SAMPLE_ID``).
    require    : if given, keep only conditions whose ``Results/{condition}/``
                 folder contains this file (e.g. ``"stage2_kk.xlsx"`` before
                 stage 3, ``"stage3_fit.xlsx"`` before stage 4).

    Returns
    -------
    list[str] : condition folder names (empty if no source folder exists).
    """
    base = Path(sample_dir)

    def _filtered(names: list[str]) -> list[str]:
        if require is None:
            return sorted(names)
        return sorted(n for n in names if (base / "Results" / n / require).exists())

    for source in ["ISM validation", "Raw data", "input_spectra"]:
        folder = base / source
        if folder.exists():
            return _filtered([d.name for d in folder.iterdir() if d.is_dir()])
    for folder in sorted(base.glob("Raw oven*")):
        if folder.is_dir():
            return _filtered([d.name for d in folder.iterdir() if d.is_dir()])
    return []


def discover_conditions_from_session(cfg: dict) -> list[str]:
    return sorted(cfg.get("conditions", []))


def make_condition_selector(
    conditions:  Sequence[str],
    title:       str = "Select gas conditions:",
    temps:       Sequence[int] | None = None,
    set_focus_t: Callable[[int | None], None] | None = None,
) -> Callable[[], list[str]]:
    """
    Checkbox-per-condition panel with a select/deselect-all button,
    displayed immediately. One implementation for stages 0, 2 and 3
    (stage 1 inherits the stage 0 selection via session.json).

    When ``temps`` and ``set_focus_t`` are given, a focus-temperature
    dropdown is added: "(all T)" processes everything, a value invokes
    ``set_focus_t(T)`` so the notebook can set its ``FOCUS_T`` global
    (merge-aware exports then touch only that temperature).

    >>> get_selected = make_condition_selector(["Ar_1", "O2_1"])  # doctest: +SKIP

    Returns
    -------
    Zero-argument callable yielding the currently selected condition names.
    Degrades to "all selected" when ipywidgets is not installed.
    """
    conditions = list(conditions)
    try:
        import ipywidgets as W
        from IPython.display import display
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[INFO] condition selector needs ipywidgets ({exc}); "
              "all conditions selected.")
        return lambda: list(conditions)

    checkboxes = [
        W.Checkbox(value=True, description=c,
                   layout=W.Layout(width="480px"),
                   style={"description_width": "0px"})
        for c in conditions
    ]
    btn = W.Button(description="Deselect all", layout=W.Layout(width="140px"))

    def _toggle(b):
        all_checked = all(cb.value for cb in checkboxes)
        for cb in checkboxes:
            cb.value = not all_checked
        b.description = "Select all" if all_checked else "Deselect all"

    btn.on_click(_toggle)

    rows: list = [W.Label(title)] + checkboxes + [btn]
    if temps is not None and set_focus_t is not None:
        ALL_T = "(all T)"
        tdd = W.Dropdown(options=[ALL_T] + list(temps), value=ALL_T,
                         description="Focus T [°C]:",
                         layout=W.Layout(width="220px"),
                         style={"description_width": "90px"},
                         tooltip="Process only this temperature; exports merge "
                                 "into the existing xlsx rows")
        t_lbl = W.HTML()

        def _on_t(change):
            T = None if change["new"] == ALL_T else int(change["new"])
            set_focus_t(T)
            t_lbl.value = ("" if T is None else
                           f"<span style='color:#b36b00;font-size:12px'>FOCUS_T = {T} °C: "
                           "re-run the batch cell to apply.</span>")

        tdd.observe(_on_t, names="value")
        rows.append(W.HBox([tdd, t_lbl]))

    display(W.VBox(rows))
    return lambda: [cb.description for cb in checkboxes if cb.value]


