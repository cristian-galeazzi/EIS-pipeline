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

import html as _html
from pathlib import Path
from typing import Callable, Sequence

from pipeline.utils import format_pO2_value


def dialed(value: float, decimals: int | None = None) -> float:
    """Clean a number a person dialed into a widget, before it is saved.

    A widget step lands on binary noise (0.76 + 0.01 gives 0.7700000000000001),
    and that noise then reaches session.json and every metadata sheet exported
    afterwards. With ``decimals`` the value is rounded to what the widget shows;
    without it, every digit a person could have typed survives and only the
    noise is dropped, which is what a log slider needs.

    Numbers the program computed must never pass through here: their digits are
    a result, not typing.

    >>> dialed(0.76 + 0.01)
    0.77
    >>> dialed(2.9000000000000004, 1)
    2.9
    >>> dialed(1e-06)
    1e-06
    """
    if decimals is not None:
        return round(float(value), decimals)
    return float(f"{float(value):.12g}")


def pre_html(text: str) -> str:
    """Escaped monospace block for a widgets.HTML value (replaced on each
    update, so it never doubles).

    >>> pre_html("a < b")
    "<pre style='margin:0;font:12px/1.4 monospace;white-space:pre-wrap'>a &lt; b</pre>"
    """
    return (f"<pre style='margin:0;font:12px/1.4 monospace;white-space:pre-wrap'>"
            f"{_html.escape(text)}</pre>")


def discover_samples(base_dir: Path | str) -> list[str]:
    """Return sorted list of sample folder names under base_dir.

    A folder qualifies if it contains a ``Raw data/`` or ``Raw oven*/``
    subdirectory. ``sample_template/`` is always excluded.

    >>> discover_samples("EIS program")  # doctest: +SKIP
    ['SAMPLE_A', 'SAMPLE_B']
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
    folder name typed directly. A typed name outside the discovered list is
    accepted only if that folder exists under ``notebook_dir`` (CSV-only
    samples have no ``Raw data/`` so they never appear in the list). Empty or
    unknown input re-prompts instead of being silently accepted, so a stray
    Enter can no longer create a session entry with an empty sample_id.
    One implementation for all six notebooks.

    >>> SAMPLE_ID = select_sample(".", show_list=True)  # doctest: +SKIP
    """
    import sys

    base = Path(notebook_dir)
    found = discover_samples(base)
    if show_list and found:
        print("Available samples:")
        for i, name in enumerate(found, 1):
            print(f"  {i}. {name}")
        sys.stdout.flush()
    prompt = "Sample number (or name): " if found else "Sample folder name: "
    while True:
        sel = input(prompt).strip()
        if sel.isdigit() and 1 <= int(sel) <= len(found):
            return found[int(sel) - 1]
        if sel and (sel in found or (base / sel).is_dir()):
            return sel
        hint = f"1-{len(found)} or an existing folder name" if found else "an existing folder name"
        print(f"Invalid sample {sel!r}: enter {hint}.")


def _normalize_param_mode(mode: str | bool) -> str:
    """Map the legacy ``USE_SAVED_PARAMS`` bool onto the 3-state string.

    ``True`` -> ``"lock"``, ``False`` -> ``"continue"`` (its ``"reset"`` state
    has no bool equivalent, notebooks not yet migrated to the 3-state switch
    never had it). Strings pass through unchanged.

    >>> _normalize_param_mode(True)
    'lock'
    >>> _normalize_param_mode("reset")
    'reset'
    """
    if isinstance(mode, bool):
        return "lock" if mode else "continue"
    return mode


def _param_source_message(mode: str | bool, stage: str) -> str:
    """One-line mode banner for the config cell. The full explanation of each
    mode lives in the Configuration markdown above the cell, so the banner is
    kept to a glanceable status line.

    >>> "LOCK MODE" in _param_source_message("lock", "Stage 3")
    True
    >>> "CONTINUE MODE" in _param_source_message("continue", "Stage 3")
    True
    >>> "RESET MODE" in _param_source_message("reset", "Stage 3")
    True
    """
    label = {"lock": "LOCK MODE", "continue": "CONTINUE MODE",
             "reset": "RESET MODE"}[_normalize_param_mode(mode)]
    return f"{stage}: {label}"


def param_source_banner(mode: str | bool, stage: str) -> str:
    """Show a colored banner for the PARAM_MODE switch and return the
    plain-text summary.

    ``mode`` is one of ``"lock"`` (green: everything, scalars and
    per-condition overrides, loads from session.json and nothing writes
    back), ``"continue"`` (amber: starting values load from session.json
    when present, every widget/Apply edit is merge-saved) or ``"reset"``
    (red: session.json is ignored, starting values are the notebook's own
    literals, and the next save overwrites the saved history on purpose). A
    bool is accepted for notebooks not yet migrated to the 3-state switch
    (see ``_normalize_param_mode``); those can only reach lock/continue.
    Falls back to a plain print when IPython/HTML rendering is unavailable,
    so it never breaks a headless run.

    >>> param_source_banner("reset", "Stage 3")  # doctest: +SKIP
    """
    norm = _normalize_param_mode(mode)
    msg = _param_source_message(norm, stage)
    fg, bg, icon = {
        "lock":     ("#1a7f37", "#dafbe1", "●"),
        "continue": ("#9a6700", "#fff8c5", "⚠"),
        "reset":    ("#cf222e", "#ffebe9", "⟲"),
    }[norm]
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

    >>> discover_conditions("EIS program/SAMPLE_A",
    ...                     require="stage2_kk.xlsx")  # doctest: +SKIP
    ['Ar_100', 'O2_100']
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
    """Sorted condition names stored in a session.json sample entry.

    >>> discover_conditions_from_session({"conditions": ["O2_100", "Ar_100"]})
    ['Ar_100', 'O2_100']
    >>> discover_conditions_from_session({})
    []
    """
    return sorted(cfg.get("conditions", []))


def format_pO2(x: float | None) -> str:
    """p(O2) label with unit, e.g. "0.21 bar" or "1.0×10⁻³ bar", "" when absent.
    Delegates the number to ``format_pO2_value`` (decimal down to 0.01, a power
    of ten below), so labels and figure titles share one formatting rule. The
    exponent is Unicode here because a widget description is HTML, not mathtext.

    >>> format_pO2(0.21)
    '0.21 bar'
    >>> format_pO2(3.42e-18)
    '3.4×10⁻¹⁸ bar'
    >>> format_pO2(None)
    ''
    """
    v = format_pO2_value(x)
    return f"{v} bar" if v else ""


def order_conditions_by_pO2(
    conditions: Sequence[str],
    pO2_map: dict[str, float | None],
) -> list[str]:
    """Conditions ordered by representative p(O2) high to low; conditions with
    no p(O2) trail in alphabetical order.

    >>> order_conditions_by_pO2(["b", "a", "c"], {"a": 1e-2, "b": 1.0, "c": None})
    ['b', 'a', 'c']
    """
    with_p = [c for c in conditions if pO2_map.get(c) is not None]
    without = sorted(c for c in conditions if pO2_map.get(c) is None)
    with_p.sort(key=lambda c: pO2_map[c], reverse=True)
    return with_p + without


def make_condition_selector(
    conditions:  Sequence[str],
    title:       str = "Select gas conditions:",
    temps:       Sequence[int] | None = None,
    set_focus_t: Callable[[int | None], None] | None = None,
    pO2_map:     dict[str, float | None] | None = None,
) -> Callable[[], list[str]]:
    """
    Checkbox-per-condition panel with a select/deselect-all button,
    displayed immediately. One implementation for stages 0, 2 and 3
    (stage 1 inherits the stage 0 selection via session.json).

    When ``temps`` and ``set_focus_t`` are given, a focus-temperature
    dropdown is added: "(all T)" processes everything, a value invokes
    ``set_focus_t(T)`` so the notebook can set its ``FOCUS_T`` global.
    The selector only selects; the batch cell does the processing.

    When ``pO2_map`` is given (condition -> representative p(O2) in bar, or
    None), conditions are ordered high-to-low pressure (no-p(O2) ones trail
    alphabetically) and each checkbox shows ``p(O₂) = <value> bar - name``.
    The folder name stays the identity: the callable still returns folder names.
    Without ``pO2_map`` the behaviour is unchanged.

    >>> get_selected = make_condition_selector(["Ar_1", "O2_1"])  # doctest: +SKIP

    Returns
    -------
    Zero-argument callable yielding the currently selected condition names.
    Degrades to "all selected" when ipywidgets is not installed.
    """
    conditions = list(conditions)
    if pO2_map is not None:
        conditions = order_conditions_by_pO2(conditions, pO2_map)
    try:
        import ipywidgets as W
        from IPython.display import display
    except Exception as exc:  # pragma: no cover - depends on environment
        print(f"[INFO] condition selector needs ipywidgets ({exc}); "
              "all conditions selected.")
        return lambda: list(conditions)

    def _label(c: str) -> str:
        p = format_pO2(pO2_map.get(c)) if pO2_map is not None else ""
        return f"p(O₂) = {p} - {c}" if p else c

    checkboxes = [
        W.Checkbox(value=True, description=_label(c),
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
    return lambda: [conditions[i] for i, cb in enumerate(checkboxes) if cb.value]


