"""Shared helpers used by the notebook export cells.

Two responsibilities:
- merge_sheet_by_T : safe per-temperature update of a sheet inside an Excel file
- build_metadata_sheet : standardised 2-column DataFrame recording fixed parameters
- condition_label : one naming rule for every condition shown to the user

These helpers exist so that NB02 and NB03 export logic stays identical and the
Metadata schema is uniform across all stages (NB01, NB02, NB03).
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import warnings
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


# Libraries whose version can affect numeric results - recorded in every
# Metadata sheet so a run is reproducible against the exact environment that
# produced it (Wilson et al., "Good enough practices in scientific computing").
_TRACKED_LIBS = (
    "numpy", "scipy", "pandas", "matplotlib",
    "impedance", "pyDRTtools", "zahner_analysis", "openpyxl",
)


def _library_versions() -> list[tuple[str, str]]:
    """Return (``version_<pkg>``, version) rows for the tracked libraries."""
    import importlib.metadata as _md
    rows: list[tuple[str, str]] = []
    for pkg in _TRACKED_LIBS:
        try:
            rows.append((f"version_{pkg}", _md.version(pkg)))
        except Exception:
            rows.append((f"version_{pkg}", "not installed"))
    return rows


def check_replica_overrides(
    sample_dir: Path | str,
    overrides:  Mapping[str, Mapping],
    conditions: list[str],
) -> list[str]:
    """
    Cross-check session.json replica overrides against the stage2 "Selected"
    sheets actually on disk.

    A mismatch means the stage2 export predates (or never saw) the override:
    downstream stages would silently analyse a different replica than the one
    forced in session.json. Returns one human-readable message per mismatch;
    an empty list means the state is consistent.

    >>> check_replica_overrides("missing_dir", {"Ar_100": {"600": "a.ism"}},
    ...                         ["Ar_100"])
    []
    """
    msgs: list[str] = []
    for cond in conditions:
        ov = overrides.get(cond) or {}
        if not ov:
            continue
        xlsx = Path(sample_dir) / "Results" / cond / "stage2_kk.xlsx"
        if not xlsx.exists():
            continue
        try:
            sel = pd.read_excel(xlsx, sheet_name="Selected")
        except Exception as exc:
            msgs.append(f"{cond}: cannot read Selected sheet "
                        f"({type(exc).__name__}: {exc})")
            continue
        by_T = {int(r["T_nominal"]): str(r["file"]) for _, r in sel.iterrows()}
        for T, forced in ov.items():
            try:
                t_int = int(T)
            except (TypeError, ValueError):
                continue
            actual = by_T.get(t_int)
            if actual is not None and actual != forced:
                msgs.append(
                    f"{cond} T={t_int}: override forces '{forced}' but the "
                    f"stage2 export on disk selected '{actual}'. Re-run the "
                    f"stage 2 export for this condition, or delete the override."
                )
    return msgs


# Where the per-spectrum pO2_mean column can be read, most processed first.
# Same column everywhere (born at matching, stage 1), so any source gives the
# same value; the list just makes the helper work from stage 2 onward.
def format_pO2_value(x: float | None) -> str:
    """Number-only p(O2) string [bar implied]: two significant figures, decimal
    down to 0.01 and scientific below 1e-2 (so 0.21, 0.01, then 1.0e-03). "" when
    the value is absent, NaN or nonpositive. Single source of the p(O2) number so
    the selector labels and the figure titles never disagree.

    >>> format_pO2_value(0.21)
    '0.21'
    >>> format_pO2_value(1e-3)
    '1.0e-03'
    >>> format_pO2_value(None)
    ''
    """
    if x is None or x != x or x <= 0:  # None, NaN, or nonpositive
        return ""
    return f"{x:.2g}" if x >= 1e-2 else f"{x:.1e}"


_PO2_SOURCES: tuple[tuple[str, str], ...] = (
    ("stage3_fit.xlsx", "Peaks"),
    ("stage2_kk.xlsx", "Selected"),
    ("stage1_labeling.xlsx", "VALID"),
)


def condition_pO2_map(
    sample_dir: Path | str,
    conditions: list[str],
    *,
    enabled: bool = True,
) -> dict[str, float | None]:
    """Representative oxygen partial pressure [bar] per condition.

    The value is ``median(pO2_mean)`` over the condition's spectra, read from
    the most processed stage xlsx available (the same ``pO2_mean`` column the
    Brouwer figures use). Median because p(O2) is constant per condition to a
    few percent, so the median is robust to a stray furnace-startup reading and
    needs no acquisition timestamp. Returns ``None`` for a condition with no
    p(O2) (the non-Zahner CSV path), so callers can fall back to the folder
    name unchanged.

    ``enabled=False`` reports no pressure at all: the run's lambda probe was
    off, so the stored column holds idle-probe readings rather than data.

    >>> condition_pO2_map("does_not_exist", ["Ar"])
    {'Ar': None}
    >>> condition_pO2_map("any_dir", ["Ar"], enabled=False)
    {'Ar': None}
    """
    if not enabled:
        return dict.fromkeys(conditions)
    base = Path(sample_dir)
    out: dict[str, float | None] = {}
    for cond in conditions:
        value: float | None = None
        for fname, sheet in _PO2_SOURCES:
            xlsx = base / "Results" / cond / fname
            if not xlsx.exists():
                continue
            try:
                df = pd.read_excel(xlsx, sheet_name=sheet)
            except Exception:
                continue
            if "pO2_mean" not in df.columns:
                continue
            s = pd.to_numeric(df["pO2_mean"], errors="coerce").dropna()
            s = s[s > 0]
            if not s.empty:
                value = float(s.median())
                break
        out[cond] = value
    return out


def merge_sheet_by_T(
    xlsx_path: Path,
    sheet_name: str,
    new_df: pd.DataFrame,
    focus_t: int | float | None,
) -> pd.DataFrame:
    """Return the DataFrame that should be written to ``sheet_name``.

    Behaviour:
    - ``focus_t is None`` → return ``new_df`` unchanged (full overwrite).
    - ``focus_t`` is set and the file does not exist → return ``new_df``.
    - ``focus_t`` is set and the file exists → read the existing sheet,
      drop rows with ``T_nominal == focus_t``, append ``new_df``, sort by
      ``T_nominal`` descending, return the result.

    Failures while reading the existing sheet are swallowed (treated as
    "no existing data") so that adding the helper to a notebook that has
    a partial file never causes a hard crash.

    >>> df = pd.DataFrame({"T_nominal": [600], "R": [1.0]})
    >>> merge_sheet_by_T(Path("missing.xlsx"), "Fit", df, focus_t=None) is df
    True
    >>> merge_sheet_by_T(Path("missing.xlsx"), "Fit", df, focus_t=600) is df
    True
    """
    if focus_t is None or not Path(xlsx_path).exists():
        return new_df

    try:
        existing = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception as exc:
        warnings.warn(
            f"merge_sheet_by_T: could not read existing sheet '{sheet_name}' in "
            f"{xlsx_path} ({exc}); falling back to overwrite - rows for other "
            f"temperatures may be lost. Verify the file before trusting the merge.",
            stacklevel=2,
        )
        return new_df

    if "T_nominal" not in existing.columns:
        return new_df

    existing = existing[existing["T_nominal"] != focus_t]
    merged = pd.concat([existing, new_df], ignore_index=True)
    return merged.sort_values("T_nominal", ascending=False).reset_index(drop=True)


def build_metadata_sheet(
    sample_id: str,
    stage_name: str,
    params: Mapping[str, Any],
) -> pd.DataFrame:
    """Return a 2-column DataFrame (parameter, value) describing fixed config.

    The schema is identical across stages so any methods section can be
    written by reading any ``Metadata`` sheet in the pipeline outputs.

    A ``processed_at`` timestamp and the stage name are prepended automatically.

    >>> df = build_metadata_sheet("S1", "stage2_kk", {"KK_C": 0.76})
    >>> list(df.columns)
    ['parameter', 'value']
    >>> df.iloc[3].tolist()
    ['KK_C', 0.76]
    """
    rows: list[tuple[str, Any]] = [
        ("sample_id", sample_id),
        ("stage", stage_name),
        ("processed_at", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    rows.extend((str(k), v) for k, v in params.items())
    rows.extend(_library_versions())
    return pd.DataFrame(rows, columns=["parameter", "value"])


_GAS_RE = re.compile(r"^(Ar|O2|N2|H2|Air)(?![A-Za-z0-9])", re.IGNORECASE)


def condition_label(condition: str, sample_id: str) -> str:
    """Short human label for a condition folder: the gas and the ramp range.

    A folder name repeats the sample prefix, a bank letter and the ramp
    endpoints (``S1_B_Ar-80_O2-20_600_400_25``); only the gas and the range are
    worth showing. The prefix is stripped by exact match, or by shared prefix
    when the sample id carries a run suffix the folder does not have
    (``S1_Tvar`` against ``S1_B_...``), so repeated runs of one pellet can live
    side by side without losing their labels.

    A sample id sharing no prefix with the folder cannot be stripped; see the
    naming convention in ``docs/INPUT_FORMAT.md``.

    >>> condition_label("S1_B_Ar-80_O2-20_600_400_25", "S1")
    'Ar-80 O2-20 | 400-600C'
    >>> condition_label("S1_B_Air_600_400_25", "S1_Tvar")
    'Air | 400-600C'
    """
    if condition.startswith(sample_id):
        stripped = condition[len(sample_id):].lstrip("_")
    else:
        shared = os.path.commonprefix([condition, sample_id])
        shared = shared[:shared.rfind("_") + 1] if "_" in shared else ""
        stripped = condition[len(shared):]

    parts = stripped.split("_")
    # a bank or position letter (B, A1) precedes the gas; a gas never does
    if parts and len(parts[0]) <= 3 and not _GAS_RE.match(parts[0]):
        stripped = "_".join(parts[1:])

    parts = stripped.split("_")
    if len(parts) >= 4 and parts[-3].isdigit() and parts[-2].isdigit():
        return f"{' '.join(parts[:-3])} | {parts[-2]}-{parts[-3]}C"
    return stripped
