"""Shared helpers used by the notebook export cells.

Two responsibilities:
- merge_sheet_by_T : safe per-temperature update of a sheet inside an Excel file
- build_metadata_sheet : standardised 2-column DataFrame recording fixed parameters

These helpers exist so that NB02 and NB03 export logic stays identical and the
Metadata schema is uniform across all stages (NB01, NB02, NB03).
"""
from __future__ import annotations

import datetime as _dt
import warnings
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


# Libraries whose version can affect numeric results — recorded in every
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
    """
    if focus_t is None or not Path(xlsx_path).exists():
        return new_df

    try:
        existing = pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception as exc:
        warnings.warn(
            f"merge_sheet_by_T: could not read existing sheet '{sheet_name}' in "
            f"{xlsx_path} ({exc}); falling back to overwrite — rows for other "
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
    """
    rows: list[tuple[str, Any]] = [
        ("sample_id", sample_id),
        ("stage", stage_name),
        ("processed_at", _dt.datetime.now().strftime("%Y-%m-%d %H:%M")),
    ]
    rows.extend((str(k), v) for k, v in params.items())
    rows.extend(_library_versions())
    return pd.DataFrame(rows, columns=["parameter", "value"])
