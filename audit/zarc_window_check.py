"""Zarc constraint-window boundary check.

Method
------
Stage 3 fits each Zarc element inside a box around its DRT seed: R within
R_dec decades of the integrated peak area, tau within tau_dec decades of the
peak position. A fit that lands ON that boundary is bound-limited: the
optimizer pushed against the wall instead of settling where the data wanted
it, so the reported parameter is an artifact of the window, not a
measurement. A fit that sits comfortably inside the window but still
reproduces the spectrum poorly points instead at the DRT seed itself (peak
detection), not at the window.

For every (condition, T, peak) this script recomputes the bounds from the
saved DRT seed and the condition's R_dec/tau_dec, then measures where the
fitted value sits as a fraction of the log-window half-width (0 = on a
boundary, 1 = dead centre). Values below --margin are flagged as pinned.

Interpretation
--------------
- Many pinned tau at the same peak across temperatures: the seed windows are
  too narrow for how fast that process moves; widen tau_dec or fix the seed.
- Pinned only at the series edges (lowest/highest T): the DRT peak drifts
  out of the detection window there; check the frequency cuts first.
- Nothing pinned but fits still poor: the constraint windows are not the
  bottleneck; look at peak detection or the model itself.

Outputs
-------
Only file paths and pinned counts go to stdout. Per-point detail (condition,
T, peak_id, distance to bound) goes to audit/output/{sample_id}/
zarc_window_check.csv, which is gitignored.

Usage (from the repository root)
--------------------------------
  .venv/bin/python audit/zarc_window_check.py                # all samples
  .venv/bin/python audit/zarc_window_check.py --samples MY_SAMPLE \\
      --margin 0.15
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audit._common import DEFAULTS
from pipeline.fitting import resolve_condition_entry

DEFAULT_MARGIN = 0.15   # within 15% of the boundary (in decades) = "pinned"


def bounds_from_seed(seed: float, dec: float) -> tuple[float, float]:
    """Fit window around a DRT seed: seed / 10**dec to seed * 10**dec.

    >>> bounds_from_seed(1.0, 1.0)
    (0.1, 10.0)
    """
    return seed / 10**dec, seed * 10**dec


def edge_distance(value: float, lo: float, hi: float) -> float:
    """Fraction of the log-window half-width separating value from its edge.

    0 means the value sits on a boundary, 1 dead centre. Degenerate windows
    or non-positive values return NaN: cannot judge, never flag.

    >>> round(edge_distance(5.0, 1.0, 10.0), 3)
    0.602
    >>> edge_distance(1.0, 0.0, 10.0)
    nan
    """
    if not (0 < lo < hi and value > 0):
        return float("nan")
    log_lo, log_hi, log_v = math.log10(lo), math.log10(hi), math.log10(value)
    half = (log_hi - log_lo) / 2
    return min(log_v - log_lo, log_hi - log_v) / half if half > 0 else 0.0


def check_rows(drt: pd.DataFrame, fit: pd.DataFrame, R_dec: float,
               tau_dec: float, condition: str, margin: float) -> list[dict]:
    """Boundary check of one condition from its DRT-seed and fit tables.

    `drt` needs columns T_nominal, peak_id, tau, R_approx; `fit` needs
    T_nominal, peak_id, tau_i, R_i. Rows are matched on (T_nominal, peak_id).

    >>> drt = pd.DataFrame([{"T_nominal": 600, "peak_id": 1,
    ...                      "tau": 1e-4, "R_approx": 100.0}])
    >>> fit = pd.DataFrame([{"T_nominal": 600, "peak_id": 1,
    ...                      "tau_i": 1e-4, "R_i": 100.0 * 10**0.7}])
    >>> row = check_rows(drt, fit, 0.7, 0.7, "c", 0.15)[0]
    >>> row["pinned_R"], row["pinned_tau"]
    (True, False)
    """
    merged = fit.merge(
        drt[["T_nominal", "peak_id", "tau", "R_approx"]],
        on=["T_nominal", "peak_id"], how="inner", suffixes=("", "_seed"))
    rows = []
    for _, r in merged.iterrows():
        tau_lo, tau_hi = bounds_from_seed(r["tau"], tau_dec)
        R_lo, R_hi = bounds_from_seed(r["R_approx"], R_dec)
        d_tau = edge_distance(r["tau_i"], tau_lo, tau_hi)
        d_R = edge_distance(r["R_i"], R_lo, R_hi)
        rows.append({
            "condition": condition,
            "T_nominal": r["T_nominal"],
            "peak_id": r["peak_id"],
            "tau_edge_frac": round(d_tau, 4),
            "R_edge_frac": round(d_R, 4),
            # NaN edge distances compare False: degenerate rows never flag
            "pinned_tau": bool(d_tau <= margin),
            "pinned_R": bool(d_R <= margin),
        })
    return rows


def _condition_windows(entry: dict, condition: str) -> tuple[float, float]:
    """R_dec/tau_dec for a condition: per-condition override, then
    stage3_params, then the notebook defaults."""
    p3 = entry.get("stage3_params", {})
    over = resolve_condition_entry(entry.get("condition_params", {}), condition)
    R_dec = over.get("R_dec", p3.get("ZARC_R_DEC", DEFAULTS["R_dec"]))
    tau_dec = over.get("tau_dec", p3.get("ZARC_TAU_DEC", DEFAULTS["tau_dec"]))
    return R_dec, tau_dec


def check_sample(sample_dir: Path, margin: float,
                 session_path: Path) -> pd.DataFrame | None:
    """Run the boundary check on every fitted condition of one sample.

    Reads {sample}/Results/{condition}/stage3_drt.xlsx (Peaks sheet, seeds)
    and stage3_fit.xlsx (Peaks sheet, fitted values). Returns None when the
    sample has nothing to check.

    >>> # See tests/test_audit_zarc_window_check.py for a runnable example.
    """
    try:
        cfg = json.loads(session_path.read_text())
        entry = next((s for s in cfg
                      if s.get("sample_id") == sample_dir.name), {})
    except (OSError, ValueError):
        entry = {}

    results_dir = sample_dir / "Results"
    if not results_dir.is_dir():
        return None
    rows: list[dict] = []
    for cond_dir in sorted(d for d in results_dir.iterdir() if d.is_dir()):
        drt_xlsx = cond_dir / "stage3_drt.xlsx"
        fit_xlsx = cond_dir / "stage3_fit.xlsx"
        if not (drt_xlsx.exists() and fit_xlsx.exists()):
            continue
        try:
            drt = pd.read_excel(drt_xlsx, sheet_name="Peaks")
            fit = pd.read_excel(fit_xlsx, sheet_name="Peaks")
        except (ValueError, OSError) as exc:
            print(f"[WARN] {cond_dir.name}: {exc}", file=sys.stderr)
            continue
        R_dec, tau_dec = _condition_windows(entry, cond_dir.name)
        rows.extend(check_rows(drt, fit, R_dec, tau_dec,
                               cond_dir.name, margin))
    return pd.DataFrame(rows) if rows else None


def main() -> None:
    """CLI entry point; see the module docstring for the method.

    >>> # .venv/bin/python audit/zarc_window_check.py --margin 0.15
    """
    parser = argparse.ArgumentParser(
        description="Flag Zarc fits pinned at their constraint-window bounds.")
    parser.add_argument("--samples", nargs="+", default=None,
                        help="sample folder names; default: every folder "
                             "with a Results/ directory")
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN,
                        help="boundary distance (fraction of window "
                             "half-width) below which a fit counts as pinned")
    parser.add_argument("--session", type=Path, default=REPO / "session.json")
    parser.add_argument("--output", type=Path, default=REPO / "audit" / "output")
    args = parser.parse_args()

    if args.samples is not None:
        sample_dirs = [REPO / s for s in args.samples]
    else:
        sample_dirs = sorted(
            d for d in REPO.iterdir()
            if d.is_dir() and (d / "Results").is_dir()
            and d.name != "sample_template")

    total_pinned = 0
    checked_any = False
    for sample_dir in sample_dirs:
        df = check_sample(sample_dir, args.margin, args.session)
        if df is None or df.empty:
            continue
        checked_any = True
        out_dir = args.output / sample_dir.name
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "zarc_window_check.csv"
            df.to_csv(out_csv, index=False)
        except OSError as exc:
            sys.exit(f"cannot write results: {exc}")
        n_pinned = int((df["pinned_tau"] | df["pinned_R"]).sum())
        total_pinned += n_pinned
        print(f"{sample_dir.name}: {len(df)} (condition, T, peak) points "
              f"checked, {n_pinned} pinned to a bound -> {out_csv}")

    if not checked_any:
        print("No stage-3 results found to check.")
    elif total_pinned == 0:
        print("No points pinned to their constraint window: bounds are not "
              "the bottleneck anywhere checked.")


if __name__ == "__main__":
    main()
