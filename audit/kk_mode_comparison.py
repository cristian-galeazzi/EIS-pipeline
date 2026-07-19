"""Compare Lin-KK M-selection modes over every validated spectrum.

Method
------
The Lin-KK validity test (Schoenleber et al. 2014) fits the spectrum with a
chain of M RC elements whose time constants are fixed log-spaced; only their
weights are free. Because the chain is causal by construction, systematic
misfit flags a Kramers-Kronig violation (drift, nonlinearity, artifacts). The
one methodological knob is how M is chosen:

- pct{XX} : Percentage mode, M = round(XX% of the point count). This is the
            RelaxIS "Percentage" mode; the KK-View default is 50%.
- auto    : binary search for the smallest M whose residual sign-change
            fraction mu reaches 0.5, the criterion recommended by Schoenleber
            (under-fitting leaves correlated residuals, mu << 0.5;
            over-fitting chases noise, mu -> 1).

This script runs every requested mode on every spectrum of the given samples
and tabulates the Shapiro-Wilk KK score, the residual normality per part, M,
mu, and the adaptive IQR frequency cuts, so the operator can judge which mode
keeps the widest clean window on their instrument and material.

Interpretation
--------------
A mode is better on your data when it keeps a comparable kk_score with a
wider retained frequency window (later f_min_cut rise, earlier f_max_cut
fall are both losses). If auto and a percentage mode disagree strongly on M
but agree on kk_score, the spectrum is over-sampled and the cheaper mode is
fine. The pipeline default (Percentage, KK_C = 0.76) was chosen this way on
the authors' dataset; rerun this script before trusting it on yours.

Outputs
-------
audit/output/{sample_id}/kk_mode_comparison.csv (one row per spectrum and
mode) plus an aggregate table on stdout. Read-only: no pipeline output is
touched.

Usage (from the repository root)
--------------------------------
  .venv/bin/python audit/kk_mode_comparison.py --samples MY_SAMPLE
  .venv/bin/python audit/kk_mode_comparison.py --samples EXAMPLE_SAMPLE \\
      --pcts 0.76 0.50
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audit._common import _load_record, _spectrum_dirs, load_session_entry
from pipeline.quality import run_linkk, strip_inductive

CSV_FIELDS = ["sample", "condition", "file", "mode", "N", "M", "mu",
              "kk_score", "W_re", "W_im", "f_min_cut", "f_max_cut"]


def build_modes(pcts: list[float], include_auto: bool = True) -> dict[str, dict]:
    """Mode table: run_linkk keyword sets keyed by a short mode name.

    >>> sorted(build_modes([0.76, 0.50]).keys())
    ['auto', 'pct50', 'pct76']
    """
    modes = {f"pct{round(p * 100)}": dict(c=p, use_binary_M=False)
             for p in pcts}
    if include_auto:
        modes["auto"] = dict(use_binary_M=True, mu_target=0.50)
    return modes


def stage2_hard_limits(session_path: Path,
                       sample_id: str) -> tuple[float | None, float | None]:
    """Per-sample hard frequency limits from session.json stage2_params.

    >>> stage2_hard_limits(Path("does_not_exist.json"), "X")
    (None, None)
    """
    p2 = load_session_entry(session_path, sample_id).get("stage2_params", {})
    return p2.get("KK_F_MIN_HARD"), p2.get("KK_F_MAX_HARD")


def compare_sample(sample_dir: Path, modes: dict[str, dict],
                   iqr_fence: float, iqr_window: int,
                   f_min_hard: float | None,
                   f_max_hard: float | None) -> list[dict]:
    """Run every mode on every spectrum of one sample; one dict per run.

    Spectra are taken from ISM validation/ (Zahner) or input_spectra/ (CSV),
    every condition, every replica.

    >>> # See tests/test_audit_kk_mode_comparison.py for a runnable example.
    """
    sample_id = sample_dir.name
    conditions = sorted({d.name
                         for root in ("ISM validation", "input_spectra")
                         if (sample_dir / root).is_dir()
                         for d in (sample_dir / root).iterdir() if d.is_dir()})
    rows: list[dict] = []
    for condition in conditions:
        for spec_dir in _spectrum_dirs(sample_dir, condition):
            for path in sorted(spec_dir.iterdir()):
                if path.suffix.lower() not in {".ism", ".csv", ".txt"}:
                    continue
                try:
                    rec = _load_record(path)
                except Exception as exc:
                    print(f"[WARN] {path.name}: {exc}", file=sys.stderr)
                    continue
                freq, Z_re, Z_im, _ = strip_inductive(rec.freq, rec.Z_re,
                                                      rec.Z_im)
                for mode, kw in modes.items():
                    try:
                        res = run_linkk(freq, Z_re, Z_im,
                                        iqr_fence_factor=iqr_fence,
                                        iqr_window=iqr_window,
                                        f_min_hard=f_min_hard,
                                        f_max_hard=f_max_hard, **kw)
                    except Exception as exc:
                        print(f"[WARN] {path.name} {mode}: {exc}",
                              file=sys.stderr)
                        continue
                    rows.append({
                        "sample": sample_id, "condition": condition,
                        "file": path.name, "mode": mode,
                        "N": len(freq), "M": res["M"],
                        "mu": round(res["mu"], 3),
                        "kk_score": round(res["kk_score"], 4),
                        "W_re": round(res["W_re"], 4),
                        "W_im": round(res["W_im"], 4),
                        "f_min_cut": round(res["f_min_cut"], 1),
                        "f_max_cut": round(res["f_max_cut"], 1),
                    })
    return rows


def print_aggregate(rows: list[dict]) -> None:
    """Mean metrics per (sample, mode) as a fixed-width stdout table.

    >>> print_aggregate([])
    (no results)
    """
    if not rows:
        print("(no results)")
        return
    agg: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        agg[(r["sample"], r["mode"])].append(r)
    print(f"\n{'sample':<16} {'mode':<6} {'n':>4} {'kk_score':>9} "
          f"{'W_re':>7} {'W_im':>7} {'M/N':>6} {'f_min_cut':>10} "
          f"{'f_max_cut':>12}")
    for (sample, mode), rs in sorted(agg.items()):
        n = len(rs)
        def mean(key: str) -> float:
            return sum(r[key] for r in rs) / n
        print(f"{sample:<16} {mode:<6} {n:>4} {mean('kk_score'):>9.4f} "
              f"{mean('W_re'):>7.4f} {mean('W_im'):>7.4f} "
              f"{mean('M') / mean('N'):>6.2f} {mean('f_min_cut'):>10.1f} "
              f"{mean('f_max_cut'):>12.1f}")


def main() -> None:
    """CLI entry point; see the module docstring for the method.

    >>> # .venv/bin/python audit/kk_mode_comparison.py --samples EXAMPLE_SAMPLE
    """
    parser = argparse.ArgumentParser(
        description="Compare Lin-KK M-selection modes on your spectra.")
    parser.add_argument("--samples", nargs="+", required=True,
                        help="sample folder names (repo-root relative)")
    parser.add_argument("--pcts", nargs="+", type=float, default=[0.76, 0.50],
                        help="Percentage-mode RC densities to compare")
    parser.add_argument("--no-auto", action="store_true",
                        help="skip the automatic mu=0.5 mode")
    parser.add_argument("--iqr-fence", type=float, default=2.0)
    parser.add_argument("--iqr-window", type=int, default=5)
    parser.add_argument("--session", type=Path, default=REPO / "session.json")
    parser.add_argument("--output", type=Path, default=REPO / "audit" / "output")
    args = parser.parse_args()

    modes = build_modes(args.pcts, include_auto=not args.no_auto)
    for sample_id in args.samples:
        sample_dir = REPO / sample_id
        if not sample_dir.is_dir():
            print(f"[WARN] sample folder not found: {sample_dir}",
                  file=sys.stderr)
            continue
        f_min_hard, f_max_hard = stage2_hard_limits(args.session, sample_id)
        rows = compare_sample(sample_dir, modes, args.iqr_fence,
                              args.iqr_window, f_min_hard, f_max_hard)
        if not rows:
            print(f"[WARN] no spectra found for {sample_id}", file=sys.stderr)
            continue
        out_dir = args.output / sample_id
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            out_csv = out_dir / "kk_mode_comparison.csv"
            with out_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            sys.exit(f"cannot write results: {exc}")
        print(f"saved {len(rows)} rows -> {out_csv}")
        print_aggregate(rows)


if __name__ == "__main__":
    main()
