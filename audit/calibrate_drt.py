"""Stage-3 DRT parameter calibration by physics-based ranking.

Method
------
The stage-3 defaults (RBF derivative order, regularization lambda, HF weight,
peak cap) are not universal constants: the best values depend on the noise
level, the frequency window, and how strongly the relaxation processes of a
given material overlap. This script sweeps a grid of those parameters over one
or more measured conditions of one sample and ranks each combination by a
physics score instead of the raw fit residual.

Why not rank by rmse: a lower residual can always be bought by letting the DRT
split noise into extra peaks, so rmse systematically rewards overfitting. A
real thermally activated process must instead move smoothly with temperature.
The score therefore chains fitted peaks across the temperature series
(nearest-neighbour matching in log10 tau) and measures, per track, the
Arrhenius linearity of tau(T) and R(T):

    score = 0.40 * R2_arrh(tau)   weighted by track resistance
          + 0.25 * R2_arrh(R)     weighted by track resistance
          + 0.20 * coverage       (fraction of total R inside long tracks)
          + 0.10 * convergence fraction
          + 0.05 * peak-count stability across temperatures

Interpretation
--------------
The top-ranked combination is the one whose DRT peaks behave like physical
processes over the whole temperature series. If two combinations score within
noise of each other, prefer the smoother one (higher lambda, lower cap): it is
the more conservative choice. A low best score (< ~0.7) usually means the
temperature series is too short for tracking or the peaks genuinely overlap;
in that case widen the grid rather than trusting the ranking.

Outputs
-------
audit/output/{sample_id}/calibrate_drt_ranking_{set}.csv, one row per
combination, sorted by score; {set} encodes the condition set, so runs on
different sets never overwrite each other. Only paths and progress go to stdout; nothing derived from
your data enters the repository.

Usage (from the repository root)
--------------------------------
  .venv/bin/python audit/calibrate_drt.py --sample MY_SAMPLE \\
      --conditions Ar-80_O2-20_600_400_50 O2-100_600_400_50

Synthetic known-answer example (also run as a test in tests/):
  .venv/bin/python audit/calibrate_drt.py --sample EXAMPLE_SAMPLE \\
      --lambdas 1e-4 --rbf-ders "2nd order" --hf-weights 0.0 --caps 2 free \\
      --workers 1
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audit._common import (
    apply_peak_cap,
    default_min_track_points,
    discover_conditions,
    load_condition_spectra,
    run_jobs,
    run_slug,
    sample_settings,
    score_condition,
)
from pipeline.drt import compute_drt, find_drt_peaks
from pipeline.fitting import fit_condition_batch


def drt_job(args: tuple) -> tuple:
    """Worker: DRT + peak detection for one (rbf_der, lambda, condition)."""
    rbf_der, lam, condition, spectra, settings = args
    results = []
    for sp in spectra:
        try:
            entry = compute_drt(sp["freq"], sp["Z_re"], sp["Z_im"],
                                cv_type="custom", rbf_der=rbf_der,
                                shape_s=0.5, lambda_val=lam,
                                suppress_output=True)
            peaks = find_drt_peaks(
                entry,
                min_height_frac=settings["peak_height_frac"],
                min_dist_decades=settings["peak_min_dist_decades"])
        except Exception as exc:
            print(f"[DRT ERROR] {condition} T={sp['T_nominal']}: {exc}",
                  file=sys.stderr)
            peaks = []
        results.append({**sp, "peaks": peaks})
    return rbf_der, lam, condition, results


def fit_job(args: tuple) -> tuple:
    """Worker: Zarc fit of one condition for one full parameter combo."""
    combo, condition, drt_tasks, settings, L_m, D_m = args
    _rbf_der, _lam, hf_w, cap = combo
    tasks = []
    for t in drt_tasks:
        peaks = apply_peak_cap(t["peaks"], cap)
        if not peaks:
            continue
        tasks.append({
            "T_nominal": t["T_nominal"], "fname": t["fname"],
            "ism_path": t["ism_path"], "pO2": t["pO2"],
            "freq": t["freq"], "Z_re": t["Z_re"], "Z_im": t["Z_im"],
            "peaks": peaks,
            "R_dec": settings["R_dec"], "tau_dec": settings["tau_dec"],
            "alpha_init": settings["alpha_init"],
            "alpha_min": settings["alpha_min"],
            "alpha_max": settings["alpha_max"],
            "hf_weight": hf_w, "fix_params": None, "ov_tag": "",
        })
    try:
        res = fit_condition_batch(condition, tasks,
                                  include_r0=settings["include_r0"],
                                  r0_max=settings["r0_max"],
                                  n_restarts=settings["n_restarts"],
                                  rmse_tol=settings["rmse_tol"],
                                  L_m=L_m, D_m=D_m)
    except Exception as exc:
        print(f"[FIT ERROR] {combo} {condition}: {exc}", file=sys.stderr)
        res = {"fit_peaks": [], "fit_summary": []}
    return combo, condition, res["fit_peaks"], res["fit_summary"]


def run_grid(sample_dir: Path, conditions: list[str],
             rbf_ders: list[str], lambdas: list[float],
             hf_weights: list[float], caps: list[int | None],
             settings: dict, L_m: float, D_m: float,
             min_track_points: int | None, workers: int,
             use_stage2: bool = True) -> pd.DataFrame:
    """Sweep the full grid and return the ranking DataFrame (best first).

    With workers=1 everything runs serially in-process, which is what the
    synthetic known-answer test uses.

    >>> # See tests/test_audit_calibrate_drt.py for a runnable example.
    """
    t0 = time.time()
    spectra = {c: load_condition_spectra(sample_dir, c, use_stage2)
               for c in conditions}
    for c, sp in spectra.items():
        if not sp:
            raise RuntimeError(f"no spectra loaded for condition '{c}'")
        print(f"  {c}: {len(sp)} temperatures", flush=True)
    if min_track_points is None:
        n_T = min(len(sp) for sp in spectra.values())
        min_track_points = default_min_track_points(n_T)

    drt_jobs = [(d, lam, c, spectra[c], settings)
                for d, lam, c in product(rbf_ders, lambdas, conditions)]
    drt_cache: dict[tuple, list[dict]] = {}
    print(f"Phase 1: {len(drt_jobs)} DRT jobs...", flush=True)
    for d, lam, c, res in run_jobs(drt_job, drt_jobs, workers):
        drt_cache[(d, lam, c)] = res
        if workers > 1:
            print(f"  DRT done {d} lambda={lam:.1e} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    combos = list(product(rbf_ders, lambdas, hf_weights, caps))
    fit_jobs = [(combo, c, drt_cache[(combo[0], combo[1], c)],
                 settings, L_m, D_m)
                for combo in combos for c in conditions]
    print(f"Phase 2: {len(fit_jobs)} fit jobs...", flush=True)
    raw: dict[tuple, dict] = {}
    done = 0
    for combo, cond, pk, sm in run_jobs(fit_job, fit_jobs, workers):
        raw[(combo, cond)] = {"peaks": pk, "summary": sm}
        done += 1
        if workers > 1:
            print(f"  [{done}/{len(fit_jobs)}] fitted "
                  f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for combo in combos:
        d, lam, h, cap = combo
        per_cond = [score_condition(raw[(combo, c)]["peaks"],
                                    raw[(combo, c)]["summary"],
                                    min_track_points) for c in conditions]
        agg = {k: round(float(np.nanmean([pc[k] for pc in per_cond])), 4)
               for k in per_cond[0]}
        rows.append({"rbf_der": d, "lambda": lam, "hf_weight": h,
                     "n_cap": cap if cap is not None else "free", **agg})
    df = pd.DataFrame(rows).sort_values(
        "score", ascending=False).reset_index(drop=True)
    print(f"Grid done in {time.time() - t0:.0f}s", flush=True)
    return df


def _parse_cap(value: str) -> int | None:
    if value.lower() == "free":
        return None
    return int(value)


def main() -> None:
    """CLI entry point; see the module docstring for the method.

    >>> # .venv/bin/python audit/calibrate_drt.py --sample EXAMPLE_SAMPLE \\
    >>> #     --lambdas 1e-4 --caps 2 free --workers 1
    """
    parser = argparse.ArgumentParser(
        description="Physics-based calibration of the stage-3 DRT parameters.")
    parser.add_argument("--sample", required=True,
                        help="sample folder name (repo-root relative)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="condition folder names; default: the VALIDATED "
                             "conditions (those with a stage-2 selection); one "
                             "run with several conditions ranks by the mean "
                             "score, which is the intended use")
    parser.add_argument("--all-conditions", action="store_true",
                        help="with no --conditions: use every spectra folder, "
                             "validated or not")
    parser.add_argument("--rbf-ders", nargs="+",
                        default=["1st order", "2nd order"])
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[1e-5, 1e-4, 1e-3])
    parser.add_argument("--hf-weights", nargs="+", type=float,
                        default=[0.0, 0.3, 1.0])
    parser.add_argument("--caps", nargs="+", type=_parse_cap,
                        default=[None, 4],
                        help="peak caps to try; integers or 'free'")
    parser.add_argument("--min-track-points", type=int, default=None,
                        help="temperatures a track needs to enter the score "
                             "(default: 2/3 of the series length)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--session", type=Path, default=REPO / "session.json")
    parser.add_argument("--output", type=Path, default=REPO / "audit" / "output")
    args = parser.parse_args()

    sample_dir = REPO / args.sample
    if not sample_dir.is_dir():
        sys.exit(f"sample folder not found: {sample_dir}")
    conditions = args.conditions
    if conditions is None:
        conditions = discover_conditions(
            sample_dir, validated_only=not args.all_conditions)
        if not conditions:
            sys.exit(f"no conditions found under {sample_dir}")
    print(f"conditions in this run ({len(conditions)}):")
    for c in conditions:
        print(f"  {c}")

    settings, L_m, D_m = sample_settings(args.session, args.sample)
    df = run_grid(sample_dir, conditions, args.rbf_ders, args.lambdas,
                  args.hf_weights, args.caps, settings, L_m, D_m,
                  args.min_track_points, args.workers)
    # self-documenting output: which conditions produced this ranking
    df.insert(0, "conditions", ";".join(sorted(conditions)))

    out_dir = args.output / args.sample
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # per-set filename: runs on different condition sets never overwrite
        out_csv = out_dir / f"calibrate_drt_ranking_{run_slug(conditions)}.csv"
        df.to_csv(out_csv, index=False)
    except OSError as exc:
        sys.exit(f"cannot write ranking: {exc}")

    pd.set_option("display.width", 200)
    print("\n========== RANKING (mean over conditions) ==========")
    print(df.to_string(index=True))
    best = df.iloc[0]
    print(f"\nBEST: rbf_der={best['rbf_der']}  lambda={best['lambda']:.1e}  "
          f"hf_weight={best['hf_weight']}  n_cap={best['n_cap']}")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
