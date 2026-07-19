"""Second-pass calibration of the stage-3 fit knobs, DRT frozen.

Method
------
Run this after calibrate_drt.py has fixed the DRT set (RBF derivative,
lambda, peak cap). With the deconvolution frozen, only the Zarc-fit knobs are
swept: the high-frequency weight and the R/tau seed windows (R_dec, tau_dec,
in decades around the DRT seed). Combinations are ranked by high-frequency
fidelity under two guards that catch the failure modes a pure residual
ranking would miss:

- physics guard: the physics score (Arrhenius consistency of tracked peaks,
  same definition as calibrate_drt.py) must stay within 0.01 of the best
  combination; a fit that buys HF fidelity by breaking the temperature
  systematics is rejected;
- alpha stress signal: the fraction of fitted Zarc exponents pinned at their
  bounds. A pinned alpha means the optimizer pushed against the box instead
  of settling inside it, the classic signature of an over-constrained or
  overfitting-prone parametrization.

Metrics
-------
hf_res       : mean relative |Z| residual over the top quarter of the log-f
               range (the HF band where electrode/inductive artifacts and
               seed errors concentrate)
all_res      : same over the full spectrum
hf_phase_deg : mean absolute phase error in the HF band [degrees]
physics      : physics score, see calibrate_drt.py
alpha_pinned : fraction of fitted alphas at a bound

Interpretation
--------------
Pick the row with the lowest hf_res among those whose physics score is within
0.01 of the maximum and whose alpha_pinned is low. If every low-hf_res row has
alpha_pinned near 1, the alpha window itself is the bottleneck and the
constraint bounds deserve a second look before any knob does.

Outputs
-------
audit/output/{sample_id}/calibrate_fit_ranking_{set}.csv, where {set}
encodes the condition set (runs on different sets never overwrite each
other). Only paths and progress go to stdout.

Usage (from the repository root)
--------------------------------
  .venv/bin/python audit/calibrate_fit.py --sample MY_SAMPLE \\
      --rbf-der "2nd order" --lambda-val 1e-4 --cap 4

Synthetic known-answer example (also run as a test in tests/):
  .venv/bin/python audit/calibrate_fit.py --sample EXAMPLE_SAMPLE \\
      --rbf-der "2nd order" --lambda-val 1e-4 --cap 2 \\
      --hf-weights 0.0 1.0 --r-decs 0.7 --tau-decs 0.7 --workers 1
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

HF_BAND_FRAC = 0.25   # top quarter of the log-f range counts as "HF"
PIN_TOL = 5e-3        # |alpha - bound| below this counts as pinned


def drt_job(args: tuple) -> tuple:
    """Worker: frozen-set DRT + capped peak detection for one condition."""
    condition, spectra, rbf_der, lambda_val, cap, settings = args
    results = []
    for sp in spectra:
        try:
            entry = compute_drt(sp["freq"], sp["Z_re"], sp["Z_im"],
                                cv_type="custom", rbf_der=rbf_der,
                                shape_s=0.5, lambda_val=lambda_val,
                                suppress_output=True)
            peaks = apply_peak_cap(
                find_drt_peaks(
                    entry,
                    min_height_frac=settings["peak_height_frac"],
                    min_dist_decades=settings["peak_min_dist_decades"],
                    min_prom_decades=settings["peak_min_prom_decades"] or None),
                cap)
        except Exception as exc:
            print(f"[DRT ERROR] {condition} T={sp['T_nominal']}: {exc}",
                  file=sys.stderr)
            peaks = []
        results.append({**sp, "peaks": peaks})
    return condition, results


def fit_job(args: tuple) -> tuple:
    """Worker: Zarc fit of one condition for one (hf_weight, R_dec, tau_dec)."""
    combo, condition, drt_tasks, settings, L_m, D_m = args
    hf_w, r_dec, tau_dec = combo
    tasks = [{
        "T_nominal": t["T_nominal"], "fname": t["fname"],
        "ism_path": t["ism_path"], "pO2": t["pO2"],
        "freq": t["freq"], "Z_re": t["Z_re"], "Z_im": t["Z_im"],
        "peaks": t["peaks"],
        "R_dec": r_dec, "tau_dec": tau_dec,
        "alpha_init": settings["alpha_init"],
        "alpha_min": settings["alpha_min"],
        "alpha_max": settings["alpha_max"],
        "hf_weight": hf_w, "fix_params": None, "ov_tag": "",
    } for t in drt_tasks if t["peaks"]]
    try:
        res = fit_condition_batch(condition, tasks,
                                  include_r0=settings["include_r0"],
                                  r0_max=settings["r0_max"],
                                  n_restarts=settings["n_restarts"],
                                  rmse_tol=settings["rmse_tol"],
                                  L_m=L_m, D_m=D_m)
    except Exception as exc:
        print(f"[FIT ERROR] {combo} {condition}: {exc}", file=sys.stderr)
        res = {"fit_peaks": [], "fit_summary": [], "nyq_fits": {}}
    return combo, condition, res


def hf_metrics(drt_tasks: list[dict], nyq_fits: dict,
               band_frac: float = HF_BAND_FRAC) -> dict:
    """Frequency-resolved residuals: full band, HF band, HF Bode phase.

    >>> out = hf_metrics([], {})
    >>> bool(np.isnan(out["hf_res"]))
    True
    """
    hf_res, all_res, hf_phase = [], [], []
    for t in drt_tasks:
        nf = nyq_fits.get(t["T_nominal"])
        if nf is None or nf.get("Z_fit") is None:
            continue
        freq = np.asarray(t["freq"], dtype=float)
        Z_exp = np.asarray(t["Z_re"]) - 1j * np.asarray(t["Z_im"])
        Z_fit = np.asarray(nf["Z_fit"])
        if len(Z_fit) != len(freq):
            continue
        lo, hi = np.log10(freq.min()), np.log10(freq.max())
        hf_mask = np.log10(freq) >= hi - band_frac * (hi - lo)
        rel = np.abs(Z_fit - Z_exp) / np.maximum(np.abs(Z_exp), 1e-12)
        all_res.append(float(np.mean(rel)))
        hf_res.append(float(np.mean(rel[hf_mask])))
        dphase = np.degrees(np.angle(Z_fit) - np.angle(Z_exp))
        hf_phase.append(float(np.mean(np.abs(dphase[hf_mask]))))
    return {
        "hf_res": float(np.mean(hf_res)) if hf_res else np.nan,
        "all_res": float(np.mean(all_res)) if all_res else np.nan,
        "hf_phase_deg": float(np.mean(hf_phase)) if hf_phase else np.nan,
    }


def alpha_pinned_fraction(peak_rows: list[dict], alpha_min: float,
                          alpha_max: float, tol: float = PIN_TOL) -> float:
    """Fraction of fitted Zarc exponents sitting at an alpha bound.

    >>> rows = [{"alpha_i": 0.5}, {"alpha_i": 0.75}]
    >>> alpha_pinned_fraction(rows, 0.5, 1.0)
    0.5
    """
    alphas = np.array([r["alpha_i"] for r in peak_rows], dtype=float)
    if not len(alphas):
        return float("nan")
    pinned = (np.abs(alphas - alpha_min) < tol) | (np.abs(alphas - alpha_max) < tol)
    return float(np.mean(pinned))


def run_fit_grid(sample_dir: Path, conditions: list[str], rbf_der: str,
                 lambda_val: float, cap: int | None,
                 hf_weights: list[float], r_decs: list[float],
                 tau_decs: list[float], settings: dict,
                 L_m: float, D_m: float,
                 min_track_points: int | None, workers: int,
                 use_stage2: bool = True) -> pd.DataFrame:
    """Sweep the fit-knob grid and return the ranking (best hf_res first).

    With workers=1 everything runs serially in-process, which is what the
    synthetic known-answer test uses.

    >>> # See tests/test_audit_calibrate_fit.py for a runnable example.
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

    drt_jobs = [(c, spectra[c], rbf_der, lambda_val, cap, settings)
                for c in conditions]
    drt_cache: dict[str, list[dict]] = {}
    print(f"Phase 1: DRT (frozen set) on {len(drt_jobs)} conditions...",
          flush=True)
    for c, res in run_jobs(drt_job, drt_jobs, workers):
        drt_cache[c] = res
        if workers > 1:
            print(f"  DRT done ({time.time() - t0:.0f}s)", flush=True)

    combos = list(product(hf_weights, r_decs, tau_decs))
    fit_jobs = [(combo, c, drt_cache[c], settings, L_m, D_m)
                for combo in combos for c in conditions]
    print(f"Phase 2: {len(fit_jobs)} fit jobs...", flush=True)
    raw: dict[tuple, dict] = {}
    done = 0
    for combo, cond, res in run_jobs(fit_job, fit_jobs, workers):
        raw[(combo, cond)] = res
        done += 1
        if workers > 1:
            print(f"  [{done}/{len(fit_jobs)}] fitted "
                  f"({time.time() - t0:.0f}s)", flush=True)

    rows = []
    for combo in combos:
        hf_w, r_dec, tau_dec = combo
        cells = []
        for c in conditions:
            res = raw[(combo, c)]
            m = hf_metrics(drt_cache[c], res.get("nyq_fits", {}))
            m["physics"] = score_condition(res["fit_peaks"],
                                           res["fit_summary"],
                                           min_track_points)["score"]
            m["alpha_pinned"] = alpha_pinned_fraction(
                res["fit_peaks"], settings["alpha_min"], settings["alpha_max"])
            cells.append(m)
        agg = {k: round(float(np.nanmean([cl[k] for cl in cells])), 4)
               for k in cells[0]}
        rows.append({"hf_weight": hf_w, "R_dec": r_dec,
                     "tau_dec": tau_dec, **agg})
    df = pd.DataFrame(rows).sort_values("hf_res").reset_index(drop=True)
    print(f"Grid done in {time.time() - t0:.0f}s", flush=True)
    return df


def best_with_physics_guard(df: pd.DataFrame,
                            guard: float = 0.01) -> pd.Series:
    """Lowest-hf_res row whose physics score is within `guard` of the best.

    >>> df = pd.DataFrame({"hf_res": [0.01, 0.02], "physics": [0.5, 0.99]})
    >>> float(best_with_physics_guard(df)["hf_res"])
    0.02
    """
    ok = df[df["physics"] >= df["physics"].max() - guard]
    return ok.sort_values("hf_res").iloc[0]


def main() -> None:
    """CLI entry point; see the module docstring for the method.

    >>> # .venv/bin/python audit/calibrate_fit.py --sample EXAMPLE_SAMPLE \\
    >>> #     --rbf-der "2nd order" --lambda-val 1e-4 --cap 2 --workers 1
    """
    parser = argparse.ArgumentParser(
        description="Fit-knob calibration with the DRT frozen.")
    parser.add_argument("--sample", required=True,
                        help="sample folder name (repo-root relative)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="condition folder names; default: the VALIDATED "
                             "conditions (those with a stage-2 selection)")
    parser.add_argument("--all-conditions", action="store_true",
                        help="with no --conditions: use every spectra folder, "
                             "validated or not")
    parser.add_argument("--rbf-der", default="2nd order",
                        help="frozen DRT RBF derivative order")
    parser.add_argument("--lambda-val", type=float, default=1e-4,
                        help="frozen DRT regularization lambda")
    parser.add_argument("--cap", type=int, default=4,
                        help="frozen peak cap (0 = uncapped)")
    parser.add_argument("--hf-weights", nargs="+", type=float,
                        default=[0.0, 0.3, 1.0, 2.0])
    parser.add_argument("--r-decs", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--tau-decs", nargs="+", type=float, default=[0.5, 0.7])
    parser.add_argument("--min-track-points", type=int, default=None)
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
    cap = args.cap if args.cap > 0 else None
    df = run_fit_grid(sample_dir, conditions, args.rbf_der, args.lambda_val,
                      cap, args.hf_weights, args.r_decs, args.tau_decs,
                      settings, L_m, D_m, args.min_track_points, args.workers)
    # self-documenting output: which conditions produced this ranking
    df.insert(0, "conditions", ";".join(sorted(conditions)))

    out_dir = args.output / args.sample
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # per-set filename: runs on different condition sets never overwrite
        out_csv = out_dir / f"calibrate_fit_ranking_{run_slug(conditions)}.csv"
        df.to_csv(out_csv, index=False)
    except OSError as exc:
        sys.exit(f"cannot write ranking: {exc}")

    pd.set_option("display.width", 200)
    print("\n===== RANKING by HF residual (mean over conditions) =====")
    print(df.to_string(index=True))
    best = best_with_physics_guard(df)
    print(f"\nBEST with physics guard: hf_weight={best['hf_weight']}  "
          f"R_dec={best['R_dec']}  tau_dec={best['tau_dec']}  "
          f"hf_res={best['hf_res']:.4f}  physics={best['physics']:.3f}  "
          f"alpha_pinned={best['alpha_pinned']:.2f}")
    print(f"saved -> {out_csv}")


if __name__ == "__main__":
    main()
