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
audit/output/{sample_id}/calibrate_drt_ranking.csv, one row per combination,
sorted by score. Only paths and progress go to stdout; nothing derived from
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
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.drt import clip_spectrum, compute_drt, find_drt_peaks
from pipeline.fitting import fit_condition_batch
from pipeline.ingest import load_csv_spectrum, load_ism

K_B_EV = 8.617333e-5  # Boltzmann constant [eV/K]

# Notebook defaults, overridable per sample via session.json stage3_params.
DEFAULTS = {
    "peak_height_frac": 0.05,
    "peak_min_dist_decades": 0.3,
    "include_r0": False,
    "r0_max": 200.0,
    "n_restarts": 5,
    "rmse_tol": 0.02,
    "R_dec": 0.7,
    "tau_dec": 0.7,
    "alpha_init": 0.7,
    "alpha_min": 0.5,
    "alpha_max": 1.0,
}

TRACK_MATCH_DECADES = 0.6  # max |delta log10 tau| between adjacent T steps


def load_condition_spectra(sample_dir: Path, condition: str) -> list[dict]:
    """Load one spectrum per temperature for a condition, hottest first.

    Preferred source is the stage-2 output (best replica per T with the
    Lin-KK frequency cuts applied). When stage2_kk.xlsx is absent, every
    spectrum found under input_spectra/{condition}/ or
    ISM validation/{condition}/ is loaded uncut, replica 1 only.

    >>> spectra = load_condition_spectra(Path("EXAMPLE_SAMPLE"),
    ...                                  "Ar-80_O2-20_600_400_50")
    >>> [s["T_nominal"] for s in spectra]
    [600, 550, 500, 450, 400]
    """
    stage2 = sample_dir / "Results" / condition / "stage2_kk.xlsx"
    if stage2.exists():
        return _load_stage2_selected(sample_dir, condition, stage2)
    return _load_raw_directory(sample_dir, condition)


def _load_record(path: Path):
    """Dispatch on extension: Zahner binary or plain CSV/TXT."""
    if path.suffix.lower() == ".ism":
        return load_ism(path)
    return load_csv_spectrum(path)


def _spectrum_dirs(sample_dir: Path, condition: str) -> list[Path]:
    candidates = [sample_dir / "ISM validation" / condition,
                  sample_dir / "input_spectra" / condition]
    return [d for d in candidates if d.is_dir()]


def _load_stage2_selected(sample_dir: Path, condition: str,
                          stage2: Path) -> list[dict]:
    try:
        df = pd.read_excel(stage2, sheet_name="Selected")
    except (ValueError, OSError) as exc:
        raise RuntimeError(f"cannot read {stage2}: {exc}") from exc
    dirs = _spectrum_dirs(sample_dir, condition)
    out: list[dict] = []
    for _, row in df.sort_values("T_nominal", ascending=False).iterrows():
        path = next((d / row["file"] for d in dirs
                     if (d / row["file"]).exists()), None)
        if path is None:
            print(f"[WARN] {condition}: missing spectrum file {row['file']}",
                  file=sys.stderr)
            continue
        try:
            rec = _load_record(path)
        except Exception as exc:
            print(f"[WARN] {path.name}: {exc}", file=sys.stderr)
            continue
        f_min = row["f_min_cut"] if pd.notna(row.get("f_min_cut")) else None
        f_max = row["f_max_cut"] if pd.notna(row.get("f_max_cut")) else None
        freq, Z_re, Z_im = clip_spectrum(rec.freq, rec.Z_re, rec.Z_im,
                                         f_min, f_max)
        out.append({"T_nominal": int(row["T_nominal"]), "fname": row["file"],
                    "ism_path": str(path), "pO2": row.get("pO2_mean"),
                    "freq": freq, "Z_re": Z_re, "Z_im": Z_im})
    return out


def _load_raw_directory(sample_dir: Path, condition: str) -> list[dict]:
    dirs = _spectrum_dirs(sample_dir, condition)
    if not dirs:
        raise FileNotFoundError(
            f"no ISM validation/ or input_spectra/ folder for condition "
            f"'{condition}' under {sample_dir}")
    records = []
    for d in dirs:
        for path in sorted(d.iterdir()):
            if path.suffix.lower() not in {".ism", ".csv", ".txt"}:
                continue
            try:
                rec = _load_record(path)
            except Exception as exc:
                print(f"[WARN] {path.name}: {exc}", file=sys.stderr)
                continue
            if rec.T_nominal is None or (rec.replica or 1) != 1:
                continue
            records.append((path, rec))
    records.sort(key=lambda pr: -pr[1].T_nominal)
    return [{"T_nominal": int(rec.T_nominal), "fname": path.name,
             "ism_path": str(path), "pO2": None,
             "freq": rec.freq, "Z_re": rec.Z_re, "Z_im": rec.Z_im}
            for path, rec in records]


def apply_peak_cap(peaks: list[dict], cap: int | None) -> list[dict]:
    """Keep the `cap` largest-R peaks, renumbered in ascending tau.

    >>> pk = [{"tau": 1e-5, "R_approx": 1.0, "peak_id": 1},
    ...       {"tau": 1e-3, "R_approx": 9.0, "peak_id": 2}]
    >>> [p["peak_id"] for p in apply_peak_cap(pk, 1)]
    [1]
    """
    if cap is None or len(peaks) <= cap:
        return peaks
    by_R = sorted(peaks, key=lambda p: p["R_approx"], reverse=True)
    kept = sorted(by_R[:cap], key=lambda p: p["tau"])
    return [{**p, "peak_id": k + 1} for k, p in enumerate(kept)]


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


def build_tracks(peak_rows: list[dict],
                 match_decades: float = TRACK_MATCH_DECADES) -> list[list[dict]]:
    """Chain fitted peaks across temperatures by log10(tau) proximity.

    Greedy nearest-neighbour matching, hottest temperature first: each fitted
    peak joins the open track whose last tau is closest in log10, if within
    `match_decades`; otherwise it opens a new track.

    >>> rows = [{"T_nominal": 600, "tau_i": 1e-5},
    ...         {"T_nominal": 550, "tau_i": 2e-5}]
    >>> [len(t) for t in build_tracks(rows)]
    [2]
    """
    by_T: dict[float, list[dict]] = {}
    for r in peak_rows:
        by_T.setdefault(r["T_nominal"], []).append(r)
    tracks: list[dict] = []
    for T in sorted(by_T, reverse=True):
        rows = sorted(by_T[T], key=lambda r: r["tau_i"])
        free = set(range(len(tracks)))
        for r in rows:
            lt = np.log10(r["tau_i"])
            best, best_d = None, match_decades
            for i in free:
                d = abs(tracks[i]["last"] - lt)
                if d < best_d:
                    best, best_d = i, d
            if best is None:
                tracks.append({"last": lt, "points": [r]})
            else:
                tracks[best]["points"].append(r)
                tracks[best]["last"] = lt
                free.discard(best)
    return [t["points"] for t in tracks]


def track_activation_energy(points: list[dict]) -> float:
    """Activation energy [eV] from the Arrhenius slope of ln(tau) vs 1/T.

    For a thermally activated process tau = tau0 * exp(Ea / kB T), so the
    slope of ln(tau) against 1/T is Ea / kB.

    >>> pts = [{"T_K": 873.15, "tau_i": 2e-6},
    ...        {"T_K": 823.15, "tau_i": 2e-6 * np.exp(0.9 / K_B_EV
    ...                        * (1 / 823.15 - 1 / 873.15))}]
    >>> round(track_activation_energy(pts), 3)
    0.9
    """
    invT = np.array([1.0 / p["T_K"] for p in points])
    slope = linregress(invT, np.log([p["tau_i"] for p in points])).slope
    return float(slope * K_B_EV)


def score_condition(peak_rows: list[dict], summary_rows: list[dict],
                    min_track_points: int) -> dict:
    """Physics score of one fitted condition (see module docstring).

    >>> score_condition([], [], 4)["score"]
    0.0
    """
    if not peak_rows:
        return {"score": 0.0, "r2_tau": np.nan, "r2_R": np.nan,
                "coverage": 0.0, "n_long_tracks": 0, "n_tracks": 0,
                "conv_frac": 0.0, "rmse_mean": np.nan, "npeaks_std": np.nan}
    tracks = build_tracks(peak_rows)
    R_total = sum(r["R_i"] for r in peak_rows)
    r2t_n = r2R_n = w_sum = cover = 0.0
    n_long = 0
    for pts in tracks:
        track_R = sum(p["R_i"] for p in pts)
        if len(pts) < min_track_points:
            continue
        n_long += 1
        cover += track_R
        invT = np.array([1.0 / p["T_K"] for p in pts])
        r2_tau = linregress(invT, np.log([p["tau_i"] for p in pts])).rvalue ** 2
        r2_R = linregress(invT, np.log([p["R_i"] for p in pts])).rvalue ** 2
        r2t_n += track_R * r2_tau
        r2R_n += track_R * r2_R
        w_sum += track_R
    r2_tau_w = r2t_n / w_sum if w_sum else 0.0
    r2_R_w = r2R_n / w_sum if w_sum else 0.0
    coverage = cover / R_total if R_total else 0.0
    conv = float(np.mean([bool(s["converged"]) for s in summary_rows]))
    rmse = float(np.nanmean([s["rmse_rel"] for s in summary_rows]))
    np_std = float(np.std([s["N_peaks"] for s in summary_rows]))
    score = (0.40 * r2_tau_w + 0.25 * r2_R_w + 0.20 * coverage
             + 0.10 * conv + 0.05 * max(0.0, 1.0 - np_std / 2.0))
    return {"score": round(score, 4), "r2_tau": round(r2_tau_w, 3),
            "r2_R": round(r2_R_w, 3), "coverage": round(coverage, 3),
            "n_long_tracks": n_long, "n_tracks": len(tracks),
            "conv_frac": round(conv, 2), "rmse_mean": round(rmse, 4),
            "npeaks_std": round(np_std, 2)}


def _sample_settings(session_path: Path, sample_id: str) -> tuple[dict, float, float]:
    """Merge session.json stage3_params over the notebook defaults.

    Geometry falls back to dummy values when the sample has no session entry:
    the ranking only uses R, tau, alpha and convergence, so the conductivity
    columns computed from L_m/D_m never influence a score.
    """
    settings = dict(DEFAULTS)
    L_m, D_m = 1e-3, 1e-2
    try:
        cfg = json.loads(session_path.read_text())
        entry = next(s for s in cfg if s.get("sample_id") == sample_id)
    except (OSError, ValueError, StopIteration):
        return settings, L_m, D_m
    p3 = entry.get("stage3_params", {})
    key_map = {"peak_height_frac": "PEAK_HEIGHT_FRAC",
               "peak_min_dist_decades": "PEAK_MIN_DIST_DECADES",
               "include_r0": "ZARC_INCLUDE_R0", "r0_max": "ZARC_R0_MAX",
               "R_dec": "ZARC_R_DEC", "tau_dec": "ZARC_TAU_DEC",
               "alpha_init": "ZARC_ALPHA_INIT"}
    for ours, theirs in key_map.items():
        if theirs in p3:
            settings[ours] = p3[theirs]
    L_m = entry.get("L_m", L_m)
    D_m = entry.get("D_m", D_m)
    return settings, L_m, D_m


def run_grid(sample_dir: Path, conditions: list[str],
             rbf_ders: list[str], lambdas: list[float],
             hf_weights: list[float], caps: list[int | None],
             settings: dict, L_m: float, D_m: float,
             min_track_points: int | None, workers: int) -> pd.DataFrame:
    """Sweep the full grid and return the ranking DataFrame (best first).

    With workers=1 everything runs serially in-process, which is what the
    synthetic known-answer test uses.

    >>> # See tests/test_audit_calibrate_drt.py for a runnable example.
    """
    t0 = time.time()
    spectra = {c: load_condition_spectra(sample_dir, c) for c in conditions}
    for c, sp in spectra.items():
        if not sp:
            raise RuntimeError(f"no spectra loaded for condition '{c}'")
        print(f"  {c}: {len(sp)} temperatures", flush=True)
    if min_track_points is None:
        # 2/3 of the series: a physical track must span most temperatures,
        # but demanding all of them would punish one failed fit too hard
        n_T = min(len(sp) for sp in spectra.values())
        min_track_points = max(3, round(2 * n_T / 3))

    drt_jobs = [(d, lam, c, spectra[c], settings)
                for d, lam, c in product(rbf_ders, lambdas, conditions)]
    drt_cache: dict[tuple, list[dict]] = {}
    print(f"Phase 1: {len(drt_jobs)} DRT jobs...", flush=True)
    if workers == 1:
        for job in drt_jobs:
            d, lam, c, res = drt_job(job)
            drt_cache[(d, lam, c)] = res
    else:
        from pipeline._worker import limit_blas_threads
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=limit_blas_threads) as ex:
            for fut in as_completed([ex.submit(drt_job, j) for j in drt_jobs]):
                d, lam, c, res = fut.result()
                drt_cache[(d, lam, c)] = res
                print(f"  DRT done {d} lambda={lam:.1e} "
                      f"({time.time() - t0:.0f}s)", flush=True)

    combos = list(product(rbf_ders, lambdas, hf_weights, caps))
    fit_jobs = [(combo, c, drt_cache[(combo[0], combo[1], c)],
                 settings, L_m, D_m)
                for combo in combos for c in conditions]
    print(f"Phase 2: {len(fit_jobs)} fit jobs...", flush=True)
    raw: dict[tuple, dict] = {}
    if workers == 1:
        for job in fit_jobs:
            combo, cond, pk, sm = fit_job(job)
            raw[(combo, cond)] = {"peaks": pk, "summary": sm}
    else:
        from pipeline._worker import limit_blas_threads
        done = 0
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=limit_blas_threads) as ex:
            for fut in as_completed([ex.submit(fit_job, j) for j in fit_jobs]):
                combo, cond, pk, sm = fut.result()
                raw[(combo, cond)] = {"peaks": pk, "summary": sm}
                done += 1
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
    >>> #     --lambdas 1e-4 --caps 4 --workers 1
    """
    parser = argparse.ArgumentParser(
        description="Physics-based calibration of the stage-3 DRT parameters.")
    parser.add_argument("--sample", required=True,
                        help="sample folder name (repo-root relative)")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="condition folder names; default: every condition "
                             "found under the sample's spectra folders")
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
        conditions = sorted({d.name for root in ("ISM validation",
                                                 "input_spectra")
                             if (sample_dir / root).is_dir()
                             for d in (sample_dir / root).iterdir()
                             if d.is_dir()})
        if not conditions:
            sys.exit(f"no conditions found under {sample_dir}")

    settings, L_m, D_m = _sample_settings(args.session, args.sample)
    df = run_grid(sample_dir, conditions, args.rbf_ders, args.lambdas,
                  args.hf_weights, args.caps, settings, L_m, D_m,
                  args.min_track_points, args.workers)

    out_dir = args.output / args.sample
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_csv = out_dir / "calibrate_drt_ranking.csv"
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
