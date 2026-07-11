"""Shared helpers for the audit calibration scripts.

Spectrum loading (stage-2 selected or raw directory), peak capping,
temperature tracking of fitted peaks, and the physics score used to rank
parameter combinations. Kept private to audit/ (leading underscore): the
public entry points are the scripts that import from here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import linregress

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pipeline.drt import clip_spectrum
from pipeline.ingest import load_csv_spectrum, load_ism

K_B_EV = 8.617333e-5  # Boltzmann constant [eV/K]

# Notebook defaults, overridable per sample via session.json stage3_params.
DEFAULTS = {
    "peak_height_frac": 0.05,
    "peak_min_dist_decades": 0.3,
    "include_r0": False,
    "r0_max": 200.0,
    "n_restarts": 4,
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

    >>> spectra = load_condition_spectra(REPO / "EXAMPLE_SAMPLE",
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
        if records:
            # a condition duplicated in both source folders would otherwise
            # contribute every temperature twice
            break
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
    """Physics score of one fitted condition.

    Ranks Arrhenius linearity of tau(T) and R(T) along tracked peaks
    (weighted by track resistance), coverage of total R by long tracks,
    convergence, and peak-count stability. See calibrate_drt.py for why this
    replaces rmse as the ranking criterion.

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


def sample_settings(session_path: Path, sample_id: str) -> tuple[dict, float, float]:
    """Merge session.json stage3_params over the notebook defaults.

    Geometry falls back to dummy values when the sample has no session entry:
    the rankings only use R, tau, alpha and convergence, so the conductivity
    columns computed from L_m/D_m never influence a score.

    >>> settings, L_m, D_m = sample_settings(Path("does_not_exist.json"), "X")
    >>> settings["R_dec"], L_m
    (0.7, 0.001)
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


def default_min_track_points(n_temperatures: int) -> int:
    """Track length required to enter the score: 2/3 of the series.

    A physical track must span most temperatures, but demanding all of them
    would punish one failed fit too hard.

    >>> default_min_track_points(9), default_min_track_points(5)
    (6, 3)
    """
    return max(3, round(2 * n_temperatures / 3))
