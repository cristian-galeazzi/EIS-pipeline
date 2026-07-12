"""A/B harness: v1 vs v2 Zarc engine on the real production inputs.

Method
------
For every fitted condition of every sample, this harness refits each
spectrum with both engines under EXACTLY the production inputs, never
redefined by hand:

- spectra: the stage-2 selected replica per temperature with the saved
  frequency cuts (stage2_kk.xlsx, "Selected" sheet);
- seeds: the saved DRT peaks (stage3_drt.xlsx, "Peaks" sheet), i.e. the very
  R_approx/tau the production fit started from;
- knobs: session.json stage3_params with per-condition and per-temperature
  overrides resolved like the stage-3 notebook (_resolve_zarc_params), plus
  the live-panel per-peak bounds (zarc_peak_bounds) and pinned parameters
  (ZARC_FIX_PARAMS) when present;
- orchestration: the same warm-start chain as fit_condition_batch
  (descending T, warm restart when the previous fit of equal peak count
  converged) and the same crc32(condition|T) restart seeding.

The only degree of freedom between the two columns of the comparison is the
optimizer path. v2 runs with loss="linear": robust losses would change the
estimator, and the gates compare optimizers, not estimators.

Per spectrum the CSV records rmse and convergence for both engines, wall
times, the largest parameter shift in units of the v1 one-sigma confidence
interval (gate G2), and the boundary distance of the closest-to-bound
parameter before/after (gate G4, same geometry as zarc_window_check.py).

PRIVACY: stdout carries only file paths and counts. Every number derived
from measured data goes to the gitignored audit/output/fitting_v2/.

Usage (from the repository root, on the fitting-v2-prototype branch):
  .venv/bin/python audit/fitting_v2/ab_harness.py            # all samples
  .venv/bin/python audit/fitting_v2/ab_harness.py --samples MY_SAMPLE
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audit._common import load_condition_spectra
from audit.zarc_window_check import bounds_from_seed, edge_distance
from audit.fitting_v2.v1_reference import fit_zarc_v1
from pipeline.fitting import (fit_zarc, resolve_condition_entry,
                              resolve_peak_windows)

PIN_MARGIN = 0.15   # same "pinned" threshold as zarc_window_check (gate G4)

CSV_FIELDS = ["condition", "T_nominal", "n_peaks",
              "rmse_v1", "rmse_v2", "conv_v1", "conv_v2",
              "wall_v1_s", "wall_v2_s",
              "max_ci_shift", "max_shift_dec", "max_alpha_shift",
              "edge_frac_v1", "edge_frac_v2",
              "pinned_v1", "pinned_v2"]


def resolve_zarc_params(entry: dict, condition: str, T_int: int) -> dict:
    """Stage-3 knob resolution: per-T override, per-condition, then defaults.

    Mirrors the notebook's _resolve_zarc_params.

    >>> e = {"stage3_params": {"ZARC_R_DEC": 0.7, "ZARC_TAU_DEC": 0.7,
    ...                        "ZARC_ALPHA_INIT": 0.7, "ZARC_HF_WEIGHT": 0.0},
    ...      "condition_params": {"c1": {"R_dec": 0.5}}}
    >>> resolve_zarc_params(e, "c1", 600)["R_dec"]
    0.5
    """
    p3 = entry.get("stage3_params", {})
    cond_ov = resolve_condition_entry(entry.get("condition_params", {}),
                                      condition)
    t_ov = cond_ov.get(str(T_int), cond_ov.get(T_int, {}))
    if not isinstance(t_ov, dict):
        t_ov = {}
    return {
        "R_dec":      t_ov.get("R_dec",      cond_ov.get("R_dec",      p3.get("ZARC_R_DEC", 1.5))),
        "tau_dec":    t_ov.get("tau_dec",    cond_ov.get("tau_dec",    p3.get("ZARC_TAU_DEC", 1.5))),
        "alpha_init": t_ov.get("alpha_init", cond_ov.get("alpha_init", p3.get("ZARC_ALPHA_INIT", 0.8))),
        "hf_weight":  t_ov.get("hf_weight",  cond_ov.get("hf_weight",  p3.get("ZARC_HF_WEIGHT", 0.0))),
    }


def _per_peak(store: dict, condition: str, T_int: int, name: str,
              n_peaks: int, scalar):
    """Legacy per-(condition, T) bound list when saved and length-consistent."""
    t_map = store.get(condition, {}) or {}
    pp = t_map.get(str(T_int), t_map.get(T_int))
    if pp and isinstance(pp.get(name), (list, tuple)) \
            and len(pp[name]) == n_peaks:
        return list(pp[name])
    return scalar


def build_tasks(entry: dict, condition: str, spectra: list[dict],
                seeds_by_T: dict[int, list[dict]]) -> list[dict]:
    """Production-equivalent task list for one condition, T descending.

    >>> # See tests/test_fitting_v2_ab.py for a runnable example.
    """
    p3 = entry.get("stage3_params", {})
    peak_bounds = entry.get("zarc_peak_bounds", {}) or {}
    peak_windows = entry.get("zarc_peak_windows", {}) or {}
    fix_all = p3.get("ZARC_FIX_PARAMS", {}) or {}
    tasks = []
    for sp in spectra:
        T = sp["T_nominal"]
        peaks = seeds_by_T.get(T)
        if not peaks:
            continue
        zp = resolve_zarc_params(entry, condition, T)
        n = len(peaks)
        fix_map = fix_all.get(condition, {}) or {}
        # per-peak windows exactly as the stage-3 batch resolves them
        # (peak_id-keyed maps, legacy per-(condition, T) lists honoured)
        r_dec, tau_dec = resolve_peak_windows(
            peaks, condition, T,
            windows=peak_windows, legacy=peak_bounds,
            r_dec_default=zp["R_dec"], tau_dec_default=zp["tau_dec"])
        tasks.append({
            **sp, "peaks": peaks,
            "R_dec": r_dec,
            "tau_dec": tau_dec,
            "alpha_init": zp["alpha_init"],
            "alpha_min": _per_peak(peak_bounds, condition, T, "alpha_min",
                                   n, 0.5),
            "alpha_max": _per_peak(peak_bounds, condition, T, "alpha_max",
                                   n, 1.0),
            "hf_weight": zp["hf_weight"],
            "fix_params": fix_map.get(str(T), fix_map.get(T)),
        })
    return tasks


def run_condition_pair(condition: str, tasks: list[dict],
                       include_r0: bool, r0_max: float | None,
                       n_restarts: int, rmse_tol: float,
                       v2_loss: str = "linear",
                       v2_f_scale: float = 1.0) -> list[dict]:
    """Fit every task with both engines, replicating the warm-start chain.

    Each engine keeps its OWN warm-start state, exactly as it would in its
    own production run of fit_condition_batch. v2_loss/v2_f_scale are passed
    to the v2 engine (pipeline.fitting.fit_zarc) only: v1 has no robust-loss option, so the comparison
    stays meaningful (v1 = plain L2 baseline vs v2 under the chosen loss).
    """
    engines = {"v1": fit_zarc_v1, "v2": fit_zarc}
    prev: dict[str, dict | None] = {"v1": None, "v2": None}
    rows: list[dict] = []
    for t in tasks:
        T = t["T_nominal"]
        seed = zlib.crc32(f"{condition}|{T}".encode())
        rec: dict = {"condition": condition, "T_nominal": T,
                     "n_peaks": len(t["peaks"])}
        fits: dict[str, dict] = {}
        for name, fn in engines.items():
            p = prev[name]
            if p is not None and len(p["R"]) == len(t["peaks"]):
                peaks_seeded = [{**pk, "R_approx": float(p["R"][i]),
                                 "tau": float(p["tau"][i])}
                                for i, pk in enumerate(t["peaks"])]
                R0_seed = float(p["R0"])
            else:
                peaks_seeded, R0_seed = t["peaks"], None
            extra = ({"loss": v2_loss, "f_scale": v2_f_scale}
                    if name == "v2" else {})
            t1 = time.time()
            fit = fn(t["freq"], t["Z_re"], t["Z_im"], peaks_seeded,
                     R0_guess=R0_seed, R_dec=t["R_dec"],
                     tau_dec=t["tau_dec"], alpha_init=t["alpha_init"],
                     alpha_min=t["alpha_min"], alpha_max=t["alpha_max"],
                     include_r0=include_r0, r0_max=r0_max,
                     fix_params=t["fix_params"], hf_weight=t["hf_weight"],
                     n_restarts=n_restarts, rmse_tol=rmse_tol, seed=seed,
                     **extra)
            rec[f"wall_{name}_s"] = round(time.time() - t1, 3)
            rec[f"rmse_{name}"] = round(float(fit["rmse_rel"]), 6)
            rec[f"conv_{name}"] = bool(fit["converged"])
            fits[name] = fit
            if fit["converged"]:
                prev[name] = fit
            # G4: distance of the closest R/tau parameter to its window edge
            # (windows around the ORIGINAL DRT seeds, as in production)
            edges = []
            for i, pk in enumerate(t["peaks"]):
                for key, dec in (("R", t["R_dec"]), ("tau", t["tau_dec"])):
                    dec_i = dec[i] if isinstance(dec, (list, tuple)) else dec
                    seed_v = pk["R_approx"] if key == "R" else pk["tau"]
                    lo, hi = bounds_from_seed(seed_v, dec_i)
                    edges.append(edge_distance(float(fit[key][i]), lo, hi))
            edge = float(np.nanmin(edges)) if edges else float("nan")
            rec[f"edge_frac_{name}"] = round(edge, 4)
            rec[f"pinned_{name}"] = bool(edge <= PIN_MARGIN)
        # G2: parameter shift in units of the v1 one-sigma CI
        v1, v2 = fits["v1"], fits["v2"]
        conf = np.asarray(v1["conf"], dtype=float)
        p1 = np.asarray(v1["params"], dtype=float)
        p2 = np.asarray(v2["params"], dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            shifts = np.abs(p2 - p1) / conf
        shifts = shifts[np.isfinite(shifts)]
        rec["max_ci_shift"] = (round(float(np.max(shifts)), 4)
                               if len(shifts) else float("nan"))
        # physical size of the disagreement: decades on R/tau (and R0 when
        # fitted), linear on alpha; the CI-based measure alone exaggerates
        # wherever the v1 CI is tiny or invalid (parameter on a bound)
        dec = [abs(np.log10(b / a)) for a, b in
               [*zip(v1["R"], v2["R"]), *zip(v1["tau"], v2["tau"])]
               if a > 0 and b > 0]
        if include_r0 and v1["R0"] > 0 and v2["R0"] > 0:
            dec.append(abs(np.log10(v2["R0"] / v1["R0"])))
        rec["max_shift_dec"] = round(max(dec), 4) if dec else float("nan")
        rec["max_alpha_shift"] = round(
            float(np.max(np.abs(np.asarray(v2["alpha"])
                                - np.asarray(v1["alpha"])))), 4)
        rows.append(rec)
    return rows


def run_sample(sample_dir: Path, session_path: Path,
              v2_loss: str = "linear",
              v2_f_scale: float = 1.0) -> list[dict]:
    """All fitted conditions of one sample; returns the per-spectrum rows.

    v2_loss/v2_f_scale let v2 use a robust loss (soft_l1, huber) instead of
    the default plain L2, to test whether down-weighting noisy points
    changes the picture on the degenerate low-T fits. v1 is unaffected: it
    has no robust-loss option, so it stays the fixed baseline.

    >>> # See tests/test_fitting_v2_ab.py for a runnable example.
    """
    try:
        cfg = json.loads(session_path.read_text())
        entry = next((s for s in cfg
                      if s.get("sample_id") == sample_dir.name), {})
    except (OSError, ValueError):
        entry = {}
    p3 = entry.get("stage3_params", {})
    include_r0 = p3.get("ZARC_INCLUDE_R0", False)
    r0_max = p3.get("ZARC_R0_MAX")
    n_restarts = p3.get("ZARC_N_RESTARTS", 5)
    rmse_tol = p3.get("ZARC_RMSE_TOL", 0.02)

    rows: list[dict] = []
    results_dir = sample_dir / "Results"
    if not results_dir.is_dir():
        return rows
    for cond_dir in sorted(d for d in results_dir.iterdir() if d.is_dir()):
        drt_xlsx = cond_dir / "stage3_drt.xlsx"
        if not (drt_xlsx.exists()
                and (cond_dir / "stage2_kk.xlsx").exists()):
            continue
        condition = cond_dir.name
        try:
            drt = pd.read_excel(drt_xlsx, sheet_name="Peaks")
            spectra = load_condition_spectra(sample_dir, condition)
        except Exception as exc:
            print(f"[WARN] condition skipped: {exc}", file=sys.stderr)
            continue
        seeds_by_T: dict[int, list[dict]] = {}
        for T, grp in drt.groupby("T_nominal"):
            seeds_by_T[int(T)] = [
                {"peak_id": int(r["peak_id"]), "tau": float(r["tau"]),
                 "R_approx": float(r["R_approx"])}
                for _, r in grp.sort_values("peak_id").iterrows()]
        tasks = build_tasks(entry, condition, spectra, seeds_by_T)
        if not tasks:
            continue
        rows.extend(run_condition_pair(condition, tasks, include_r0,
                                       r0_max, n_restarts, rmse_tol,
                                       v2_loss, v2_f_scale))
        print(f"  condition done: {len(tasks)} spectra "
              f"({len(rows)} total)", flush=True)
    return rows


def gate_table(rows: list[dict]) -> dict[str, str]:
    """G1/G2/G4 verdicts from the per-spectrum rows (counts only).

    >>> r = {"rmse_v1": 0.02, "rmse_v2": 0.019, "conv_v1": True,
    ...      "conv_v2": True, "max_ci_shift": 0.5,
    ...      "pinned_v1": False, "pinned_v2": False}
    >>> gate_table([r])["G1"].startswith("PASS")
    True
    """
    n = len(rows)
    better = sum(1 for r in rows if r["rmse_v2"] <= r["rmse_v1"])
    worst_ratio = max((r["rmse_v2"] / r["rmse_v1"]
                       for r in rows if r["rmse_v1"] > 0), default=1.0)
    g1 = (better / n >= 0.95) and (worst_ratio <= 1.05)
    clean = [r for r in rows
             if r["conv_v1"] and np.isfinite(r["max_ci_shift"])]
    within = sum(1 for r in clean if r["max_ci_shift"] <= 1.0)
    g2 = within == len(clean)
    pin1 = sum(1 for r in rows if r["pinned_v1"])
    pin2 = sum(1 for r in rows if r["pinned_v2"])
    g4 = pin2 <= pin1
    return {
        "G1": (f"{'PASS' if g1 else 'FAIL'}: rmse_v2 <= rmse_v1 on "
               f"{better}/{n} spectra, worst ratio {worst_ratio:.4f}"),
        "G2": (f"{'PASS' if g2 else 'FAIL'}: {within}/{len(clean)} clean-v1 "
               f"spectra with every parameter within the v1 68% CI"),
        "G4": f"{'PASS' if g4 else 'FAIL'}: pinned fits v1={pin1} v2={pin2}",
    }


def main() -> None:
    """CLI entry point; see the module docstring for the method.

    >>> # .venv/bin/python audit/fitting_v2/ab_harness.py
    """
    parser = argparse.ArgumentParser(
        description="Paired v1/v2 refit of the production dataset.")
    parser.add_argument("--samples", nargs="+", default=None)
    parser.add_argument("--loss", default="linear",
                        choices=["linear", "soft_l1", "huber"],
                        help="v2 loss; linear reproduces the baseline "
                             "comparison, soft_l1/huber down-weight points "
                             "whose relative misfit exceeds --f-scale")
    parser.add_argument("--f-scale", type=float, default=1.0,
                        help="relative-residual scale where the robust loss "
                             "engages; ~1-2x the rmse_rel of your clean fits "
                             "(irrelevant when --loss linear)")
    parser.add_argument("--session", type=Path, default=REPO / "session.json")
    parser.add_argument("--output", type=Path,
                        default=REPO / "audit" / "output" / "fitting_v2")
    args = parser.parse_args()

    if args.samples is not None:
        sample_dirs = [REPO / s for s in args.samples]
    else:
        sample_dirs = sorted(
            d for d in REPO.iterdir()
            if d.is_dir() and (d / "Results").is_dir()
            and d.name != "sample_template")

    # robust-loss runs get their own files so the linear baseline CSVs
    # are never overwritten by an experiment
    tag = "" if args.loss == "linear" else f"_{args.loss}_{args.f_scale:g}"
    all_rows: list[dict] = []
    for sample_dir in sample_dirs:
        print(f"sample: {sample_dir.name}", flush=True)
        rows = run_sample(sample_dir, args.session,
                          v2_loss=args.loss, v2_f_scale=args.f_scale)
        if not rows:
            print("  (no fitted conditions found)")
            continue
        try:
            args.output.mkdir(parents=True, exist_ok=True)
            out_csv = args.output / f"ab_{sample_dir.name}{tag}.csv"
            with out_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            sys.exit(f"cannot write results: {exc}")
        print(f"  saved {len(rows)} rows -> {out_csv}")
        all_rows.extend(rows)

    if not all_rows:
        sys.exit("nothing to compare: no sample with stage-3 results")

    t1 = sum(r["wall_v1_s"] for r in all_rows)
    t2 = sum(r["wall_v2_s"] for r in all_rows)
    print(f"\nspectra compared: {len(all_rows)}")
    print(f"total wall time: v1 {t1:.1f}s, v2 {t2:.1f}s")
    print("\nGate table (G3 comes from synthetic_gate.py, G5 holds by "
          "construction: numpy + scipy only):")
    for gate, verdict in gate_table(all_rows).items():
        print(f"  {gate}: {verdict}")


if __name__ == "__main__":
    main()
