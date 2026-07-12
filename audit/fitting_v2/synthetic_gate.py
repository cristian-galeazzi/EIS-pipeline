"""Synthetic ground-truth gate for the v2 Zarc engine (gate G3).

Method
------
This is the only place where fitting accuracy is measurable absolutely: the
spectra are generated from known parameters, so "recovery error" is a fact,
not a comparison between two opinions. Both engines fit the same spectra
with the same seeds, the same decade windows, the same weighting and the
same restart budget; the only difference is the optimizer path (v1: linear
space, finite differences, curve_fit; v2: log space, analytic Jacobian,
direct TRF).

Cases: 1 to 4 Zarc elements with decade-spread parameters, plus two
overlapping-tau stress cases (0.8 and 0.5 decades of separation, where the
least-squares valley is narrowest). Noise: multiplicative uniform at three
levels (0.1%, 0.5%, 2%), several seeded replicates each. Seeds mimic the DRT:
R_approx and tau are the true values displaced by up to +-0.3 decades, drawn
once per replicate and shared by both engines.

Recovery error per fit (dimensionless):

    err = mean_k |log10(R_k / R_k_true)| + mean_k |log10(tau_k / tau_k_true)|
        + mean_k |alpha_k - alpha_k_true|

Gate criterion (G3, fixed in fitting_v2_proposal.md before any number was
produced): the v2 median recovery error must be <= v1 at EVERY noise level.

Outputs
-------
audit/output/fitting_v2/synthetic_gate.csv (one row per case, noise,
replicate, engine) and a per-noise summary table on stdout ending with
"G3 PASS" or "G3 FAIL".

Usage (from the repository root, on the fitting-v2-prototype branch):
  .venv/bin/python audit/fitting_v2/synthetic_gate.py
  .venv/bin/python audit/fitting_v2/synthetic_gate.py --replicates 3 --quick
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from audit.fitting_v2.v1_reference import fit_zarc_v1
from pipeline.fitting import fit_zarc, zarc_model

FREQ = np.logspace(6, -1, 60)
NOISE_LEVELS = (0.001, 0.005, 0.02)
SEED_DISPLACE_DEC = 0.3     # DRT-like seed error, up to +-0.3 decades

# (case name, R0 or None, [(R, tau, alpha), ...]); tau separations of the
# stress cases are 0.8 and 0.5 decades, everything else is well separated
CASES: list[tuple[str, float | None, list[tuple[float, float, float]]]] = [
    ("1zarc",        None,  [(1e4, 1e-4, 0.90)]),
    ("2zarc",        None,  [(8e3, 2e-6, 0.92), (2.5e4, 3e-4, 0.88)]),
    ("3zarc_r0",     120.0, [(5e3, 1e-6, 0.95), (2e4, 1e-4, 0.85),
                             (8e4, 1e-2, 0.80)]),
    ("4zarc",        None,  [(3e3, 5e-7, 0.95), (1e4, 3e-5, 0.90),
                             (4e4, 2e-3, 0.85), (1e5, 1e-1, 0.75)]),
    ("overlap_0.8d", None,  [(1e4, 1e-4, 0.90), (1.5e4, 6.3e-4, 0.85)]),
    ("overlap_0.5d", None,  [(1e4, 1e-4, 0.90), (1.5e4, 3.2e-4, 0.85)]),
]

FIT_KW = dict(R_dec=0.7, tau_dec=0.7, alpha_init=0.7, alpha_min=0.5,
              alpha_max=1.0, r0_max=200.0, n_restarts=5, rmse_tol=0.02)


def recovery_error(fit: dict, true: list[tuple[float, float, float]]) -> float:
    """Scalar recovery error of one fit against the ground truth.

    >>> fake = {"R": np.array([1e4]), "tau": np.array([1e-4]),
    ...         "alpha": np.array([0.9])}
    >>> recovery_error(fake, [(1e4, 1e-4, 0.9)])
    0.0
    """
    R_t = np.array([p[0] for p in true])
    tau_t = np.array([p[1] for p in true])
    a_t = np.array([p[2] for p in true])
    return float(np.mean(np.abs(np.log10(fit["R"] / R_t)))
                 + np.mean(np.abs(np.log10(fit["tau"] / tau_t)))
                 + np.mean(np.abs(fit["alpha"] - a_t)))


def make_spectrum(case: tuple, noise: float,
                  rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Noisy synthetic spectrum in the IsmRecord sign convention.

    >>> rng = np.random.default_rng(0)
    >>> Z_re, Z_im = make_spectrum(CASES[0], 0.0, rng)
    >>> bool(Z_im.max() > 0)
    True
    """
    _, R0, procs = case
    params = ([R0] if R0 is not None else []) + [v for p in procs for v in p]
    Z = zarc_model(FREQ, np.array(params, dtype=float), R0 is not None)
    if noise:
        Z = Z * (1 + rng.uniform(-noise, noise, len(Z)))
    return Z.real, -Z.imag


def make_seeds(procs: list[tuple[float, float, float]],
               rng: np.random.Generator) -> list[dict]:
    """DRT-like peak seeds: true values displaced up to +-0.3 decades."""
    return [{"R_approx": R * 10 ** rng.uniform(-SEED_DISPLACE_DEC,
                                               SEED_DISPLACE_DEC),
             "tau": tau * 10 ** rng.uniform(-SEED_DISPLACE_DEC,
                                            SEED_DISPLACE_DEC)}
            for R, tau, _ in procs]


def run_gate(replicates: int, noise_levels: tuple[float, ...],
             cases: list[tuple]) -> tuple[list[dict], dict]:
    """Run both engines over the full grid; returns (rows, medians).

    medians maps (engine, noise) -> median recovery error.

    >>> # See tests/test_fitting_v2_gate.py for a reduced runnable example.
    """
    rows: list[dict] = []
    t0 = time.time()
    for case in cases:
        name, R0, procs = case
        include_r0 = R0 is not None
        for noise in noise_levels:
            for rep in range(replicates):
                # one deterministic stream per (case, noise, rep), shared
                # by both engines: same spectrum, same seeds
                stream = np.random.default_rng(
                    zlib.crc32(f"{name}|{noise}|{rep}".encode()))
                Z_re, Z_im = make_spectrum(case, noise, stream)
                seeds = make_seeds(procs, stream)
                fit_seed = zlib.crc32(f"{name}|{noise}|{rep}|fit".encode())
                for engine, fn in (("v1", fit_zarc_v1), ("v2", fit_zarc)):
                    t1 = time.time()
                    try:
                        fit = fn(FREQ, Z_re, Z_im, seeds,
                                 include_r0=include_r0, seed=fit_seed,
                                 **FIT_KW)
                        err = recovery_error(fit, procs)
                        conv = bool(fit["converged"])
                        rmse = float(fit["rmse_rel"])
                    except Exception as exc:
                        print(f"[ERROR] {name} {noise} rep{rep} {engine}: "
                              f"{exc}", file=sys.stderr)
                        err, conv, rmse = np.inf, False, np.inf
                    rows.append({"case": name, "noise": noise, "rep": rep,
                                 "engine": engine,
                                 "recovery_err": round(err, 6),
                                 "rmse_rel": round(rmse, 6),
                                 "converged": conv,
                                 "wall_s": round(time.time() - t1, 3)})
            print(f"  {name} noise={noise} done ({time.time() - t0:.0f}s)",
                  flush=True)
    medians = {}
    for engine in ("v1", "v2"):
        for noise in noise_levels:
            errs = [r["recovery_err"] for r in rows
                    if r["engine"] == engine and r["noise"] == noise]
            medians[(engine, noise)] = float(np.median(errs))
    return rows, medians


def main() -> None:
    """CLI entry point; see the module docstring for the gate definition.

    >>> # .venv/bin/python audit/fitting_v2/synthetic_gate.py
    """
    parser = argparse.ArgumentParser(
        description="Ground-truth recovery gate: v1 vs v2 Zarc engine.")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--quick", action="store_true",
                        help="first three cases only (smoke run)")
    parser.add_argument("--output", type=Path,
                        default=REPO / "audit" / "output" / "fitting_v2")
    args = parser.parse_args()

    cases = CASES[:3] if args.quick else CASES
    rows, medians = run_gate(args.replicates, NOISE_LEVELS, cases)

    try:
        args.output.mkdir(parents=True, exist_ok=True)
        out_csv = args.output / "synthetic_gate.csv"
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    except OSError as exc:
        sys.exit(f"cannot write results: {exc}")
    print(f"\nsaved {len(rows)} rows -> {out_csv}")

    print(f"\n{'noise':>7} {'median err v1':>14} {'median err v2':>14} "
          f"{'v2<=v1':>7} {'within 1%':>10}")
    g3_literal = True
    g3_amended = True
    for noise in NOISE_LEVELS:
        m1, m2 = medians[("v1", noise)], medians[("v2", noise)]
        ok_lit = m2 <= m1
        ok_amd = m2 <= m1 * 1.01
        g3_literal &= ok_lit
        g3_amended &= ok_amd
        print(f"{noise:>7} {m1:>14.5f} {m2:>14.5f} "
              f"{'yes' if ok_lit else 'NO':>7} {'yes' if ok_amd else 'NO':>10}")
    # wall-time context (not a gate, but the expected payoff)
    for engine in ("v1", "v2"):
        tot = sum(r["wall_s"] for r in rows if r["engine"] == engine)
        print(f"total wall time {engine}: {tot:.1f}s")
    # Two verdicts: the literal pre-registered criterion, and the operator-
    # approved amendment (design.md section 4b): paired non-inferiority with
    # a 1% relative equivalence band. At finite noise the exact LS minimum
    # does not coincide with the ground truth, so sub-band excesses are
    # realization noise, not optimizer quality.
    print(f"\nG3 literal (v2 median <= v1 at every noise): "
          f"{'PASS' if g3_literal else 'FAIL'}")
    print(f"G3 amended (non-inferiority, 1% band; see design.md 4b): "
          f"{'PASS' if g3_amended else 'FAIL'}")


if __name__ == "__main__":
    main()
