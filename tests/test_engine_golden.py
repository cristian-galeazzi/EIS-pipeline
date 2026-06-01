"""
Golden-master regression tests for the EIS calculation engine.

Purpose
-------
The pipeline relies on the engine (``drt.py`` / ``fitting.py``) staying
numerically frozen. These tests turn that guarantee into automatic checks so an
accidental change is caught without a manual md5 inspection.

They cover:
1. C_eff = τ/R is exact for the Zarc parametrisation (the documented identity).
2. ``fit_zarc`` recovers the parameters of a synthetic single-Zarc spectrum.
3. Saved ``stage3_fit.xlsx`` rows still satisfy C_eff_i == τ_i / R_i
   (skipped automatically when no sample data is present).

Run with either::

    pytest tests/test_engine_golden.py
    python tests/test_engine_golden.py     # no pytest needed

References: Wilson et al., "Best Practices for Scientific Computing" (2014) —
"turn bugs into test cases"; Scopatz & Huff, "Effective Computation in Physics".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.fitting import fit_zarc, conductivity, build_circuit_string


def _synthetic_zarc(R0, R, tau, alpha, freq):
    """Z(ω) = R0 + R / (1 + (jωτ)^α) for one Zarc element."""
    w = 2 * np.pi * freq
    Z = R0 + R / (1.0 + (1j * w * tau) ** alpha)
    return Z.real, -Z.imag  # store -Z'' (pipeline sign convention)


def test_ceff_identity_exact():
    """C_eff returned by the fit equals τ/R to floating-point precision."""
    freq = np.logspace(5, -1, 60)
    R0, R, tau, alpha = 20.0, 5000.0, 1e-3, 0.85
    Z_re, Z_im = _synthetic_zarc(R0, R, tau, alpha, freq)
    peaks = [{"R_approx": R, "tau": tau, "peak_id": 1}]
    fit = fit_zarc(freq, Z_re, Z_im, peaks, R0_guess=R0, r0_max=200)
    c_eff = fit["C_eff"][0]
    assert np.isclose(c_eff, fit["tau"][0] / fit["R"][0], rtol=1e-9), \
        f"C_eff identity broken: {c_eff} != tau/R"


def test_fit_recovers_synthetic_params():
    """fit_zarc recovers R, τ, α of a clean single-Zarc spectrum."""
    freq = np.logspace(5, -1, 80)
    R0, R, tau, alpha = 15.0, 8000.0, 2e-3, 0.80
    Z_re, Z_im = _synthetic_zarc(R0, R, tau, alpha, freq)
    peaks = [{"R_approx": R, "tau": tau, "peak_id": 1}]
    fit = fit_zarc(freq, Z_re, Z_im, peaks, R0_guess=R0, r0_max=200)
    assert fit["converged"], f"fit failed: {fit.get('fit_error')}"
    assert np.isclose(fit["R"][0],     R,     rtol=0.05), fit["R"][0]
    assert np.isclose(fit["tau"][0],   tau,   rtol=0.10), fit["tau"][0]
    assert np.isclose(fit["alpha"][0], alpha, rtol=0.05), fit["alpha"][0]
    assert fit["rmse_rel"] < 1e-3, fit["rmse_rel"]


def test_circuit_string():
    assert build_circuit_string(2, include_r0=True) == "R0-Zarc1-Zarc2"
    assert build_circuit_string(2, include_r0=False) == "Zarc1-Zarc2"


def test_conductivity_formula():
    # σ = L / (R·A),  A = π(D/2)²
    L, D, R = 1.26e-3, 10.98e-3, 1000.0
    A = np.pi * (D / 2) ** 2
    assert np.isclose(conductivity(R, L, D), L / (R * A), rtol=1e-12)


def test_saved_fit_ceff_invariant():
    """Every saved Peaks row must satisfy C_eff_i == τ_i / R_i (engine drift guard)."""
    import pandas as pd
    files = list(_ROOT.glob("*/Results/*/stage3_fit.xlsx"))
    if not files:
        print("  [SKIP] no stage3_fit.xlsx sample data present")
        return
    checked = 0
    for f in files:
        df = pd.read_excel(f, sheet_name="Peaks")
        ok = np.isclose(df["C_eff_i"], df["tau_i"] / df["R_i"], rtol=1e-6)
        assert ok.all(), f"C_eff != tau/R in {f.name} rows {np.where(~ok)[0].tolist()}"
        checked += len(df)
    print(f"  C_eff invariant holds across {checked} saved peak rows in {len(files)} file(s)")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"ERROR {t.__name__}:")
            traceback.print_exc()
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
