"""Golden-master tests for the Stage 5 global MIEC conductivity model.

These mirror the engine golden tests: they pin the numerical behaviour of
``pipeline/model.py`` so an accidental change is caught automatically.

They cover:
1. recovery of the six parameters from a synthetic surface;
2. the pure-ionic degenerate case (no electronic channels);
3. non-negativity of the fitted prefactors;
4. the closed-form stoichiometric-pO2 minimum;
5. that the global fit uses every selected point (no per-temperature dropping);
6. a descriptive error when too few points are supplied;
7. the reduced fit with an operator-excluded channel (channels=("ion", "p")).

Run with either::

    pytest tests/test_model_golden.py
    python tests/test_model_golden.py     # no pytest needed
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.model import (
    ModelParams,
    fit_global_conductivity,
    stoichiometric_pO2,
    total_conductivity,
)

# A truth set where each channel dominates somewhere in the window, so all six
# parameters are identifiable: n-type at low pO2, p-type at high pO2, ionic mid.
_TRUTH = ModelParams(
    sigma0_ion=1.0e7, Ea_ion=0.90,
    sigma0_p=1.0e8, Ea_p=1.10,
    sigma0_n=1.0e6, Ea_n=1.30, x=0.25,
)
_T_C = np.array([475, 525, 575, 625, 675], dtype=float)
_PO2 = np.logspace(-5, 1, 7)  # bar


def _make_df(p: ModelParams, noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic Peaks-style frame for one peak from the model + optional noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for tc in _T_C:
        for po2 in _PO2:
            sig = float(total_conductivity(po2, tc + 273.15, p))
            sig *= 1.0 + rng.normal(0.0, noise) if noise else 1.0
            rows.append({"peak_id": 1, "T_nominal": tc, "pO2_mean": po2, "sigma_Sm_i": sig})
    return pd.DataFrame(rows)


def test_recovers_synthetic_6params():
    """Clean synthetic surface: the fit recovers the six parameters."""
    out = fit_global_conductivity(_make_df(_TRUTH), x=0.25)
    p = out["params"]
    assert out["converged"], "polish did not converge"
    assert out["r2"] > 0.999, f"R2 too low on clean data: {out['r2']}"
    for name, true in [("Ea_ion", 0.90), ("Ea_p", 1.10), ("Ea_n", 1.30)]:
        assert np.isclose(getattr(p, name), true, atol=0.05), f"{name}={getattr(p, name)}"
    for name, true in [("sigma0_ion", 1.0e7), ("sigma0_p", 1.0e8), ("sigma0_n", 1.0e6)]:
        assert np.isclose(getattr(p, name), true, rtol=0.15), f"{name}={getattr(p, name)}"
    # all three channels are present, so every uncertainty must be finite
    assert all(np.isfinite(v) for v in out["perr"].values()), out["perr"]


def test_pure_ionic_degenerate():
    """No electronic channels: flat in pO2; ionic recovered, p/n driven to ~0."""
    truth = ModelParams(sigma0_ion=1.0e7, Ea_ion=0.90,
                        sigma0_p=0.0, Ea_p=1.0, sigma0_n=0.0, Ea_n=1.0, x=0.25)
    out = fit_global_conductivity(_make_df(truth), x=0.25)
    p = out["params"]
    assert out["r2"] > 0.999, out["r2"]
    assert np.isclose(p.Ea_ion, 0.90, atol=0.05), p.Ea_ion
    assert np.isclose(p.sigma0_ion, 1.0e7, rtol=0.15), p.sigma0_ion
    assert p.sigma0_p < 0.01 * p.sigma0_ion, p.sigma0_p
    assert p.sigma0_n < 0.01 * p.sigma0_ion, p.sigma0_n
    # the present (ionic) channel keeps a finite error bar; absent channels are
    # NaN (unidentifiable), and must NOT poison the ionic uncertainty
    assert np.isfinite(out["perr"]["Ea_ion"]), out["perr"]
    assert np.isfinite(out["perr"]["sigma0_ion"]), out["perr"]
    assert np.isnan(out["perr"]["Ea_p"]) and np.isnan(out["perr"]["Ea_n"]), out["perr"]


def test_sigma0_nonnegative():
    """Even on noisy data the fitted prefactors are never negative."""
    out = fit_global_conductivity(_make_df(_TRUTH, noise=0.05, seed=3), x=0.25)
    p = out["params"]
    assert p.sigma0_ion >= 0 and p.sigma0_p >= 0 and p.sigma0_n >= 0


def test_stoichiometric_pO2_formula():
    """pO2_min(T) matches the closed form (sigma_n == sigma_p crossover)."""
    T_K = 800.0 + 273.15
    expected = ((_TRUTH.sigma0_n / _TRUTH.sigma0_p)
                * np.exp(-(_TRUTH.Ea_n - _TRUTH.Ea_p) / (8.617e-5 * T_K))) ** (1.0 / (2 * _TRUTH.x))
    got = float(stoichiometric_pO2(_TRUTH, T_K))
    assert np.isclose(got, expected, rtol=1e-9), f"{got} != {expected}"


def test_uses_all_points():
    """A temperature with only 2 pO2 points still enters the fit (no dropping)."""
    df = _make_df(_TRUTH)
    # keep every point at the other temperatures, but only 2 pO2 at 475 C
    sparse = df[(df["T_nominal"] != 475) | (df["pO2_mean"].isin(_PO2[:2]))].reset_index(drop=True)
    out = fit_global_conductivity(sparse, x=0.25)
    assert out["n_points"] == len(sparse), f"{out['n_points']} != {len(sparse)}"
    assert (out["residuals"]["T_C"] == 475).sum() == 2, "the sparse temperature was dropped"


def test_two_channel_fit_excludes_n():
    """channels=("ion","p"): n is absent by construction, ion/p still recovered."""
    truth = ModelParams(sigma0_ion=1.0e7, Ea_ion=0.90,
                        sigma0_p=1.0e8, Ea_p=1.10,
                        sigma0_n=0.0, Ea_n=float("nan"), x=0.25)
    out = fit_global_conductivity(_make_df(truth), x=0.25, channels=("ion", "p"))
    p = out["params"]
    assert out["r2"] > 0.999, out["r2"]
    assert np.isclose(p.Ea_ion, 0.90, atol=0.05), p.Ea_ion
    assert np.isclose(p.Ea_p, 1.10, atol=0.05), p.Ea_p
    assert np.isclose(p.sigma0_ion, 1.0e7, rtol=0.15), p.sigma0_ion
    assert np.isclose(p.sigma0_p, 1.0e8, rtol=0.15), p.sigma0_p
    # the excluded channel is marked absent, not fitted to a leftover seed
    assert p.sigma0_n == 0.0 and np.isnan(p.Ea_n), (p.sigma0_n, p.Ea_n)
    assert np.isnan(out["perr"]["sigma0_n"]) and np.isnan(out["perr"]["Ea_n"]), out["perr"]
    assert np.isfinite(out["perr"]["Ea_ion"]) and np.isfinite(out["perr"]["Ea_p"]), out["perr"]
    # without both electronic channels there is no conductivity minimum
    assert np.isnan(float(stoichiometric_pO2(p, 900.0 + 273.15)))
    # and the excluded channel cannot poison the forward model with NaN
    assert np.all(np.isfinite(total_conductivity(_PO2, 900.0 + 273.15, p)))


def test_too_few_points_raises():
    """Below the minimum point count the fit raises a descriptive ValueError."""
    df = _make_df(_TRUTH).head(4)
    try:
        fit_global_conductivity(df, x=0.25)
    except ValueError as exc:
        assert "points" in str(exc).lower()
    else:
        raise AssertionError("expected ValueError for too few points")


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
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
