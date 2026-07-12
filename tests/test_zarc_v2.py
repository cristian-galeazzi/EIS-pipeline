"""Tests for the v2 Zarc engine (pipeline/fitting.py::fit_zarc).

Covers, in this order: the analytic Jacobian against central finite
differences (the load-bearing piece), exact parameter recovery on noiseless
synthetic spectra, output-schema parity with v1, the C_eff = tau/R identity,
fix_params pinning, determinism under seeding, and the weighting modes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audit.fitting_v2.v1_reference import fit_zarc_v1
from pipeline.fitting import (
    _to_internal,
    _to_linear,
    fit_zarc,
    zarc_model,
    zarc_model_jac,
)

RNG = np.random.default_rng(20260711)
FREQ = np.logspace(6, -1, 40)


def _random_params(n_peaks: int, include_r0: bool) -> np.ndarray:
    params = [float(RNG.uniform(1, 200))] if include_r0 else []
    for _ in range(n_peaks):
        params += [float(10 ** RNG.uniform(1, 5)),      # R
                   float(10 ** RNG.uniform(-6, -1)),    # tau
                   float(RNG.uniform(0.55, 0.98))]      # alpha
    return np.array(params)


@pytest.mark.parametrize("n_peaks,include_r0",
                         [(1, False), (1, True), (2, False), (3, True),
                          (4, False)])
def test_jacobian_vs_finite_differences(n_peaks, include_r0):
    """Max relative error of the analytic Jacobian < 1e-6 (gate criterion)."""
    for _ in range(5):
        params = _random_params(n_peaks, include_r0)
        x = _to_internal(params, include_r0)
        Z0, dZ = zarc_model_jac(FREQ, params, include_r0)
        scale = np.max(np.abs(dZ))
        for k in range(len(x)):
            h = 1e-6 * max(1.0, abs(x[k]))
            xp, xm = x.copy(), x.copy()
            xp[k] += h
            xm[k] -= h
            Zp = zarc_model(FREQ, _to_linear(xp, include_r0), include_r0)
            Zm = zarc_model(FREQ, _to_linear(xm, include_r0), include_r0)
            num = (Zp - Zm) / (2 * h)
            err = np.max(np.abs(num - dZ[:, k])) / scale
            assert err < 1e-6, f"param {k}: rel err {err:.2e}"


def test_model_matches_v1_predict():
    """zarc_model must equal impedance.py's evaluation of the same circuit."""
    from impedance.models.circuits import CustomCircuit
    params = [120.0, 8e3, 2e-6, 0.92, 2.5e4, 3e-4, 0.88]
    c = CustomCircuit("R0-Zarc1-Zarc2", initial_guess=params)
    c.parameters_ = np.array(params)
    Z_ref = c.predict(FREQ, use_initial=False)
    Z_new = zarc_model(FREQ, np.array(params), include_r0=True)
    assert np.allclose(Z_new, Z_ref, rtol=1e-12)


def _synth(params: np.ndarray, include_r0: bool,
           noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    Z = zarc_model(FREQ, params, include_r0)
    if noise:
        rng = np.random.default_rng(7)
        Z = Z * (1 + rng.uniform(-noise, noise, len(Z)))
    return Z.real, -Z.imag   # IsmRecord convention


def test_noiseless_recovery_two_zarc():
    true = np.array([8e3, 2e-6, 0.92, 2.5e4, 3e-4, 0.88])
    Z_re, Z_im = _synth(true, include_r0=False)
    peaks = [{"R_approx": 5e3, "tau": 4e-6}, {"R_approx": 4e4, "tau": 1e-4}]
    out = fit_zarc(FREQ, Z_re, Z_im, peaks, include_r0=False,
                      R_dec=1.0, tau_dec=1.0, alpha_init=0.7)
    assert out["converged"]
    assert out["rmse_rel"] < 1e-8
    fitted = np.concatenate([[r, t, a] for r, t, a in
                             zip(out["R"], out["tau"], out["alpha"])])
    assert np.allclose(fitted, true, rtol=1e-6)


def test_output_schema_matches_v1():
    true = np.array([100.0, 1e4, 1e-4, 0.9])
    Z_re, Z_im = _synth(true, include_r0=True, noise=0.002)
    peaks = [{"R_approx": 8e3, "tau": 2e-4}]
    kw = dict(include_r0=True, r0_max=200.0, n_restarts=2, seed=42)
    v1 = fit_zarc_v1(FREQ, Z_re, Z_im, peaks, **kw)
    v2 = fit_zarc(FREQ, Z_re, Z_im, peaks, **kw)
    assert set(v2) == set(v1)
    assert v2["circuit_str"] == v1["circuit_str"]
    assert v2["param_names"] == list(v1["param_names"])
    assert v2["n_peaks"] == v1["n_peaks"]
    for key in ("R", "tau", "alpha", "C_eff", "params", "conf"):
        assert np.shape(v2[key]) == np.shape(v1[key]), key
    assert v2["Z_fit"].shape == v1["Z_fit"].shape
    assert isinstance(v2["R0"], float)


def test_ceff_identity_exact():
    true = np.array([8e3, 2e-6, 0.92, 2.5e4, 3e-4, 0.88])
    Z_re, Z_im = _synth(true, include_r0=False, noise=0.003)
    peaks = [{"R_approx": 8e3, "tau": 2e-6}, {"R_approx": 2.5e4, "tau": 3e-4}]
    out = fit_zarc(FREQ, Z_re, Z_im, peaks, include_r0=False)
    assert np.allclose(out["C_eff"], out["tau"] / out["R"], rtol=0, atol=0)


def test_fix_params_held_exactly():
    true = np.array([8e3, 2e-6, 0.92, 2.5e4, 3e-4, 0.88])
    Z_re, Z_im = _synth(true, include_r0=False, noise=0.002)
    peaks = [{"R_approx": 8e3, "tau": 2e-6}, {"R_approx": 2.5e4, "tau": 3e-4}]
    fix = {"tau": [None, 2.718e-4], "alpha": [0.9, None]}
    out = fit_zarc(FREQ, Z_re, Z_im, peaks, include_r0=False,
                      fix_params=fix)
    assert out["tau"][1] == 2.718e-4      # exact, no ULP drift
    assert out["alpha"][0] == 0.9
    # params order without R0: [R1, tau1, alpha1, R2, tau2, alpha2]
    assert out["conf"][4] == 0.0          # pinned tau2 has zero CI
    assert out["conf"][2] == 0.0          # pinned alpha1 has zero CI
    assert out["conf"][5] > 0.0           # free alpha2 keeps a real CI
    assert out["converged"]


def test_determinism_same_seed():
    true = np.array([8e3, 2e-6, 0.92, 2.5e4, 3e-4, 0.88])
    Z_re, Z_im = _synth(true, include_r0=False, noise=0.02)
    # deliberately bad seeds so the restarts actually engage
    peaks = [{"R_approx": 1e3, "tau": 5e-5}, {"R_approx": 1e5, "tau": 5e-3}]
    kw = dict(include_r0=False, n_restarts=5, rmse_tol=1e-12, seed=1234)
    a = fit_zarc(FREQ, Z_re, Z_im, peaks, **kw)
    b = fit_zarc(FREQ, Z_re, Z_im, peaks, **kw)
    assert np.array_equal(a["params"], b["params"])
    assert a["rmse_rel"] == b["rmse_rel"]


def test_bounds_respected():
    true = np.array([8e3, 2e-6, 0.92])
    Z_re, Z_im = _synth(true, include_r0=False)
    # seed far from truth with a window that excludes it
    peaks = [{"R_approx": 1e2, "tau": 1e-3}]
    out = fit_zarc(FREQ, Z_re, Z_im, peaks, include_r0=False,
                      R_dec=0.5, tau_dec=0.5)
    assert 1e2 / 10**0.5 <= out["R"][0] <= 1e2 * 10**0.5
    assert 1e-3 / 10**0.5 <= out["tau"][0] <= 1e-3 * 10**0.5


def test_robust_loss_downweights_outliers():
    true = np.array([8e3, 2e-6, 0.92, 2.5e4, 3e-4, 0.88])
    Z_re, Z_im = _synth(true, include_r0=False, noise=0.002)
    # corrupt two low-frequency points by 30%
    Z_re[-2:] *= 1.3
    Z_im[-2:] *= 1.3
    peaks = [{"R_approx": 8e3, "tau": 2e-6}, {"R_approx": 2.5e4, "tau": 3e-4}]
    kw = dict(include_r0=False, R_dec=1.0, tau_dec=1.0)
    lin = fit_zarc(FREQ, Z_re, Z_im, peaks, loss="linear", **kw)
    rob = fit_zarc(FREQ, Z_re, Z_im, peaks, loss="soft_l1",
                      f_scale=0.01, **kw)
    true_R = true[[0, 3]]
    err_lin = np.max(np.abs(lin["R"] - true_R) / true_R)
    err_rob = np.max(np.abs(rob["R"] - true_R) / true_R)
    assert err_rob < err_lin


def test_invalid_loss_rejected():
    with pytest.raises(ValueError):
        fit_zarc(FREQ, FREQ, FREQ, [{"R_approx": 1, "tau": 1}],
                    loss="cauchy")
    with pytest.raises(ValueError):
        fit_zarc(FREQ, FREQ, FREQ, [], include_r0=False)
