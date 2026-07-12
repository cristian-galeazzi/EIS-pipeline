"""
pipeline/zarc_v2.py
===================
Prototype v2 Zarc fitting engine: log-space parametrization, analytic
Jacobian, direct scipy least_squares (TRF), optional robust loss.

This module is a NEW code path. It imports v1 helpers read-only and does not
modify pipeline.fitting in any way; the golden-master suite is untouched.
`fit_zarc_v2` accepts the exact argument set of `pipeline.fitting.fit_zarc`
plus `loss` and `f_scale`, and returns the same output dict schema, so
`fit_to_rows()` consumes either engine unchanged.

Model and parametrization
-------------------------
The circuit is R0 - Zarc1 - ... - ZarcN (R0 optional), with

    Z_k(w) = R_k / (1 + u_k),        u_k = (j w tau_k)^alpha_k

Internally the optimizer works on x = (ln R0?, ln R_k, ln tau_k, alpha_k):
R and tau span decades, so in linear space the least-squares valley is a
long, badly conditioned trench; in log space it is well-conditioned and the
decade box bounds of v1 map to plain intervals:

    ln(R_lo) <= ln R <= ln(R_hi),  same for tau; alpha stays linear.

Analytic Jacobian
-----------------
With p = ln R, q = ln tau, a = alpha and u = (j w tau)^a = (j w)^a e^{a q}:

    dZ/dp = Z                          (Z is linear in R = e^p)
    du/dq = a u                        (only e^{a q} depends on q)
    dZ/dq = -R u a / (1 + u)^2         (chain rule through 1/(1+u))
    du/da = u ln(j w tau)              (u = e^{a ln(j w tau)})
    dZ/da = -R u ln(j w tau) / (1+u)^2

where ln(j w tau) = ln(w tau) + j pi/2 on the principal branch (w, tau > 0).
For the optional series term, dZ/d(ln R0) = R0. The scalar residual vector
stacks weighted real and imaginary parts,

    r = [ Re(Z_model - Z_exp) / sig ; Im(Z_model - Z_exp) / sig ],

and sig is a per-frequency weight, so every Jacobian column is just the
same stacking of dZ/dx / sig: the weighting is linear and commutes with
differentiation. Verified against central finite differences in
tests/test_zarc_v2.py (max relative error < 1e-6).

Weighting semantics (identical to v1)
-------------------------------------
hf_weight > 0 : sig = |Z_exp| / (1 + hf_weight * normalized log10 f)
                (RelaxIS "High freq. modulus" mode)
else, weight_by_modulus=True : sig = |Z_exp|  (proportional weighting)
else : sig = 1  (unit weighting)

Robust loss
-----------
`loss` is passed straight to scipy.optimize.least_squares: "linear" is the
plain L2 of v1; "soft_l1" and "huber" bound the influence of outlier points.
Because the residuals are relative (dimensionless), `f_scale` is the
relative-error scale beyond which a point counts as an outlier; with the
default 1.0 the robust losses are effectively linear for well-fitted
spectra, so set f_scale to a few times the expected relative noise (e.g.
0.01) to actually engage them.

Determinism
-----------
Restart guesses use the same numpy Generator seeding and the same
log-uniform sampling helper as v1 (`_sample_guess`), so a given seed
produces the identical restart sequence in both engines.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import least_squares

from pipeline.fitting import (
    _sample_guess,
    build_bounds,
    build_circuit_string,
    build_initial_guess,
)

_LOSSES = ("linear", "soft_l1", "huber")


def zarc_model(freq: np.ndarray, params: np.ndarray,
               include_r0: bool) -> np.ndarray:
    """Series-Zarc impedance in the physical convention (Im Z < 0 capacitive).

    `params` is the linear-space vector (R0?, R_1, tau_1, alpha_1, ...).

    >>> Z = zarc_model(np.array([1.0]), np.array([100.0, 1.0, 0.9]), False)
    >>> bool(abs(Z[0] - 100 / (1 + (2j * np.pi) ** 0.9)) < 1e-12)
    True
    """
    freq = np.asarray(freq, dtype=float)
    w = 2.0 * np.pi * freq
    off = 1 if include_r0 else 0
    Z = np.full(len(freq), params[0] if include_r0 else 0.0, dtype=complex)
    for k in range(off, len(params), 3):
        R, tau, alpha = params[k], params[k + 1], params[k + 2]
        u = (1j * w * tau) ** alpha
        Z += R / (1.0 + u)
    return Z


def zarc_model_jac(freq: np.ndarray, params: np.ndarray,
                   include_r0: bool) -> tuple[np.ndarray, np.ndarray]:
    """Model and complex Jacobian w.r.t. x = (ln R0?, ln R, ln tau, alpha).

    Returns (Z, dZ) with dZ of shape (n_freq, n_params), column order equal
    to the linear parameter order. See the module docstring for the
    derivation.

    >>> Z, dZ = zarc_model_jac(np.array([10.0, 100.0]),
    ...                        np.array([50.0, 1e3, 1e-3, 0.8]), True)
    >>> dZ.shape
    (2, 4)
    >>> bool(np.allclose(dZ[:, 0], 50.0))   # dZ/dlnR0 = R0
    True
    """
    freq = np.asarray(freq, dtype=float)
    w = 2.0 * np.pi * freq
    n = len(params)
    Z = np.zeros(len(freq), dtype=complex)
    dZ = np.zeros((len(freq), n), dtype=complex)
    off = 0
    if include_r0:
        Z += params[0]
        dZ[:, 0] = params[0]
        off = 1
    for k in range(off, n, 3):
        R, tau, alpha = params[k], params[k + 1], params[k + 2]
        jwt = 1j * w * tau
        u = jwt ** alpha
        denom = (1.0 + u) ** 2
        Zk = R / (1.0 + u)
        # principal branch: ln(j w tau) = ln(w tau) + j pi/2 for w, tau > 0
        ln_jwt = np.log(w * tau) + 1j * (np.pi / 2.0)
        Z += Zk
        dZ[:, k] = Zk
        dZ[:, k + 1] = -R * alpha * u / denom
        dZ[:, k + 2] = -R * u * ln_jwt / denom
    return Z, dZ


def _weights(freq: np.ndarray, Z_exp: np.ndarray, hf_weight: float,
             weight_by_modulus: bool) -> np.ndarray:
    """Per-frequency sigma replicating the v1 weighting modes.

    >>> f = np.array([1.0, 10.0, 100.0])
    >>> Z = np.array([3.0 + 0j, 4.0 + 0j, 5.0 + 0j])
    >>> _weights(f, Z, 0.0, True).tolist()
    [3.0, 4.0, 5.0]
    """
    if hf_weight and hf_weight > 0:
        mod = np.abs(Z_exp)
        lf = np.log10(freq)
        lfn = (lf - lf.min()) / ((lf.max() - lf.min()) + 1e-12)
        return mod / (1.0 + hf_weight * lfn)
    if weight_by_modulus:
        return np.abs(Z_exp)
    return np.ones(len(freq))


def _to_internal(params: np.ndarray, include_r0: bool) -> np.ndarray:
    """Linear parameters -> optimizer space (ln R0?, ln R, ln tau, alpha)."""
    x = np.array(params, dtype=float)
    off = 1 if include_r0 else 0
    if include_r0:
        x[0] = np.log(x[0])
    for k in range(off, len(x), 3):
        x[k] = np.log(x[k])
        x[k + 1] = np.log(x[k + 1])
    return x


def _to_linear(x: np.ndarray, include_r0: bool) -> np.ndarray:
    """Optimizer space -> linear parameters; inverse of _to_internal.

    >>> p = np.array([50.0, 1e3, 1e-3, 0.8])
    >>> bool(np.allclose(_to_linear(_to_internal(p, True), True), p))
    True
    """
    params = np.array(x, dtype=float)
    off = 1 if include_r0 else 0
    if include_r0:
        params[0] = np.exp(params[0])
    for k in range(off, len(params), 3):
        params[k] = np.exp(params[k])
        params[k + 1] = np.exp(params[k + 1])
    return params


def _quality(Z_fit: np.ndarray, Z_exp: np.ndarray) -> tuple[float, float]:
    """(rmse_rel, max_rel_err), computed exactly as v1 does."""
    mod_exp = np.maximum(np.abs(Z_exp), 1e-12)
    rel_re = (Z_fit.real - Z_exp.real) / mod_exp
    rel_im = (Z_fit.imag - Z_exp.imag) / mod_exp
    rmse_rel = float(np.sqrt(np.mean(rel_re**2 + rel_im**2)))
    max_rel = float(np.max(np.abs(Z_fit - Z_exp) / mod_exp))
    return rmse_rel, max_rel


def _confidence(res, free_idx: np.ndarray, params_lin: np.ndarray,
                n_params: int, include_r0: bool,
                n_residuals: int) -> np.ndarray:
    """1-sigma confidence intervals in LINEAR units from the TRF Jacobian.

    The covariance in optimizer space is s2 * inv(J^T J); the delta method
    maps ln-space deviations back to linear units (sigma_R = sigma_lnR * R).
    Fixed parameters get conf = 0 like v1's collapsed-bound pinning.
    """
    conf = np.zeros(n_params)
    dof = n_residuals - len(free_idx)
    if dof <= 0 or res.jac is None:
        return np.full(n_params, np.nan)
    try:
        JTJ = res.jac.T @ res.jac
        cov = np.linalg.pinv(JTJ) * (2.0 * res.cost / dof)
        sig_x = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except np.linalg.LinAlgError:
        return np.full(n_params, np.nan)
    off = 1 if include_r0 else 0
    for pos, k in enumerate(free_idx):
        is_log = (include_r0 and k == 0) or ((k - off) % 3 in (0, 1))
        conf[k] = sig_x[pos] * (params_lin[k] if is_log else 1.0)
    return conf


def fit_zarc_v2(
    freq:        np.ndarray,
    Z_re:        np.ndarray,
    Z_im:        np.ndarray,
    peaks:       list[dict],
    R0_guess:    float | None = None,
    R_dec:       float | list[float] = 1.5,
    tau_dec:     float | list[float] = 1.5,
    alpha_init:  float | list[float] = 0.8,
    alpha_min:   float | list[float] = 0.5,
    alpha_max:   float | list[float] = 1.0,
    include_r0:  bool = True,
    r0_max:      float | None = None,
    fix_params:  dict | None = None,
    weight_by_modulus: bool = True,
    hf_weight:   float = 0.0,
    n_restarts:  int = 0,
    rmse_tol:    float = 0.02,
    seed:        int | None = None,
    loss:        str = "linear",
    f_scale:     float = 1.0,
) -> dict:
    """Fit a series-Zarc circuit with the v2 engine (log-space TRF).

    Same argument set, semantics and output dict schema as
    pipeline.fitting.fit_zarc; see that docstring for the shared parameters.
    Additional parameters:

    loss    : "linear" (plain L2, the v1 behaviour), "soft_l1" or "huber"
              (robust losses that bound the pull of outlier points)
    f_scale : residual scale for the robust losses; residuals are relative,
              so ~0.01 means "points worse than 1% relative error count as
              outliers". Irrelevant when loss="linear".

    >>> f = np.logspace(5, -1, 30)
    >>> Z = zarc_model(f, np.array([1e3, 1e-4, 0.9]), False)
    >>> pk = [{"R_approx": 8e2, "tau": 2e-4}]
    >>> out = fit_zarc_v2(f, Z.real, -Z.imag, pk, include_r0=False)
    >>> bool(out["converged"]), round(float(out["alpha"][0]), 3)
    (True, 0.9)
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
    n_peaks = len(peaks)
    if n_peaks == 0:
        raise ValueError("No peaks provided — cannot build circuit.")

    freq = np.asarray(freq, dtype=float)
    Z_re = np.asarray(Z_re, dtype=float)
    Z_im = np.asarray(Z_im, dtype=float)
    # IsmRecord stores Z_im positive in the capacitive region
    Z_exp = Z_re - 1j * Z_im

    if R0_guess is None:
        # identical HF-intercept heuristic to v1 (see fit_zarc)
        n_hf = max(5, int(0.3 * len(freq)))
        idx_hf = np.argsort(freq)[-n_hf:]
        hf_pos = Z_re[idx_hf][Z_re[idx_hf] > 0]
        if hf_pos.size >= 2:
            R0_guess = float(np.percentile(hf_pos, 10))
        elif hf_pos.size == 1:
            R0_guess = float(hf_pos[0])
        else:
            R0_guess = (max(float(np.median(Z_re[Z_re > 0])), 0.5)
                        if np.any(Z_re > 0) else 1.0)
            warnings.warn(
                "fit_zarc_v2: no HF points with Z_re > 0 found; "
                "R0_guess derived from LF data and may be unreliable.",
                UserWarning, stacklevel=2)
        if r0_max is not None:
            R0_guess = float(np.clip(R0_guess, 0.01, r0_max))

    circuit_str = build_circuit_string(n_peaks, include_r0=include_r0)
    initial_guess = build_initial_guess(R0_guess, peaks, alpha_init,
                                        include_r0=include_r0)
    lower, upper = build_bounds(R0_guess, peaks, R_dec, tau_dec,
                                alpha_min=alpha_min, alpha_max=alpha_max,
                                include_r0=include_r0, r0_max=r0_max)
    n_params = len(initial_guess)

    # fix_params pins parameters by removing them from the free vector
    # (numerically cleaner in log space than v1's one-ULP bound collapse,
    # same observable effect: the value is held exactly)
    fixed = np.full(n_params, np.nan)
    if fix_params:
        off = 0
        if include_r0:
            if fix_params.get("R0") is not None:
                fixed[0] = float(fix_params["R0"])
            off = 1
        for j in range(n_peaks):
            base = off + j * 3
            for k, key in enumerate(("R", "tau", "alpha")):
                vals = fix_params.get(key) or []
                if j < len(vals) and vals[j] is not None:
                    fixed[base + k] = float(vals[j])
    free_idx = np.flatnonzero(np.isnan(fixed))
    guess0 = np.array(initial_guess, dtype=float)
    guess0[~np.isnan(fixed)] = fixed[~np.isnan(fixed)]

    sig = _weights(freq, Z_exp, hf_weight, weight_by_modulus)
    x_lo = _to_internal(np.asarray(lower, dtype=float), include_r0)[free_idx]
    x_hi = _to_internal(np.asarray(upper, dtype=float), include_r0)[free_idx]

    x_base = _to_internal(guess0, include_r0)

    def _assemble(x_free: np.ndarray) -> np.ndarray:
        x_full = x_base.copy()
        x_full[free_idx] = x_free
        params = _to_linear(x_full, include_r0)
        # exact held values, immune to the ln/exp round trip
        params[~np.isnan(fixed)] = fixed[~np.isnan(fixed)]
        return params

    def _residuals(x_free: np.ndarray) -> np.ndarray:
        Z = zarc_model(freq, _assemble(x_free), include_r0)
        d = (Z - Z_exp) / sig
        return np.concatenate([d.real, d.imag])

    def _jacobian(x_free: np.ndarray) -> np.ndarray:
        _, dZ = zarc_model_jac(freq, _assemble(x_free), include_r0)
        dZw = dZ[:, free_idx] / sig[:, None]
        return np.vstack([dZw.real, dZw.imag])

    def _attempt(guess_lin: np.ndarray) -> dict:
        x0 = _to_internal(guess_lin, include_r0)[free_idx]
        x0 = np.clip(x0, x_lo, x_hi)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # tolerances one order tighter than the scipy defaults:
                # with the analytic Jacobian the extra iterations are cheap
                # and the solution lands measurably closer to the exact
                # weighted-LS minimum than v1's curve_fit stopping point
                res = least_squares(_residuals, x0, jac=_jacobian,
                                    bounds=(x_lo, x_hi), method="trf",
                                    x_scale="jac", loss=loss,
                                    f_scale=f_scale,
                                    xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception as exc:
            return {"converged": False, "params": guess_lin,
                    "conf": np.full(n_params, np.nan), "Z_fit": None,
                    "rmse_rel": np.inf, "max_rel_err": np.inf,
                    "fit_error": f"{type(exc).__name__}: {exc}", "res": None}
        params = _assemble(res.x)
        Z_fit = zarc_model(freq, params, include_r0)
        rmse_rel, max_rel = _quality(Z_fit, Z_exp)
        conf = _confidence(res, free_idx, params, n_params, include_r0,
                           2 * len(freq))
        return {"converged": bool(res.success), "params": params,
                "conf": conf, "Z_fit": Z_fit, "rmse_rel": rmse_rel,
                "max_rel_err": max_rel, "fit_error": "", "res": res}

    best = _attempt(guess0)

    if n_restarts > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            if best["rmse_rel"] < rmse_tol:
                break
            rnd = np.array(_sample_guess(list(lower), list(upper), n_peaks,
                                         include_r0, rng), dtype=float)
            cand = _attempt(rnd)
            if cand["converged"] and cand["rmse_rel"] < best["rmse_rel"]:
                best = cand

    params = np.asarray(best["params"], dtype=float)
    conf = best["conf"]
    Z_fit = best["Z_fit"]
    if Z_fit is None:
        params = guess0.copy()
        conf = np.full(n_params, np.nan)
        Z_fit = np.zeros(len(freq), dtype=complex)

    if include_r0:
        R0_fit = params[0]
        R_arr, tau_arr, alpha_arr = params[1::3], params[2::3], params[3::3]
    else:
        R0_fit = 0.0
        R_arr, tau_arr, alpha_arr = params[0::3], params[1::3], params[2::3]

    # C_eff = tau / R, exact for the Zarc parametrization (see fitting.py)
    C_eff_arr = tau_arr / R_arr

    if np.any((alpha_arr <= 0) | (alpha_arr > 1)):
        warnings.warn(
            "fit_zarc_v2: one or more alpha values are outside (0, 1] after "
            "fitting. Check fix_params or bounds — C_eff values may be "
            "physically meaningless.", UserWarning, stacklevel=2)

    param_names = []
    if include_r0:
        param_names.append("R0")
    for i in range(1, n_peaks + 1):
        param_names += [f"Zarc{i}_0", f"Zarc{i}_1", f"Zarc{i}_2"]

    return {
        "converged":   best["converged"],
        "circuit_str": circuit_str,
        "param_names": param_names,
        "params":      params,
        "conf":        conf,
        "R0":          float(R0_fit),
        "R":           R_arr,
        "tau":         tau_arr,
        "alpha":       alpha_arr,
        "C_eff":       C_eff_arr,
        "Z_fit":       Z_fit,
        "rmse_rel":    best["rmse_rel"],
        "max_rel_err": best["max_rel_err"],
        "n_peaks":     n_peaks,
        "fit_error":   best["fit_error"],
    }
