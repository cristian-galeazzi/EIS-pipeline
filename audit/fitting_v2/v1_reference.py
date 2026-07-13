"""Frozen v1 Zarc engine (linear-space curve_fit through impedance.py).

Kept ONLY as the reference arm of the engine-migration validation record:
`synthetic_gate.py` and `ab_harness.py` refit against this implementation
so the comparison stays reproducible at any commit. It is NOT part of the
pipeline; production code must import `pipeline.fitting.fit_zarc`.

This is a verbatim copy of `pipeline/fitting.py::fit_zarc` as of the last
v1 commit, with the function renamed to `fit_zarc_v1`; the shared,
engine-neutral helpers (circuit string, seeds, bounds, restart sampling)
are imported from `pipeline.fitting` unchanged.
"""
from __future__ import annotations

import warnings

import numpy as np
from impedance.models.circuits import CustomCircuit

from pipeline.fitting import (
    _sample_guess,
    build_bounds,
    build_circuit_string,
    build_initial_guess,
)


def _try_fit(
    circuit,
    guess:             list[float],
    lower:             list[float],
    upper:             list[float],
    Z_exp:             np.ndarray,
    freq:              np.ndarray,
    hf_weight:         float,
    weight_by_modulus: bool,
) -> dict:
    """Single optimizer call. Mutates circuit.initial_guess; returns dict with converged, params, conf, rmse_rel."""
    circuit.initial_guess = list(guess)
    converged = True
    fit_error = ""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hf_weight and hf_weight > 0:
                _mod = np.abs(Z_exp)
                _lf  = np.log10(freq)
                _lfn = (_lf - _lf.min()) / ((_lf.max() - _lf.min()) + 1e-12)
                _sig = _mod / (1.0 + hf_weight * _lfn)
                circuit.fit(freq, Z_exp, bounds=(lower, upper),
                            weight_by_modulus=False,
                            sigma=np.hstack([_sig, _sig]))
            else:
                circuit.fit(freq, Z_exp, bounds=(lower, upper),
                            weight_by_modulus=weight_by_modulus)
    except Exception as exc:
        converged = False
        fit_error = f"{type(exc).__name__}: {exc}"

    params = circuit.parameters_
    if params is None:
        converged = False
        params = np.array(guess)
        conf   = np.full(len(params), np.nan)
        rmse_rel = np.inf
        max_rel  = np.inf
        Z_fit    = None
    else:
        conf  = circuit.conf_ if hasattr(circuit, "conf_") else np.full_like(params, np.nan)
        Z_fit = circuit.predict(freq)
        # floor avoids inf/nan in the quality metrics if a degenerate point
        # has |Z| = 0; real spectra are orders of magnitude above 1e-12 Ohm
        mod_exp  = np.maximum(np.abs(Z_exp), 1e-12)
        rel_re   = (Z_fit.real - Z_exp.real) / mod_exp
        rel_im   = (Z_fit.imag - Z_exp.imag) / mod_exp
        rmse_rel = float(np.sqrt(np.mean(rel_re**2 + rel_im**2)))
        max_rel  = float(np.max(np.abs(Z_fit - Z_exp) / mod_exp))

    return {
        "converged":    converged,
        "params":       params,
        "conf":         conf,
        "Z_fit":        Z_fit,
        "rmse_rel":     rmse_rel,
        "max_rel_err":  max_rel,
        "fit_error":    fit_error,
        "param_names":  circuit.get_param_names()[0],
    }



def fit_zarc_v1(
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
) -> dict:
    """
    Fit a series-Zarc equivalent circuit to Z(f) data.

    The number of Zarc elements equals len(peaks).
    Initial guesses and bounds are derived from the DRT peaks.

    Parameters
    ----------
    freq      : frequency [Hz] (after Stage 2 frequency clipping)
    Z_re      : real impedance [Ohm]
    Z_im      : imaginary impedance [Ohm], positive in capacitive region (−Z″)
    peaks     : DRT peak list from find_drt_peaks() - sets N and initial bounds
    R0_guess  : estimate of R∞ (Z_re at highest frequency); auto if None
    R_dec     : R bounds in log-decades (default 1.5 = ±1.5 decades, factor ~32)
    tau_dec   : τ bounds in log-decades (default 1.5 = ±1.5 decades)
    alpha_init: initial ⍺ (default 0.8)
    alpha_min : lower ⍺ bound (default 0.5)
    alpha_max : upper ⍺ bound (default 1.0)
    weight_by_modulus : default True = modulus-weighted (proportional) residuals,
                matching the RelaxIS "Proportional weighting (recommended)" mode
                (manual §10.3). The optimizer minimizes the relative error, which
                fits the small-|Z| high-frequency bulk arc (the primary datum)
                far better at every temperature than unit weighting. Set False
                for legacy unit weighting (absolute residual, dominated by the
                large low-frequency arcs).
    hf_weight : RelaxIS "High freq. modulus" mode (manual §10.3). 0 = plain
                proportional weighting. >0 adds a high-frequency emphasis,
                sigma = |Z| / (1 + hf_weight * normalized_log10 f), pinning the
                small-|Z| bulk arc. A mild value (~1.0) both sharpens the bulk
                fit and stabilises overlapping mid-frequency Zarcs across
                temperature (less C_eff crossing). Overrides weight_by_modulus
                when > 0.
    n_restarts: number of additional random restarts within the bounds (default 0 =
                DRT-seeded guess only). Each restart samples R and tau log-uniformly,
                alpha linearly. The attempt with the lowest rmse_rel is returned.
                Values of 5-10 close most local-minimum traps for overlapping peaks.
    rmse_tol  : early-exit threshold for restarts. If the current best rmse_rel is
                already below this value, remaining restarts are skipped (default 0.02).

    R_dec, tau_dec, alpha_init, alpha_min and alpha_max each accept a scalar
    (same for every peak) or a per-peak list of length N, so each Zarc element
    can be given its own R / tau / alpha range instead of one global range.

    Returns
    -------
    dict with keys:
        converged     : bool
        circuit_str   : impedance.py circuit string used
        param_names   : list of parameter name strings
        params        : np.ndarray of fitted values
        conf          : np.ndarray of 1σ confidence intervals
        R0, R, tau, alpha, C_eff : arrays of derived quantities
        Z_fit         : complex impedance from fit (same freq as input)
        rmse_rel      : RMSE of relative residuals (modulus-weighted)
        max_rel_err   : max |Z_fit - Z_exp| / |Z_exp|
        n_peaks       : number of Zarc elements fitted
    """
    n_peaks = len(peaks)
    if n_peaks == 0:
        raise ValueError("No peaks provided - cannot build circuit.")

    # Convention: impedance.py uses Z = Z_re + j*Z_im_physical
    # where Z_im_physical < 0 in capacitive region
    # Our IsmRecord stores Z_im = -Z_im_physical > 0 in capacitive region
    Z_exp = Z_re - 1j * Z_im

    if R0_guess is None:
        # Improved HF intercept: 10th percentile of POSITIVE Z_re values among
        # the top 30% highest-frequency points. Excludes inductive noise points
        # (Z_re < 0) which previously inflated the guess by orders of magnitude
        # in high-resistance ceramic samples where the true R0 (contact + wire) is 1–100 Ω.
        n_hf  = max(5, int(0.3 * len(freq)))
        idx_hf = np.argsort(freq)[-n_hf:]
        hf_pos = Z_re[idx_hf][Z_re[idx_hf] > 0]
        if hf_pos.size >= 2:
            R0_guess = float(np.percentile(hf_pos, 10))
        elif hf_pos.size == 1:
            R0_guess = float(hf_pos[0])
        else:
            R0_guess = max(float(np.median(Z_re[Z_re > 0])), 0.5) if np.any(Z_re > 0) else 1.0
            warnings.warn(
                "fit_zarc_v1: no HF points with Z_re > 0 found; "
                "R0_guess derived from LF data and may be unreliable. "
                "Consider setting R0_guess manually or using r0_max.",
                UserWarning, stacklevel=2,
            )
        # Clamp into the explicit R0_max window when set
        if r0_max is not None:
            R0_guess = float(np.clip(R0_guess, 0.01, r0_max))

    circuit_str   = build_circuit_string(n_peaks, include_r0=include_r0)
    initial_guess = build_initial_guess(R0_guess, peaks, alpha_init, include_r0=include_r0)
    lower, upper  = build_bounds(R0_guess, peaks, R_dec, tau_dec,
                                 alpha_min=alpha_min, alpha_max=alpha_max,
                                 include_r0=include_r0, r0_max=r0_max)

    # Apply fix_params: pin individual parameters by collapsing bounds.
    # fix_params schema:
    #     {"R0":    value or None,
    #      "R":     [value or None per peak],
    #      "tau":   [value or None per peak],
    #      "alpha": [value or None per peak]}
    if fix_params:
        # scipy.least_squares requires lower < upper strictly, so a pinned
        # parameter gets a one-ULP window via nextafter instead of lower==upper
        # (which would make every fit fail with "infeasible bounds").
        offset = 0
        if include_r0:
            fixed_R0 = fix_params.get("R0")
            if fixed_R0 is not None:
                lower[0] = initial_guess[0] = float(fixed_R0)
                upper[0] = np.nextafter(float(fixed_R0), np.inf)
            offset = 1
        for j, peak in enumerate(peaks):
            base = offset + j * 3
            for k, key in enumerate(("R", "tau", "alpha")):
                vals = fix_params.get(key) or []
                if j < len(vals) and vals[j] is not None:
                    val = float(vals[j])
                    lower[base + k] = initial_guess[base + k] = val
                    upper[base + k] = np.nextafter(val, np.inf)

    circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)

    _fit_kw = dict(
        Z_exp=Z_exp, freq=freq,
        hf_weight=hf_weight, weight_by_modulus=weight_by_modulus,
    )

    best = _try_fit(circuit, initial_guess, lower, upper, **_fit_kw)

    if n_restarts > 0:
        rng = np.random.default_rng(seed)
        for _ in range(n_restarts):
            if best["rmse_rel"] < rmse_tol:
                break
            rnd_guess = _sample_guess(lower, upper, n_peaks, include_r0, rng)
            candidate = _try_fit(circuit, rnd_guess, lower, upper, **_fit_kw)
            if candidate["converged"] and candidate["rmse_rel"] < best["rmse_rel"]:
                best = candidate

    params = best["params"]
    conf   = best["conf"]
    Z_fit  = best["Z_fit"]
    if Z_fit is None:
        # All attempts failed; fall back to initial guess for downstream safety
        params = np.array(initial_guess)
        conf   = np.full(len(params), np.nan)
        Z_fit  = np.zeros(len(freq), dtype=complex)
        best["rmse_rel"]    = np.inf
        best["max_rel_err"] = np.inf

    # Parse parameters depending on whether R0 is included
    if include_r0:
        R0_fit    = params[0]
        R_arr     = params[1::3]
        tau_arr   = params[2::3]
        alpha_arr = params[3::3]
    else:
        R0_fit    = 0.0
        R_arr     = params[0::3]
        tau_arr   = params[1::3]
        alpha_arr = params[2::3]

    # C_eff = tau / R  (exact for Zarc parametrization, independent of ⍺)
    # See module docstring for derivation.
    C_eff_arr = tau_arr / R_arr

    if np.any((alpha_arr <= 0) | (alpha_arr > 1)):
        warnings.warn(
            "fit_zarc_v1: one or more alpha values are outside (0, 1] after fitting. "
            "Check fix_params or bounds - C_eff values may be physically meaningless.",
            UserWarning, stacklevel=2,
        )

    return {
        "converged":   best["converged"],
        "circuit_str": circuit_str,
        "param_names": best["param_names"],
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


# ---------------------------------------------------------------------------
# Conductivity
# ---------------------------------------------------------------------------
