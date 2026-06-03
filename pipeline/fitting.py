"""
pipeline/fitting.py
===================
Zarc equivalent circuit fitting using impedance.py.

The circuit model is:  R0 - Zarc1 - Zarc2 - ... - ZarcN
where each Zarc element has impedance:

    Z_Zarc(ω) = R / (1 + (jωτ)^α)

Zarc parameters per element: R [Ohm], τ [s], α (dimensionless ∈ [0,1])

Physical quantities derived from fit
-------------------------------------
C_eff = Q^(1/α) · R^((1-α)/α)

For the Zarc parametrization (R, τ, α) the CPE admittance is:
    Y_CPE = Q·(jω)^α  with  Q = τ^α / R

Substituting into the C_eff formula:

    C_eff = Q^(1/α) · R^((1-α)/α)
          = (τ^α/R)^(1/α) · R^((1-α)/α)
          = τ · R^(-1/α) · R^((1-α)/α)
          = τ · R^(-1/α + 1/α - 1)
          = τ · R^(-1)
          = τ / R

This simplification holds exactly for any α.  C_eff = τ/R is therefore
the correct formula when using impedance.py's built-in Zarc element.

Conductivity (requires sample geometry L_m, D_m from config):
    A_m  = π·(D_m/2)²
    σ_i  = L_m / (R_i · A_m)   [S/m]
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from impedance.models.circuits import CustomCircuit


# ---------------------------------------------------------------------------
# Circuit construction
# ---------------------------------------------------------------------------

def build_circuit_string(n_peaks: int, include_r0: bool = True) -> str:
    """
    Build the impedance.py circuit string for N Zarc elements in series.

    Parameters
    ----------
    n_peaks    : number of Zarc elements
    include_r0 : when True (default) the model is R0 - Zarc1 - ... - ZarcN.
                 When False the model is pure Zarc1 - ... - ZarcN with no series
                 resistance (RelaxIS-style for ceramics where R0 is negligible).

    Examples
    --------
    include_r0=True,  n_peaks=2  ->  'R0-Zarc1-Zarc2'
    include_r0=False, n_peaks=2  ->  'Zarc1-Zarc2'
    """
    zarcs = "-".join(f"Zarc{i+1}" for i in range(n_peaks))
    return f"R0-{zarcs}" if include_r0 else zarcs


def _broadcast(val, n: int, name: str) -> list[float]:
    """
    Coerce a fitting control into a per-peak list of length n.

    Accepts a scalar (repeated for every peak — the original global behavior)
    or a per-peak sequence (one value per Zarc element). This is what lets the
    caller set R_dec / tau_dec / alpha bounds individually per peak instead of
    globally.
    """
    if np.isscalar(val):
        return [float(val)] * n
    arr = [float(x) for x in val]
    if len(arr) != n:
        raise ValueError(f"{name} has length {len(arr)} but there are {n} peaks")
    return arr


def build_initial_guess(
    R0_guess: float,
    peaks:    list[dict],
    alpha_init: float | list[float] = 0.8,
    include_r0: bool = True,
) -> list[float]:
    """
    Build initial-guess array from DRT peak information.

    Parameter order (impedance.py Zarc convention):
        include_r0=True  -> R0, [R_1, tau_1, alpha_1], [R_2, tau_2, alpha_2], ...
        include_r0=False -> [R_1, tau_1, alpha_1], [R_2, tau_2, alpha_2], ...

    alpha_init accepts a scalar (same for all peaks) or a per-peak list.

    Returns
    -------
    List[float] of length (1 if include_r0 else 0) + 3*N
    """
    guess = [float(R0_guess)] if include_r0 else []
    alpha_arr = _broadcast(alpha_init, len(peaks), "alpha_init")
    for i, p in enumerate(peaks):
        guess.extend([p["R_approx"], p["tau"], alpha_arr[i]])
    return guess


def build_bounds(
    R0_guess:  float,
    peaks:     list[dict],
    R_dec:     float | list[float] = 1.5,
    tau_dec:   float | list[float] = 1.5,
    alpha_min: float | list[float] = 0.5,
    alpha_max: float | list[float] = 1.0,
    include_r0: bool = True,
    r0_max:    float | None = None,
) -> tuple[list[float], list[float]]:
    """
    Build parameter bounds derived from DRT peak positions and areas.

    Bounds strategy
    ---------------
    R_i  : [R_drt / 10^R_dec,   R_drt * 10^R_dec]   ← ±R_dec log-decades (default ±1.5)
    τ_i  : [τ_drt / 10^tau_dec, τ_drt * 10^tau_dec] ← ±tau_dec log-decades (default ±1.5)
    α_i  : [alpha_min,          alpha_max]            ← [0.5, 1.0]
    R0   : [max(R0_guess*0.1, 0.5),  max(R0_guess*20, 100)]

    Why the R0 floor at 0.1 × R0_guess (or 0.5 Ω)?
    With many free parameters (5+ Zarc elements), the optimizer can find a
    degenerate solution where R0 → 0 and Zarc1 absorbs the HF resistance.
    Anchoring R0 to a positive minimum prevents this collapse and keeps R0
    physically meaningful as the ohmic series resistance.

    Why log-decades for R?
    ----------------------
    Tikhonov DRT areas are reliable indicators of ORDER OF MAGNITUDE only.
    Near frequency boundaries or for overlapping processes, the area can be off
    by a factor of 3–10×.  Log-decade bounds guarantee the optimizer can find
    the true R even when the DRT estimate is a rough guide.

    τ bounds are tighter (±1.5 decades) because peak POSITION from DRT is much
    more reliable than peak AREA.

    Parameters
    ----------
    R0_guess  : high-frequency Z_re estimate [Ohm]
    peaks     : DRT peak list from find_drt_peaks()
    R_dec     : R bound in log-decades (default 1.5 → factor ~32 in each direction)
    tau_dec   : τ bound in log-decades (default 1.5 → ±1.5 decades)
    alpha_min : lower bound for α
    alpha_max : upper bound for α

    R_dec, tau_dec, alpha_min and alpha_max each accept a scalar (applied to
    every peak — the original global behavior) or a per-peak list of length N,
    which lets each Zarc element have its own resistance / tau / alpha range.

    Returns
    -------
    (lower_bounds, upper_bounds) — lists of floats, length 1 + 3*N
    """
    if include_r0:
        if r0_max is not None:
            # Explicit bound: R0 confined to [0.01, r0_max] regardless of guess.
            # For high-resistance ceramics where R0 (contact + wire + electrode material) is 1-100 Ω, this avoids
            # the bug where a noisy HF intercept inflates the upper bound.
            R0_lo = 0.01
            R0_hi = float(r0_max)
        else:
            # Backward-compatible default.
            R0_lo = max(R0_guess * 0.1, 0.5)
            R0_hi = max(R0_guess * 20.0, 100.0)
        lower = [R0_lo]
        upper = [R0_hi]
    else:
        lower = []
        upper = []

    n = len(peaks)
    R_dec_a   = _broadcast(R_dec,   n, "R_dec")
    tau_dec_a = _broadcast(tau_dec, n, "tau_dec")
    a_min_a   = _broadcast(alpha_min, n, "alpha_min")
    a_max_a   = _broadcast(alpha_max, n, "alpha_max")

    for i, p in enumerate(peaks):
        R_lo   = max(p["R_approx"] / (10 ** R_dec_a[i]), 1e-3)
        R_hi   = p["R_approx"] * (10 ** R_dec_a[i])
        tau_lo = p["tau"] / (10 ** tau_dec_a[i])
        tau_hi = p["tau"] * (10 ** tau_dec_a[i])

        lower.extend([R_lo,  tau_lo, a_min_a[i]])
        upper.extend([R_hi,  tau_hi, a_max_a[i]])

    return lower, upper


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_zarc(
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
    peaks     : DRT peak list from find_drt_peaks() — sets N and initial bounds
    R0_guess  : estimate of R∞ (Z_re at highest frequency); auto if None
    R_dec     : R bounds in log-decades (default 1.5 = ±1.5 decades, factor ~32)
    tau_dec   : τ bounds in log-decades (default 1.5 = ±1.5 decades)
    alpha_init: initial α (default 0.8)
    alpha_min : lower α bound (default 0.5)
    alpha_max : upper α bound (default 1.0)

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
        raise ValueError("No peaks provided — cannot build circuit.")

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
        offset = 0
        if include_r0:
            fixed_R0 = fix_params.get("R0")
            if fixed_R0 is not None:
                lower[0] = upper[0] = initial_guess[0] = float(fixed_R0)
            offset = 1
        for j, peak in enumerate(peaks):
            base = offset + j * 3
            for k, key in enumerate(("R", "tau", "alpha")):
                vals = fix_params.get(key) or []
                if j < len(vals) and vals[j] is not None:
                    val = float(vals[j])
                    lower[base + k] = upper[base + k] = initial_guess[base + k] = val

    circuit = CustomCircuit(circuit_str, initial_guess=initial_guess)

    converged = True
    fit_error = ""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            circuit.fit(freq, Z_exp, bounds=(lower, upper))
    except Exception as exc:
        converged = False
        fit_error = f"{type(exc).__name__}: {exc}"
        warnings.warn(f"Zarc fit did not converge ({circuit_str}): {fit_error}",
                      stacklevel=2)

    params = circuit.parameters_
    if params is None:
        # Fit raised an exception before writing any parameters; fall back to initial guess
        # so downstream indexing does not crash on None.
        params = np.array(initial_guess)
        conf   = np.full(len(params), np.nan)
    else:
        conf = circuit.conf_ if hasattr(circuit, "conf_") else np.full_like(params, np.nan)

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

    # C_eff = tau / R  (exact for Zarc parametrization, independent of α)
    # See module docstring for derivation.
    C_eff_arr = tau_arr / R_arr

    # Predicted impedance
    Z_fit = circuit.predict(freq)

    # Modulus-weighted relative residuals
    mod_exp  = np.abs(Z_exp)
    rel_re   = (Z_fit.real - Z_exp.real) / mod_exp
    rel_im   = (Z_fit.imag - Z_exp.imag) / mod_exp
    rmse_rel = float(np.sqrt(np.mean(rel_re**2 + rel_im**2)))
    max_rel  = float(np.max(np.abs(Z_fit - Z_exp) / mod_exp))

    return {
        "converged":   converged,
        "circuit_str": circuit_str,
        "param_names": circuit.get_param_names()[0],
        "params":      params,
        "conf":        conf,
        "R0":          float(R0_fit),
        "R":           R_arr,
        "tau":         tau_arr,
        "alpha":       alpha_arr,
        "C_eff":       C_eff_arr,
        "Z_fit":       Z_fit,
        "rmse_rel":    rmse_rel,
        "max_rel_err": max_rel,
        "n_peaks":     n_peaks,
        "fit_error":   fit_error,
    }


# ---------------------------------------------------------------------------
# Conductivity
# ---------------------------------------------------------------------------

def conductivity(R_ohm: float, L_m: float, D_m: float) -> float:
    """
    Compute conductivity from a resistance and sample geometry.

        σ = L / (R · A)   [S/m]

    Parameters
    ----------
    R_ohm : resistance [Ohm]
    L_m   : sample thickness [m]
    D_m   : sample diameter [m]

    Returns
    -------
    float : conductivity [S/m]
    """
    A_m2 = np.pi * (D_m / 2.0) ** 2
    return L_m / (R_ohm * A_m2)


# ---------------------------------------------------------------------------
# Results → DataFrame rows
# ---------------------------------------------------------------------------

def fit_to_rows(
    fit:        dict,
    condition:  str,
    file:       str,
    full_path:  str,
    T_nominal:  float,
    pO2_mean:   float,
    L_m:        float,
    D_m:        float,
) -> tuple[list[dict], dict]:
    """
    Convert a fit_zarc() result to exportable records.

    Returns
    -------
    (peak_rows, summary_row)

    peak_rows : one dict per Zarc element with:
        condition, file, T_nominal, T_K, pO2_mean,
        R0, peak_id, R_i, tau_i, alpha_i, C_eff_i, sigma_Sm_i,
        conf_R, conf_tau, conf_alpha

    summary_row : one dict per spectrum with:
        condition, file, T_nominal, T_K, pO2_mean,
        R0, N_peaks, circuit_str, rmse_rel, max_rel_err, converged
    """
    T_K = T_nominal + 273.15

    peak_rows = []
    for i in range(fit["n_peaks"]):
        R_i     = float(fit["R"][i])
        tau_i   = float(fit["tau"][i])
        alpha_i = float(fit["alpha"][i])
        C_eff_i = float(fit["C_eff"][i])
        sigma_i = conductivity(R_i, L_m, D_m)

        # Confidence intervals (1σ) — params order: R0, R1,tau1,a1, R2,tau2,a2,...
        base = 1 + i * 3
        conf_R   = float(fit["conf"][base])     if len(fit["conf"]) > base   else np.nan
        conf_tau = float(fit["conf"][base + 1]) if len(fit["conf"]) > base+1 else np.nan
        conf_a   = float(fit["conf"][base + 2]) if len(fit["conf"]) > base+2 else np.nan

        peak_rows.append({
            "condition":   condition,
            "file":        file,
            "full_path":   full_path,
            "T_nominal":   T_nominal,
            "T_K":         T_K,
            "pO2_mean":    pO2_mean,
            "R0":          fit["R0"],
            "peak_id":     i + 1,
            "R_i":         R_i,
            "tau_i":       tau_i,
            "alpha_i":     alpha_i,
            "C_eff_i":     C_eff_i,
            "sigma_Sm_i":  sigma_i,
            "conf_R":      conf_R,
            "conf_tau":    conf_tau,
            "conf_alpha":  conf_a,
        })

    summary_row = {
        "condition":   condition,
        "file":        file,
        "full_path":   full_path,
        "T_nominal":   T_nominal,
        "T_K":         T_K,
        "pO2_mean":    pO2_mean,
        "R0":          fit["R0"],
        "N_peaks":     fit["n_peaks"],
        "circuit_str": fit["circuit_str"],
        "rmse_rel":    fit["rmse_rel"],
        "max_rel_err": fit["max_rel_err"],
        "converged":   fit["converged"],
    }

    return peak_rows, summary_row
