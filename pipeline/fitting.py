"""
pipeline/fitting.py
===================
Zarc equivalent circuit fitting (log-space TRF, analytic Jacobian).

The circuit model is:  R0 - Zarc1 - Zarc2 - ... - ZarcN
where each Zarc element has impedance:

    Z_Zarc(ω) = R / (1 + (jωτ)^⍺) 

Zarc parameters per element: R [Ohm], τ [s], ⍺ (dimensionless ∈ [0,1])

Physical quantities derived from fit
-------------------------------------
C_eff = Q^(1/⍺) · R^((1-⍺)/⍺)

For the Zarc parametrization (R, τ, ⍺) the CPE admittance is:
    Y_CPE = Q·(jω)^⍺  with  Q = τ^⍺ / R

Substituting into the C_eff formula:

    C_eff = Q^(1/⍺) · R^((1-⍺)/⍺)
          = (τ^⍺/R)^(1/⍺) · R^((1-⍺)/⍺)
          = τ · R^(-1/⍺) · R^((1-⍺)/⍺)
          = τ · R^(-1/⍺ + 1/⍺ - 1)
          = τ · R^(-1)
          = τ / R

This simplification holds exactly for any ⍺.  C_eff = τ/R is therefore
the correct formula when using impedance.py's built-in Zarc element.

Conductivity (requires sample geometry L_m, D_m from config):
    A_m  = π·(D_m/2)²
    σ_i  = L_m / (R_i · A_m)   [S/m]
"""

from __future__ import annotations

import warnings
import zlib

import numpy as np
from scipy.optimize import least_squares


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
    >>> build_circuit_string(2)
    'R0-Zarc1-Zarc2'
    >>> build_circuit_string(2, include_r0=False)
    'Zarc1-Zarc2'
    """
    zarcs = "-".join(f"Zarc{i+1}" for i in range(n_peaks))
    return f"R0-{zarcs}" if include_r0 else zarcs


def _broadcast(val, n: int, name: str) -> list[float]:
    """
    Coerce a fitting control into a per-peak list of length n.

    Accepts a scalar (repeated for every peak - the original global behavior)
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

    >>> build_initial_guess(10.0, [{"R_approx": 100.0, "tau": 1e-3}])
    [10.0, 100.0, 0.001, 0.8]
    >>> build_initial_guess(10.0, [{"R_approx": 100.0, "tau": 1e-3}],
    ...                     include_r0=False)
    [100.0, 0.001, 0.8]
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
    ⍺_i  : [alpha_min,          alpha_max]            ← [0.5, 1.0]
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
    alpha_min : lower bound for ⍺
    alpha_max : upper bound for ⍺

    R_dec, tau_dec, alpha_min and alpha_max each accept a scalar (applied to
    every peak - the original global behavior) or a per-peak list of length N,
    which lets each Zarc element have its own resistance / tau / alpha range.

    Returns
    -------
    (lower_bounds, upper_bounds) - lists of floats, length 1 + 3*N

    >>> lo, hi = build_bounds(10.0, [{"R_approx": 100.0, "tau": 1e-3}],
    ...                       R_dec=1.0, tau_dec=1.0)
    >>> lo
    [1.0, 10.0, 0.0001, 0.5]
    >>> hi
    [200.0, 1000.0, 0.01, 1.0]
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
# Fitting helpers
# ---------------------------------------------------------------------------

def _sample_guess(
    lower:    list[float],
    upper:    list[float],
    n_peaks:  int,
    has_r0:   bool,
    rng:      np.random.Generator,
) -> list[float]:
    """
    Sample a random initial guess within bounds.
    R and tau are log-uniform (physically meaningful for quantities spanning decades).
    Alpha is linear-uniform (already in [0,1]).
    """
    guess = []
    idx = 0
    if has_r0:
        lo, hi = lower[idx], upper[idx]
        guess.append(float(np.exp(rng.uniform(np.log(max(lo, 1e-12)), np.log(max(hi, 1e-12))))))
        idx += 1
    for _ in range(n_peaks):
        # R: log-uniform
        lo, hi = lower[idx], upper[idx]
        guess.append(float(np.exp(rng.uniform(np.log(max(lo, 1e-12)), np.log(max(hi, 1e-12))))))
        idx += 1
        # tau: log-uniform
        lo, hi = lower[idx], upper[idx]
        guess.append(float(np.exp(rng.uniform(np.log(max(lo, 1e-12)), np.log(max(hi, 1e-12))))))
        idx += 1
        # alpha: linear-uniform
        lo, hi = lower[idx], upper[idx]
        guess.append(float(rng.uniform(lo, hi)))
        idx += 1
    return guess


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
        # floor keeps a malformed zero-impedance point from zeroing a weight
        mod = np.maximum(np.abs(Z_exp), 1e-12)
        lf = np.log10(freq)
        lfn = (lf - lf.min()) / ((lf.max() - lf.min()) + 1e-12)
        return mod / (1.0 + hf_weight * lfn)
    if weight_by_modulus:
        return np.maximum(np.abs(Z_exp), 1e-12)
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
    weight_by_modulus: bool = True,
    hf_weight:   float = 0.0,
    n_restarts:  int = 0,
    rmse_tol:    float = 0.02,
    seed:        int | None = None,
    loss:        str = "linear",
    f_scale:     float = 1.0,
) -> dict:
    """
    Fit a series-Zarc equivalent circuit to Z(f) data.

    The number of Zarc elements equals len(peaks). Initial guesses and
    bounds are derived from the DRT peaks. The optimizer works in log space
    (ln R0?, ln R, ln tau, alpha) with an analytic Jacobian and bounded TRF
    least squares; the mathematics and the migration record are in
    docs/MATHEMATICS.md section 3 and audit/fitting_v2/.

    Parameters
    ----------
    freq      : frequency [Hz] (after Stage 2 frequency clipping)
    Z_re      : real impedance [Ohm]
    Z_im      : imaginary impedance [Ohm], positive in capacitive region (-Z'')
    peaks     : DRT peak list from find_drt_peaks(); sets N and initial bounds
    R0_guess  : estimate of R_inf (Z_re at highest frequency); auto if None
    R_dec     : R bounds in log-decades around the seed (default 1.5)
    tau_dec   : tau bounds in log-decades around the seed (default 1.5)
    alpha_init: initial alpha (default 0.8)
    alpha_min : lower alpha bound (default 0.5)
    alpha_max : upper alpha bound (default 1.0)
    weight_by_modulus : default True = modulus-weighted (proportional)
                residuals, matching the RelaxIS "Proportional weighting
                (recommended)" mode. The optimizer minimizes the relative
                error, which fits the small-|Z| high-frequency bulk arc far
                better at every temperature than unit weighting. Set False
                for legacy unit weighting.
    hf_weight : RelaxIS "High freq. modulus" mode. 0 = plain proportional
                weighting. >0 adds a high-frequency emphasis,
                sigma = |Z| / (1 + hf_weight * normalized_log10 f), pinning
                the small-|Z| bulk arc. Overrides weight_by_modulus when > 0.
    n_restarts: number of additional random restarts within the bounds
                (default 0 = DRT-seeded guess only). Each restart samples R
                and tau log-uniformly, alpha linearly; the attempt with the
                lowest rmse_rel is returned.
    rmse_tol  : early-exit threshold for restarts (default 0.02).
    seed      : seed for the restart RNG (deterministic re-runs).
    loss      : "linear" (plain L2, default), "soft_l1" or "huber" (robust
                losses that bound the pull of outlier points).
    f_scale   : residual scale for the robust losses; residuals are
                relative, so ~0.01 means "points worse than 1% relative
                error count as outliers". Irrelevant when loss="linear".

    R_dec, tau_dec, alpha_init, alpha_min and alpha_max each accept a scalar
    (same for every peak) or a per-peak list of length N, so each Zarc
    element can be given its own R / tau / alpha range.

    Returns
    -------
    dict with keys:
        converged     : bool
        circuit_str   : circuit string (impedance.py-style notation)
        param_names   : list of parameter name strings
        params        : np.ndarray of fitted values
        conf          : np.ndarray of 1-sigma confidence intervals
        R0, R, tau, alpha, C_eff : arrays of derived quantities
        Z_fit         : complex impedance from fit (same freq as input)
        rmse_rel      : RMSE of relative residuals (modulus-weighted)
        max_rel_err   : max |Z_fit - Z_exp| / |Z_exp|
        n_peaks       : number of Zarc elements fitted

    >>> f = np.logspace(5, -1, 30)
    >>> Z = zarc_model(f, np.array([1e3, 1e-4, 0.9]), False)
    >>> pk = [{"R_approx": 8e2, "tau": 2e-4}]
    >>> out = fit_zarc(f, Z.real, -Z.imag, pk, include_r0=False)
    >>> bool(out["converged"]), round(float(out["alpha"][0]), 3)
    (True, 0.9)
    """
    if loss not in _LOSSES:
        raise ValueError(f"loss must be one of {_LOSSES}, got {loss!r}")
    n_peaks = len(peaks)
    if n_peaks == 0:
        raise ValueError("No peaks provided - cannot build circuit.")

    freq = np.asarray(freq, dtype=float)
    Z_re = np.asarray(Z_re, dtype=float)
    Z_im = np.asarray(Z_im, dtype=float)
    # IsmRecord stores Z_im positive in the capacitive region
    Z_exp = Z_re - 1j * Z_im

    # Underdetermined fits can report converged=True with meaningless
    # parameters; fail loud like Lin-KK (>=4 points) and the Stage-5 model.
    _n_fit_params = 3 * n_peaks + (1 if include_r0 else 0)
    if 2 * len(freq) <= _n_fit_params:
        raise ValueError(
            f"fit_zarc: {len(freq)} frequency points give {2 * len(freq)} "
            f"residuals for {_n_fit_params} parameters; the fit is "
            f"underdetermined.")

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
                "fit_zarc: no HF points with Z_re > 0 found; "
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
    with np.errstate(divide="ignore", invalid="ignore"):
        C_eff_arr = tau_arr / R_arr

    if np.any(R_arr <= 0):
        warnings.warn(
            "fit_zarc: one or more R values are <= 0 after fitting (pinned "
            "via fix_params?). C_eff = tau/R is not physically meaningful "
            "for those peaks.", UserWarning, stacklevel=2)

    if np.any((alpha_arr <= 0) | (alpha_arr > 1)):
        warnings.warn(
            "fit_zarc: one or more alpha values are outside (0, 1] after "
            "fitting. Check fix_params or bounds - C_eff values may be "
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

    >>> round(conductivity(100.0, 1e-3, 1e-2), 4)   # 1 mm thick, 1 cm diameter
    0.1273
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

    >>> fit = {"n_peaks": 1, "R": [100.0], "tau": [1e-3], "alpha": [0.9],
    ...        "C_eff": [1e-5], "conf": [1.0, 0.5, 1e-5, 0.01], "R0": 5.0,
    ...        "circuit_str": "R0-Zarc1", "rmse_rel": 0.01,
    ...        "max_rel_err": 0.02, "converged": True}
    >>> rows, summary = fit_to_rows(fit, "Ar", "a.ism", "/a.ism",
    ...                             500.0, 0.21, 1e-3, 1e-2)
    >>> rows[0]["peak_id"], rows[0]["conf_R"], summary["N_peaks"]
    (1, 0.5, 1)
    """
    T_K = T_nominal + 273.15

    peak_rows = []
    for i in range(fit["n_peaks"]):
        R_i     = float(fit["R"][i])
        tau_i   = float(fit["tau"][i])
        alpha_i = float(fit["alpha"][i])
        C_eff_i = float(fit["C_eff"][i])
        sigma_i = conductivity(R_i, L_m, D_m)

        # Confidence intervals (1σ) - params order depends on include_r0
        _has_r0 = fit["circuit_str"].startswith("R0")
        base = (1 + i * 3) if _has_r0 else (i * 3)
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


def resolve_condition_entry(condition_params: dict, condition: str) -> dict:
    """
    Return the per-condition override entry matching ``condition``.

    Keys may be the full condition folder name or a shorter label made of
    complete underscore-separated tokens (e.g. ``"Ar-100_O2-1"``); the first
    match wins. Matching is token-aligned so a short key cannot match inside
    an unrelated token (e.g. key ``"5"`` must not match ``"850"``).
    Single source of truth for this lookup: the batch cell and the live
    tuning panel must resolve parameters identically.

    >>> cp = {"Ar-100_O2-1": {"R_dec": 1.0}}
    >>> resolve_condition_entry(cp, "P1_B_Ar-100_O2-1_850_500_50")
    {'R_dec': 1.0}
    >>> resolve_condition_entry(cp, "P1_B_N2-100_850_500_50")
    {}
    """
    short = condition.split("_B_")[-1] if "_B_" in condition else condition
    for key, val in condition_params.items():
        if f"_{key}_" in f"_{condition}_" or f"_{key}_" in f"_{short}_":
            return val
    return {}


def resolve_peak_windows(
    peaks:           list[dict],
    condition:       str,
    T_nominal:       int,
    *,
    windows:         dict | None = None,
    legacy:          dict | None = None,
    r_dec_default:   float = 0.7,
    tau_dec_default: float = 0.7,
) -> tuple[float | list[float], float | list[float]]:
    """
    Resolve the R/tau constraint-window half-widths (in decades) per Zarc.

    Windows are keyed by ``peak_id``, so any peak count works and a window
    follows its process when the number of detected peaks changes with T.
    Priority: a legacy per-(condition, T) entry (position-based lists saved
    by older panel versions) wins wholesale when its length matches the
    spectrum; otherwise each peak looks up ``windows["conditions"][condition]``
    first, ``windows["sample"]`` next, and falls back to the scalar defaults.
    When no per-peak information exists at all, the scalar defaults are
    returned unchanged (identical tasks to a pipeline without this feature).

    >>> pk = [{"peak_id": 1}, {"peak_id": 2}]
    >>> w = {"sample": {"2": {"R_dec": 1.0, "tau_dec": 1.2}}}
    >>> resolve_peak_windows(pk, "c1", 500, windows=w)
    ([0.7, 1.0], [0.7, 1.2])
    >>> resolve_peak_windows(pk, "c1", 500)
    (0.7, 0.7)
    """
    n = len(peaks)
    t_map = (legacy or {}).get(condition, {}) or {}
    lp = t_map.get(str(T_nominal), t_map.get(T_nominal))
    if (isinstance(lp, dict)
            and isinstance(lp.get("R_dec"), (list, tuple)) and len(lp["R_dec"]) == n
            and isinstance(lp.get("tau_dec"), (list, tuple)) and len(lp["tau_dec"]) == n):
        try:
            return [float(v) for v in lp["R_dec"]], [float(v) for v in lp["tau_dec"]]
        except (TypeError, ValueError):
            pass   # corrupted legacy entry: fall through to the per-peak maps

    cond_map = ((windows or {}).get("conditions") or {}).get(condition) or {}
    samp_map = (windows or {}).get("sample") or {}
    if not cond_map and not samp_map:
        return float(r_dec_default), float(tau_dec_default)

    def _one(pid, key: str, default: float) -> float:
        for m in (cond_map, samp_map):
            e = m.get(str(pid), m.get(pid))
            if isinstance(e, dict) and e.get(key) is not None:
                try:
                    return float(e[key])
                except (TypeError, ValueError):
                    continue
        return float(default)

    pids = [p.get("peak_id", i + 1) for i, p in enumerate(peaks)]
    return ([_one(pid, "R_dec", r_dec_default) for pid in pids],
            [_one(pid, "tau_dec", tau_dec_default) for pid in pids])


def fit_condition_batch(
    condition:  str,
    tasks:      list[dict],
    include_r0: bool,
    r0_max:     float | None,
    n_restarts: int,
    rmse_tol:   float,
    L_m:        float,
    D_m:        float,
) -> dict:
    """
    Fit all temperatures of one condition sequentially with warm-start.

    Plain-data in, plain-data out, so it can run in a worker process and
    several conditions can be fitted in parallel. Within a condition the
    temperature order (descending) and the warm-start chain are identical
    to the serial notebook loop, so results do not depend on parallelism.

    Each task dict (ordered by descending T) must contain:
        T_nominal, fname, ism_path, pO2, freq, Z_re, Z_im, peaks,
        R_dec, tau_dec, alpha_init, alpha_min, alpha_max,
        hf_weight, fix_params, ov_tag

    Returns
    -------
    dict with keys: condition, fit_peaks, fit_summary, nyq_fits, log
        nyq_fits : {T_nominal: {R0, R, tau, alpha, Z_fit, rmse_rel, converged}}
        log      : printable per-temperature report (the worker cannot print)

    >>> out = fit_condition_batch(          # doctest: +SKIP
    ...     "Ar-100", tasks, include_r0=False, r0_max=None,
    ...     n_restarts=0, rmse_tol=0.02, L_m=1e-3, D_m=1e-2)
    >>> sorted(out["nyq_fits"])             # doctest: +SKIP
    [500, 550, 600]
    """
    log:         list[str] = []
    fit_peaks:   list[dict] = []
    fit_summary: list[dict] = []
    nyq_fits:    dict[int, dict] = {}
    prev_fit: dict | None = None
    prev_T:   float | None = None

    for t in tasks:
        T_nom   = t["T_nominal"]
        peaks   = t["peaks"]
        n_peaks = len(peaks)
        log.append(f"\n  T = {T_nom} °C  |  {t['fname']}")
        line = (f"  Fitting {build_circuit_string(n_peaks, include_r0=include_r0)}"
                f" ...{t.get('ov_tag', '')} ")

        # Warm-start from the previous (hotter) T when the peak count matches
        if prev_fit is not None and len(prev_fit["R"]) == n_peaks:
            peaks_seeded = [
                {**p, "R_approx": float(prev_fit["R"][i]), "tau": float(prev_fit["tau"][i])}
                for i, p in enumerate(peaks)
            ]
            R0_seed = float(prev_fit["R0"])
            line += f"[warm T={prev_T}] "
        else:
            if prev_fit is not None:
                line += f"[cold n_peaks {len(prev_fit['R'])}→{n_peaks}] "
            peaks_seeded = peaks
            R0_seed = None

        fit = fit_zarc(
            t["freq"], t["Z_re"], t["Z_im"], peaks_seeded,
            R0_guess=R0_seed,
            R_dec=t["R_dec"],
            tau_dec=t["tau_dec"],
            alpha_init=t["alpha_init"],
            alpha_min=t["alpha_min"],
            alpha_max=t["alpha_max"],
            include_r0=include_r0,
            r0_max=r0_max,
            fix_params=t["fix_params"],
            hf_weight=t["hf_weight"],
            n_restarts=n_restarts,
            rmse_tol=rmse_tol,
            # stable per-(condition, T) seed: restart guesses become
            # reproducible across runs and independent of parallelism
            seed=zlib.crc32(f"{condition}|{T_nom}".encode()),
        )

        line += ("converged" if fit["converged"] else "NOT CONVERGED")
        line += f"  rmse_rel={fit['rmse_rel']:.4f}"
        log.append(line)
        log.append(f"    R0={fit['R0']:.4g} Ω")
        for i in range(n_peaks):
            C = float(fit["C_eff"][i])
            log.append(f"    Zarc{i+1}: R={fit['R'][i]:.4g} Ω  τ={fit['tau'][i]:.3e} s  "
                       f"α={fit['alpha'][i]:.3f}  C_eff={C:.2e} F")

        if fit["converged"]:
            prev_fit, prev_T = fit, T_nom

        nyq_fits[T_nom] = {
            "R0": fit["R0"], "R": fit["R"],
            "tau": fit["tau"], "alpha": fit["alpha"],
            "Z_fit": fit["Z_fit"],
            "rmse_rel": fit["rmse_rel"],
            "converged": fit["converged"],
        }

        peak_rows, summary_row = fit_to_rows(
            fit, condition, t["fname"], t["ism_path"], T_nom, t["pO2"], L_m, D_m,
        )
        fit_peaks.extend(peak_rows)
        fit_summary.append(summary_row)

    return {
        "condition":   condition,
        "fit_peaks":   fit_peaks,
        "fit_summary": fit_summary,
        "nyq_fits":    nyq_fits,
        "log":         "\n".join(log),
    }

