"""Stage 5 engine: global mixed ionic-electronic conductivity (MIEC) model.

Fits the measured conductivity surface sigma(pO2, T) of one process (peak) with
the three-channel MIEC model in a single global fit. This is the "backward"
approach: from the measured data to the six physical parameters, instead of
predicting sigma from assumed parameters.

Model (three parallel channels; mobility ~ 1/T, Arrhenius-activated):

    sigma(pO2, T) = (sigma0_ion / T) * exp(-Ea_ion / kT)
                  + (sigma0_p   / T) * exp(-Ea_p   / kT) * pO2**(+x)
                  + (sigma0_n   / T) * exp(-Ea_n   / kT) * pO2**(-x)

x is the Brouwer exponent (1/4 in the dilute defect regime). The six parameters
(three prefactors sigma0_*, three activation energies Ea_*) are constants of the
material: a single set must describe the whole (pO2, T) surface. That constancy
is the physical-validity test.

Fit method (variable projection, VARPRO):
- at fixed activation energies the model is LINEAR in the three prefactors, so
  they are solved exactly and non-negatively with weighted NNLS (inner problem);
- only the three activation energies are optimised non-linearly (outer problem).
A final 6-parameter polish, seeded at the VARPRO optimum, yields the covariance
(parameter uncertainties). This is more robust than a blind 6-parameter fit (no
false minima, prefactors non-negative by construction) and reuses scipy's nnls.

Conventions:
- conductivities in S/m (same as the stage-4 Arrhenius ln(sigma*T)); pO2 in bar;
  T_nominal in the input frame is in degrees Celsius (converted to K internally).
- the prefactor sigma0 has the 1/T mobility factor pulled out: it is the
  intercept exp(ln sigma0) of ln(sigma*T) vs 1/T.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares, nnls

KB_EV: float = 8.617e-5  # Boltzmann constant (eV/K)

# Minimum distinct (pO2, T) points required to constrain the fit. The model has
# up to 6 free parameters, so we ask for a clear margin above that.
MIN_POINTS: int = 8

# Canonical channel order and Brouwer-exponent sign of each channel. Channel
# selection is an operator decision driven by defect chemistry (e.g. drop "n"
# when the measured pO2 window is never reducing enough to create electrons);
# the fit never adds or removes channels on its own.
CHANNELS: tuple[str, str, str] = ("ion", "p", "n")
_EXPO_SIGN: dict[str, float] = {"ion": 0.0, "p": +1.0, "n": -1.0}


def _canonical_channels(channels) -> tuple[str, ...]:
    """Validate and reorder a channel selection to the canonical order.

    >>> _canonical_channels(["p", "ion"])
    ('ion', 'p')
    """
    unknown = set(channels) - set(CHANNELS)
    if unknown:
        raise ValueError(f"unknown channel(s) {sorted(unknown)}; valid: {CHANNELS}")
    sel = tuple(ch for ch in CHANNELS if ch in set(channels))
    if not sel:
        raise ValueError(f"at least one channel required; valid: {CHANNELS}")
    return sel


@dataclass(frozen=True)
class ModelParams:
    """Six MIEC parameters for one process, plus the Brouwer exponent used."""

    sigma0_ion: float
    Ea_ion: float
    sigma0_p: float
    Ea_p: float
    sigma0_n: float
    Ea_n: float
    x: float = 0.25


def _channel_sigma0T(s0: float, Ea: float, T_K: NDArray) -> NDArray[np.float64]:
    """One channel's sigma0/T * exp(-Ea/kT) term, exactly 0 when the channel is absent.

    An excluded channel stores s0 = 0 and Ea = NaN; without the short-circuit
    the term would be 0 * exp(NaN) = NaN and poison the whole surface.
    """
    if s0 == 0.0:
        return np.zeros_like(T_K)
    return s0 / T_K * np.exp(-Ea / (KB_EV * T_K))


def total_conductivity(pO2, T_K, p: ModelParams) -> NDArray[np.float64]:
    """Forward model sigma(pO2, T) in S/m. pO2 and T_K broadcast together (T in K)."""
    pO2 = np.asarray(pO2, dtype=float)
    T_K = np.asarray(T_K, dtype=float)
    ion = _channel_sigma0T(p.sigma0_ion, p.Ea_ion, T_K)
    pol = _channel_sigma0T(p.sigma0_p, p.Ea_p, T_K) * pO2 ** (+p.x)
    ele = _channel_sigma0T(p.sigma0_n, p.Ea_n, T_K) * pO2 ** (-p.x)
    return ion + pol + ele


def _design_matrix(
    pO2: NDArray, T_K: NDArray, Ea, x: float,
    channels: tuple[str, ...] = CHANNELS,
) -> NDArray[np.float64]:
    """Channel columns whose non-negative combination gives sigma, at fixed Ea.

    Column k = (1/T) * exp(-Ea_k / kT) * pO2**expo_k for the selected channels
    (ionic: expo 0; p-type: +x; n-type: -x). The unknowns are the prefactors
    sigma0, which is why the inner problem is linear. ``Ea`` has one entry per
    selected channel, in canonical order.
    """
    kT = KB_EV * T_K
    base = 1.0 / T_K
    cols = []
    for k, ch in enumerate(channels):
        col = base * np.exp(-Ea[k] / kT)
        sign = _EXPO_SIGN[ch]
        if sign:
            col = col * pO2 ** (sign * x)
        cols.append(col)
    return np.column_stack(cols)


def _solve_sigma0(A: NDArray, y: NDArray, w: NDArray) -> NDArray[np.float64]:
    """Weighted non-negative least squares for the prefactors (>= 0), one per column."""
    coef, _ = nnls(A * w[:, None], y * w)
    return coef


def _covariance_errors(res) -> NDArray[np.float64]:
    """1-sigma parameter errors from a least_squares result.

    Only the parameters the residual is actually sensitive to get an error bar.
    An absent channel (prefactor driven to ~0) makes both its prefactor and its
    activation energy unidentifiable; those are reported as NaN instead of
    poisoning the whole covariance, because a single zero Jacobian column makes
    JᵀJ singular. Active columns are scaled to unit norm before inversion to stay
    well-conditioned despite the very different magnitudes (σ0 ~ 1e7 vs Eₐ ~ 1).
    """
    J = res.jac
    m, n = J.shape
    dof = max(m - n, 1)
    s2 = 2.0 * res.cost / dof  # res.cost = 0.5 * sum(residuals**2)
    err = np.full(n, np.nan)

    col_norm = np.sqrt((J ** 2).sum(axis=0))
    active = col_norm > 1e-10 * max(col_norm.max(), 1e-300)
    if not active.any():
        return err

    Jn = J[:, active] / col_norm[active]  # unit-norm columns
    try:
        cov_n = np.linalg.inv(Jn.T @ Jn)
    except np.linalg.LinAlgError:
        cov_n = np.linalg.pinv(Jn.T @ Jn)
    cov = cov_n / np.outer(col_norm[active], col_norm[active]) * s2
    err[active] = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    return err


def stoichiometric_pO2(p: ModelParams, T_K) -> NDArray[np.float64]:
    """Electronic conductivity minimum pO2_min(T) [bar]: where sigma_n == sigma_p.

    Derived from d(sigma_el)/d(pO2) = 0:
        pO2_min = ( (sigma0_n/sigma0_p) * exp(-(Ea_n - Ea_p)/kT) )**(1/(2x)).
    Returns NaN where either electronic channel is absent.
    """
    T_K = np.asarray(T_K, dtype=float)
    if p.sigma0_p <= 0.0 or p.sigma0_n <= 0.0:
        return np.full_like(T_K, np.nan)
    ratio = (p.sigma0_n / p.sigma0_p) * np.exp(-(p.Ea_n - p.Ea_p) / (KB_EV * T_K))
    return ratio ** (1.0 / (2.0 * p.x))


def predict_grid(p: ModelParams, pO2_grid, T_K_grid) -> NDArray[np.float64]:
    """sigma on a (pO2, T) mesh, for the 3-D surface plot. Returns shape (nT, npO2)."""
    PO2, TT = np.meshgrid(np.asarray(pO2_grid, float), np.asarray(T_K_grid, float))
    return total_conductivity(PO2, TT, p)


def fit_global_conductivity(
    df_peak: pd.DataFrame,
    x: float = 0.25,
    seed: tuple[float, ...] | None = None,
    t_min: float | None = None,
    t_max: float | None = None,
    channels: tuple[str, ...] = ("ion", "p", "n"),
) -> dict:
    """Global VARPRO fit of the MIEC model to one peak's sigma(pO2, T).

    Parameters
    ----------
    df_peak : rows for a single process; needs columns ``T_nominal`` [C],
              ``pO2_mean`` [bar], ``sigma_Sm_i`` [S/m].
    x       : Brouwer exponent (1/4 dilute regime, 1/6 elsewhere).
    seed    : initial guess for the Ea of the SELECTED channels, in eV and in
              canonical (ion, p, n) order (default 1 eV each).
    t_min,
    t_max   : optional temperature window [C]; points outside are excluded
              (e.g. below the temperature where the peaks no longer separate).
    channels: which channels exist, a subset of ("ion", "p", "n"). A physics
              decision by the operator (defect chemistry), not a fit outcome:
              excluded channels get sigma0 = 0.0 and Ea = NaN in the result.

    >>> out = fit_global_conductivity(df_peak, channels=("ion", "p"))  # doctest: +SKIP
    >>> out["params"].sigma0_n, out["params"].Ea_n                     # doctest: +SKIP
    (0.0, nan)

    Returns
    -------
    dict with ``params`` (ModelParams), ``perr`` (1-sigma errors per name),
    ``r2`` (on ln sigma, over the whole surface), ``residuals`` (tidy DataFrame),
    ``n_points`` and ``converged``.

    Raises
    ------
    ValueError if required columns are missing, if too few usable points remain,
    if ``channels`` or ``seed`` are invalid, or if the optimiser fails.
    """
    chans = _canonical_channels(channels)
    n_ch = len(chans)
    need = {"T_nominal", "pO2_mean", "sigma_Sm_i"}
    missing = need - set(df_peak.columns)
    if missing:
        raise ValueError(f"fit_global_conductivity: missing columns {sorted(missing)}")

    Tc = pd.to_numeric(df_peak["T_nominal"], errors="coerce").to_numpy(float)
    pO2 = pd.to_numeric(df_peak["pO2_mean"], errors="coerce").to_numpy(float)
    sig = pd.to_numeric(df_peak["sigma_Sm_i"], errors="coerce").to_numpy(float)

    good = np.isfinite(Tc) & np.isfinite(pO2) & np.isfinite(sig) & (pO2 > 0) & (sig > 0)
    if t_min is not None:
        good &= Tc >= t_min
    if t_max is not None:
        good &= Tc <= t_max
    Tc, pO2, sig = Tc[good], pO2[good], sig[good]

    n_points = int(pO2.size)
    if n_points < MIN_POINTS:
        raise ValueError(
            f"fit_global_conductivity: only {n_points} usable (pO2, T) points after "
            f"filtering (need >= {MIN_POINTS}); widen the temperature/condition selection."
        )

    T_K = Tc + 273.15
    w = 1.0 / sig  # relative-residual weights: weak low-sigma channels count too

    def _outer(Ea):
        A = _design_matrix(pO2, T_K, Ea, x, chans)
        s0 = _solve_sigma0(A, sig, w)
        return (A @ s0 - sig) * w

    def _full(theta):
        A = _design_matrix(pO2, T_K, theta[n_ch:], x, chans)
        return (A @ theta[:n_ch] - sig) * w

    Ea0 = np.asarray(seed if seed is not None else (1.0,) * n_ch, dtype=float)
    if Ea0.size != n_ch:
        raise ValueError(
            f"fit_global_conductivity: seed has {Ea0.size} value(s) but "
            f"{n_ch} channel(s) are selected {chans}"
        )
    try:
        outer = least_squares(_outer, Ea0, bounds=(0.0, 3.0))
        A = _design_matrix(pO2, T_K, outer.x, x, chans)
        s0 = _solve_sigma0(A, sig, w)
        # Full polish (seeded at the VARPRO optimum) for the covariance.
        theta0 = np.concatenate([s0, outer.x])
        bounds = (np.zeros(2 * n_ch),
                  np.concatenate([np.full(n_ch, np.inf), np.full(n_ch, 3.0)]))
        polish = least_squares(_full, theta0, bounds=bounds)
    except Exception as exc:  # solver may fail on degenerate data
        raise ValueError(f"fit_global_conductivity: solver failed ({type(exc).__name__}: {exc})") from exc

    # Scatter the fitted values back onto the full 3-channel parameter set;
    # excluded channels are marked absent (sigma0 = 0, Ea = NaN), not zero-energy.
    s0_map = {ch: 0.0 for ch in CHANNELS}
    Ea_map = {ch: float("nan") for ch in CHANNELS}
    for k, ch in enumerate(chans):
        s0_map[ch] = float(polish.x[k])
        Ea_map[ch] = float(polish.x[n_ch + k])
    params = ModelParams(
        sigma0_ion=s0_map["ion"], Ea_ion=Ea_map["ion"],
        sigma0_p=s0_map["p"], Ea_p=Ea_map["p"],
        sigma0_n=s0_map["n"], Ea_n=Ea_map["n"], x=x,
    )

    model = total_conductivity(pO2, T_K, params)
    ln_obs, ln_fit = np.log(sig), np.log(model)
    ss_tot = float(np.sum((ln_obs - ln_obs.mean()) ** 2))
    r2 = 1.0 - float(np.sum((ln_obs - ln_fit) ** 2)) / ss_tot if ss_tot > 0 else np.nan

    err = _covariance_errors(polish)  # order: s0 of selected channels, then their Ea
    perr = {f"sigma0_{ch}": float("nan") for ch in CHANNELS}
    perr.update({f"Ea_{ch}": float("nan") for ch in CHANNELS})
    for k, ch in enumerate(chans):
        perr[f"sigma0_{ch}"] = float(err[k])
        perr[f"Ea_{ch}"] = float(err[n_ch + k])
    residuals = pd.DataFrame({
        "pO2": pO2, "T_C": Tc, "sigma_exp_Sm": sig,
        "sigma_model_Sm": model, "resid_rel": (model - sig) / sig,
    })
    return {
        "params": params, "perr": perr, "r2": float(r2),
        "residuals": residuals, "n_points": n_points, "converged": bool(polish.success),
    }


def global_transference_table(
    df_peak: pd.DataFrame, params: ModelParams, exponent: float = 0.25,
) -> pd.DataFrame:
    """Per-(T, pO2) ionic/electronic split FROM the fitted global model.

    Returns the same columns as ``plots.fit_transference`` so the Stage-4
    Brouwer/transference figure can be redrawn from the refined global model
    instead of the per-isotherm NNLS. The channel conductivities are the model
    prefactor terms ``sigma0/T * exp(-Ea/kT)`` in S/cm (constant per T); the
    pO2 dependence ``pO2**(+-x)`` is applied by the plot, exactly as for
    ``fit_transference``. Input columns: ``T_nominal`` [C], ``pO2_mean`` [bar],
    ``sigma_Sm_i`` [S/m].
    """
    Tc = pd.to_numeric(df_peak["T_nominal"], errors="coerce").to_numpy(float)
    pO2 = pd.to_numeric(df_peak["pO2_mean"], errors="coerce").to_numpy(float)
    sig = pd.to_numeric(df_peak["sigma_Sm_i"], errors="coerce").to_numpy(float)
    pid = int(df_peak["peak_id"].iloc[0]) if "peak_id" in df_peak.columns and len(df_peak) else 1

    good = np.isfinite(Tc) & np.isfinite(pO2) & np.isfinite(sig) & (pO2 > 0) & (sig > 0)
    Tc, pO2, sig = Tc[good], pO2[good], sig[good]
    T_K = Tc + 273.15
    to_Scm = 100.0  # S/m -> S/cm

    s_ion = _channel_sigma0T(params.sigma0_ion, params.Ea_ion, T_K) / to_Scm
    s_p = _channel_sigma0T(params.sigma0_p, params.Ea_p, T_K) / to_Scm
    s_n = _channel_sigma0T(params.sigma0_n, params.Ea_n, T_K) / to_Scm
    tot = s_ion + s_p * pO2 ** exponent + s_n * pO2 ** (-exponent)
    t_ion = np.divide(s_ion, tot, out=np.zeros_like(tot), where=tot > 0)

    return pd.DataFrame({
        "peak_id": pid, "T_nominal": Tc.astype(int), "pO2": pO2,
        "sigma_Scm": sig / to_Scm, "sigma_ion": s_ion, "sigma_p": s_p, "sigma_n": s_n,
        "R2": np.nan, "t_ion": t_ion, "t_el": 1.0 - t_ion,
    })
