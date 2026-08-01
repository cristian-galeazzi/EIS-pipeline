"""
pipeline/quality.py
===================
Lin-KK (Kramers-Kronig) quality assessment and replica selection.

Implements the linearized KK test as used in relaXIS (Zahner), based on:
    Schönleber, M. et al. - "A Method for Improving the Robustness of
    Linear Kramers-Kronig Validity Tests" - Electrochimica Acta 131 (2014) 20-27.

Validation criterion (matching relaXIS workflow):
    Both magnitude-normalized real AND imaginary residuals must pass the
    Shapiro-Wilk normality test at W ≥ 0.95 (95% confidence level),
    evaluated on the frequency-trimmed spectrum (edge artifacts removed).

Magnitude-normalized residual definition:
    r_re[i] = (Z_re[i] − Z_fit_re[i]) / |Z[i]|
    r_im[i] = (Z_im[i] − Z_fit_im[i]) / |Z[i]|
    |Z[i]|  = sqrt(Z_re[i]² + Z_im[i]²)

Two-pass approach (mirrors RelaxIS interactive trimming):
    Pass 1 - fit full spectrum → identify edge frequencies where |r| > threshold
    Pass 2 - re-fit on trimmed spectrum → compute kk_score from trimmed residuals
    Frequency cutoffs from Pass 1 are propagated to Stage 3.

M selection, automatic mode (use_binary_M=True):
    Finds the smallest M such that the sign-change fraction μ ≥ mu_target (0.50).
    μ = fraction of adjacent RC weight pairs with opposite signs:
        μ → 0  for small M (underfitting: all RC weights same sign)
        μ → 1  for large M (overfitting: alternating RC weight signs)
    The trend is not monotonic, so M is found by a linear scan from M = 3
    upward (a bisection could skip valid M values).
    Target μ = 0.50 balances the two regimes (RelaxIS default).
    Fixed mode (use_binary_M=False): M = round(c × N_freq), c default 0.85.

Adaptive frequency cutoffs (IQR-based, mirrors RelaxIS KK Filter §7.1.6):
    iqr_fence_factor - fence = Q3_interior + factor × IQR_interior.  Default 2.0.
    iqr_window       - consecutive clean points required to confirm the cut edge.
    No manual threshold needed; the fence adapts to each spectrum's noise floor.

Replaces the original Bayesian Hilbert Transform (BHT) approach which required
~8 min/spectrum on 8 GB RAM. Lin-KK runs in < 5 ms/spectrum.

Main functions
--------------
run_linkk(freq, Z_re, Z_im, ...)        -> KK results dict
select_best_replica(kk_results)         -> index of best replica
compute_frequency_cutoffs(kk_result)    -> (f_min, f_max) to KEEP
kk_summary_table(records, results, idx) -> summary DataFrame
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import shapiro


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

def strip_inductive(
    freq:  np.ndarray,
    Z_re:  np.ndarray,
    Z_im:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Drop non-physical points before Lin-KK.

    Two kinds of points violate KK causality and must be removed first:
      - inductive points (Z_im < 0): IsmRecord stores Z_im with positive sign
        for capacitive points, so Z_im < 0 marks the highest-frequency
        inductive tail.
      - negative-real points (Z_re < 0): a high-frequency measurement artifact
        (lead inductance resonating with stray capacitance) seen in
        high-impedance ceramic spectra. No passive circuit can produce
        Z_re < 0, so these points break the KK fit and any later Zarc fit.

    Returns (freq, Z_re, Z_im, n_stripped) where n_stripped counts all points
    removed for either reason.

    >>> f  = np.array([1e5, 1e3, 10.0])
    >>> zr = np.array([-2.0, 50.0, 80.0])
    >>> zi = np.array([-1.0, 30.0, 5.0])
    >>> f2, zr2, zi2, n = strip_inductive(f, zr, zi)
    >>> n, f2.tolist()
    (1, [1000.0, 10.0])
    """
    mask = (Z_im >= 0) & (Z_re >= 0)
    n_stripped = int((~mask).sum())
    return freq[mask], Z_re[mask], Z_im[mask], n_stripped


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fit_linkk(
    freq:           np.ndarray,
    Z_re:           np.ndarray,
    Z_im:           np.ndarray,
    c:              float = 0.85,
    M_override:     Optional[int] = None,
    add_inductance: bool = False,
) -> dict:
    """
    Single-pass Lin-KK fit on a given (possibly trimmed) frequency set.

    M_override takes precedence over c when provided.
    Frequencies must be sorted ascending.

    Returns fitted impedance, magnitude-normalized residuals, and
    Shapiro-Wilk statistics.
    """
    N = len(freq)
    if N < 4:
        # With N <= 3 the clamp below would force M = 3 > N - 1 and the fit
        # would run over-parametrized on a near-empty spectrum.
        raise ValueError(f"Lin-KK needs at least 4 frequency points, got {N}")
    omega = 2.0 * np.pi * freq

    M = M_override if M_override is not None else max(3, round(c * N))
    M = max(3, min(M, N - 1))   # clamp to valid range

    tau = np.logspace(
        np.log10(1.0 / omega.max()),
        np.log10(1.0 / omega.min()),
        M,
    )

    omega_tau = np.outer(omega, tau)
    denom     = 1.0 + omega_tau ** 2

    A_re = np.hstack([np.ones((N, 1)),  1.0 / denom])
    A_im = np.hstack([np.zeros((N, 1)), omega_tau / denom])

    if add_inductance:
        A_re = np.hstack([A_re, np.zeros((N, 1))])
        A_im = np.hstack([A_im, -omega.reshape(-1, 1)])

    A = np.vstack([A_re, A_im])
    b = np.concatenate([Z_re, Z_im])

    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    Z_fit_re = A_re @ x
    Z_fit_im = A_im @ x

    Z_mag = np.sqrt(Z_re ** 2 + Z_im ** 2)
    Z_mag = np.where(Z_mag > 0, Z_mag, np.finfo(float).eps)
    res_re = (Z_re - Z_fit_re) / Z_mag
    res_im = (Z_im - Z_fit_im) / Z_mag

    W_re, _ = shapiro(res_re)
    W_im, _ = shapiro(res_im)

    return {
        "Z_fit_re":  Z_fit_re,
        "Z_fit_im":  Z_fit_im,
        "res_re":    res_re,
        "res_im":    res_im,
        "W_re":      float(W_re),
        "W_im":      float(W_im),
        "M":         M,
        "R_inf":     float(x[0]),
        "R_weights": x[1 : M + 1],   # RC weights only (excludes R_inf)
    }


def _mu_sign_changes(R_weights: np.ndarray) -> float:
    """
    Sign-change fraction μ of adjacent RC weights.

    μ = (number of sign changes between consecutive weights) / (M − 1)

    Tends towards 0 for small M (underfitting: all RC weights same sign)
    and towards 1 for large M (overfitting: alternating signs), but the
    trend is NOT monotonic: individual M values can dip below or jump
    above their neighbours. Used as the stopping criterion in the linear
    M scan of _find_optimal_M().
    """
    if len(R_weights) < 2:
        return 0.0
    signs     = np.sign(R_weights)
    n_changes = int(np.sum(signs[:-1] != signs[1:]))
    return n_changes / (len(R_weights) - 1)


def _find_optimal_M(
    freq:           np.ndarray,
    Z_re:           np.ndarray,
    Z_im:           np.ndarray,
    mu_target:      float = 0.50,
    add_inductance: bool  = False,
) -> int:
    """
    Linear scan for the optimal number of RC elements M.

    Returns the smallest M such that μ(M) ≥ mu_target, where μ is the
    sign-change fraction of adjacent RC weights.

    A linear scan (not bisection) is required because μ(M) is not
    monotonic in M: bisection can skip over valid M values and settle on
    a larger one. Scanning upward from M = 3 guarantees the true minimum
    and stops at the first hit, so in practice it costs only a few more
    Lin-KK fits than bisection did.

    Search range: M ∈ [3, N−1].

    Parameters
    ----------
    freq           : frequency array [Hz], sorted ascending
    Z_re, Z_im     : impedance arrays [Ω]
    mu_target      : target sign-change fraction (default 0.50)
    add_inductance : pass-through to _fit_linkk

    Returns
    -------
    int : optimal M (clamped to [3, N−1])
    """
    N = len(freq)
    for M in range(3, N):
        r  = _fit_linkk(freq, Z_re, Z_im,
                        M_override=M,
                        add_inductance=add_inductance)
        mu = _mu_sign_changes(r["R_weights"])
        if mu >= mu_target:
            return M

    return max(3, round(0.85 * N))   # no M satisfied μ ≥ target


def _edge_cutoffs_adaptive(
    freq:         np.ndarray,
    res_re:       np.ndarray,
    res_im:       np.ndarray,
    fence_factor: float = 2.0,
    window:       int   = 3,
) -> tuple[Optional[float], Optional[float], float]:
    """
    Adaptive IQR-based frequency cutoffs (mirrors RelaxIS KK Filter §7.1.6).

    Algorithm
    ---------
    1. Per-point worst-case magnitude: mag[i] = max(|res_re[i]|, |res_im[i]|).
    2. Compute Q3 and IQR on the central 60 % of the spectrum (interior only,
       so edge artefacts do not inflate the fence estimate).
    3. Upper fence = max(Q3 + fence_factor × IQR, 0.05).
       The 0.05 floor (5 % of |Z|) prevents cutting clean edge points when the
       interior noise is extremely small.
    4. Mark points as "bad" where mag > fence.
    5. Continuous-range finder (edge inward):
         HF: scan from highest freq downward; cut until the first run of
             `window` consecutive clean points - f_max = top of that run.
         LF: scan from lowest freq upward; cut until the first run of
             `window` consecutive clean points - f_min = bottom of that run.
       This matches RelaxIS "Apply as continuous range / window size" option.

    Parameters
    ----------
    fence_factor : IQR fence multiplier. Default 2.0 (RelaxIS default).
                   Lower → tighter fence → more aggressive cut.
    window       : consecutive clean points required to confirm the cut edge.
                   Default 3. Prevents triggering on a single isolated good point.

    Returns
    -------
    (f_min, f_max, fence) : frequency limits to KEEP [Hz] and fence value used.
    """
    res_mag = np.maximum(np.abs(res_re), np.abs(res_im))
    N = len(res_mag)

    # Interior IQR on the central 60 % of the spectrum
    i_lo = max(0, int(0.20 * N))
    i_hi = min(N, int(0.80 * N))
    core = res_mag[i_lo:i_hi] if (i_hi - i_lo) >= 4 else res_mag
    q1, q3 = np.percentile(core, [25, 75])
    iqr    = q3 - q1
    fence  = max(float(q3 + fence_factor * iqr), 0.05)

    bad = res_mag > fence

    # HF cutoff: scan downward from highest frequency
    f_max      = float(freq[-1])
    good_count = 0
    hf_found   = False
    for i in range(N - 1, -1, -1):
        if not bad[i]:
            good_count += 1
            if good_count >= window:
                f_max = float(freq[i + (window - 1)])
                hf_found = True
                break
        else:
            good_count = 0

    # LF cutoff: scan upward from lowest frequency
    f_min      = float(freq[0])
    good_count = 0
    lf_found   = False
    for i in range(N):
        if not bad[i]:
            good_count += 1
            if good_count >= window:
                f_min = float(freq[i - (window - 1)])
                lf_found = True
                break
        else:
            good_count = 0

    if not (hf_found and lf_found):
        warnings.warn(
            f"adaptive edge cutoff found no run of {window} consecutive "
            f"clean points; full frequency range kept and Pass 2 will refit "
            f"the untrimmed spectrum",
            stacklevel=2,
        )

    return f_min, f_max, fence


# ---------------------------------------------------------------------------
# Lin-KK main function
# ---------------------------------------------------------------------------

def run_linkk(
    freq:             np.ndarray,
    Z_re:             np.ndarray,
    Z_im:             np.ndarray,
    c:                float = 0.85,
    use_binary_M:     bool  = True,
    mu_target:        float = 0.50,
    add_inductance:   bool  = False,
    iqr_fence_factor: float = 2.0,
    iqr_window:       int   = 3,
    f_min_hard:       Optional[float] = None,
    f_max_hard:       Optional[float] = None,
) -> dict:
    """
    Two-pass Linearized Kramers-Kronig test (Schönleber et al. 2014).

    Pass 1: fit full spectrum → identify edge frequency cutoffs.
    Pass 2: re-fit on trimmed spectrum → compute Shapiro-Wilk on clean data.

    This mirrors the relaXIS interactive workflow where the operator trims
    edge points until the KK residuals pass the normality test.

    M selection
    -----------
    use_binary_M=True (default, automatic mode):
        Linear scan upward from M = 3 for the smallest M where the
        sign-change fraction μ ≥ mu_target (0.50).  A scan and not a
        bisection: μ(M) is not monotonic in M, so bisection can skip valid
        M and settle on a larger one.  Run independently for Pass 1 and
        Pass 2.  The flag name is historical and predates the switch away
        from bisection; the behavior it selects is the scan.
    use_binary_M=False (fixed-density mode):
        M = round(c × N_freq), same for both passes.

    Frequency cutoffs
    -----------------
    Adaptive IQR-based approach (mirrors RelaxIS KK Filter §7.1.6):
        The fence = max(Q3_interior + iqr_fence_factor × IQR_interior, 0.05).
        Scanning from each edge inward, the cutoff is placed at the start of
        the first run of `iqr_window` consecutive clean points.

    f_min_hard overrides the adaptive LF cutoff when set.  Use this when the
    electrode contribution starts at a physically known frequency (e.g. non-
    sintered electrode/wire/cement): the adaptive IQR stops at the first clean
    GB points, leaving electrode noise in the window; f_min_hard forces the
    floor regardless of what the residuals look like below it.

    Parameters
    ----------
    freq             : frequency array [Hz], any order
    Z_re             : real part [Ω]
    Z_im             : imaginary part [Ω], positive for capacitive half
                       (IsmRecord convention: Z_im = −Z_complex.imag)
    c                : RC density (only used when use_binary_M=False). Default 0.85.
    use_binary_M     : if True, linear scan for the smallest M with μ ≥ mu_target
                       (the name is historical). If False, M = round(c × N_freq).
                       Default True.
    mu_target        : sign-change fraction target for binary search. Default 0.50.
    add_inductance   : add inductive element for HF inductive loops. Default False.
    iqr_fence_factor : IQR fence multiplier.  Default 2.0 (RelaxIS default).
                       Lower → tighter fence → more points cut at the edges.
    iqr_window       : consecutive clean points to confirm the cut edge. Default 3.
    f_min_hard       : hard lower frequency floor [Hz].  If the adaptive IQR
                       cutoff falls below this value it is raised to f_min_hard.
                       None = adaptive only (default).
    f_max_hard       : hard upper frequency floor [Hz].  If the adaptive IQR
                       cutoff falls below this value it is raised to f_max_hard,
                       preserving HF data (e.g. the bulk arc).  The IQR is also
                       re-estimated on freq ≤ f_max_hard to prevent HF boundary
                       artifacts from inflating the fence.
                       None = adaptive only (default).

    Returns
    -------
    dict with keys
        W_re, W_im        : Shapiro-Wilk W on trimmed spectrum (0–1)
        pass_re           : bool - W_re ≥ 0.95
        pass_im           : bool - W_im ≥ 0.95
        kk_score          : (W_re + W_im) / 2 - primary ranking metric
        mu                : sign-change fraction on the final (Pass 2) fit
        M                 : number of RC elements (Pass 2)
        res_re            : magnitude-normalized real residuals (full spectrum, Pass 1)
        res_im            : magnitude-normalized imag residuals (full spectrum, Pass 1)
        freq              : sorted frequency array (full spectrum)
        f_min_cut         : lowest frequency to keep [Hz]  (≥ f_min_hard if set)
        f_max_cut         : highest frequency to keep [Hz] (≥ f_max_hard if set)
        cutoff_fence      : adaptive IQR fence value used [fraction of |Z|]
        Z_fit_re          : Lin-KK fitted real part (full spectrum, Pass 1)
        Z_fit_im          : Lin-KK fitted imag part (full spectrum, Pass 1)
        R_inf             : fitted series resistance [Ω] (Pass 2)
        R_weights         : fitted RC weights [Ω] (Pass 2, M-element array)

    >>> res = run_linkk(freq, Z_re, Z_im, use_binary_M=True)     # doctest: +SKIP
    >>> res["pass_re"], res["pass_im"], round(res["kk_score"], 3)  # doctest: +SKIP
    (True, True, 0.985)
    """
    # Sort by ascending frequency
    idx  = np.argsort(freq)
    freq = freq[idx]
    Z_re = Z_re[idx]
    Z_im = Z_im[idx]

    # ── Pass 1: full spectrum ─────────────────────────────────────────────
    if use_binary_M:
        M1 = _find_optimal_M(freq, Z_re, Z_im,
                             mu_target=mu_target,
                             add_inductance=add_inductance)
    else:
        M1 = None   # _fit_linkk will use c

    p1 = _fit_linkk(freq, Z_re, Z_im,
                    c=c, M_override=M1,
                    add_inductance=add_inductance)

    # IQR cutoffs - computed on the physically meaningful interior only.
    # f_min_hard and f_max_hard pre-clip the residuals before IQR estimation so
    # that noisy LF/HF boundary regions don't inflate the fence and cause
    # under-cutting in the valid interior. Full-spectrum Pass 1 residuals are
    # still returned for plotting so the operator can see the excluded region.
    _hard_mask = np.ones(len(freq), dtype=bool)
    if f_min_hard is not None:
        _hard_mask &= freq >= float(f_min_hard)
    if f_max_hard is not None:
        _hard_mask &= freq <= float(f_max_hard)

    _freq_iqr = freq[_hard_mask]
    _re_iqr   = p1["res_re"][_hard_mask]
    _im_iqr   = p1["res_im"][_hard_mask]

    if len(_freq_iqr) >= 8:
        f_min, f_max, cutoff_fence = _edge_cutoffs_adaptive(
            _freq_iqr, _re_iqr, _im_iqr, iqr_fence_factor, iqr_window
        )
    else:
        f_min, f_max, cutoff_fence = _edge_cutoffs_adaptive(
            freq, p1["res_re"], p1["res_im"], iqr_fence_factor, iqr_window
        )

    # Hard limits define the keep-window: no adaptive cut may cross them.
    # f_min_hard is a floor (excludes electrode noise from below), f_max_hard
    # a ceiling (excludes HF cable/inductive artifacts from above). The clamp
    # also covers the <8-point fallback above, where the adaptive scan ran on
    # the full spectrum instead of the hard-masked one.
    if f_min_hard is not None:
        f_min = max(f_min, float(f_min_hard))
    if f_max_hard is not None:
        f_max = min(f_max, float(f_max_hard))

    # ── Pass 2: trimmed spectrum ──────────────────────────────────────────
    mask   = (freq >= f_min) & (freq <= f_max)
    freq_t = freq[mask]
    Z_re_t = Z_re[mask]
    Z_im_t = Z_im[mask]

    if len(freq_t) >= 4:
        if use_binary_M:
            M2 = _find_optimal_M(freq_t, Z_re_t, Z_im_t,
                                 mu_target=mu_target,
                                 add_inductance=add_inductance)
        else:
            M2 = None

        p2    = _fit_linkk(freq_t, Z_re_t, Z_im_t,
                           c=c, M_override=M2,
                           add_inductance=add_inductance)
    else:
        # Not enough points after trimming - fall back to Pass 1
        p2 = p1

    W_re  = p2["W_re"]
    W_im  = p2["W_im"]
    M_out = p2["M"]
    R_inf = p2["R_inf"]
    R_w   = p2["R_weights"]

    return {
        "W_re":      float(W_re),
        "W_im":      float(W_im),
        "pass_re":   bool(W_re >= 0.95),
        "pass_im":   bool(W_im >= 0.95),
        "kk_score":  float((W_re + W_im) / 2.0),
        "mu":        float(_mu_sign_changes(R_w)),
        "M":         M_out,
        "res_re":       p1["res_re"],    # full spectrum residuals (for plotting)
        "res_im":       p1["res_im"],
        "freq":         freq,
        "f_min_cut":    f_min,
        "f_max_cut":    f_max,
        "cutoff_fence": cutoff_fence,
        "Z_fit_re":     p1["Z_fit_re"],
        "Z_fit_im":     p1["Z_fit_im"],
        "R_inf":        R_inf,
        "R_weights":    R_w,
    }


# ---------------------------------------------------------------------------
# Replica selection
# ---------------------------------------------------------------------------

def select_best_replica(kk_results: list[dict]) -> int:
    """
    Return the index of the replica with the highest kk_score.

    kk_score = (W_re + W_im) / 2 on the trimmed spectrum.
    Higher = more normally distributed residuals = better KK compliance.

    Parameters
    ----------
    kk_results : list of dicts from run_linkk()

    Returns
    -------
    int : 0-based index of the best replica

    Raises
    ------
    ValueError : if kk_results is empty (all replicas rejected upstream).

    >>> select_best_replica([{"kk_score": 0.91}, {"kk_score": 0.97}])
    1
    """
    if not kk_results:
        raise ValueError(
            "select_best_replica: kk_results is empty; no replicas to score. "
            "All spectra for this (condition, T) were likely rejected upstream."
        )
    scores = np.asarray([r["kk_score"] for r in kk_results], dtype=float)
    if np.isnan(scores).all():
        raise ValueError(
            "select_best_replica: every kk_score is NaN; Shapiro-Wilk "
            "degenerated on all replicas for this (condition, T)."
        )
    # NaN wins a plain argmax in numpy; a degenerate replica must never
    # outrank a finite-scored one.
    return int(np.argmax(np.where(np.isnan(scores), -np.inf, scores)))


# ---------------------------------------------------------------------------
# Frequency cutoffs (used by Stage 3)
# ---------------------------------------------------------------------------

def compute_frequency_cutoffs(
    kk_result: dict,
) -> tuple[Optional[float], Optional[float]]:
    """
    Return the frequency cutoffs stored in the Lin-KK result.

    Cutoffs are computed during Pass 1 of run_linkk() and stored directly
    in the result dict. This function simply retrieves them for compatibility
    with the Stage 2 export and Stage 3 clipping.

    Parameters
    ----------
    kk_result : dict from run_linkk()

    Returns
    -------
    (f_min, f_max) : frequency range to KEEP [Hz]

    >>> compute_frequency_cutoffs({"f_min_cut": 10.0, "f_max_cut": 1e5})
    (10.0, 100000.0)
    """
    return kk_result.get("f_min_cut"), kk_result.get("f_max_cut")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def kk_summary_table(
    records:    list,
    kk_results: list[dict],
    best_idx:   int,
) -> pd.DataFrame:
    """
    Build a per-replica summary DataFrame for one (condition, T_nominal) group.

    Parameters
    ----------
    records    : list of IsmRecord objects (same order as kk_results)
    kk_results : list of dicts from run_linkk()
    best_idx   : 0-based index of the auto-selected replica

    Returns
    -------
    pd.DataFrame with columns:
        file, replica, kk_score, W_re, W_im, pass_re, pass_im,
        mu, M, max_res_re, max_res_im, f_min_cut, f_max_cut, selected

    >>> from types import SimpleNamespace
    >>> from pathlib import Path
    >>> rec = SimpleNamespace(path=Path("r1.ism"), replica=1)
    >>> res = {"kk_score": 0.96, "W_re": 0.97, "W_im": 0.95,
    ...        "pass_re": True, "pass_im": True,
    ...        "res_re": np.array([0.01]), "res_im": np.array([0.02]),
    ...        "f_min_cut": 1.0, "f_max_cut": 1e5}
    >>> df = kk_summary_table([rec], [res], best_idx=0)
    >>> df.loc[0, "file"], float(df.loc[0, "kk_score"]), bool(df.loc[0, "selected"])
    ('r1.ism', 0.96, True)
    """
    rows = []
    for i, (rec, res) in enumerate(zip(records, kk_results)):
        f_min, f_max = compute_frequency_cutoffs(res)
        rows.append({
            "file":       rec.path.name,
            "replica":    rec.replica,
            "kk_score":   round(res["kk_score"],  4),
            "W_re":       round(res["W_re"],       4),
            "W_im":       round(res["W_im"],       4),
            "pass_re":    res["pass_re"],
            "pass_im":    res["pass_im"],
            "mu":         round(res.get("mu", float("nan")), 3),
            "M":          res.get("M"),
            "max_res_re":    round(float(np.abs(res["res_re"]).max()), 4),
            "max_res_im":    round(float(np.abs(res["res_im"]).max()), 4),
            "cutoff_fence":  round(res.get("cutoff_fence", float("nan")), 4),
            "f_min_cut":     f_min,
            "f_max_cut":     f_max,
            "selected":      i == best_idx,
        })
    return pd.DataFrame(rows)
