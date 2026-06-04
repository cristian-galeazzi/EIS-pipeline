"""
pipeline/drt.py
===============
DRT computation (Tikhonov / mGCV) and peak extraction.

Main functions
--------------
compute_drt(freq, Z_re, Z_im, ...)   -> EIS_object with DRT result
find_drt_peaks(entry, ...)           -> list of peak dicts
clip_spectrum(freq, Z_re, Z_im, ...) -> clipped arrays

Peak dict keys
--------------
peak_id     : 1-based index sorted by ascending τ (= descending frequency)
              peak_id=1 → highest-frequency process (e.g., bulk)
tau         : characteristic timescale [s]  (= 1 / (2π f_peak))
gamma_peak  : DRT value at peak maximum [Ohm]
R_approx    : area under the peak [Ohm]  — used as initial guess for Zarc R
tau_left    : left integration boundary [s]
tau_right   : right integration boundary [s]
"""

from __future__ import annotations

import io
import contextlib

import numpy as np
from scipy.signal import find_peaks
from scipy.integrate import trapezoid

import sys
from unittest.mock import MagicMock
# stub pyDRTtools GUI submodules to avoid requiring PyQt5
for _mod in ("pyDRTtools.GUI", "pyDRTtools.cli", "pyDRTtools.layout"):
    sys.modules.setdefault(_mod, MagicMock())

from pyDRTtools.runs import EIS_object, simple_run


# ---------------------------------------------------------------------------
# Spectrum clipping
# ---------------------------------------------------------------------------

def clip_spectrum(
    freq:    np.ndarray,
    Z_re:    np.ndarray,
    Z_im:    np.ndarray,
    f_min:   float | None = None,
    f_max:   float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Restrict frequency range to [f_min, f_max].

    Parameters
    ----------
    freq, Z_re, Z_im : raw arrays from IsmRecord
    f_min, f_max     : cutoff frequencies from Stage 2 Lin-KK (None = no cutoff)

    Returns
    -------
    (freq_clip, Z_re_clip, Z_im_clip) — same dtype as inputs
    """
    mask = np.ones(len(freq), dtype=bool)
    if f_min is not None:
        mask &= freq >= f_min
    if f_max is not None:
        mask &= freq <= f_max
    return freq[mask], Z_re[mask], Z_im[mask]


# ---------------------------------------------------------------------------
# DRT computation
# ---------------------------------------------------------------------------

def compute_drt(
    freq:          np.ndarray,
    Z_re:          np.ndarray,
    Z_im:          np.ndarray,
    cv_type:       str   = "mGCV",
    der_used:      str   = "2nd order",
    induct_used:   int   = 0,
    coeff:         float = 0.5,
    reg_param:     float | None = None,
    suppress_output: bool = True,
) -> EIS_object:
    """
    Compute DRT via Tikhonov regularization with automatic λ selection.

    Parameters
    ----------
    freq        : frequency [Hz], any order
    Z_re        : real impedance [Ohm]
    Z_im        : imaginary impedance [Ohm], positive in capacitive region (−Z″)
    cv_type     : regularization parameter selection method
                  'mGCV' (default) — modified Generalized Cross-Validation
                  'GCV'            — Generalized Cross-Validation
                  'LC'             — L-curve
                  'custom'         — use reg_param directly (reg_param must be set)
    der_used    : derivative order for regularization penalty
                  '2nd order' (default) gives smoother, better-shaped peaks
                  '1st order' gives sharper peaks
    induct_used : inductance handling
                  0 = no inductance (default — appropriate for ceramics at 400-600 °C)
                  1 = include inductance term
    coeff       : FWHM coefficient for Gaussian RBF (default 0.5)
    reg_param   : fixed regularization λ used when cv_type='custom'.
                  Ignored for all other cv_type values.
                  Typical range: 1e-4 (sharp peaks) to 1e-2 (very smooth).
    suppress_output : hide pyDRTtools λ print messages

    Returns
    -------
    EIS_object with attributes populated by simple_run:
        .out_tau_vec  : τ grid [s]
        .gamma        : DRT γ(logτ) [Ohm]
        .lambda_value : regularization parameter λ chosen
        .res_re       : fit residuals, real part
        .res_im       : fit residuals, imaginary part
    """
    if cv_type == "custom" and reg_param is None:
        raise ValueError(
            "cv_type='custom' requires reg_param to be set. "
            "Example: reg_param=1e-3  (higher = smoother DRT)"
        )

    # pyDRTtools requires frequency in DESCENDING order (high → low) so that
    # 1/freq (= tau) is ascending and compute_epsilon returns a positive shape factor.
    # IsmRecord stores Z_im sign-flipped vs physical convention (positive in
    # capacitive region); pyDRTtools expects the physical sign (negative).
    sort_idx = np.argsort(freq)[::-1]
    entry = EIS_object(freq[sort_idx], Z_re[sort_idx], -Z_im[sort_idx])

    _kwargs = dict(
        rbf_type      = "Gaussian",
        data_used     = "Combined Re-Im Data",
        induct_used   = induct_used,
        der_used      = der_used,
        cv_type       = cv_type,
        shape_control = "FWHM Coefficient",
        coeff         = coeff,
    )
    if reg_param is not None:
        _kwargs["reg_param"] = reg_param

    if suppress_output:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            simple_run(entry, **_kwargs)
    else:
        simple_run(entry, **_kwargs)

    return entry


# ---------------------------------------------------------------------------
# Peak extraction
# ---------------------------------------------------------------------------

def find_drt_peaks(
    entry:             EIS_object,
    min_height_frac:   float = 0.05,
    min_distance:      int   = 5,
    min_prom_decades:  float | None = None,
    min_dist_decades:  float | None = None,
) -> list[dict]:
    """
    Detect peaks in the DRT spectrum and estimate their areas.

    Peak area R_i = ∫ γ(ln τ) d(ln τ) integrated between valley points
    on either side of the peak.  This is the Tikhonov-DRT estimate of the
    resistance associated with that relaxation process.  It serves as the
    initial guess for the Zarc R_i parameter (Stage 3 fitting).
    pyDRTtools defines γ w.r.t. d(ln τ) (natural log); integrating over
    d(log10 τ) would underestimate R by a factor of ln(10) ≈ 2.303.

    Parameters
    ----------
    entry             : EIS_object after compute_drt()
    min_height_frac   : peaks below this fraction of γ_max are ignored
                        (default 0.05 = 5 % of max — filters noise floor and
                        HF/LF boundary artifacts that appear at the grid edges)
    min_distance      : minimum number of grid points between peaks
                        (default 5 on the fine log-τ grid). Used only when
                        min_dist_decades is None.
    min_dist_decades  : minimum peak separation expressed in decades of τ
                        (a factor of 10 in τ = 1.0 decade). This is the
                        RelaxIS convention (§7.2.5.2: minimum distance in
                        ln τ units), made grid-resolution independent by
                        converting to grid points from the actual log-τ
                        spacing. When set, overrides min_distance. Typical:
                        0.3–0.5 decades for overlapping electrode processes.
    min_prom_decades  : when not None, switch to log10(γ)-based detection with
                        prominence threshold in decades. Recovers sub-peaks that
                        the absolute height threshold misses when one peak
                        dominates γ_max (e.g. a large electrode arc at τ≈10⁻² s
                        masking sub-GB peaks). Typical value: 0.05 (≈12 % local
                        rise above neighbouring valleys). When None (default),
                        the original height-threshold algorithm is used.

    Returns
    -------
    List of dicts sorted by ascending τ (= descending frequency):
        peak_id     : int, 1-based (peak_id=1 = highest-frequency process)
        tau         : float [s], position of peak maximum
        gamma_peak  : float [Ohm], DRT value at peak
        R_approx    : float [Ohm], integrated area (initial Zarc bound)
        tau_left    : float [s], left integration boundary
        tau_right   : float [s], right integration boundary
    Returns [] if no peaks are found.
    """
    tau   = entry.out_tau_vec   # fine log-spaced τ grid [s]
    gamma = entry.gamma         # DRT values [Ohm]

    if gamma.max() == 0:
        return []

    # Convert a decade-based minimum separation (RelaxIS convention) into the
    # grid-point distance that scipy.find_peaks expects. The τ grid is
    # log-spaced, so the step in log10(τ) per point is constant.
    distance_pts = min_distance
    if min_dist_decades is not None and len(tau) > 1:
        log10_tau = np.log10(tau)
        dlog10    = abs(log10_tau[-1] - log10_tau[0]) / (len(tau) - 1)
        if dlog10 > 0:
            distance_pts = max(1, int(round(min_dist_decades / dlog10)))

    if min_prom_decades is not None:
        # Log-prominence detection: works on log10(γ) so a peak at γ = 5 % of γ_max
        # is still found if its local rise above surrounding valleys is significant.
        # Keep an absolute floor at min_height_frac × γ_max to reject true noise.
        log_gamma     = np.log10(np.maximum(gamma, 1.0))
        height_floor  = np.log10(max(min_height_frac * gamma.max(), 1.0))
        peak_indices, _ = find_peaks(
            log_gamma,
            prominence=float(min_prom_decades),
            distance=distance_pts,
            height=height_floor,
        )
    else:
        # Original behaviour preserved for backward compatibility.
        height_thresh = min_height_frac * gamma.max()
        peak_indices, _ = find_peaks(gamma, height=height_thresh,
                                      distance=distance_pts)

    if len(peak_indices) == 0:
        return []

    ln_tau = np.log(tau)   # natural log — matches pyDRTtools' d(ln τ) convention
    n = len(tau)

    def valley_idx(i_left: int, i_right: int) -> int:
        """Return index of minimum γ between two peak indices."""
        segment = gamma[i_left : i_right + 1]
        return i_left + int(np.argmin(segment))

    peaks = []
    for k, idx in enumerate(peak_indices):
        # Left boundary: valley between previous peak and this one
        if k == 0:
            left_idx = 0
        else:
            left_idx = valley_idx(peak_indices[k - 1], idx)

        # Right boundary: valley between this peak and the next one
        if k == len(peak_indices) - 1:
            right_idx = n - 1
        else:
            right_idx = valley_idx(idx, peak_indices[k + 1])

        # Integrate γ over d(ln τ) — correct for pyDRTtools' normalization
        R_i = float(trapezoid(
            gamma[left_idx : right_idx + 1],
            ln_tau[left_idx : right_idx + 1],
        ))

        peaks.append({
            "peak_id":    k + 1,          # will be renumbered after sort
            "tau":        float(tau[idx]),
            "gamma_peak": float(gamma[idx]),
            "R_approx":   max(R_i, 1e-6), # guard against zero or negative area
            "tau_left":   float(tau[left_idx]),
            "tau_right":  float(tau[right_idx]),
        })

    # Sort by ascending τ and assign final peak_id
    peaks.sort(key=lambda p: p["tau"])
    for k, p in enumerate(peaks):
        p["peak_id"] = k + 1

    return peaks
