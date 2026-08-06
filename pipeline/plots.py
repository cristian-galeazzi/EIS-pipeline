"""
pipeline/plots.py
=================
Publication-quality plot functions for the EIS pipeline.

Visual style is adapted exactly from the existing reference notebooks:
  - DRT_Analysis_SAMPLE_ID_(Ar-SCCM_O2-SCCM).ipynb  → DRT stacked, Arrhenius panel
  - Analysis_SAMPLE_ID_(Ar-SCCM_O2-SCCM).ipynb      → Nyquist, Bode
  - Brouwer_pO2_Dependence_SAMPLE_ID.ipynb           → Brouwer diagram

All functions:
  apply_pub_style()               - set rcParams once per session
  plot_drt_stacked()              - stacked DRT γ(τ) with vertical offset
  plot_nyquist_multipanel()       - data circles + fit dashes, all temperatures
  plot_bode()                     - |Z| and phase Bode, all temperatures
  plot_arrhenius_panel()          - 2×2 panel: ln(σT), ln(τ), ln(C), log₁₀(εᵣ)
  plot_brouwer()                  - Brouwer p(O₂) diagram (multi-condition)

Supporting helpers (public API):
  build_arrhenius_results()       - compute Ea, R², pre-exponentials from df_peaks
  COLOR_MAP                       - T [°C] → hex colour dict
  PEAK_COLORS, PEAK_MARKERS       - per-peak visual style
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator
from scipy import stats, optimize
from pathlib import Path

from pipeline.utils import format_pO2_value

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
from pipeline.model import KB_EV as KB  # Boltzmann constant [eV/K], shared
EPS_0 = 8.854e-12  # vacuum permittivity [F/m]

# ---------------------------------------------------------------------------
# Visual constants - must match existing notebooks exactly
# ---------------------------------------------------------------------------

#: Temperature [°C] → hex colour (9 steps, 400–600 °C)
COLOR_MAP: dict[int, str] = {
    400: "#0000CC", 425: "#0066FF", 450: "#00AAFF",
    475: "#00CCAA", 500: "#00AA44", 525: "#AAAA00",
    550: "#FF8800", 575: "#FF4400", 600: "#CC0000",
}

#: Per-peak colours for Arrhenius / multi-peak panels
PEAK_COLORS:  list[str] = ["#0066FF", "#00AA44", "#CC0000",
                             "#FF8800", "#9900CC", "#00CCCC"]

#: Per-peak marker styles
PEAK_MARKERS: list[str] = ["o", "s", "^", "d", "v", "P"]

# Brouwer diagram temperature style (distinct from COLOR_MAP, taken from reference notebook)
_TEMP_STYLE_BROUWER: dict[int, dict] = {
    400: dict(color="#1f4e79", marker="s",  ms=7, label="400 °C"),
    425: dict(color="#2e75b6", marker="s",  ms=7, label="425 °C"),
    450: dict(color="#00b0f0", marker="o",  ms=7, label="450 °C"),
    475: dict(color="#00b050", marker="o",  ms=7, label="475 °C"),
    500: dict(color="#70ad47", marker="^",  ms=7, label="500 °C"),
    525: dict(color="#ffc000", marker="^",  ms=7, label="525 °C"),
    550: dict(color="#ff7c00", marker="D",  ms=6, label="550 °C"),
    575: dict(color="#d04000", marker="D",  ms=6, label="575 °C"),
    600: dict(color="#7f0000", marker="v",  ms=7, label="600 °C"),
}

# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------

def apply_pub_style() -> None:
    """
    Apply publication-quality matplotlib rcParams.
    Call once at the top of each notebook or script.

    Serif (Times New Roman / DejaVu Serif) with STIX math, the Times-metric
    companion: every axis label is one math expression, so a math font of a
    different family would set the whole label in a second serif. Computer
    Modern, used before, is lighter than Times and made the labels recede.
    Inward ticks, top+right tick marks, dpi=150 display / 300 export.

    >>> apply_pub_style()
    >>> mpl.rcParams["figure.dpi"]
    150.0
    """
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":   "stix",
        "font.size":          10,
        "axes.labelsize":     12,
        "axes.titlesize":     12,
        "legend.fontsize":    8,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "xtick.direction":    "in",
        "ytick.direction":    "in",
        "xtick.top":          True,
        "ytick.right":        True,
        "xtick.major.size":   3,
        "xtick.minor.size":   1.5,
        "xtick.major.width":  1.0,
        "xtick.minor.width":  0.7,
        "axes.linewidth":     1.0,
        "lines.linewidth":    1.2,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
    })


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, save_dir: Path | str, stem: str, *, tight: bool = True) -> None:
    """Export figure as PNG and PDF in save_dir.

    tight=False keeps the full figure canvas (no "tight" crop). Needed for 3-D
    axes, where the tight bounding box drops the rotated z-axis label.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    bbox = "tight" if tight else None
    for ext in ("png", "pdf"):
        fig.savefig(save_dir / f"{stem}.{ext}", dpi=300, bbox_inches=bbox)


def _reconstruct_Z_fit(
    freq:      np.ndarray,
    R0:        float,
    R_arr:     np.ndarray,
    tau_arr:   np.ndarray,
    alpha_arr: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct complex Z(f) from Zarc fit parameters without re-fitting.

    Parameters
    ----------
    freq      : frequency array [Hz]
    R0        : ohmic resistance [Ohm]
    R_arr     : per-Zarc resistance values [Ohm]
    tau_arr   : per-Zarc time constants [s]
    alpha_arr : per-Zarc exponents (dimensionless)

    Returns
    -------
    Z_fit : complex array, shape (len(freq),)
    """
    from pipeline.fitting import build_circuit_string
    from impedance.models.circuits import CustomCircuit

    n = len(R_arr)
    circuit_str = build_circuit_string(n)
    params = [float(R0)]
    for R_i, tau_i, a_i in zip(R_arr, tau_arr, alpha_arr):
        params.extend([float(R_i), float(tau_i), float(a_i)])

    circuit = CustomCircuit(circuit_str, initial_guess=params)
    circuit.parameters_ = np.array(params)
    return circuit.predict(freq)


def _arrhenius_linreg(
    inv_T: np.ndarray,
    y:     np.ndarray,
) -> tuple[float, float, float, float, float]:
    """
    Linear regression of y vs 1/T [K⁻¹].

    Returns (Ea_eV, Ea_err_eV, R2, slope, intercept).
    Note: returns -slope * KB as Ea, which is correct only for conductivity
    (where slope < 0). Callers must use slope directly for τ and C_eff.
    """
    valid = ~np.isnan(y) & ~np.isnan(inv_T)
    if valid.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    # This function always returns Ea = -slope*KB, the right sign for
    # ln(σT) vs 1/T (slope < 0). For ln(τ) and ln(C_eff), where the slope
    # sign differs, callers recompute Ea from the returned raw slope.
    slope, intercept, r, _, se = stats.linregress(inv_T[valid], y[valid])
    return -slope * KB, se * KB, r**2, slope, intercept


# ---------------------------------------------------------------------------
# 1. DRT stacked plot
# ---------------------------------------------------------------------------

def plot_drt_stacked(
    df_spectra:    pd.DataFrame,
    condition:     str,
    save_dir:      Path | str,
    tau_max:       float       = 1.0,
    tau_min:       float | None = None,
    offset_step:   float       = 1.2,
    label_tau:     float | None = None,
    exclude_temps: list[int] | None = None,
    save:          bool        = True,
    df_peaks:      pd.DataFrame | None = None,
    label:         str         = "",
    show_pO2:      bool        = True,
) -> plt.Figure:
    """
    Stacked DRT plot - normalized γ(ln τ) with vertical offset per temperature.

    Each trace is normalized to its own maximum, then offset vertically so
    temperatures stack from bottom (low T) to top (high T) with no overlap.

    Parameters
    ----------
    df_spectra    : DataFrame with columns [T_nominal, tau, gamma]
                    (from stage3_drt.xlsx sheet "DRT_Spectra")
    condition     : condition name used for the figure title and filename
    save_dir      : directory where PNG + PDF are written
    tau_max       : x-axis upper limit [s] (default 1.0). Points above it are
                    dropped before each trace is normalized, so it also decides
                    what each maximum is taken over.
    tau_min       : x-axis lower limit [s]. Framing only: no point is dropped
                    and no trace is renormalized, so a window narrowed from the
                    left shows the same curves at the same heights. None keeps
                    the label-driven edge this plot has always used.
    offset_step   : vertical spacing between traces (default 1.2)
    label_tau     : τ position for the T-label text. None puts the labels at
                    ``tau_min`` when that is set, at 5e-8 otherwise, so they
                    never fall outside a narrowed window.
    exclude_temps : list of T_nominal [°C] to skip
    save          : when False, do not write PNG/PDF (diagnostic preview only).
    df_peaks      : optional stage3_fit.xlsx "Peaks" DataFrame; when given and
                    it carries pO2_mean, the median p(O2) is shown as a
                    suptitle, same value the condition selector and the
                    Arrhenius figures use.
    label         : condition label for the suptitle ("" = pressure only)
    show_pO2      : False for a run whose lambda probe was off

    >>> fig = plot_drt_stacked(df_spectra, "Ar_100", "Results/Ar_100/plots",
    ...                        exclude_temps=[400])  # doctest: +SKIP

    Returns
    -------
    matplotlib Figure
    """
    exclude_temps = exclude_temps or []
    temps = sorted(df_spectra["T_nominal"].unique())
    temps = [int(t) for t in temps if int(t) not in exclude_temps]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200, layout="constrained")

    # The labels ride the left edge of the view: pinned to 5e-8, a window
    # narrowed from the left would leave them outside it.
    lbl_x = label_tau if label_tau is not None else (
        tau_min if tau_min is not None else 5e-8)
    x_lo  = tau_min if tau_min is not None else lbl_x * 0.8

    for i, T in enumerate(temps):
        sub     = df_spectra[df_spectra["T_nominal"] == T].sort_values("tau")
        mask    = sub["tau"].values < tau_max
        tau_f   = sub["tau"].values[mask]
        gamma_f = sub["gamma"].values[mask]

        if len(gamma_f) == 0 or gamma_f.max() == 0:
            continue

        gamma_norm = gamma_f / gamma_f.max()
        baseline   = i * offset_step
        color      = COLOR_MAP.get(T, "black")

        ax.axhline(y=baseline, color="gray", linestyle="-",
                   linewidth=0.4, alpha=0.4, zorder=1)
        ax.plot(tau_f, gamma_norm + baseline, color=color, linewidth=0.5, zorder=10)
        ax.fill_between(tau_f, baseline, gamma_norm + baseline,
                        color=color, alpha=0.2, zorder=5)

        ax.text(lbl_x, baseline + 0.5, f"{T} °C",
                fontsize=8, va="center", ha="left", color="black")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau\:/\:\mathrm{s}$", fontsize=8, labelpad=10)
    ax.set_xlim([x_lo, tau_max])
    ax.set_ylabel(r"$\gamma(\ln\tau)\:/\:\gamma_\mathrm{max}$", fontsize=8, labelpad=10)
    y_max = (len(temps) - 1) * offset_step + 1.5
    ax.set_ylim([-0.3, y_max])
    ax.set_yticks([])
    ax.yaxis.set_tick_params(left=False, right=False)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=10))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))
    ax.xaxis.set_major_formatter(LogFormatterMathtext())
    ax.tick_params(axis="x", labelsize=8)
    ax.spines["top"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if df_peaks is not None:
        _title = _condition_suptitle(df_peaks, label, show_pO2=show_pO2)
        if _title:
            fig.suptitle(_title, fontsize=_suptitle_size(fig))

    if save:
        _save(fig, save_dir, f"DRT_{condition}_Stacked")
    return fig


# ---------------------------------------------------------------------------
# 2. Nyquist overlay (data + fit, all temperatures)
# ---------------------------------------------------------------------------

def plot_nyquist_multipanel(
    records:    dict[int, tuple],
    fit_params: dict[int, dict],
    condition:  str,
    save_dir:   Path | str,
    xlim:       tuple[float, float] | None = None,
    ylim:       tuple[float, float] | None = None,
    hf_inset:   bool = True,
    save:       bool = True,
    df_peaks:   pd.DataFrame | None = None,
    label:      str  = "",
    show_pO2:   bool = True,
) -> plt.Figure:
    """
    Nyquist plot with all temperatures overlaid on a single panel.

    Data are plotted as filled circles; fits as solid lines, both using
    the temperature colour palette (blue→red). An upper-right inset zooms the
    high-frequency region so the bunched first semicircles stay readable.

    Parameters
    ----------
    records    : {T_nominal: (freq, Z_re, Z_im)}
                 freq [Hz], Z_re [Ohm], Z_im [Ohm] with Z_im > 0 in cap. region
    fit_params : {T_nominal: dict}
                 Each dict must have keys: R0, R (list), tau (list), alpha (list)
    condition  : condition label (title + filename)
    save_dir   : output directory
    xlim, ylim : optional (min, max) in kOhm to crop the display window.
                 The fit overlay still spans the full data range - only the
                 axes are limited. None = auto-scale from data (default).
    hf_inset   : when True (default) draw an upper-right inset zoomed on the
                 high-frequency arcs. The zoom range auto-adapts from the
                 highest-frequency 40 % of points across all temperatures.
    save       : when False, do not write PNG/PDF (interactive preview only).
    df_peaks   : optional stage3_fit.xlsx "Peaks" DataFrame; when given and it
                 carries pO2_mean, the median p(O2) is shown as a suptitle,
                 same value the condition selector and the Arrhenius figures use.
    label      : condition label for the suptitle ("" = pressure only)
    show_pO2   : False for a run whose lambda probe was off

    >>> fig = plot_nyquist_multipanel(records, fit_params, "Ar_100",
    ...                               "Results/Ar_100/plots")  # doctest: +SKIP

    Returns
    -------
    matplotlib Figure
    """
    temps = sorted(records.keys())
    fig, ax = plt.subplots(figsize=(6, 6), layout="constrained")
    vmax = 0.0

    fits: dict[int, np.ndarray] = {}
    for T in temps:
        freq, Z_re, Z_im = records[T]
        color   = COLOR_MAP.get(T, "gray")
        Zr_k    = Z_re / 1e3
        Zi_k    = Z_im / 1e3
        vmax    = max(vmax, float(Zr_k.max()), float(Zi_k.max()))

        ax.plot(Zr_k, Zi_k, "o", color=color, ms=3.5, label=f"{T} °C", alpha=0.9)

        fp = fit_params.get(T)
        if fp is not None:
            try:
                Z_fit = _reconstruct_Z_fit(
                    freq, fp["R0"],
                    np.asarray(fp["R"]),
                    np.asarray(fp["tau"]),
                    np.asarray(fp["alpha"]),
                )
                fits[T] = Z_fit
                # Draw the fit line in frequency order: CSV-ingested spectra
                # carry no order guarantee and an unsorted connected line
                # self-intersects. Data markers and fits[T] keep native order.
                o = np.argsort(freq)
                ax.plot(Z_fit.real[o] / 1e3, -Z_fit.imag[o] / 1e3, "-", color=color, lw=1.0)
            except Exception as exc:
                warnings.warn(f"Nyquist fit overlay skipped for T={T}: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)

    ax.set_xlabel(r"$Z'\:/\:\mathrm{k}\Omega$")
    ax.set_ylabel(r"$-Z''\:/\:\mathrm{k}\Omega$")
    ax.set_aspect("equal", adjustable="box")
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=0, right=vmax * 1.1)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        ax.set_ylim(bottom=0, top=vmax * 1.1)
    ax.legend(loc="upper left", ncol=2, fontsize=9, frameon=False)

    # HF-zoom inset (upper-right): the first semicircles are crushed near the
    # origin in the full view; this resolves them. Range auto-set from the
    # highest-frequency 40 % of points across all temperatures.
    if hf_inset and temps:
        hf_per_T = []
        for T in temps:
            freq, Z_re, Z_im = records[T]
            if len(freq) == 0:
                continue
            sel = freq >= np.percentile(freq, 60.0)
            if np.any(sel):
                hf_per_T.append(max(float((Z_re[sel] / 1e3).max()),
                                    float((Z_im[sel] / 1e3).max())))
        # Favor the smaller (warm) first arcs so they read clearly; the larger
        # cold arcs run off the inset edge (acceptable - the full panel shows them).
        hf_max = float(np.percentile(hf_per_T, 50)) if hf_per_T else vmax * 0.1
        if hf_max <= 0:
            hf_max = vmax * 0.1
        axin = ax.inset_axes([0.62, 0.63, 0.35, 0.35])
        for T in temps:
            freq, Z_re, Z_im = records[T]
            color = COLOR_MAP.get(T, "gray")
            axin.plot(Z_re / 1e3, Z_im / 1e3, "o", color=color, ms=2.4)
            if T in fits:
                axin.plot(fits[T].real / 1e3, -fits[T].imag / 1e3, "-",
                          color=color, lw=0.9)
        axin.set_xlim(0, hf_max * 1.15)
        axin.set_ylim(0, hf_max * 1.15)
        axin.set_aspect("equal", "box")
        axin.tick_params(labelsize=7)

    if df_peaks is not None:
        _title = _condition_suptitle(df_peaks, label, show_pO2=show_pO2)
        if _title:
            fig.suptitle(_title, fontsize=_suptitle_size(fig))

    if save:
        _save(fig, save_dir, f"Nyquist_{condition}")
    return fig


# ---------------------------------------------------------------------------
# 3. Bode plot (|Z| and phase)
# ---------------------------------------------------------------------------

def plot_bode(
    records:    dict[int, tuple],
    fit_params: dict[int, dict],
    condition:  str,
    save_dir:   Path | str,
    freq_lim:   tuple[float, float] | None = None,
    mag_lim:    tuple[float, float] | None = None,
    phase_lim:  tuple[float, float] | None = None,
    save:       bool = True,
    model_label: str | None = None,
    df_peaks:   pd.DataFrame | None = None,
    label:      str  = "",
    show_pO2:   bool = True,
) -> plt.Figure:
    """
    Bode plot: |Z| [kΩ] (loglog) and phase [°] (semilog) for all temperatures.

    Parameters
    ----------
    records, fit_params, condition, save_dir
        Same meaning as in plot_nyquist_multipanel.
    freq_lim   : optional (min_Hz, max_Hz) display window. None = auto.
    mag_lim    : optional (min_kOhm, max_kOhm) |Z| window. None = auto.
    phase_lim  : optional (min_deg, max_deg) phase window. None = auto.
    save       : when False, do not write PNG/PDF (interactive preview only).
    model_label: equivalent-circuit annotation drawn in a wheat box on the
                 magnitude panel. None = auto-build from fit_params (number of
                 Zarc elements). Pass "" to disable the box.
    df_peaks   : optional stage3_fit.xlsx "Peaks" DataFrame; when given and it
                 carries pO2_mean, the median p(O2) is shown as a suptitle,
                 same value the condition selector and the Arrhenius figures use.
    label      : condition label for the suptitle ("" = pressure only)
    show_pO2   : False for a run whose lambda probe was off

    >>> fig = plot_bode(records, fit_params, "Ar_100",
    ...                 "Results/Ar_100/plots")  # doctest: +SKIP

    Returns
    -------
    matplotlib Figure (2 stacked panels, shared x-axis)
    """
    temps = sorted(records.keys())
    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, figsize=(6, 8), sharex=True, layout="constrained")

    for T in temps:
        freq, Z_re, Z_im = records[T]
        color   = COLOR_MAP.get(T, "gray")
        Z_mag   = np.sqrt(Z_re**2 + Z_im**2) / 1e3
        # Standard EIS phase: φ = arctan(-Z''/Z') < 0 in capacitive region.
        # IsmRecord stores Z_im = +Z'' (positive for capacitive), so negate it.
        phase   = np.degrees(np.arctan2(-Z_im, Z_re))

        ax_mag.loglog(freq, Z_mag, "o", color=color, ms=3, label=f"{T} °C")
        ax_ph.semilogx(freq, phase, "o", color=color, ms=3)

        fp = fit_params.get(T)
        if fp is not None:
            try:
                Z_fit   = _reconstruct_Z_fit(
                    freq, fp["R0"],
                    np.asarray(fp["R"]),
                    np.asarray(fp["tau"]),
                    np.asarray(fp["alpha"]),
                )
                Z_mag_f = np.abs(Z_fit) / 1e3
                # Z_fit.imag < 0 in capacitive region (physical convention) → φ < 0 directly
                phase_f = np.degrees(np.arctan2(Z_fit.imag, Z_fit.real))
                # Sorted for the same reason as the Nyquist fit overlay.
                o = np.argsort(freq)
                ax_mag.loglog(freq[o], Z_mag_f[o], "--", color=color, lw=1)
                ax_ph.semilogx(freq[o], phase_f[o], "--", color=color, lw=1)
            except Exception as exc:
                warnings.warn(f"Bode fit overlay skipped for T={T}: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)

    ax_mag.set_ylabel(r"$|Z|\:/\:\mathrm{k}\Omega$")
    ax_mag.legend(loc="upper right", ncol=2, fontsize=8)
    ax_mag.grid(True, which="both", ls=":", alpha=0.4)

    # Equivalent-circuit annotation (number of Zarc elements). Auto-built from
    # the fit when not supplied; drawn in a wheat box like the reference notebook.
    if model_label is None:
        n_set = sorted({len(fp["R"]) for fp in fit_params.values() if fp})
        # R0 appears in the label only when it is actually in the circuit
        # (ZARC_INCLUDE_R0=False stores R0=0 in every fit)
        has_r0 = any(abs(fp.get("R0") or 0) > 0 for fp in fit_params.values() if fp)
        prefix = "R0–" if has_r0 else ""
        if len(n_set) == 1:
            model_label = prefix + "–".join(["Zarc"] * n_set[0])
        elif n_set:
            model_label = f"{prefix}Zarc×{n_set[0]}–{n_set[-1]}"
        else:
            model_label = ""
    if model_label:
        ax_mag.text(0.03, 0.08, model_label, transform=ax_mag.transAxes,
                    fontsize=9, ha="left", va="bottom",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    ax_ph.set_xlabel(r"$f\:/\:\mathrm{Hz}$")
    ax_ph.set_ylabel(r"$\varphi\:/\:^\circ$")
    ax_ph.grid(True, which="both", ls=":", alpha=0.4)

    if freq_lim is not None:
        ax_mag.set_xlim(left=freq_lim[0], right=freq_lim[1])
        ax_ph.set_xlim(left=freq_lim[0], right=freq_lim[1])
    if mag_lim is not None:
        ax_mag.set_ylim(bottom=mag_lim[0], top=mag_lim[1])
    if phase_lim is not None:
        ax_ph.set_ylim(bottom=phase_lim[0], top=phase_lim[1])

    if df_peaks is not None:
        _title = _condition_suptitle(df_peaks, label, show_pO2=show_pO2)
        if _title:
            fig.suptitle(_title, fontsize=_suptitle_size(fig))

    if save:
        _save(fig, save_dir, f"Bode_{condition}")
    return fig


# ---------------------------------------------------------------------------
# 4. Arrhenius analysis helpers
# ---------------------------------------------------------------------------

def build_arrhenius_results(
    df_peaks: pd.DataFrame,
    L_m:      float,
    D_m:      float,
    t_min:    float | None = None,
) -> list[dict]:
    """
    Compute Arrhenius fit results for each peak_id from stage3_fit.xlsx Peaks sheet.

    Physical quantities derived
    ---------------------------
    σ  = L / (R · A)       [S/m], returned in S/cm  (conductivity)
    C  = C_eff_i = τ / R   [F]            (effective capacitance)
    εᵣ = C · L / (ε₀ · A)  (dimensionless) (relative permittivity)

    Arrhenius plots
    ---------------
    ln(σT) vs 1/T  →  Eₐᶜᵒⁿᵈ   (long-range charge transport)
    ln(τ)  vs 1/T  →  Eₐᵖᵒˡ    (local dipole reorientation)
    ln(C)  vs 1/T  →  Eₐᶜ       (net: Eₐᵖᵒˡ − Eₐᶜᵒⁿᵈ)

    Parameters
    ----------
    df_peaks : DataFrame from stage3_fit.xlsx sheet "Peaks"
               Required: peak_id, T_nominal, R_i, tau_i, C_eff_i
               Optional: sigma_Sm_i (recomputed if absent)
    L_m, D_m : sample geometry [m]
    t_min    : exclude temperatures below this value [°C] from every
               Arrhenius fit. Use when peak identity is not resolved at
               low T (variable peak count or merged processes), so the
               fit covers only the range where peak_id tracks one
               physical process. None = use all temperatures.

    Returns
    -------
    List of dicts, one per peak_id, sorted by ascending peak_id.
    Each dict contains: T_C, T_K, inv_T, inv_T_fit, sigma, tau, R, C,
    ln_sigmaT, ln_tau, ln_C, Ea_cond, Ea_pol, Ea_C, R2_cond, R2_pol, R2_C,
    slope_*, int_*, color, marker.

    >>> T_K = np.array([500.0, 550.0, 600.0]) + 273.15
    >>> sigma = np.exp(-1.0 / (KB * T_K)) / T_K  # exact Ea_cond = 1 eV
    >>> df = pd.DataFrame({"peak_id": 1, "T_nominal": [500, 550, 600],
    ...     "R_i": 1e-3 / (sigma * np.pi * (1e-2 / 2) ** 2),
    ...     "tau_i": [1e-4, 3e-5, 1e-5], "C_eff_i": [5e-8, 5e-8, 5e-8]})
    >>> res = build_arrhenius_results(df, L_m=1e-3, D_m=1e-2)
    >>> round(float(res[0]["Ea_cond"]), 6), round(float(res[0]["R2_cond"]), 6)
    (1.0, 1.0)
    """
    A_m2    = np.pi * (D_m / 2) ** 2
    results = []

    if t_min is not None:
        df_peaks = df_peaks[df_peaks["T_nominal"] >= t_min]

    for i, pid in enumerate(sorted(df_peaks["peak_id"].unique())):
        sub = df_peaks[df_peaks["peak_id"] == pid].sort_values("T_nominal")
        T_C       = sub["T_nominal"].values.astype(float)
        if len(T_C) < 3:
            warnings.warn(
                f"Peak {pid}: only {len(T_C)} temperature point(s) available. "
                f"Arrhenius fit requires at least 3 temperatures for a meaningful result.",
                stacklevel=2,
            )
        T_K       = T_C + 273.15
        inv_T     = 1000.0 / T_K    # for x-axis (1000/T label convention)
        inv_T_fit = 1.0 / T_K       # actual 1/T for linregress

        R   = sub["R_i"].values.astype(float)
        tau = sub["tau_i"].values.astype(float)
        C   = sub["C_eff_i"].values.astype(float)

        # reported in S/cm like every other conductivity in the program;
        # the stored column stays S/m, only this reporting boundary converts
        if "sigma_Sm_i" in sub.columns:
            sigma = sub["sigma_Sm_i"].values.astype(float) / 100.0
        else:
            sigma = L_m / (R * A_m2) / 100.0

        def _safe_ln(vals: np.ndarray, label: str) -> np.ndarray:
            # log(0) gives -inf, which passes the NaN-only mask in
            # _arrhenius_linreg and poisons linregress; sink bad values to NaN.
            bad = ~(vals > 0)   # catches <= 0 and NaN
            if bad.any():
                warnings.warn(
                    f"Peak {pid}: {int(bad.sum())} non-positive {label} value(s) "
                    "excluded from the Arrhenius fit.", stacklevel=3)
            out = np.full(vals.shape, np.nan)
            out[~bad] = np.log(vals[~bad])
            return out

        ln_sigmaT = _safe_ln(sigma * T_K, "sigma*T")
        ln_tau    = _safe_ln(tau,         "tau")
        ln_C      = _safe_ln(C,           "C_eff")

        Ea_cond, Ea_cond_err, R2_cond, slope_cond, int_cond = _arrhenius_linreg(inv_T_fit, ln_sigmaT)
        _,       Ea_pol_err,  R2_pol,  slope_pol,  int_pol  = _arrhenius_linreg(inv_T_fit, ln_tau)
        _,       Ea_C_err,    R2_C,    slope_C,    int_C    = _arrhenius_linreg(inv_T_fit, ln_C)
        # τ = τ₀·exp(+Ea/kT) → slope of ln(τ) vs 1/T is +Ea/k (positive)
        Ea_pol = slope_pol * KB
        # C_eff = τ/R → slope_C = (Ea_pol - Ea_cond)/k; sign follows data
        Ea_C   = slope_C   * KB

        results.append({
            "Peak":        f"Peak {pid}",
            "peak_id":     pid,
            "T_C":         T_C,
            "T_K":         T_K,
            "inv_T":       inv_T,
            "inv_T_fit":   inv_T_fit,
            "sigma":       sigma,
            "tau":         tau,
            "R":           R,
            "C":           C,
            "ln_sigmaT":   ln_sigmaT,
            "ln_tau":      ln_tau,
            "ln_C":        ln_C,
            # Conductivity Arrhenius
            "Ea_cond":     Ea_cond,   "Ea_cond_err": Ea_cond_err, "R2_cond":  R2_cond,
            "slope_cond":  slope_cond, "int_cond":    int_cond,
            # Polarisation Arrhenius (tau)
            "Ea_pol":      Ea_pol,    "Ea_pol_err":  Ea_pol_err,  "R2_pol":   R2_pol,
            "slope_pol":   slope_pol,  "int_pol":     int_pol,
            # Capacitance Arrhenius
            "Ea_C":        Ea_C,      "Ea_C_err":    Ea_C_err,    "R2_C":     R2_C,
            "slope_C":     slope_C,    "int_C":       int_C,
            # Visual
            "color":       PEAK_COLORS[i % len(PEAK_COLORS)],
            "marker":      PEAK_MARKERS[i % len(PEAK_MARKERS)],
        })

    return results


def _draw_arrhenius_panel(
    ax:          plt.Axes,
    results:     list[dict],
    y_key:       str,
    slope_key:   str,
    int_key:     str,
    ea_key:      str,
    ea_label:    str,
    ylabel:      str,
) -> None:
    """Draw one Arrhenius sub-panel onto ax (internal helper)."""
    for r in results:
        if np.isnan(r[ea_key]):
            continue
        valid = ~np.isnan(r[y_key])
        ax.plot(
            r["inv_T"][valid], r[y_key][valid],
            r["marker"], color=r["color"],
            markersize=7, markeredgecolor="black", markeredgewidth=0.5,
            label=f"{r['Peak']}: ${ea_label}$ = {r[ea_key]:.2f} eV",
        )
        fit_x = np.linspace(r["inv_T"].min() - 0.02, r["inv_T"].max() + 0.02, 100)
        # axis shows 1000/T while the regression was done on 1/T,
        # hence the /1000 when evaluating the fit line in display units
        fit_y = r[int_key] + r[slope_key] * (fit_x / 1000)
        ax.plot(fit_x, fit_y, "-", color=r["color"], linewidth=1.0, alpha=0.7)

    ax.set_xlabel(r"$1000\,T^{-1}\:/\:\mathrm{K^{-1}}$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=10, frameon=True, ncol=2)
    ax.tick_params(direction="in", top=True, right=True)


# ---------------------------------------------------------------------------
# 5. Arrhenius 2×2 panel
# ---------------------------------------------------------------------------

def _suptitle_size(fig: plt.Figure) -> float:
    """Suptitle size in points, proportional to the figure width.

    Equal point sizes are not equal on the page. A figure is scaled to a fixed
    column width before it is read, so the same 11 pt lands 2.8 times larger on
    a 4-inch figure than on an 11-inch one. Tying the size to the width keeps
    the apparent size constant across the whole set. The constant is 12 pt on
    the 6.5-inch figure, the reference width of this module.

    >>> from matplotlib.figure import Figure
    >>> round(_suptitle_size(Figure(figsize=(6.5, 5))), 1)
    12.0
    >>> round(_suptitle_size(Figure(figsize=(4, 4))), 1)
    7.4
    """
    return 12.0 / 6.5 * float(fig.get_size_inches()[0])


def _condition_suptitle(df_peaks: pd.DataFrame, label: str = "", *,
                        show_pO2: bool = True) -> str:
    """Figure suptitle: the condition label, and the median p(O2) when there is one.

    This module defines no ``set_title``, so the suptitle is the only text on a
    figure that identifies the measurement; it therefore falls back to the label
    rather than to nothing. The median is the one the condition selector shows,
    so the figure and the selector never disagree.

    ``show_pO2=False`` is for a run whose lambda probe was off: the recorded
    pressures are readings of an idle probe, not measurements.

    >>> _condition_suptitle(pd.DataFrame({"pO2_mean": [0.21]}), "Air")
    'Air,  $p$(O$_2$) = 0.21 bar'
    >>> _condition_suptitle(pd.DataFrame({"pO2_mean": [8715.0]}), "Air", show_pO2=False)
    'Air'
    """
    value = ""
    if show_pO2 and "pO2_mean" in df_peaks.columns:
        s = pd.to_numeric(df_peaks["pO2_mean"], errors="coerce").dropna()
        s = s[s > 0]
        if not s.empty:
            value = format_pO2_value(s.median())
    if not value:
        return label
    pressure = f"$p$(O$_2$) = {value} bar"
    return f"{label},  {pressure}" if label else pressure


def plot_arrhenius_panel(
    df_peaks:     pd.DataFrame,
    L_m:          float,
    D_m:          float,
    condition:    str,
    save_dir:     Path | str,
    t_min:        float | None = None,
    label:        str = "",
    show_pO2:     bool = True,
) -> tuple[plt.Figure, list[dict]]:
    """
    2×2 Arrhenius panel: ln(σT), ln(τ), ln(C), log₁₀(εᵣ).

    Panel layout
    ------------
    [0,0]  ln(σT)  vs 1000/T  →  Eₐᶜᵒⁿᵈ  (long-range transport)
    [0,1]  ln(τ)   vs 1000/T  →  Eₐᵖᵒˡ   (polarization / local hop)
    [1,0]  ln(C)   vs 1000/T  →  Eₐᶜ      (net: Eₐᵖᵒˡ − Eₐᶜᵒⁿᵈ)
    [1,1]  log₁₀(εᵣ) vs 1000/T (no Arrhenius line - diagnostic only)

    Parameters
    ----------
    df_peaks  : DataFrame from stage3_fit.xlsx sheet "Peaks"
    L_m, D_m  : sample geometry [m]
    condition : filename stem for the saved figure
    save_dir  : output directory
    t_min     : exclude temperatures below this value [°C] from the
                Arrhenius fits (see build_arrhenius_results); the active
                range is annotated on the figure. None = all temperatures.
    label     : condition label for the suptitle ("" = pressure only)
    show_pO2  : False for a run whose lambda probe was off

    Returns
    -------
    (fig, results)
    results is the full list from build_arrhenius_results(); every peak
    is drawn regardless of its Arrhenius R².

    >>> fig, results = plot_arrhenius_panel(df_peaks, L_m=1.2e-3, D_m=1.0e-2,
    ...     condition="Ar_100", save_dir="Results/Ar_100/plots")  # doctest: +SKIP
    """
    results = build_arrhenius_results(df_peaks, L_m, D_m, t_min=t_min)
    A_m2 = np.pi * (D_m / 2) ** 2

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150, layout="constrained")

    _title = _condition_suptitle(df_peaks, label, show_pO2=show_pO2)
    if _title:
        fig.suptitle(_title, fontsize=_suptitle_size(fig))

    # The figure must declare its own fit range when low-T points are excluded
    if t_min is not None:
        axes[0, 0].text(0.03, 0.03, f"Arrhenius fit: T ≥ {t_min:g} °C",
                        transform=axes[0, 0].transAxes, fontsize=8, color="#555")

    _draw_arrhenius_panel(
        axes[0, 0], results, "ln_sigmaT", "slope_cond", "int_cond", "Ea_cond",
        r"E_a^{cond}", r"$\ln\!\left[\sigma T\:/\:\mathrm{(S\,K\,cm^{-1})}\right]$")

    _draw_arrhenius_panel(
        axes[0, 1], results, "ln_tau", "slope_pol", "int_pol", "Ea_pol",
        r"E_a^{pol}", r"$\ln[\tau\:/\:\mathrm{s}]$")

    _draw_arrhenius_panel(
        axes[1, 0], results, "ln_C", "slope_C", "int_C", "Ea_C",
        r"E_a^{C}", r"$\ln[C\:/\:\mathrm{F}]$")

    # Panel [1,1]: log₁₀(εᵣ) - diagnostic, no fit line
    ax4 = axes[1, 1]
    for r in results:
        eps_r     = (r["C"] * L_m) / (EPS_0 * A_m2)
        log10_eps = np.log10(eps_r)
        valid     = ~np.isnan(log10_eps)
        ax4.plot(
            r["inv_T"][valid], log10_eps[valid],
            r["marker"] + "-", color=r["color"],
            markersize=7, markeredgecolor="black", markeredgewidth=0.5,
            linewidth=1, label=r["Peak"],
        )
    ax4.set_xlabel(r"$1000\,T^{-1}\:/\:\mathrm{K^{-1}}$", fontsize=12)
    ax4.set_ylabel(r"$\log_{10}\![\varepsilon_\mathrm{r}]$", fontsize=12)
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               fontsize=10, frameon=True, ncol=2)
    ax4.tick_params(direction="in", top=True, right=True)

    _save(fig, save_dir, f"Arrhenius_{condition}")
    return fig, results


def plot_arrhenius_sigma(
    df_peaks:     pd.DataFrame,
    L_m:          float,
    D_m:          float,
    condition:    str,
    save_dir:     Path | str,
    t_min:        float | None = None,
    sum_peak_ids: list[int] | None = None,
    label:        str = "",
    show_pO2:     bool = True,
) -> plt.Figure | None:
    """
    Single-panel conductivity Arrhenius for the HF block: ln(σT) vs 1000/T.

    Two layers:
    - one branch per peak in ``sum_peak_ids``, restricted to T ≥ ``t_min``
      (below it the R split between close peaks is not validated);
    - the series sum, σ = L / (Σ Rᵢ · A), over the FULL temperature range:
      series resistances add, so the sum stays well defined even where the
      individual split is degenerate.

    The sum mixes processes with different Eₐ, so its Arrhenius line may
    curve slightly; its Eₐ is an effective value for the whole HF block.

    Parameters
    ----------
    df_peaks     : stage3_fit.xlsx "Peaks" sheet for one condition
    sum_peak_ids : peaks forming the block (e.g. [1, 2]); None disables
                   the figure (returns None)
    t_min        : threshold above which the split is validated [°C];
                   None = branches drawn over the full range
    label        : condition label for the suptitle ("" = pressure only)
    show_pO2     : False for a run whose lambda probe was off

    >>> fig = plot_arrhenius_sigma(df_peaks, 1.2e-3, 1.0e-2, "Ar_100",
    ...     "Results/Ar_100/plots", t_min=500, sum_peak_ids=[1, 2])  # doctest: +SKIP
    """
    if not sum_peak_ids or len(sum_peak_ids) < 2:
        return None
    A_m2 = np.pi * (D_m / 2) ** 2

    sub = df_peaks[df_peaks["peak_id"].isin(sum_peak_ids)]
    if sub.empty:
        warnings.warn(f"plot_arrhenius_sigma: no rows for peaks {sum_peak_ids}",
                      stacklevel=2)
        return None

    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=150, layout="constrained")

    _title = _condition_suptitle(df_peaks, label, show_pO2=show_pO2)
    if _title:
        fig.suptitle(_title, fontsize=_suptitle_size(fig))

    # branches: validated range only
    branch_results = build_arrhenius_results(sub, L_m, D_m, t_min=t_min)
    _draw_arrhenius_panel(
        ax, branch_results, "ln_sigmaT", "slope_cond", "int_cond", "Ea_cond",
        r"E_a", r"$\ln\!\left[\sigma T\:/\:\mathrm{(S\,K\,cm^{-1})}\right]$")

    # series sum: full range, only at T where every block peak was fitted
    wide = sub.pivot_table(index="T_nominal", columns="peak_id",
                           values="R_i", aggfunc="first")
    wide = wide.dropna()
    if len(wide) >= 3:
        T_K     = wide.index.to_numpy(dtype=float) + 273.15
        R_sum   = wide.sum(axis=1).to_numpy(dtype=float)
        # S/cm, same unit as the per-branch curves it is drawn beside
        sigma   = L_m / (R_sum * A_m2) / 100.0
        ln_sT   = np.log(sigma * T_K)
        inv_T   = 1000.0 / T_K
        Ea, _, _, slope, intercept = _arrhenius_linreg(1.0 / T_K, ln_sT)
        ids_lab = "+".join(f"P{int(p)}" for p in sorted(sum_peak_ids))
        ax.plot(inv_T, ln_sT, "D", color="#333333", markersize=6,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"{ids_lab} (series sum): $E_a$ = {Ea:.2f} eV")
        fit_x = np.linspace(inv_T.min() - 0.02, inv_T.max() + 0.02, 100)
        ax.plot(fit_x, intercept + slope * (fit_x / 1000), "--",
                color="#333333", linewidth=1.0, alpha=0.8)
    else:
        warnings.warn("plot_arrhenius_sigma: fewer than 3 temperatures with "
                      "all block peaks fitted; sum line skipped", stacklevel=2)

    if t_min is not None:
        ax.text(0.03, 0.03, f"split validated for T ≥ {t_min:g} °C; "
                            "sum drawn over the full range",
                transform=ax.transAxes, fontsize=8, color="#555")
    # No in-figure title: the file name carries the identity (publication style)
    # _draw_arrhenius_panel puts the legend below the axis; redo it inside
    ax.legend(fontsize=9, frameon=True, loc="best")

    _save(fig, save_dir, f"Arrhenius_sigma_HF_{condition}")
    return fig


# ---------------------------------------------------------------------------
# 6. Brouwer p(O₂) diagram
# ---------------------------------------------------------------------------

def _slope_line(
    ax:         plt.Axes,
    x0:         float,
    x1:         float,
    y_center:   float,
    slope:      float,
    label:      str | None = None,
    label_side: str  = "right",
    label_pad:  tuple = (0.15, 0.0),
    **kw,
) -> None:
    """Draw a reference slope guide line on a Brouwer diagram."""
    half = (x1 - x0) / 2.0
    ys   = np.array([y_center - slope * half, y_center + slope * half])
    ax.plot([x0, x1], ys, **kw)
    if label:
        if label_side == "right":
            ax.text(x1 + label_pad[0], ys[1] + label_pad[1],
                    label, fontsize=10, va="center", ha="left")
        else:
            ax.text(x0 - label_pad[0], ys[0] + label_pad[1],
                    label, fontsize=10, va="center", ha="right")


def plot_brouwer(
    df_all:        pd.DataFrame,
    save_dir:      Path | str,
    sample_name:   str       = "",
    peak_id:       int       = 1,
    temps_to_plot: list[int] | None = None,
    add_slopes:    bool      = True,
    slopes:        tuple[str, ...] = ("-1/4", "-1/6", "0", "+1/6", "+1/4"),
) -> plt.Figure:
    """
    Brouwer p(O₂)-dependence diagram: log₁₀(σ₁) vs log₁₀(p(O₂)).

    Aggregates data from ALL atmospheric conditions for one sample.
    Each temperature is shown as a distinct symbol; conditions are not
    colour-coded (the diagram maps σ vs pO₂ independently of condition label).

    Parameters
    ----------
    df_all        : DataFrame aggregated from all conditions' stage3_fit.xlsx
                    Required columns: T_nominal, peak_id, pO2_mean, sigma_Sm_i
    save_dir      : output directory
    sample_name   : sample id, appended to the filename stem
    peak_id       : which peak to use (default 1 = highest-frequency process)
    temps_to_plot : temperatures to show (default None = all available)
    add_slopes    : draw the reference slope guides (default True)
    slopes        : which guides to draw, any of "-1/4", "-1/6", "0",
                    "+1/6", "+1/4" (default: all)

    >>> fig = plot_brouwer(df_all, "Results/plots", sample_name="S1",
    ...                    peak_id=1, temps_to_plot=[500, 550, 600])  # doctest: +SKIP

    Returns
    -------
    matplotlib Figure
    """
    sub = df_all[df_all["peak_id"] == peak_id].copy()
    sub["sigma_Scm"] = sub["sigma_Sm_i"] / 100.0      # S/m → S/cm

    # log10 of zero/negative/NaN values would silently corrupt the axis
    # limits below (NaN min/max), so drop those rows up front
    _pO2   = pd.to_numeric(sub["pO2_mean"], errors="coerce")
    _sigma = pd.to_numeric(sub["sigma_Scm"], errors="coerce")
    _ok    = (_pO2 > 0) & (_sigma > 0)
    if (~_ok).any():
        warnings.warn(f"Brouwer: dropping {int((~_ok).sum())} row(s) with "
                      f"non-positive or missing pO2/sigma", stacklevel=2)
    sub = sub[_ok]
    sub["lg_pO2"]   = np.log10(sub["pO2_mean"].astype(float))
    sub["lg_sigma"] = np.log10(sub["sigma_Scm"])

    if temps_to_plot is not None:
        sub = sub[sub["T_nominal"].isin(temps_to_plot)]

    if sub.empty:
        raise ValueError(
            f"Brouwer: no plottable rows for peak_id={peak_id} "
            f"(check pO2_mean/sigma values and temps_to_plot)"
        )
    if sub["pO2_mean"].nunique() < 2:
        # one pressure leaves one point per temperature: an axis, not a diagram
        raise ValueError(
            f"Brouwer: peak_id={peak_id} has {sub['pO2_mean'].nunique()} distinct "
            f"pO2 value(s); the diagram needs at least 2 conditions"
        )

    # raw Figure, never pyplot-registered: this function runs inside the
    # stage-4 replot widget callback, where a pyplot figure would be
    # re-rendered by the inline backend's post-execute flush (the June 2026
    # live-panel rule: callbacks use matplotlib.figure.Figure only)
    fig = Figure(figsize=(7, 5.5), dpi=150, layout="constrained")
    ax = fig.add_subplot()

    for T in sorted(sub["T_nominal"].unique()):
        group = sub[sub["T_nominal"] == T].sort_values("lg_pO2")
        sty   = _TEMP_STYLE_BROUWER.get(
            int(T), dict(color="gray", marker="o", ms=7, label=f"{T:.0f} °C"))
        ax.plot(
            group["lg_pO2"], group["lg_sigma"],
            marker=sty["marker"], linestyle="none",
            color=sty["color"], markersize=sty["ms"],
            markeredgecolor="none", label=sty["label"], zorder=5,
        )

    y_lo = float(sub["lg_sigma"].min())
    y_hi = float(sub["lg_sigma"].max())

    ax.set_xlim(-5.5, 0.5)
    ax.set_ylim(y_lo - 0.25, y_hi + 0.40)
    ax.set_xticks([-5, -2.5, 0])
    ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))

    if add_slopes and slopes:
        y_ref = y_hi + 0.22
        # the 1/6 guides share the fan origin of the 1/4 ones (right end
        # for negative, left end for positive slopes) so the two candidate
        # Brouwer exponents can be compared by eye
        _guides = {  # name: (x0, x1, y_center, slope, label_side)
            "-1/4": (-3.6, -3.0, y_ref,         -1 / 4, "left"),
            "-1/6": (-3.6, -3.0, y_ref - 0.025, -1 / 6, "left"),
            "0":    (-2.0, -1.4, y_ref,          0.0,   "left"),
            "+1/6": (-0.8, -0.2, y_ref - 0.025,  1 / 6, "right"),
            "+1/4": (-0.8, -0.2, y_ref,          1 / 4, "right"),
        }
        _bad = [s for s in slopes if s not in _guides]
        if _bad:
            warnings.warn(f"Brouwer: ignoring unknown slope guide(s) {_bad}; "
                          f"valid: {list(_guides)}", stacklevel=2)
            slopes = tuple(s for s in slopes if s in _guides)
        for name in slopes:
            x0, x1, yc, sl, side = _guides[name]
            grey = name.endswith("1/6")
            _slope_line(ax, x0, x1, yc, slope=sl,
                        label=r"$plateau$" if name == "0" else rf"${name}$",
                        label_side=side,
                        label_pad=(0.15, -0.10 if grey else 0.0),
                        color="#888888" if grey else "black",
                        lw=1.0, ls="--", zorder=3)

    ax.set_xlabel(r"$\log_{10}\![p(\mathrm{O_2})\:/\:\mathrm{bar}]$", fontsize=12)
    ax.set_ylabel(
        r"$\log_{10}\!\left[\sigma_{" + str(peak_id) + r"}\:/\:\mathrm{(S\,cm^{-1})}\right]$", fontsize=12)
    ax.legend(loc="upper left", frameon=True, fontsize=8,
              ncol=2, handlelength=1.3, borderpad=0.5,
              labelspacing=0.3, columnspacing=0.8)
    # Tick / spine weights matched to the Brouwer reference notebook (Cell 3).
    ax.tick_params(direction="in", which="major", labelsize=10, width=1.2, length=4)
    ax.tick_params(direction="in", which="minor", width=1.2, length=2.5)
    for _sp in ax.spines.values():
        _sp.set_linewidth(1.2)

    stem = f"Brouwer_Peak{peak_id}_{sample_name}" if sample_name else f"Brouwer_Peak{peak_id}"
    _save(fig, save_dir, stem)
    return fig


def fit_transference(
    df_all:   pd.DataFrame,
    peak_id:  int = 1,
    exponent: float = 0.25,
    temps:    list[int] | None = None,
) -> pd.DataFrame:
    """
    Decompose σ(pO₂) of one process into ionic and electronic partial
    conductivities (Patterson analysis).

    Model (dilute defect regime):

        σ(pO₂) = σ_ion + σ_p · pO₂^(+x) + σ_n · pO₂^(−x)      x = ``exponent``

    The model is linear in the three coefficients, so each temperature is
    solved with non-negative least squares (σᵢ ≥ 0, no initial guesses).
    The local Brouwer slope then satisfies d(log σ)/d(log pO₂) = x·(t_p − t_n):
    a plateau is purely ionic, a +x slope is purely p-type electronic
    (polaron hopping), so the ionic transference number follows directly:

        t_ion(pO₂) = σ_ion / σ(pO₂)          t_el = 1 − t_ion

    Parameters
    ----------
    df_all   : aggregated stage3_fit Peaks rows from all conditions
               (required columns: peak_id, T_nominal, pO2_mean, sigma_Sm_i)
    peak_id  : process to decompose
    exponent : Brouwer exponent x (0.25 in the dilute regime; 1/6 elsewhere)
    temps    : restrict to these temperatures [°C] (None = all)

    Returns
    -------
    Tidy DataFrame, one row per (T, pO₂): peak_id, T_nominal, pO2,
    sigma_Scm, sigma_ion, sigma_p, sigma_n [S/cm], R2 (per-T fit),
    t_ion, t_el. Temperatures with fewer than 4 pO₂ points are skipped.

    >>> p = [1e-4, 1e-2, 1.0, 1e2]
    >>> df = pd.DataFrame({"peak_id": 1, "T_nominal": 600, "pO2_mean": p,
    ...     "sigma_Sm_i": [(1.0 + 2.0 * x ** 0.25) * 100.0 for x in p]})
    >>> out = fit_transference(df, peak_id=1)
    >>> round(float(out["sigma_ion"].iloc[0]), 6), round(float(out["sigma_p"].iloc[0]), 6)
    (1.0, 2.0)
    """
    sub = df_all[df_all["peak_id"] == peak_id].copy()
    sub["sigma_Scm"] = pd.to_numeric(sub["sigma_Sm_i"], errors="coerce") / 100.0
    _pO2 = pd.to_numeric(sub["pO2_mean"], errors="coerce")
    sub  = sub[(_pO2 > 0) & (sub["sigma_Scm"] > 0)]
    if temps is not None:
        sub = sub[sub["T_nominal"].isin(temps)]

    rows: list[dict] = []
    for T in sorted(sub["T_nominal"].unique()):
        g = sub[sub["T_nominal"] == T].sort_values("pO2_mean")
        p = g["pO2_mean"].to_numpy(dtype=float)
        y = g["sigma_Scm"].to_numpy(dtype=float)
        if len(p) < 4:
            warnings.warn(f"transference peak {peak_id} T={T}: "
                          f"only {len(p)} pO2 point(s), skipped", stacklevel=2)
            continue
        A = np.column_stack([np.ones_like(p), p ** exponent, p ** (-exponent)])
        try:
            coef, _ = optimize.nnls(A, y)
        except Exception as exc:
            warnings.warn(f"transference peak {peak_id} T={T}: NNLS failed "
                          f"({type(exc).__name__}: {exc})", stacklevel=2)
            continue
        s_ion, s_p, s_n = (float(c) for c in coef)
        y_fit  = A @ coef
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2     = 1.0 - float(np.sum((y - y_fit) ** 2)) / ss_tot if ss_tot > 0 else np.nan
        for p_i, y_i in zip(p, y):
            tot   = s_ion + s_p * p_i ** exponent + s_n * p_i ** (-exponent)
            t_ion = s_ion / tot if tot > 0 else np.nan
            rows.append({
                "peak_id": peak_id, "T_nominal": int(T), "pO2": p_i,
                "sigma_Scm": y_i, "sigma_ion": s_ion, "sigma_p": s_p,
                "sigma_n": s_n, "R2": r2, "t_ion": t_ion, "t_el": 1.0 - t_ion,
            })
    return pd.DataFrame(rows)


def plot_brouwer_transference(
    df_all:        pd.DataFrame,
    save_dir:      Path | str,
    sample_name:   str       = "",
    peak_id:       int       = 1,
    exponent:      float     = 0.25,
    temps_to_plot: list[int] | None = None,
    df_t:          pd.DataFrame | None = None,
    params         = None,
    perr:          dict | None = None,
) -> plt.Figure:
    """
    Two-panel ionic/electronic decomposition of the Brouwer diagram.

    Left  : log₁₀(σ) vs log₁₀(pO₂) with the fitted total curve (solid) and
            the ionic component σ_ion (dashed horizontal) per temperature.
    Right : t_ion vs log₁₀(pO₂), one curve per temperature.

    Fit model and conventions: see fit_transference(). The exponent in use
    is annotated on the figure so the assumed defect regime is explicit.

    ``df_t``: optional precomputed transference table (same columns as
    ``fit_transference``). When given it is used directly instead of the
    per-isotherm NNLS, so Stage 5 can redraw this figure from the refined
    global model (``model.global_transference_table``). Default behaviour
    (Stage 4) is unchanged.

    ``params`` / ``perr``: optional ``ModelParams`` and error dict from
    ``model.fit_global_conductivity``. When given, the sigma panel gets a
    summary box with the pO2 exponent x and the activation energy of each
    active channel. A channel excluded from the model (Ea = NaN) is omitted;
    a selected channel whose fitted contribution to sigma is negligible over
    the measured window is kept and flagged as sigma ~ 0. The box is docked in
    a headroom band added above the data, so it never overlaps the curves even
    when the model is flat and fills the panel.

    >>> fig = plot_brouwer_transference(df_all, "Results/plots",
    ...     sample_name="S1", peak_id=1, exponent=0.25)  # doctest: +SKIP
    """
    if df_t is None:
        df_t = fit_transference(df_all, peak_id=peak_id, exponent=exponent,
                                temps=temps_to_plot)
    elif "peak_id" in df_t.columns:
        df_t = df_t[df_t["peak_id"] == peak_id]
    if df_t.empty:
        raise ValueError(f"transference: no usable data for peak_id={peak_id}")

    fig, (ax_s, ax_t) = plt.subplots(1, 2, figsize=(12, 5.2), dpi=150,
                                     layout="constrained")

    _t_curves: list[tuple] = []
    for T in sorted(df_t["T_nominal"].unique()):
        g     = df_t[df_t["T_nominal"] == T].sort_values("pO2")
        sty   = _TEMP_STYLE_BROUWER.get(
            int(T), dict(color="gray", marker="o", ms=7, label=f"{T:.0f} °C"))
        s_ion, s_p, s_n = g["sigma_ion"].iloc[0], g["sigma_p"].iloc[0], g["sigma_n"].iloc[0]
        p_grid = np.logspace(np.log10(g["pO2"].min()), np.log10(g["pO2"].max()), 120)
        s_grid = s_ion + s_p * p_grid ** exponent + s_n * p_grid ** (-exponent)

        ax_s.plot(np.log10(g["pO2"]), np.log10(g["sigma_Scm"]),
                  marker=sty["marker"], linestyle="none", color=sty["color"],
                  markersize=sty["ms"], markeredgecolor="none",
                  label=sty["label"], zorder=5)
        ax_s.plot(np.log10(p_grid), np.log10(s_grid),
                  "-", color=sty["color"], lw=1.0, alpha=0.9, zorder=4)
        if s_ion > 0:
            ax_s.axhline(np.log10(s_ion), color=sty["color"],
                         lw=0.8, ls="--", alpha=0.5, zorder=2)

        t_grid = s_ion / (s_ion + s_p * p_grid ** exponent + s_n * p_grid ** (-exponent))
        _t_curves.append((T, sty, p_grid, t_grid, g))

    # Curves that coincide numerically (typically the sigma_ion = 0 group on
    # the zero line) are drawn with interleaved dash offsets and rotated
    # markers, so every temperature stays visible on the shared line.
    _groups: dict = {}
    for entry in _t_curves:
        key = (len(entry[3]), tuple(np.round(entry[3][::24], 5)))
        _groups.setdefault(key, []).append(entry)
    for members in _groups.values():
        n = len(members)
        for i, (T, sty, p_grid, t_grid, g) in enumerate(members):
            if n == 1:
                ax_t.plot(np.log10(p_grid), t_grid, "-", color=sty["color"], lw=1.4)
                _mev = None
            else:
                ax_t.plot(np.log10(p_grid), t_grid,
                          linestyle=(i * 5, (5, 5 * (n - 1))),
                          color=sty["color"], lw=1.8)
                _mev = (i, n)
            ax_t.plot(np.log10(g["pO2"]), g["t_ion"],
                      marker=sty["marker"], linestyle="none", color=sty["color"],
                      markersize=sty["ms"] - 1, markeredgecolor="none",
                      markevery=_mev, zorder=5)

    # No panel titles (publication style): the Patterson model formula goes in
    # the figure caption / README. The shared legend sits in a framed box above
    # the figure, outside both panels, so it can never collide with the data.
    ax_s.set_xlabel(r"$\log_{10}\![p(\mathrm{O_2})\:/\:\mathrm{bar}]$", fontsize=12)
    ax_s.set_ylabel(
        r"$\log_{10}\!\left[\sigma_{" + str(peak_id) + r"}\:/\:\mathrm{(S\,cm^{-1})}\right]$", fontsize=12)
    _handles, _labels = ax_s.get_legend_handles_labels()
    # No dashed sigma_ion guide in the legend when the channel is absent
    # everywhere (no line was drawn), so a reduced model shows no phantom entry.
    if (df_t["sigma_ion"] > 0).any():
        _handles.append(plt.Line2D([], [], color="gray", lw=0.8, ls="--", alpha=0.7))
        _labels.append(r"$\sigma_{ion}$ (fit)")
    fig.legend(_handles, _labels, loc="outside upper center", frameon=True,
               fontsize=9, ncol=min(len(_labels), 7), handlelength=1.3,
               columnspacing=0.8)

    ax_t.set_xlabel(r"$\log_{10}\![p(\mathrm{O_2})\:/\:\mathrm{bar}]$", fontsize=12)
    ax_t.set_ylabel(r"$t_\mathrm{ion} = \sigma_\mathrm{ion}/\sigma_\mathrm{tot}$", fontsize=12)
    # Zoom on the data range so overlapping curves separate; the 0/1 guides
    # and their labels appear only when they fall inside the window.
    _t_lo, _t_hi = float(df_t["t_ion"].min()), float(df_t["t_ion"].max())
    _pad = max(0.06, 0.15 * (_t_hi - _t_lo))
    y_lo, y_hi = max(-0.04, _t_lo - _pad), min(1.04, _t_hi + _pad)
    ax_t.set_ylim(y_lo, y_hi)
    if y_hi >= 1.0:
        ax_t.axhline(1.0, color="black", lw=0.8, ls=":", alpha=0.6)
        ax_t.annotate("purely ionic", xy=(0.985, 1.0),
                      xycoords=ax_t.get_yaxis_transform(),
                      xytext=(0, 3), textcoords="offset points",
                      fontsize=9, ha="right", va="bottom", fontstyle="italic",
                      color="#444444",
                      bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))
    if y_lo <= 0.0:
        ax_t.axhline(0.0, color="black", lw=0.8, ls=":", alpha=0.6)
        ax_t.annotate("purely electronic", xy=(0.985, 0.0),
                      xycoords=ax_t.get_yaxis_transform(),
                      xytext=(0, -3), textcoords="offset points",
                      fontsize=9, ha="right", va="top", fontstyle="italic",
                      color="#444444",
                      bbox=dict(fc="white", ec="none", alpha=0.7, pad=0.5))

    for ax in (ax_s, ax_t):
        ax.tick_params(direction="in", which="major", labelsize=10, width=1.2, length=4)
        ax.tick_params(direction="in", which="minor", width=1.2, length=2.5)
        for _sp in ax.spines.values():
            _sp.set_linewidth(1.2)

    if params is not None:
        from fractions import Fraction
        _fx = Fraction(params.x).limit_denominator(12)
        cells = [rf"$x = {_fx.numerator}/{_fx.denominator}$"
                 if _fx.denominator > 1 else rf"$x = {_fx.numerator}$"]
        # Per-channel peak contribution to sigma across the measured window.
        # sigma0 from the polish is never exactly 0 (a zeroed channel comes back
        # as an epsilon or a runaway pinned at an Ea bound), so classify a
        # channel by whether it actually carries current, not by sigma0 == 0.
        _po = df_t["pO2"].astype(float).to_numpy()
        _contrib = {
            "ion": df_t["sigma_ion"].astype(float).to_numpy(),
            "p":   df_t["sigma_p"].astype(float).to_numpy() * _po ** exponent,
            "n":   df_t["sigma_n"].astype(float).to_numpy() * _po ** (-exponent),
        }
        _totmax = float(np.nanmax(df_t["sigma_Scm"].astype(float).to_numpy()))
        for _ch, _sym in (("ion", r"\mathrm{ion}"), ("p", "p"), ("n", "n")):
            _ea = getattr(params, f"Ea_{_ch}")
            if not np.isfinite(_ea):
                continue  # excluded from the model: omit entirely
            _cmax = float(np.nanmax(_contrib[_ch])) if _contrib[_ch].size else 0.0
            if _totmax <= 0 or _cmax / _totmax < 1e-3:
                # selected but the fit zeroed it: a real result (no such
                # carriers in the measured pO2 window), not an absence.
                cells.append(rf"$\sigma_{{{_sym}}} \approx 0$ ($p$O$_2$ window)")
                continue
            _err = (perr or {}).get(f"Ea_{_ch}")
            _pm = (rf" \pm {_err:.2f}"
                   if _err is not None and np.isfinite(_err) else "")
            cells.append(rf"$E_{{a,{_sym}}} = {_ea:.2f}{_pm}$ eV")
        # Two-column grid, filled row by row so an omitted channel leaves no
        # hole. Columns are space-separated (mathtext, not pixel-aligned).
        _rows = [cells[i:i + 2] for i in range(0, len(cells), 2)]
        _txt = "\n".join("   ".join(r) for r in _rows)
        # Reserve an empty band above all data and dock the box there. Corner
        # placement is not robust: a flat model fills the panel and leaves no
        # free corner. Extending ylim past the data max guarantees the strip
        # under the top edge is clear whatever the curve shape.
        _y0, _y1 = ax_s.get_ylim()
        ax_s.set_ylim(_y0, _y1 + (0.04 + 0.09 * len(_rows)) * (_y1 - _y0))
        ax_s.text(0.02, 0.98, _txt, transform=ax_s.transAxes,
                  ha="left", va="top", fontsize=9.5, zorder=6,
                  bbox=dict(fc="white", ec="#888888", lw=0.6, alpha=0.85,
                            boxstyle="round,pad=0.4"))

    stem = (f"Brouwer_transference_Peak{peak_id}_{sample_name}"
            if sample_name else f"Brouwer_transference_Peak{peak_id}")
    _save(fig, save_dir, stem)
    return fig


def plot_transference_arrhenius(
    transf_df:   pd.DataFrame,
    save_dir:    Path | str,
    sample_name: str = "",
    peak_id:     int | None = None,
) -> plt.Figure | None:
    """
    Arrhenius plot of the partial conductivities from the Patterson
    decomposition: ln(σT) vs 1000/T for σ_ion and σ_p. σ_n is not drawn
    (for p-type samples it is a noise floor with no thermal activation)
    but stays in the exported stage4_transference.xlsx table.

    Straight lines here are the rigorous check that the decomposition
    separated two physically distinct channels: each partial conductivity
    must be thermally activated with its own Eₐ.

    Temperatures where NNLS set a channel exactly to zero carry no
    information about that channel and are skipped; the Eₐ line is fitted
    only when a channel keeps at least 3 non-zero temperatures.

    Parameters
    ----------
    transf_df : tidy output of fit_transference(); per-T channel values
                are repeated across pO₂ rows and collapsed internally.
    peak_id   : used for the title/filename (taken from the data if None).

    >>> fig = plot_transference_arrhenius(transf_df, "Results/plots",
    ...                                   sample_name="S1")  # doctest: +SKIP
    """
    if transf_df is None or transf_df.empty:
        warnings.warn("plot_transference_arrhenius: empty input", stacklevel=2)
        return None
    if peak_id is None:
        peak_id = int(transf_df["peak_id"].iloc[0])

    per_T = (transf_df[transf_df["peak_id"] == peak_id]
             .groupby("T_nominal")[["sigma_ion", "sigma_p", "sigma_n"]]
             .first())
    if per_T.empty:
        warnings.warn(f"plot_transference_arrhenius: no rows for peak {peak_id}",
                      stacklevel=2)
        return None

    T_K = per_T.index.to_numpy(dtype=float) + 273.15

    channels = [
        ("sigma_ion", r"$\sigma_{ion}$", "#0072B2", "o"),
        ("sigma_p",   r"$\sigma_{p}$",   "#D55E00", "s"),
    ]

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150, layout="constrained")
    drew = False
    for col, lab, color, marker in channels:
        # NNLS zeros mean "channel absent at this T", not a measured value
        sigma_Scm = per_T[col].to_numpy(dtype=float)   # already S/cm
        mask = sigma_Scm > 0
        if not mask.any():
            continue
        x  = 1000.0 / T_K[mask]
        ln = np.log(sigma_Scm[mask] * T_K[mask])
        if mask.sum() >= 3:
            Ea, Ea_err, r2, slope, intercept = _arrhenius_linreg(1.0 / T_K[mask], ln)
            label = (f"{lab}: $E_a$ = {Ea:.2f} ± {Ea_err:.2f} eV"
                     f"  (R²={r2:.3f}, n={int(mask.sum())})")
            fit_x = np.linspace(x.min() - 0.02, x.max() + 0.02, 100)
            ax.plot(fit_x, intercept + slope * (fit_x / 1000), "-",
                    color=color, linewidth=1.0, alpha=0.7)
        else:
            label = f"{lab} (n={int(mask.sum())}, no fit)"
        ax.plot(x, ln, marker, color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.5, label=label)
        drew = True

    if not drew:
        plt.close(fig)
        warnings.warn(f"plot_transference_arrhenius: all channels zero for "
                      f"peak {peak_id}", stacklevel=2)
        return None

    ax.set_xlabel(r"$1000\,T^{-1}\:/\:\mathrm{K^{-1}}$", fontsize=12)
    ax.set_ylabel(r"$\ln\!\left[\sigma T\:/\:\mathrm{(S\,K\,cm^{-1})}\right]$", fontsize=12)
    # No in-figure title: the file name carries the identity (publication style)
    ax.legend(fontsize=9, frameon=True, loc="best")
    ax.tick_params(direction="in", top=True, right=True)

    stem = (f"Transference_Arrhenius_Peak{peak_id}_{sample_name}"
            if sample_name else f"Transference_Arrhenius_Peak{peak_id}")
    _save(fig, save_dir, stem)
    return fig


# ---------------------------------------------------------------------------
# 8. Stage 5 - global MIEC model (2-D fit, 3-D surface, residual map)
# ---------------------------------------------------------------------------
# These draw the result of pipeline.model.fit_global_conductivity. The forward
# model itself lives in pipeline.model (no physics here): we only import it to
# evaluate the fitted curve/surface for display. Conductivity is shown in S/cm
# (sigma_Sm_i is stored in S/m), matching the Brouwer/transference figures.

def _clean_peak_xy(df_peak: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return finite, positive (pO2, T_C, sigma_S/cm) arrays from a peak frame."""
    Tc  = pd.to_numeric(df_peak["T_nominal"], errors="coerce").to_numpy(float)
    p   = pd.to_numeric(df_peak["pO2_mean"], errors="coerce").to_numpy(float)
    sig = pd.to_numeric(df_peak["sigma_Sm_i"], errors="coerce").to_numpy(float) / 100.0
    ok  = np.isfinite(Tc) & np.isfinite(p) & np.isfinite(sig) & (p > 0) & (sig > 0)
    return p[ok], Tc[ok], sig[ok]


def plot_conductivity_surface_3d(df_peak, params, save_dir, *, sample_name="",
                                 peak_id=None, save=True):
    """3-D surface sigma(pO2, T) (fitted) with the measured points scattered on it.

    sigma is on the vertical axis (log10, S/cm); pO2 (log10) and T on the base.

    >>> fig = plot_conductivity_surface_3d(df_peak, params, "Results/plots",
    ...     sample_name="S1", peak_id=1)  # doctest: +SKIP
    """
    from pipeline.model import predict_grid

    p_obs, Tc, sig = _clean_peak_xy(df_peak)
    if p_obs.size == 0:
        warnings.warn("plot_conductivity_surface_3d: no usable points", stacklevel=2)
        return None
    p_grid = np.logspace(np.log10(p_obs.min()), np.log10(p_obs.max()), 40)
    T_grid = np.linspace(Tc.min(), Tc.max(), 40)
    Z = predict_grid(params, p_grid, T_grid + 273.15) / 100.0  # (nT, npO2), S/cm
    PO2, TT = np.meshgrid(np.log10(p_grid), T_grid)

    fig = plt.figure(figsize=(8, 6), dpi=150)
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(PO2, TT, np.log10(Z), cmap="viridis", alpha=0.6,
                    linewidth=0, antialiased=True)
    ax.scatter(np.log10(p_obs), Tc, np.log10(sig), color="black", s=18, depthshade=True)
    ax.set_xlabel(r"$\log_{10}\![p(\mathrm{O_2})\:/\:\mathrm{bar}]$", fontsize=10)
    ax.set_ylabel(r"$T\:/\:^\circ\!\mathrm{C}$", fontsize=10)
    ax.set_zlabel(r"$\log_{10}\!\left[\sigma\:/\:\mathrm{(S\,cm^{-1})}\right]$", fontsize=10, labelpad=8)
    # pure-white cube walls instead of the default off-grey panes
    for _pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        _pane.set_facecolor("white")
        _pane.set_alpha(1.0)
    # shrink the cube slightly so the rotated z-label fits inside the canvas
    ax.set_box_aspect(None, zoom=0.85)

    if save:
        stem = f"Stage5_surface3D_Peak{peak_id}_{sample_name}" if sample_name else f"Stage5_surface3D_Peak{peak_id}"
        # tight=False: the 3-D z-label is dropped by the tight bounding box
        _save(fig, save_dir, stem, tight=False)
    return fig


def plot_fit_residuals(df_peak, params, save_dir, *, sample_name="", peak_id=None,
                       save=True):
    """Relative-residual map (sigma_fit - sigma_obs)/sigma_obs over (pO2, T).

    A structureless cloud means the model fits; systematic zones flag physics
    outside the 3-channel model (e.g. departure from the dilute regime).

    >>> fig = plot_fit_residuals(df_peak, params, "Results/plots",
    ...     sample_name="S1", peak_id=1)  # doctest: +SKIP
    """
    from pipeline.model import total_conductivity

    p_obs, Tc, sig = _clean_peak_xy(df_peak)
    if p_obs.size == 0:
        warnings.warn("plot_fit_residuals: no usable points", stacklevel=2)
        return None
    sig_fit = total_conductivity(p_obs, Tc + 273.15, params) / 100.0
    resid = (sig_fit - sig) / sig
    vmax = float(np.max(np.abs(resid))) if resid.size else 1.0

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sc = ax.scatter(p_obs, Tc, c=resid, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    s=60, edgecolor="black", linewidth=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$p(\mathrm{O_2})\:/\:\mathrm{bar}$", fontsize=12)
    ax.set_ylabel(r"$T\:/\:^\circ\!\mathrm{C}$", fontsize=12)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    fig.colorbar(sc, ax=ax, label=r"relative residual $(\sigma_\mathrm{model}-\sigma_\mathrm{exp})/\sigma_\mathrm{exp}$")

    if save:
        stem = f"Stage5_residuals_Peak{peak_id}_{sample_name}" if sample_name else f"Stage5_residuals_Peak{peak_id}"
        _save(fig, save_dir, stem)
    return fig
