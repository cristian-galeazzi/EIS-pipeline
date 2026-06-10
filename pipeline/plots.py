"""
pipeline/plots.py
=================
Publication-quality plot functions for the EIS pipeline.

Visual style is adapted exactly from the existing reference notebooks:
  - DRT_Analysis_SAMPLE_ID_(Ar-SCCM_O2-SCCM).ipynb  → DRT stacked, Arrhenius panel
  - Analysis_SAMPLE_ID_(Ar-SCCM_O2-SCCM).ipynb      → Nyquist, Bode
  - Brouwer_pO2_Dependence_SAMPLE_ID.ipynb           → Brouwer diagram

All functions:
  apply_pub_style()               — set rcParams once per session
  plot_drt_stacked()              — stacked DRT γ(τ) with vertical offset
  plot_nyquist_multipanel()       — data circles + fit dashes, all temperatures
  plot_bode()                     — |Z| and phase Bode, all temperatures
  plot_arrhenius_panel()          — 2×2 panel: ln(σT), ln(τ), ln(C), log₁₀(εᵣ)
  plot_brouwer()                  — Brouwer p(O₂) diagram (multi-condition)
  plot_tau_arrhenius_consistency()— ln(τ) vs 1000/T per peak to check physicality

Supporting helpers (public API):
  build_arrhenius_results()       — compute Ea, R², pre-exponentials from df_peaks
  COLOR_MAP                       — T [°C] → hex colour dict
  PEAK_COLORS, PEAK_MARKERS       — per-peak visual style
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import LogLocator, LogFormatterMathtext, MultipleLocator
from scipy import stats
from pathlib import Path

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KB    = 8.617e-5   # Boltzmann constant [eV/K]
EPS_0 = 8.854e-12  # vacuum permittivity [F/m]

# ---------------------------------------------------------------------------
# Visual constants — must match existing notebooks exactly
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

    Serif (Times New Roman / DejaVu Serif) with Computer Modern math —
    matches the reference notebooks (DRT_Analysis, Brouwer) exactly.
    Inward ticks, top+right tick marks, dpi=150 display / 300 export.
    """
    mpl.rcParams.update({
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset":   "cm",
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

def _save(fig: plt.Figure, save_dir: Path | str, stem: str) -> None:
    """Export figure as PNG and PDF in save_dir."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(save_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


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
    offset_step:   float       = 1.2,
    label_tau:     float | None = None,
    exclude_temps: list[int] | None = None,
    save:          bool        = True,
) -> plt.Figure:
    """
    Stacked DRT plot — normalised γ(τ) with vertical offset per temperature.

    Each trace is normalised to its own maximum, then offset vertically so
    temperatures stack from bottom (low T) to top (high T) with no overlap.

    Parameters
    ----------
    df_spectra    : DataFrame with columns [T_nominal, tau, gamma]
                    (from stage3_drt.xlsx sheet "DRT_Spectra")
    condition     : condition name used for the figure title and filename
    save_dir      : directory where PNG + PDF are written
    tau_max       : x-axis upper limit [s] (default 1.0)
    offset_step   : vertical spacing between traces (default 1.2)
    label_tau     : τ position for the T-label text (auto-set to 5e-8 if None)
    exclude_temps : list of T_nominal [°C] to skip
    save          : when False, do not write PNG/PDF (diagnostic preview only).

    Returns
    -------
    matplotlib Figure
    """
    exclude_temps = exclude_temps or []
    temps = sorted(df_spectra["T_nominal"].unique())
    temps = [int(t) for t in temps if int(t) not in exclude_temps]

    fig, ax = plt.subplots(figsize=(4, 4), dpi=200, layout="constrained")

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

        lbl_x = label_tau if label_tau is not None else 5e-8
        ax.text(lbl_x, baseline + 0.5, f"{T} °C",
                fontsize=8, va="center", ha="left", color="black")

    ax.set_xscale("log")
    ax.set_xlabel(r"$\tau$ / s", fontsize=8, labelpad=10)
    lbl_x = label_tau if label_tau is not None else 5e-8
    ax.set_xlim([lbl_x * 0.8, tau_max])
    ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$ (normalised)", fontsize=8, labelpad=10)
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
                 The fit overlay still spans the full data range — only the
                 axes are limited. None = auto-scale from data (default).
    hf_inset   : when True (default) draw an upper-right inset zoomed on the
                 high-frequency arcs. The zoom range auto-adapts from the
                 highest-frequency 40 % of points across all temperatures.
    save       : when False, do not write PNG/PDF (interactive preview only).

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
                ax.plot(Z_fit.real / 1e3, -Z_fit.imag / 1e3, "-", color=color, lw=1.0)
            except Exception as exc:
                warnings.warn(f"Nyquist fit overlay skipped for T={T}: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)

    ax.set_xlabel(r"$Z'$ / k$\Omega$")
    ax.set_ylabel(r"$-Z''$ / k$\Omega$")
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
        # cold arcs run off the inset edge (acceptable — the full panel shows them).
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
                ax_mag.loglog(freq, Z_mag_f, "--", color=color, lw=1)
                ax_ph.semilogx(freq, phase_f, "--", color=color, lw=1)
            except Exception as exc:
                warnings.warn(f"Bode fit overlay skipped for T={T}: "
                              f"{type(exc).__name__}: {exc}", stacklevel=2)

    ax_mag.set_ylabel(r"$|Z|$ / k$\Omega$")
    ax_mag.legend(loc="upper right", ncol=2, fontsize=8)
    ax_mag.grid(True, which="both", ls=":", alpha=0.4)

    # Equivalent-circuit annotation (number of Zarc elements). Auto-built from
    # the fit when not supplied; drawn in a wheat box like the reference notebook.
    if model_label is None:
        n_set = sorted({len(fp["R"]) for fp in fit_params.values() if fp})
        if len(n_set) == 1:
            model_label = "R0–" + "–".join(["Zarc"] * n_set[0])
        elif n_set:
            model_label = f"R0 + Zarc×{n_set[0]}–{n_set[-1]}"
        else:
            model_label = ""
    if model_label:
        ax_mag.text(0.03, 0.08, model_label, transform=ax_mag.transAxes,
                    fontsize=9, ha="left", va="bottom",
                    bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    ax_ph.set_xlabel("Frequency / Hz")
    ax_ph.set_ylabel("Phase / °")
    ax_ph.grid(True, which="both", ls=":", alpha=0.4)

    if freq_lim is not None:
        ax_mag.set_xlim(left=freq_lim[0], right=freq_lim[1])
        ax_ph.set_xlim(left=freq_lim[0], right=freq_lim[1])
    if mag_lim is not None:
        ax_mag.set_ylim(bottom=mag_lim[0], top=mag_lim[1])
    if phase_lim is not None:
        ax_ph.set_ylim(bottom=phase_lim[0], top=phase_lim[1])

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
) -> list[dict]:
    """
    Compute Arrhenius fit results for each peak_id from stage3_fit.xlsx Peaks sheet.

    Physical quantities derived
    ---------------------------
    σ  = L / (R · A)       [S/m]          (conductivity)
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

    Returns
    -------
    List of dicts, one per peak_id, sorted by ascending peak_id.
    Each dict contains: T_C, T_K, inv_T, inv_T_fit, sigma, tau, R, C,
    ln_sigmaT, ln_tau, ln_C, Ea_cond, Ea_pol, Ea_C, R2_cond, R2_pol, R2_C,
    slope_*, int_*, color, marker.
    """
    A_m2    = np.pi * (D_m / 2) ** 2
    results = []

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

        if "sigma_Sm_i" in sub.columns:
            sigma = sub["sigma_Sm_i"].values.astype(float)
        else:
            sigma = L_m / (R * A_m2)

        ln_sigmaT = np.log(sigma * T_K)
        ln_tau    = np.log(tau)
        ln_C      = np.log(C)

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
    ea_err_key:  str | None = None,
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
        if ea_err_key and not np.isnan(r.get(ea_err_key, np.nan)):
            # 95% CI band: propagate slope SE (eV → 1/K) to fit-line uncertainty
            se_slope = r[ea_err_key] / KB
            ci = se_slope * np.abs(fit_x / 1000)
            ax.fill_between(fit_x, fit_y - ci, fit_y + ci,
                            alpha=0.12, color=r["color"], linewidth=0)

    ax.set_xlabel(r"1000$\cdot T^{-1}$/ K$^{-1}$", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=10, frameon=True, ncol=2)
    ax.tick_params(direction="in", top=True, right=True)


# ---------------------------------------------------------------------------
# 5. Arrhenius 2×2 panel
# ---------------------------------------------------------------------------

def plot_arrhenius_panel(
    df_peaks:  pd.DataFrame,
    L_m:       float,
    D_m:       float,
    condition: str,
    save_dir:  Path | str,
) -> tuple[plt.Figure, list[dict]]:
    """
    2×2 Arrhenius panel: ln(σT), ln(τ), ln(C), log₁₀(εᵣ).

    Panel layout
    ------------
    [0,0]  ln(σT)  vs 1000/T  →  Eₐᶜᵒⁿᵈ  (long-range transport)
    [0,1]  ln(τ)   vs 1000/T  →  Eₐᵖᵒˡ   (polarisation / local hop)
    [1,0]  ln(C)   vs 1000/T  →  Eₐᶜ      (net: Eₐᵖᵒˡ − Eₐᶜᵒⁿᵈ)
    [1,1]  log₁₀(εᵣ) vs 1000/T (no Arrhenius line — diagnostic only)

    Parameters
    ----------
    df_peaks  : DataFrame from stage3_fit.xlsx sheet "Peaks"
    L_m, D_m  : sample geometry [m]
    condition : label for title and filename
    save_dir  : output directory

    Returns
    -------
    (fig, results_all)
    results_all is the list from build_arrhenius_results() — reuse for tables.
    """
    results = build_arrhenius_results(df_peaks, L_m, D_m)
    A_m2    = np.pi * (D_m / 2) ** 2

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), dpi=150, layout="constrained")

    _draw_arrhenius_panel(
        axes[0, 0], results, "ln_sigmaT", "slope_cond", "int_cond", "Ea_cond",
        r"E_a^{cond}", r"ln($\sigma T$ / S·K·m$^{-1}$)", ea_err_key="Ea_cond_err")

    _draw_arrhenius_panel(
        axes[0, 1], results, "ln_tau", "slope_pol", "int_pol", "Ea_pol",
        r"E_a^{pol}", r"ln($\tau$ / s)", ea_err_key="Ea_pol_err")

    _draw_arrhenius_panel(
        axes[1, 0], results, "ln_C", "slope_C", "int_C", "Ea_C",
        r"E_a^{C}", r"ln($C$ / F)", ea_err_key="Ea_C_err")

    # Panel [1,1]: log₁₀(εᵣ) — diagnostic, no fit line
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
    ax4.set_xlabel(r"1000$\cdot T^{-1}$/ K$^{-1}$", fontsize=12)
    ax4.set_ylabel(r"log$_{10}$($\varepsilon_r$)", fontsize=12)
    ax4.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15),
               fontsize=10, frameon=True, ncol=2)
    ax4.tick_params(direction="in", top=True, right=True)

    _save(fig, save_dir, f"Arrhenius_{condition}")
    return fig, results


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
    sample_name   : sample label for title and filename stem
    peak_id       : which peak to use (default 1 = highest-frequency process)
    temps_to_plot : temperatures to show (default None = all available)
    add_slopes    : draw −1/4, 0, +1/4 reference slopes (default True)

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

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150, layout="constrained")

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

    if add_slopes:
        y_ref = y_hi + 0.22
        _slope_line(ax, -3.6, -3.0, y_ref, slope=-1 / 4,
                    label=r"$-1/4$", label_side="left",
                    color="black", lw=1.0, ls="--", zorder=3)
        _slope_line(ax, -2.0, -1.4, y_ref, slope=0,
                    label=r"$plateau$", label_side="left",
                    color="black", lw=1.0, ls="--", zorder=3)
        _slope_line(ax, -0.8, -0.2, y_ref, slope=1 / 4,
                    label=r"$+1/4$", label_side="right",
                    color="black", lw=1.0, ls="--", zorder=3)
        ax.text(
            0.02, 0.03,
            r"$-1/4$: $n$-type    $plateau$: purely ionic    $+1/4$: $p$-type",
            transform=ax.transAxes, fontsize=7, va="bottom", ha="left",
            fontstyle="italic", color="#444444",
        )

    ax.set_xlabel(r"log$_{10}$[$p$(O$_2$) / bar]", fontsize=12)
    ax.set_ylabel(
        r"log$_{10}$($\sigma_{" + str(peak_id) + r"}$ / S cm$^{-1}$)", fontsize=12)
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


# ---------------------------------------------------------------------------
# 7. τ Arrhenius consistency check
# ---------------------------------------------------------------------------

def plot_tau_arrhenius_consistency(
    df_peaks:     pd.DataFrame,
    condition:    str,
    save_dir:     Path | str,
    r2_threshold: float = 0.97,
    save:         bool  = True,
) -> plt.Figure:
    """
    Arrhenius plot of ln(τᵢ) vs 1000/T for each DRT peak.

    Rationale
    ---------
    A physically real relaxation process obeys τ = τ₀ exp(Eₐ / kT), so
    ln(τ) vs 1/T should be linear with R² ≥ 0.97.  Flat or scattered trends
    indicate noise artefacts or overlapping processes that were incorrectly
    separated by the DRT.

    Parameters
    ----------
    df_peaks      : DataFrame from stage3_fit.xlsx sheet "Peaks"
    condition     : condition label (title + filename)
    save_dir      : output directory
    r2_threshold  : R² ≥ this value labels the peak as physically consistent
                    (default 0.97)
    save          : when False, do not write PNG/PDF (diagnostic preview only).

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150, layout="constrained")

    for i, pid in enumerate(sorted(df_peaks["peak_id"].unique())):
        sub     = df_peaks[df_peaks["peak_id"] == pid].sort_values("T_nominal")
        T_K     = sub["T_nominal"].values.astype(float) + 273.15
        inv_T   = 1000.0 / T_K
        ln_tau  = np.log(sub["tau_i"].values.astype(float))
        color   = PEAK_COLORS[i % len(PEAK_COLORS)]
        marker  = PEAK_MARKERS[i % len(PEAK_MARKERS)]

        valid = ~np.isnan(ln_tau)
        if valid.sum() < 2:
            continue

        _, _, R2, slope, intercept = _arrhenius_linreg(1.0 / T_K, ln_tau)
        r2_str  = f"R²={R2:.3f}" if not np.isnan(R2) else "R²=N/A"
        quality = "  (OK)" if (not np.isnan(R2) and R2 >= r2_threshold) else ""

        ax.plot(inv_T[valid], ln_tau[valid], marker,
                color=color, markersize=7,
                markeredgecolor="black", markeredgewidth=0.5,
                label=f"Peak {pid}  {r2_str}{quality}")

        fit_x = np.linspace(inv_T.min() - 0.02, inv_T.max() + 0.02, 100)
        fit_y = intercept + slope * (fit_x / 1000)
        ax.plot(fit_x, fit_y, "-", color=color, lw=1.0, alpha=0.7)

    ax.set_xlabel(r"1000$\cdot T^{-1}$/ K$^{-1}$", fontsize=12)
    ax.set_ylabel(r"ln($\tau$ / s)", fontsize=12)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(True, ls=":", alpha=0.3)

    if save:
        _save(fig, save_dir, f"TauConsistency_{condition}")
    return fig


# ---------------------------------------------------------------------------
# 8. τ-track diagnostic (cross-temperature peak alignment)
# ---------------------------------------------------------------------------

def plot_tau_tracks(
    df_peaks:    pd.DataFrame,
    condition:   str,
    save_dir:    Path | str,
    save:        bool  = True,
) -> plt.Figure:
    """
    Diagnostic: τᵢ (log) vs temperature for each peak_id.

    `peak_id` is assigned purely by τ-rank within each spectrum, so the same
    peak_id can represent a DIFFERENT physical process at different temperatures
    when the DRT detects a varying number of peaks (e.g. the broad, poorly
    deconvolved feature near τ≈10⁻⁴ s). This plot makes that drift visible: a
    consistent track forms a smooth, monotonic curve; where a track jumps or a
    peak_id appears/disappears, the Arrhenius / Brouwer grouping by peak_id is
    mixing peaks.

    Diagnostic ONLY — it reads the same `tau_i` column and changes no result.

    Parameters
    ----------
    df_peaks    : DataFrame from stage3_fit.xlsx sheet "Peaks"
                  (columns peak_id, T_nominal, tau_i)
    condition   : condition label (filename stem)
    save_dir    : output directory
    save        : when False, do not write PNG/PDF (diagnostic preview only).

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150, layout="constrained")

    for i, pid in enumerate(sorted(df_peaks["peak_id"].unique())):
        sub   = df_peaks[df_peaks["peak_id"] == pid].sort_values("T_nominal")
        T_C   = sub["T_nominal"].values.astype(float)
        tau   = sub["tau_i"].values.astype(float)
        valid = ~np.isnan(tau) & (tau > 0)
        if valid.sum() == 0:
            continue
        ax.plot(T_C[valid], tau[valid],
                marker=PEAK_MARKERS[i % len(PEAK_MARKERS)] + "",
                color=PEAK_COLORS[i % len(PEAK_COLORS)],
                markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                linewidth=1.0, label=f"Peak {pid}")

    ax.set_yscale("log")
    ax.set_xlabel(r"$T$ / °C", fontsize=12)
    ax.set_ylabel(r"$\tau$ / s", fontsize=12)
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, which="both", ls=":", alpha=0.3)

    if save:
        _save(fig, save_dir, f"TauTracks_{condition}")
    return fig


def plot_ceff_magnitude(
    df_peaks:  pd.DataFrame,
    condition: str,
    save_dir:  Path | str,
    save:      bool = True,
) -> plt.Figure:
    """
    Effective-capacitance magnitude vs temperature, one track per peak_id.

    Plots log₁₀(C_eff / F) against 1000/T so the operator can read the likely
    physical nature of each peak directly from its capacitance magnitude,
    without any hard-coded process-name thresholds. C_eff is read as-is from the
    Zarc fit (`C_eff_i` column); this function assigns no process label.

    Parameters
    ----------
    df_peaks  : DataFrame from stage3_fit.xlsx sheet "Peaks"
                (columns peak_id, T_nominal, C_eff_i)
    condition : condition label (title + filename stem)
    save_dir  : output directory
    save      : when False, do not write PNG/PDF (diagnostic preview only).

    Returns
    -------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150, layout="constrained")

    for i, pid in enumerate(sorted(df_peaks["peak_id"].unique())):
        sub   = df_peaks[df_peaks["peak_id"] == pid].sort_values("T_nominal")
        T_K   = sub["T_nominal"].values.astype(float) + 273.15
        inv_T = 1000.0 / T_K
        C     = sub["C_eff_i"].values.astype(float)
        valid = ~np.isnan(C) & (C > 0)
        if valid.sum() == 0:
            continue
        ax.plot(inv_T[valid], np.log10(C[valid]),
                marker=PEAK_MARKERS[i % len(PEAK_MARKERS)],
                color=PEAK_COLORS[i % len(PEAK_COLORS)],
                markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                linewidth=1.5, label=f"Peak {pid}")

    ax.set_xlabel(r"1000$\cdot T^{-1}$/ K$^{-1}$", fontsize=12)
    ax.set_ylabel(r"log$_{10}$($C_\mathrm{eff}$ / F)", fontsize=12)
    ax.legend(loc="best", fontsize=10, frameon=True, ncol=2)
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.grid(True, alpha=0.3, linestyle="--")

    if save:
        _save(fig, save_dir, f"Capacity_{condition}")
    return fig
