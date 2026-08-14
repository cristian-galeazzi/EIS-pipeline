"""Guard tests for pipeline/plots.py, what a figure says and in which unit.

Two sections. The first covers the suptitle, the only text on a figure that
identifies the measurement. The second covers the unit every conductivity is
reported in, which is a reporting boundary rather than an engine change: the
stored column stays S/m and only the read converts.

Neither section touches a computed number. They pin what is printed.
"""

import ast
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import mathtext  # noqa: E402

from pipeline.plots import (  # noqa: E402
    KB, _condition_suptitle, build_arrhenius_results, plot_arrhenius_sigma,
    plot_brouwer, plot_drt_stacked,
)

# --------------------------------------------------------------------------
# _condition_suptitle
#
# pipeline/plots.py has no set_title anywhere, so the suptitle is the only text
# on a figure that identifies the measurement. Two properties follow: it must
# keep printing exactly what it prints today when nothing is asked of it, and it
# must never collapse to nothing while a label is available.
# --------------------------------------------------------------------------

# far above any physical p(O2): pure O2 at ambient is about 1 bar, so a value
# like this can only be an idle probe's output
IDLE_PROBE_BAR = 4200.0


def _df(p: float | None = 2.1e-3) -> pd.DataFrame:
    return pd.DataFrame({"pO2_mean": [] if p is None else [p, p]})


def test_default_call_is_a_single_math_expression() -> None:
    # math and text alternating gives the "=" and the space before the unit two
    # different widths, one set by mathtext and one by the text font
    assert _condition_suptitle(_df()) == (
        r"$p(\mathrm{O_2}) = 2.1\times10^{-3}\:\mathrm{bar}$")


def test_label_precedes_the_pressure() -> None:
    assert _condition_suptitle(_df(), "Ar-100 O2-20 | 400-600C") == (
        r"Ar-100 O2-20 | 400-600C,  "
        r"$p(\mathrm{O_2}) = 2.1\times10^{-3}\:\mathrm{bar}$")


def test_probe_off_shows_the_label_alone() -> None:
    assert _condition_suptitle(_df(IDLE_PROBE_BAR), "Air | 400-600C",
                               show_pO2=False) == "Air | 400-600C"


def test_probe_off_without_a_label_is_empty_rather_than_malformed() -> None:
    # the failure this guards against is "p(O2) =  bar", a title with a hole
    assert _condition_suptitle(_df(IDLE_PROBE_BAR), "", show_pO2=False) == ""


def test_no_pO2_column_falls_back_to_the_label() -> None:
    assert _condition_suptitle(pd.DataFrame({"x": [1]}),
                               "Air | 400-600C") == "Air | 400-600C"


def test_nonpositive_pressures_are_ignored_as_before() -> None:
    df = pd.DataFrame({"pO2_mean": [0.0, -1.0]})
    assert _condition_suptitle(df, "Air") == "Air"


def test_the_median_is_used_not_the_first_row() -> None:
    df = pd.DataFrame({"pO2_mean": [0.20, 0.21, 0.22]})
    assert _condition_suptitle(df) == r"$p(\mathrm{O_2}) = 0.21\:\mathrm{bar}$"


# --------------------------------------------------------------------------
# The unit the Arrhenius family reports
#
# The engine stores conductivity in S/m as ``sigma_Sm_i`` and every figure and
# table reports S/cm. The conversion therefore lives at one reporting boundary,
# ``build_arrhenius_results``, which has two branches: the stored column and a
# fallback that recomputes sigma from the pellet geometry. A fallback that
# forgets the conversion is a hundred times off and still plots a straight line
# with the right slope, so nothing downstream would catch it.
#
# The activation energy must not notice the change at all: a constant factor in
# sigma is an additive offset in ln(sigma*T), which moves the intercept only.
# --------------------------------------------------------------------------

L_M, D_M = 1.0e-3, 1.0e-2
A_M2 = np.pi * (D_M / 2) ** 2


def _frame(with_sigma_column: bool) -> pd.DataFrame:
    """Nine isotherms of an exact 0.9 eV conductor, with or without sigma_Sm_i.

    >>> _frame(True).shape[0]
    9
    >>> "sigma_Sm_i" in _frame(False).columns
    False
    """
    T_C = np.array([400, 450, 500, 550, 600, 650, 700, 750, 800], dtype=float)
    T_K = T_C + 273.15
    sigma_Sm = np.exp(-0.9 / (KB * T_K)) / T_K
    df = pd.DataFrame({
        "peak_id":   1,
        "T_nominal": T_C,
        "R_i":       L_M / (sigma_Sm * A_M2),
        "tau_i":     1e-4 * np.exp(0.5 / (KB * T_K)),
        "C_eff_i":   5e-8,
        "sigma_Sm_i": sigma_Sm,
    })
    return df if with_sigma_column else df.drop(columns=["sigma_Sm_i"])


def test_the_stored_column_is_reported_in_S_per_cm() -> None:
    df = _frame(True)
    res = build_arrhenius_results(df, L_M, D_M)[0]
    expected = df["sigma_Sm_i"].to_numpy(dtype=float) / 100.0
    assert np.allclose(res["sigma"], expected, rtol=0, atol=0)


def test_the_geometry_fallback_reports_the_same_unit() -> None:
    # the branch taken when stage 3 wrote no sigma column; it must not be a
    # hundred times larger than the branch above
    with_col = build_arrhenius_results(_frame(True), L_M, D_M)[0]
    without  = build_arrhenius_results(_frame(False), L_M, D_M)[0]
    assert np.allclose(without["sigma"], with_col["sigma"], rtol=1e-12)


def test_the_activation_energy_ignores_the_unit() -> None:
    res = build_arrhenius_results(_frame(True), L_M, D_M)[0]
    assert abs(float(res["Ea_cond"]) - 0.9) < 1e-9
    assert abs(float(res["R2_cond"]) - 1.0) < 1e-12


def test_the_series_sum_shares_the_axis_unit(tmp_path) -> None:
    # the sum line is computed from geometry, not from the stored column, and
    # is drawn on the same axis as the per-peak branches: a missing conversion
    # would offset it by ln(100) and look like a real split between the two
    base = _frame(True)
    two_peaks = pd.concat([base, base.assign(peak_id=2)], ignore_index=True)
    fig = plot_arrhenius_sigma(two_peaks, L_M, D_M, "COND", tmp_path,
                               sum_peak_ids=[1, 2])
    assert fig is not None
    ax = fig.axes[0]
    sums = [ln for ln in ax.get_lines() if "series sum" in (ln.get_label() or "")]
    assert len(sums) == 1
    T_K = base["T_nominal"].to_numpy(dtype=float) + 273.15
    R_sum = 2.0 * base["R_i"].to_numpy(dtype=float)
    expected = np.log(L_M / (R_sum * A_M2) / 100.0 * T_K)
    assert np.allclose(np.sort(sums[0].get_ydata()), np.sort(expected))
    plt.close(fig)


def test_the_intercept_carries_the_whole_change() -> None:
    # a run of the previous unit, reproduced by feeding S/m a hundred times
    # larger, must differ by exactly ln(100) and by nothing else
    df_new = _frame(True)
    df_old = df_new.assign(sigma_Sm_i=df_new["sigma_Sm_i"] * 100.0)
    new = build_arrhenius_results(df_new, L_M, D_M)[0]
    old = build_arrhenius_results(df_old, L_M, D_M)[0]
    assert abs((float(old["int_cond"]) - float(new["int_cond"])) - np.log(100.0)) < 1e-12
    assert abs(float(old["slope_cond"]) - float(new["slope_cond"])) < 1e-9


# --------------------------------------------------------------------------
# plot_drt_stacked: the tau window
#
# tau_max has always dropped the points above it before each trace is
# normalized, so it decides both the frame and the heights. tau_min was added
# for a sample measured to lower frequencies and does neither of those two
# jobs at once: it frames, and nothing else. These tests pin that split, and
# that a call without tau_min draws exactly what it drew before.
# --------------------------------------------------------------------------


def _drt_spectra() -> pd.DataFrame:
    """Two temperatures over six decades of tau, one peak each."""
    tau = np.logspace(-8, -2, 200)
    rows = []
    for T, centre in ((600, 1e-6), (500, 1e-4)):
        gamma = np.exp(-((np.log10(tau) - np.log10(centre)) ** 2) / 0.05)
        rows.append(pd.DataFrame({"T_nominal": T, "tau": tau, "gamma": gamma}))
    return pd.concat(rows, ignore_index=True)


def _drawn(fig: plt.Figure) -> int:
    return sum(len(line.get_xdata()) for line in fig.axes[0].lines)


def test_without_tau_min_the_frame_is_the_one_drawn_before() -> None:
    fig = plot_drt_stacked(_drt_spectra(), "C", ".", tau_max=1e-2, save=False)
    assert fig.axes[0].get_xlim() == (5e-8 * 0.8, 1e-2)
    plt.close(fig)


def test_tau_min_moves_the_frame_to_exactly_what_was_asked() -> None:
    fig = plot_drt_stacked(_drt_spectra(), "C", ".", tau_max=1e-2,
                           tau_min=1e-5, save=False)
    assert fig.axes[0].get_xlim() == (1e-5, 1e-2)
    plt.close(fig)


def test_tau_min_drops_no_data() -> None:
    # the guarantee that separates it from tau_max: narrowing the frame from
    # the left must not remove a point, so no trace is renormalized
    df = _drt_spectra()
    wide = plot_drt_stacked(df, "C", ".", tau_max=1e-2, save=False)
    narrow = plot_drt_stacked(df, "C", ".", tau_max=1e-2, tau_min=1e-5, save=False)
    assert _drawn(narrow) == _drawn(wide)
    plt.close(wide)
    plt.close(narrow)


def test_the_temperature_labels_stay_inside_a_narrowed_frame() -> None:
    fig = plot_drt_stacked(_drt_spectra(), "C", ".", tau_max=1e-2,
                           tau_min=1e-5, save=False)
    lo, hi = fig.axes[0].get_xlim()
    assert [t for t in fig.axes[0].texts] != []
    assert all(lo <= t.get_position()[0] <= hi for t in fig.axes[0].texts)
    plt.close(fig)


# --------------------------------------------------------------------------
# Typography of the mathtext the module prints
#
# IUPAC (Green Book, quantity symbols) and ISO 80000-1: a script is italic only
# when it is itself a quantity or a running index, and upright when it is
# descriptive. So "cond", "pol", "ion" and the "a" of an activation energy are
# upright, while the index i and the carrier type p stay italic. Italic
# multi-letter scripts are not a style preference: set in italic they read as a
# product of that many quantities.
#
# The axis labels already followed this. The legends and the annotations did
# not, which is what these two tests pin.
# --------------------------------------------------------------------------

PLOTS_SRC = Path(__file__).resolve().parents[1] / "pipeline" / "plots.py"

# a script of two or more letters, or the bare "a" of an activation energy,
# with no \mathrm or \text in front of it
ITALIC_SCRIPT = re.compile(r"[_^]\{?(a|[A-Za-z]{2,})\}?")
# a span that is nothing but a word: italic letters with multiplication spacing
BARE_WORD = re.compile(r"^[A-Za-z]{2,}$")


def _math_spans() -> list[str]:
    """Every complete ``$...$`` span in plots.py, from the string literals.

    A literal with an odd number of ``$`` is half of a concatenation, so its
    spans are not readable on their own and it is skipped. So is a span holding
    a doubled backslash: that is a doctest quoting a returned string, not a
    label matplotlib will ever be asked to draw.

    >>> len(_math_spans()) > 20
    True
    """
    spans = []
    for node in ast.walk(ast.parse(PLOTS_SRC.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts = node.value.split("$")
            if len(parts) % 2:
                spans += [p for p in parts[1::2] if p.strip() and "\\\\" not in p]
    return spans


def test_every_descriptive_script_is_upright() -> None:
    offenders = [s for s in _math_spans()
                 if ITALIC_SCRIPT.search(s) or BARE_WORD.match(s.strip())]
    assert offenders == []


def test_every_math_span_still_parses() -> None:
    # \mathrm{} is easy to unbalance by hand, and a broken span raises only
    # when the figure is drawn, which no test of a label string would reach
    parser = mathtext.MathTextParser("path")
    for span in _math_spans():
        parser.parse(f"${span}$")


def test_the_suptitle_parses_as_mathtext() -> None:
    # the suptitle is assembled at runtime, so _math_spans never sees it: the
    # failure to catch is a literal backslash printed on the figure
    parser = mathtext.MathTextParser("path")
    parser.parse(_condition_suptitle(_df()))
    parser.parse(_condition_suptitle(_df(0.21)))


# --------------------------------------------------------------------------
# The Brouwer frame
#
# The x limits were fixed at (-5.5, 0.5) while the y limits followed the data,
# so a condition measured below 1e-5.5 bar was drawn and then left outside the
# frame, with no warning and no gap in the legend. That window is exactly the
# reducing branch an operator reads to decide whether the n channel belongs in
# the decomposition, so the figure hid the evidence for its own parameter.
#
# The frame is still the historical one whenever the data fit inside it: a
# sample measured in the oxidizing window must keep the figure it always had.
# --------------------------------------------------------------------------

HISTORICAL_FRAME = (-5.5, 0.5)


def _brouwer_df(pO2_levels: list[float]) -> pd.DataFrame:
    """Two isotherms over the given pressures, ionic plus a p-type branch."""
    rows = []
    for T, level in ((500, 1.0), (600, 2.0)):
        for p in pO2_levels:
            sigma_Scm = level * (1e-5 + 1e-4 * p ** 0.25)
            rows.append({"peak_id": 1, "T_nominal": T, "pO2_mean": p,
                         "sigma_Sm_i": sigma_Scm * 100.0})
    return pd.DataFrame(rows)


def _brouwer_fig(pO2_levels: list[float], tmp_path) -> plt.Figure:
    return plot_brouwer(df_all=_brouwer_df(pO2_levels), save_dir=tmp_path,
                        sample_name="synthetic", peak_id=1)


def test_data_inside_the_window_keeps_the_frame_drawn_before(tmp_path) -> None:
    fig = _brouwer_fig([1e-4, 1e-2, 0.2, 1.0], tmp_path)
    ax = fig.axes[0]
    assert ax.get_xlim() == HISTORICAL_FRAME
    assert list(ax.get_xticks()) == [-5, -2.5, 0]


def test_a_reducing_branch_is_never_left_outside_the_frame(tmp_path) -> None:
    fig = _brouwer_fig([1e-12, 1e-9, 1e-6, 1e-2, 1.0], tmp_path)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    drawn = [x for line in ax.lines if line.get_linestyle() == "None"
             for x in line.get_xdata()]
    assert drawn, "no markers drawn"
    assert all(lo <= x <= hi for x in drawn), (
        f"frame {lo:.2f}..{hi:.2f} cuts off {[x for x in drawn if not lo <= x <= hi]}")


def test_the_slope_guides_stay_inside_a_widened_frame(tmp_path) -> None:
    """The guides sit at fixed positions; widening must not push them out."""
    fig = _brouwer_fig([1e-12, 1e-9, 1e-6, 1e-2, 1.0], tmp_path)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    guides = [x for line in ax.lines if line.get_linestyle() == "--"
              for x in line.get_xdata()]
    assert guides, "no slope guides drawn"
    assert all(lo <= x <= hi for x in guides)
