"""Guard tests for pipeline/plots.py, what a figure says and in which unit.

Two sections. The first covers the suptitle, the only text on a figure that
identifies the measurement. The second covers the unit every conductivity is
reported in, which is a reporting boundary rather than an engine change: the
stored column stays S/m and only the read converts.

Neither section touches a computed number. They pin what is printed.
"""

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pipeline.plots import (  # noqa: E402
    KB, _condition_suptitle, build_arrhenius_results, plot_arrhenius_sigma,
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


def test_default_call_is_unchanged_from_today() -> None:
    assert _condition_suptitle(_df()) == r"$p$(O$_2$) = 2.1e-03 bar"


def test_label_precedes_the_pressure() -> None:
    assert _condition_suptitle(_df(), "Ar-100 O2-20 | 400-600C") == (
        r"Ar-100 O2-20 | 400-600C,  $p$(O$_2$) = 2.1e-03 bar")


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
    assert _condition_suptitle(df) == r"$p$(O$_2$) = 0.21 bar"


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
