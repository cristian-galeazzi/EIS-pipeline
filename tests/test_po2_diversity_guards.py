"""Guard tests for the p(O2) diversity refusals.

The stage-5 model separates its channels by their pressure exponent, so one
pressure level cannot resolve them: the fit still converges and reports a
perfect R2 for a decomposition the data does not contain. The Brouwer diagram
degenerates the same way, into one point per temperature.

These tests pin the refusals. They do not touch any computed number: a fit that
was possible before is still possible, with the same result.
"""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from pipeline.model import MIN_PO2_LEVELS, fit_global_conductivity  # noqa: E402
from pipeline.plots import plot_brouwer                             # noqa: E402


def _surface(pO2: np.ndarray) -> pd.DataFrame:
    T = np.arange(400, 625, 25)
    sigma = 1e2 * np.exp(-1.0 * 11604.5 / (T + 273.15))
    return pd.DataFrame({"T_nominal": T, "pO2_mean": pO2, "sigma_Sm_i": sigma})


def test_a_single_pressure_level_is_refused() -> None:
    with pytest.raises(ValueError, match="distinct p"):
        fit_global_conductivity(_surface(np.full(9, 0.21)), channels=("ion", "p"))


def test_two_pressure_levels_still_fit() -> None:
    out = fit_global_conductivity(_surface(np.array([0.21, 1e-3] * 4 + [0.21])),
                                  channels=("ion", "p"))
    assert out["n_points"] == 9


def test_the_threshold_is_two_levels() -> None:
    assert MIN_PO2_LEVELS == 2


def test_brouwer_refuses_a_single_pressure(tmp_path) -> None:
    df = pd.DataFrame({
        "peak_id": 1,
        "T_nominal": np.arange(400, 625, 25),
        "pO2_mean": np.full(9, 0.21),
        "sigma_Sm_i": np.linspace(1.0, 2.0, 9),
    })
    with pytest.raises(ValueError, match="distinct p"):
        plot_brouwer(df_all=df, save_dir=tmp_path, sample_name="S", peak_id=1)


def test_brouwer_still_draws_with_two_pressures(tmp_path) -> None:
    df = pd.DataFrame({
        "peak_id": 1,
        "T_nominal": [400, 400, 425, 425],
        "pO2_mean": [0.21, 1e-3, 0.21, 1e-3],
        "sigma_Sm_i": [1.0, 0.5, 1.2, 0.6],
    })
    assert plot_brouwer(df_all=df, save_dir=tmp_path, sample_name="S",
                        peak_id=1) is not None
