"""Stage 4 channel selection: the operator's choice must reach the NNLS design matrix.

Every input here is synthetic and built in the test itself: the conductivities
are generated from the Patterson model, so each assertion is about arithmetic
the fit must reproduce exactly, never about a measurement.

The selection is a defect-chemistry decision. A channel the measured pressure
window cannot populate still takes conductivity away from the ionic one when it
is left in the design matrix, which is the defect these tests pin down.
"""

import numpy as np
import pandas as pd
import pytest

from pipeline.plots import fit_transference

EXPONENT = 0.25


def _synthetic(sigma_ion: float, sigma_p: float, sigma_n: float) -> pd.DataFrame:
    """One synthetic isotherm generated from the Patterson model.

    Conductivities are given in S/cm and handed over in the S/m column the
    function reads, so a returned sigma_* is comparable with the input directly.

    >>> float(_synthetic(1.0, 0.0, 0.0)["sigma_Sm_i"].iloc[0])
    100.0
    """
    p = np.array([1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    sigma_Scm = sigma_ion + sigma_p * p ** EXPONENT + sigma_n * p ** (-EXPONENT)
    return pd.DataFrame({"peak_id": 1, "T_nominal": 500,
                         "pO2_mean": p, "sigma_Sm_i": sigma_Scm * 100.0})


def test_the_default_solves_all_three_channels():
    """No argument must keep the behaviour every existing result was produced with."""
    out = fit_transference(_synthetic(1.0, 2.0, 0.5))
    assert out["sigma_ion"].iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert out["sigma_p"].iloc[0] == pytest.approx(2.0, rel=1e-6)
    assert out["sigma_n"].iloc[0] == pytest.approx(0.5, rel=1e-6)


def test_an_excluded_channel_is_reported_as_zero():
    """Dropping n from an isotherm without n leaves the other two exact."""
    out = fit_transference(_synthetic(1.0, 2.0, 0.0), channels=("ion", "p"))
    assert out["sigma_n"].iloc[0] == 0.0
    assert out["sigma_ion"].iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert out["sigma_p"].iloc[0] == pytest.approx(2.0, rel=1e-6)


def test_the_selection_is_read_in_canonical_order():
    """("p", "ion") and ("ion", "p") are the same model, not two."""
    a = fit_transference(_synthetic(1.0, 2.0, 0.0), channels=("p", "ion"))
    b = fit_transference(_synthetic(1.0, 2.0, 0.0), channels=("ion", "p"))
    assert a["sigma_ion"].iloc[0] == pytest.approx(b["sigma_ion"].iloc[0])
    assert a["sigma_p"].iloc[0] == pytest.approx(b["sigma_p"].iloc[0])


def test_an_ion_only_model_returns_the_level_and_a_unit_transference():
    out = fit_transference(_synthetic(1.0, 0.0, 0.0), channels=("ion",))
    assert out["sigma_ion"].iloc[0] == pytest.approx(1.0, rel=1e-6)
    assert out["sigma_p"].iloc[0] == 0.0
    assert out["sigma_n"].iloc[0] == 0.0
    assert out["t_ion"].iloc[0] == pytest.approx(1.0, rel=1e-9)


def test_the_returned_columns_do_not_change():
    """Stage 5 hands its own table to the same figure; the schemas must agree."""
    expected = ["peak_id", "T_nominal", "pO2", "sigma_Scm", "sigma_ion",
                "sigma_p", "sigma_n", "R2", "t_ion", "t_el"]
    assert list(fit_transference(_synthetic(1.0, 2.0, 0.5)).columns) == expected


def test_an_unknown_channel_is_refused():
    with pytest.raises(ValueError, match="unknown channel"):
        fit_transference(_synthetic(1.0, 2.0, 0.5), channels=("ion", "hole"))


def test_an_empty_selection_is_refused():
    """Zero channels is not a reduced model, it is no model."""
    with pytest.raises(ValueError, match="at least one channel"):
        fit_transference(_synthetic(1.0, 2.0, 0.5), channels=())
