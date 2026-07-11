"""Known-answer validation of audit/calibrate_fit.py on the synthetic sample.

EXAMPLE_SAMPLE carries two Zarc processes with alpha = 0.92 and 0.88 and
uniform 0.3% noise. With the standard alpha window (0.5, 1.0) both exponents
must settle inside the bounds (alpha_pinned = 0) and the physics guard must
hold for every knob combination. Squeezing the window to alpha_max = 0.85,
below both true exponents, must drive every fitted alpha onto the bound
(alpha_pinned = 1): this is the stress signal the script exists to expose.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audit import _common as common
from audit import calibrate_fit as cf

CONDITION = "Ar-80_O2-20_600_400_50"
FROZEN = dict(rbf_der="2nd order", lambda_val=1e-4, cap=2)


def _grid(settings: dict):
    return cf.run_fit_grid(
        sample_dir=_ROOT / "EXAMPLE_SAMPLE", conditions=[CONDITION],
        hf_weights=[0.0, 1.0], r_decs=[0.7], tau_decs=[0.7],
        settings=settings, L_m=1e-3, D_m=1e-2,
        min_track_points=4, workers=1, **FROZEN)


def test_standard_bounds_no_alpha_stress():
    df = _grid(dict(common.DEFAULTS))
    assert len(df) == 2
    # true alphas 0.92/0.88 sit inside (0.5, 1.0): no exponent may pin
    assert (df["alpha_pinned"] == 0.0).all()
    assert (df["physics"] > 0.90).all()
    assert (df["hf_res"] < 0.05).all()
    best = cf.best_with_physics_guard(df)
    assert best["physics"] >= df["physics"].max() - 0.01


def test_tight_alpha_window_pins_every_exponent():
    settings = dict(common.DEFAULTS)
    settings["alpha_max"] = 0.85   # below both true exponents on purpose
    df = _grid(settings)
    assert (df["alpha_pinned"] == 1.0).all()
    # the stress must also cost measurable HF fidelity vs the true model
    relaxed = _grid(dict(common.DEFAULTS))
    assert df["hf_res"].min() > relaxed["hf_res"].min()


def test_hf_metrics_empty_input():
    out = cf.hf_metrics([], {})
    assert all(v != v for v in out.values())   # all NaN


def test_alpha_pinned_fraction():
    rows = [{"alpha_i": 0.5}, {"alpha_i": 0.999}, {"alpha_i": 0.75},
            {"alpha_i": 1.0}]
    assert cf.alpha_pinned_fraction(rows, 0.5, 1.0) == pytest.approx(0.75)
