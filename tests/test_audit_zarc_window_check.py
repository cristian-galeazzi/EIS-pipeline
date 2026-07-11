"""Known-answer validation of audit/zarc_window_check.py.

The boundary geometry is exact, so the expected answers are constructed
analytically: a fit placed exactly on a bound must flag as pinned with
edge fraction 0, a fit at the seed sits dead centre with edge fraction 1,
a fit placed at a chosen log-distance from the bound must report exactly
that fraction, and degenerate seeds must return NaN and never flag. The
xlsx round trip is validated on a fabricated sample folder.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audit import zarc_window_check as zwc


def test_edge_distance_geometry():
    lo, hi = zwc.bounds_from_seed(1e-4, 0.7)
    assert lo == pytest.approx(1e-4 / 10**0.7)
    assert hi == pytest.approx(1e-4 * 10**0.7)
    # dead centre = seed itself
    assert zwc.edge_distance(1e-4, lo, hi) == pytest.approx(1.0)
    # exactly on a bound
    assert zwc.edge_distance(lo, lo, hi) == pytest.approx(0.0)
    assert zwc.edge_distance(hi, lo, hi) == pytest.approx(0.0)
    # placed 10% of the half-width from the upper bound
    value = 10 ** (math.log10(hi) - 0.1 * 0.7)
    assert zwc.edge_distance(value, lo, hi) == pytest.approx(0.1)
    # degenerate cases: NaN, never a flag
    assert math.isnan(zwc.edge_distance(1.0, 0.0, 10.0))
    assert math.isnan(zwc.edge_distance(-1.0, 0.1, 10.0))


def _tables():
    drt = pd.DataFrame([
        {"T_nominal": 600, "peak_id": 1, "tau": 1e-5, "R_approx": 1e3},
        {"T_nominal": 600, "peak_id": 2, "tau": 1e-3, "R_approx": 1e4},
        {"T_nominal": 550, "peak_id": 1, "tau": 0.0, "R_approx": 1e3},
    ])
    fit = pd.DataFrame([
        # tau on its upper bound, R dead centre
        {"T_nominal": 600, "peak_id": 1, "tau_i": 1e-5 * 10**0.7, "R_i": 1e3},
        # both comfortably inside
        {"T_nominal": 600, "peak_id": 2, "tau_i": 2e-3, "R_i": 5e3},
        # degenerate tau seed: must not flag
        {"T_nominal": 550, "peak_id": 1, "tau_i": 1e-5, "R_i": 1e3},
    ])
    return drt, fit


def test_check_rows_known_answers():
    drt, fit = _tables()
    rows = zwc.check_rows(drt, fit, R_dec=0.7, tau_dec=0.7,
                          condition="c", margin=0.15)
    assert len(rows) == 3
    r1, r2, r3 = rows
    assert r1["pinned_tau"] and not r1["pinned_R"]
    assert r1["tau_edge_frac"] == pytest.approx(0.0)
    assert r1["R_edge_frac"] == pytest.approx(1.0)
    assert not (r2["pinned_tau"] or r2["pinned_R"])
    # log10(2e-3/1e-3) ~ 0.301 from centre -> (0.7-0.301)/0.7 from the edge
    assert r2["tau_edge_frac"] == pytest.approx((0.7 - math.log10(2)) / 0.7,
                                                abs=1e-3)
    assert math.isnan(r3["tau_edge_frac"]) and not r3["pinned_tau"]


def test_check_sample_xlsx_roundtrip(tmp_path):
    drt, fit = _tables()
    cond_dir = tmp_path / "FAKE_SAMPLE" / "Results" / "cond_A"
    cond_dir.mkdir(parents=True)
    drt.to_excel(cond_dir / "stage3_drt.xlsx", sheet_name="Peaks", index=False)
    fit.to_excel(cond_dir / "stage3_fit.xlsx", sheet_name="Peaks", index=False)
    session = tmp_path / "session.json"
    session.write_text(json.dumps([{
        "sample_id": "FAKE_SAMPLE",
        "stage3_params": {"ZARC_R_DEC": 0.7, "ZARC_TAU_DEC": 0.7},
    }]))

    df = zwc.check_sample(tmp_path / "FAKE_SAMPLE", margin=0.15,
                          session_path=session)
    assert df is not None and len(df) == 3
    assert int(df["pinned_tau"].sum()) == 1
    assert int(df["pinned_R"].sum()) == 0
    # missing sample folder gives None, not an exception
    assert zwc.check_sample(tmp_path / "NOPE", 0.15, session) is None
