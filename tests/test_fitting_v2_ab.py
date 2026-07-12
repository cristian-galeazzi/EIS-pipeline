"""Integration test of the A/B harness on a fabricated sample folder.

Builds a complete miniature production layout in tmp_path (CSV spectra from
the synthetic generator parameters, a stage-2 "Selected" sheet, a stage-3
"Peaks" seed sheet and a session entry), then runs the harness end to end
and checks the pairing guarantees: rows for every (condition, T), both
engines converged, v2 never worse on the fit objective, and the gate table
well formed.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audit.fitting_v2 import ab_harness as ab
from pipeline.zarc_v2 import zarc_model

TEMPS = (600, 500)
TRUE = {600: [(8e3, 2e-6, 0.92), (2.5e4, 3e-4, 0.88)],
        500: [(3.5e4, 9e-6, 0.92), (1.6e5, 2e-3, 0.88)]}
FREQ = np.logspace(6, -1, 40)


def _make_sample(tmp_path: Path) -> tuple[Path, Path]:
    sample = tmp_path / "AB_SAMPLE"
    cond = "gasmix_600_500_100"
    spec_dir = sample / "input_spectra" / cond
    res_dir = sample / "Results" / cond
    spec_dir.mkdir(parents=True)
    res_dir.mkdir(parents=True)

    rng = np.random.default_rng(11)
    sel_rows, seed_rows = [], []
    for T in TEMPS:
        params = np.array([v for p in TRUE[T] for v in p])
        Z = zarc_model(FREQ, params, include_r0=False)
        Z = Z * (1 + rng.uniform(-0.003, 0.003, len(Z)))
        fname = f"demo_{T}C.csv"
        pd.DataFrame({"freq": FREQ, "Z_re": Z.real,
                      "Z_im": -Z.imag}).to_csv(spec_dir / fname, index=False)
        sel_rows.append({"file": fname, "T_nominal": T,
                         "f_min_cut": np.nan, "f_max_cut": np.nan,
                         "pO2_mean": 0.2})
        for pid, (R, tau, _a) in enumerate(TRUE[T], start=1):
            # DRT-like seeds, displaced from the truth
            seed_rows.append({"T_nominal": T, "peak_id": pid,
                              "tau": tau * 1.6, "R_approx": R * 0.6})
    pd.DataFrame(sel_rows).to_excel(res_dir / "stage2_kk.xlsx",
                                    sheet_name="Selected", index=False)
    pd.DataFrame(seed_rows).to_excel(res_dir / "stage3_drt.xlsx",
                                     sheet_name="Peaks", index=False)

    session = tmp_path / "session.json"
    session.write_text(json.dumps([{
        "sample_id": "AB_SAMPLE",
        "stage3_params": {"ZARC_R_DEC": 0.7, "ZARC_TAU_DEC": 0.7,
                          "ZARC_ALPHA_INIT": 0.7, "ZARC_HF_WEIGHT": 0.0,
                          "ZARC_INCLUDE_R0": False, "ZARC_R0_MAX": 200,
                          "ZARC_N_RESTARTS": 3, "ZARC_RMSE_TOL": 0.02},
        "condition_params": {cond: {"R_dec": 0.7, "tau_dec": 0.7}},
    }]))
    return sample, session


def test_harness_end_to_end(tmp_path):
    sample, session = _make_sample(tmp_path)
    rows = ab.run_sample(sample, session)
    assert len(rows) == len(TEMPS)
    assert [r["T_nominal"] for r in rows] == [600, 500]   # T descending
    for r in rows:
        assert set(r) == set(ab.CSV_FIELDS)
        assert r["conv_v1"] and r["conv_v2"]
        assert r["n_peaks"] == 2
        # both engines minimize the same objective from the same inputs
        assert r["rmse_v2"] <= r["rmse_v1"] + 1e-9
        assert np.isfinite(r["edge_frac_v1"])
        assert np.isfinite(r["edge_frac_v2"])
    table = ab.gate_table(rows)
    assert set(table) == {"G1", "G2", "G4"}
    assert table["G1"].startswith("PASS")


def test_resolve_zarc_params_precedence():
    entry = {"stage3_params": {"ZARC_R_DEC": 0.7, "ZARC_TAU_DEC": 0.6,
                               "ZARC_ALPHA_INIT": 0.8, "ZARC_HF_WEIGHT": 0.0},
             "condition_params": {"c1": {"R_dec": 0.5,
                                         "600": {"R_dec": 0.3}}}}
    assert ab.resolve_zarc_params(entry, "c1", 600)["R_dec"] == 0.3
    assert ab.resolve_zarc_params(entry, "c1", 500)["R_dec"] == 0.5
    assert ab.resolve_zarc_params(entry, "c2", 600)["R_dec"] == 0.7
    assert ab.resolve_zarc_params(entry, "c2", 600)["tau_dec"] == 0.6
