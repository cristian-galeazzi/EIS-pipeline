"""Known-answer validation of audit/calibrate_drt.py on the synthetic sample.

EXAMPLE_SAMPLE (tools/generate_example_sample.py) has two Zarc processes: a
mixed ionic + p-type bulk and a pure-ionic grain boundary, plus 0.3% noise.
On this data the DRT also produces a small spurious satellite peak between
the two processes, which breaks the bulk track when peaks are not capped; the
ranking must prefer the cap at the true process count (2) over the uncapped
fit, and the winning combination must recover both tau(T)-slope activation
energies. The expected Ea / tau_600 are derived from the noiseless generator
model (not hard-coded), so they follow the generator if it ever changes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audit import calibrate_drt as cal
from audit import _common as common
from tools import generate_example_sample as gen

CONDITIONS = ["Ar-80_O2-20_600_400_50", "O2-100_600_400_50"]


def _generator_truth(condition):
    """(Ea_tau [eV], tau_600 [s]) per process from the noiseless generator,
    ascending tau_600 = (bulk, grain boundary)."""
    pO2 = dict(gen.CONDITIONS)[condition]
    truth = []
    for proc in gen.PROCESSES:
        pts = [{"T_K": t + 273.15,
                "tau_i": proc["C_eff"] * gen.GEOM / gen.process_sigma(proc, pO2, t)}
               for t in gen.TEMPS_C]
        tau_600 = next(p["tau_i"] for p in pts if p["T_K"] == 873.15)
        truth.append((common.track_activation_energy(pts), tau_600))
    return sorted(truth, key=lambda x: x[1])


@pytest.fixture(scope="module")
def grid_result():
    df = cal.run_grid(
        sample_dir=_ROOT / "EXAMPLE_SAMPLE",
        conditions=CONDITIONS,
        rbf_ders=["2nd order"], lambdas=[1e-4],
        hf_weights=[0.0], caps=[2, None],
        settings=dict(common.DEFAULTS), L_m=1e-3, D_m=1e-2,
        min_track_points=4, workers=1, use_stage2=False)
    return df


def test_ranking_discriminates_spurious_peak(grid_result):
    assert len(grid_result) == 2
    best, worst = grid_result.iloc[0], grid_result.iloc[1]
    # the cap at the true process count must beat the uncapped fit, whose
    # spurious satellite peak breaks the bulk track across temperatures
    assert best["n_cap"] == 2
    assert worst["n_cap"] == "free"
    assert best["score"] > worst["score"]
    # clean Arrhenius-consistent data: near-perfect physics score expected
    assert best["score"] > 0.90
    assert best["r2_tau"] > 0.99
    assert best["coverage"] > 0.95
    assert best["conv_frac"] == 1.0
    assert best["n_long_tracks"] == 2


def test_activation_energy_recovery():
    spectra = cal.load_condition_spectra(_ROOT / "EXAMPLE_SAMPLE",
                                         CONDITIONS[0], use_stage2=False)
    assert [s["T_nominal"] for s in spectra] == [600, 550, 500, 450, 400]

    _, _, _, drt_tasks = cal.drt_job(
        ("2nd order", 1e-4, CONDITIONS[0], spectra, dict(common.DEFAULTS)))
    combo = ("2nd order", 1e-4, 0.0, 2)
    _, _, peak_rows, summary = cal.fit_job(
        (combo, CONDITIONS[0], drt_tasks, dict(common.DEFAULTS), 1e-3, 1e-2))

    tracks = [t for t in common.build_tracks(peak_rows) if len(t) >= 4]
    assert len(tracks) == 2
    # ascending tau at 600 C = (bulk, grain boundary) order of the generator
    tracks.sort(key=lambda t: t[0]["tau_i"])
    for pts, (ea_true, tau_true) in zip(tracks, _generator_truth(CONDITIONS[0])):
        ea = common.track_activation_energy(pts)
        assert ea == pytest.approx(ea_true, abs=0.05)
        tau_600 = next(p["tau_i"] for p in pts if p["T_nominal"] == 600)
        assert tau_600 == pytest.approx(tau_true, rel=0.25)
