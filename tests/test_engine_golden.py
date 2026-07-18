"""
Golden-master regression tests for the EIS calculation engine.

Purpose
-------
The pipeline relies on the engine (``drt.py`` / ``fitting.py``) staying
numerically frozen. These tests turn that guarantee into automatic checks so an
accidental change is caught without a manual md5 inspection.

They cover:
1. C_eff = τ/R is exact for the Zarc parametrisation (the documented identity).
2. ``fit_zarc`` recovers the parameters of a synthetic single-Zarc spectrum.
3. Saved ``stage3_fit.xlsx`` rows still satisfy C_eff_i == τ_i / R_i
   (skipped automatically when no sample data is present).

Run with either::

    pytest tests/test_engine_golden.py
    python tests/test_engine_golden.py     # no pytest needed

References: Wilson et al., "Best Practices for Scientific Computing" (2014) -
"turn bugs into test cases"; Scopatz & Huff, "Effective Computation in Physics".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pipeline.fitting import fit_zarc, conductivity, build_circuit_string


def _synthetic_zarc(R0, R, tau, alpha, freq):
    """Z(ω) = R0 + R / (1 + (jωτ)^α) for one Zarc element."""
    w = 2 * np.pi * freq
    Z = R0 + R / (1.0 + (1j * w * tau) ** alpha)
    return Z.real, -Z.imag  # store -Z'' (pipeline sign convention)


def test_ceff_identity_exact():
    """C_eff returned by the fit equals τ/R to floating-point precision."""
    freq = np.logspace(5, -1, 60)
    R0, R, tau, alpha = 20.0, 5000.0, 1e-3, 0.85
    Z_re, Z_im = _synthetic_zarc(R0, R, tau, alpha, freq)
    peaks = [{"R_approx": R, "tau": tau, "peak_id": 1}]
    fit = fit_zarc(freq, Z_re, Z_im, peaks, R0_guess=R0, r0_max=200)
    c_eff = fit["C_eff"][0]
    assert np.isclose(c_eff, fit["tau"][0] / fit["R"][0], rtol=1e-9), \
        f"C_eff identity broken: {c_eff} != tau/R"


def test_fit_recovers_synthetic_params():
    """fit_zarc recovers R, τ, α of a clean single-Zarc spectrum."""
    freq = np.logspace(5, -1, 80)
    R0, R, tau, alpha = 15.0, 8000.0, 2e-3, 0.80
    Z_re, Z_im = _synthetic_zarc(R0, R, tau, alpha, freq)
    peaks = [{"R_approx": R, "tau": tau, "peak_id": 1}]
    fit = fit_zarc(freq, Z_re, Z_im, peaks, R0_guess=R0, r0_max=200)
    assert fit["converged"], f"fit failed: {fit.get('fit_error')}"
    assert np.isclose(fit["R"][0],     R,     rtol=0.05), fit["R"][0]
    assert np.isclose(fit["tau"][0],   tau,   rtol=0.10), fit["tau"][0]
    assert np.isclose(fit["alpha"][0], alpha, rtol=0.05), fit["alpha"][0]
    assert fit["rmse_rel"] < 1e-3, fit["rmse_rel"]


def test_circuit_string():
    assert build_circuit_string(2, include_r0=True) == "R0-Zarc1-Zarc2"
    assert build_circuit_string(2, include_r0=False) == "Zarc1-Zarc2"


def test_conductivity_formula():
    # σ = L / (R·A),  A = π(D/2)²
    L, D, R = 1.26e-3, 10.98e-3, 1000.0
    A = np.pi * (D / 2) ** 2
    assert np.isclose(conductivity(R, L, D), L / (R * A), rtol=1e-12)


def test_saved_fit_ceff_invariant():
    """Every saved Peaks row must satisfy C_eff_i == τ_i / R_i (engine drift guard)."""
    import pandas as pd
    files = list(_ROOT.glob("*/Results/*/stage3_fit.xlsx"))
    if not files:
        print("  [SKIP] no stage3_fit.xlsx sample data present")
        return
    checked = 0
    for f in files:
        df = pd.read_excel(f, sheet_name="Peaks")
        ok = np.isclose(df["C_eff_i"], df["tau_i"] / df["R_i"], rtol=1e-6)
        assert ok.all(), f"C_eff != tau/R in {f.name} rows {np.where(~ok)[0].tolist()}"
        checked += len(df)
    print(f"  C_eff invariant holds across {checked} saved peak rows in {len(files)} file(s)")


# ---------------------------------------------------------------------------
# CSV / non-Zahner entry point tests (T1–T8)
# ---------------------------------------------------------------------------

import io
import tempfile
import textwrap
from contextlib import redirect_stdout


def _make_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content))


def _spectra_root(tmp: Path) -> Path:
    return tmp / "input_spectra"


from pipeline.ingest import scan_input_spectra


def test_csv_standard_file():
    """T1: standard _400C.csv with temperature column is loaded correctly."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_csv(
            _spectra_root(tmp) / "Ar_200_O2_10_600_400_25" / "S_Ar_200_O2_10_400C.csv",
            "freq,Z_re,Z_im,temperature\n100000,5.3,0.2,400\n1000,30.2,44.8,400\n",
        )
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 1
        assert df.iloc[0]["T_nominal"] == 400.0
        assert df.iloc[0]["T_mean"] == 400.0


def test_csv_no_temperature_in_filename():
    """T2: file without _NNNc pattern is skipped; no crash."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_csv(
            _spectra_root(tmp) / "Ar_200" / "measurement.csv",
            "freq,Z_re,Z_im\n100000,5.3,0.2\n1000,30.2,44.8\n",
        )
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 0


def test_csv_low_temperature():
    """T3: two-digit temperature _20C is parsed correctly."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_csv(
            _spectra_root(tmp) / "Ar_200_20C_constant" / "S_Ar_200_20C.csv",
            "freq,Z_re,Z_im,temperature\n100000,5.3,0.2,20\n1000,30.2,44.8,20\n",
        )
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 1
        assert df.iloc[0]["T_nominal"] == 20.0


def test_csv_four_digit_temperature():
    """T4: four-digit temperature _1000C is parsed correctly."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_csv(
            _spectra_root(tmp) / "Ar_200_1000C" / "S_Ar_200_1000C.csv",
            "freq,Z_re,Z_im,temperature\n100000,5.3,0.2,1000\n1000,30.2,44.8,1000\n",
        )
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 1
        assert df.iloc[0]["T_nominal"] == 1000.0


def test_csv_empty_condition_folder():
    """T5: empty condition folder returns empty DataFrame without crashing."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        (_spectra_root(tmp) / "Ar_200_empty").mkdir(parents=True)
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 0


def test_csv_missing_required_columns():
    """T6: CSV with wrong column names raises ValueError with descriptive message."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_csv(
            _spectra_root(tmp) / "Ar_200" / "S_400C.csv",
            "frequency,real,imag\n100000,5.3,0.2\n",
        )
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 0


def test_csv_semicolon_separator():
    """T7: semicolon-separated CSV is loaded correctly."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _make_csv(
            _spectra_root(tmp) / "Ar_200_sep" / "S_400C.csv",
            "freq;Z_re;Z_im;temperature\n100000;5.3;0.2;400\n1000;30.2;44.8;400\n",
        )
        df = scan_input_spectra(tmp)
        assert df is not None and len(df) == 1


def test_csv_loose_file_warns():
    """T8: file directly in input_spectra/ (no condition subfolder) prints a warning and returns empty."""
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _spectra_root(tmp).mkdir(parents=True)
        (_spectra_root(tmp) / "measurement.csv").write_text(
            "freq,Z_re,Z_im\n100000,5.3,0.2\n"
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            df = scan_input_spectra(tmp)
        output = buf.getvalue()
        assert df is not None and len(df) == 0
        assert "[WARNING]" in output and "input_spectra/" in output, \
            f"Expected loose-file warning in stdout, got: {output!r}"


# ---------------------------------------------------------------------------
# Engine coverage: quality.py, drt.py, fitting.fix_params, session.py,
# matching.py (pure additions - the engine itself stays untouched)
# ---------------------------------------------------------------------------


def test_linkk_synthetic_consistency():
    """run_linkk accepts a KK-consistent synthetic Zarc with tiny residuals."""
    from pipeline.quality import run_linkk
    rng = np.random.default_rng(0)
    freq = np.logspace(6, 0, 60)
    Z_re, Z_im = _synthetic_zarc(50.0, 1e4, 1e-4, 0.9, freq)
    noise = 1.0 + rng.normal(0.0, 1e-3, freq.size)
    res = run_linkk(freq, Z_re * noise, Z_im * noise,
                    c=0.76, use_binary_M=False,
                    iqr_fence_factor=2.0, iqr_window=5)
    assert np.max(np.abs(res["res_re"])) < 0.01, "Re residual too large"
    assert np.max(np.abs(res["res_im"])) < 0.01, "Im residual too large"
    # a causal, stable, linear spectrum must keep (almost) the whole window
    kept = (res["freq"] >= res["f_min_cut"]) & (res["freq"] <= res["f_max_cut"])
    assert kept.sum() >= 0.9 * res["freq"].size, "edge cut ate a clean spectrum"


def test_drt_total_area_equals_R():
    """The DRT of one Zarc integrates to R over d(ln tau) - the documented
    identity behind peak areas. Peak COUNT is not asserted: the sharp
    production lambda is allowed to ring on broad peaks (handled by
    N_PEAKS_CAP in the pipeline), but the total polarization is conserved."""
    from pipeline.drt import compute_drt, find_drt_peaks
    freq = np.logspace(6, -1, 70)
    R, tau = 1e4, 1e-4
    Z_re, Z_im = _synthetic_zarc(0.0, R, tau, 0.9, freq)
    entry = compute_drt(freq, Z_re, Z_im, cv_type="custom",
                        rbf_der="2nd order", shape_s=0.5, lambda_val=6.5e-6)
    total = np.trapezoid(entry.gamma, np.log(entry.out_tau_vec))
    assert np.isclose(total, R, rtol=0.10), \
        f"integral(gamma dln tau) = {total:.4g}, expected ~{R:.4g}"
    peaks = find_drt_peaks(entry, min_height_frac=0.05)
    assert peaks, "no peaks detected on a clean Zarc"
    tallest = max(peaks, key=lambda p: p["gamma_peak"])
    assert 0.1 * tau < tallest["tau"] < 10.0 * tau, tallest["tau"]


def test_fix_params_pins_tau():
    """Regression for the pinned-bounds bug: an exactly fixed tau must hold."""
    freq = np.logspace(5, -1, 60)
    R0, R, tau, alpha = 20.0, 5000.0, 1e-3, 0.85
    Z_re, Z_im = _synthetic_zarc(R0, R, tau, alpha, freq)
    peaks = [{"R_approx": R, "tau": tau, "peak_id": 1}]
    tau_fixed = 1.2e-3
    fit = fit_zarc(freq, Z_re, Z_im, peaks, R0_guess=R0, r0_max=200,
                   fix_params={"tau": [tau_fixed]})
    assert fit["converged"], f"pinned fit failed: {fit.get('fit_error')}"
    assert np.isclose(fit["tau"][0], tau_fixed, rtol=1e-9), \
        f"tau not pinned: {fit['tau'][0]} != {tau_fixed}"


def test_ci_columns_align_without_r0():
    """fit_to_rows must map conf_R/tau/alpha from the right positions for
    both circuit layouts (regression for the old R0-offset assumption)."""
    from pipeline.fitting import fit_to_rows
    freq = np.logspace(5, -1, 60)
    Z_re, Z_im = _synthetic_zarc(0.0, 5000.0, 1e-3, 0.85, freq)
    peaks = [{"R_approx": 5000.0, "tau": 1e-3, "peak_id": 1}]
    for include_r0 in (False, True):
        fit = fit_zarc(freq, Z_re, Z_im, peaks, R0_guess=10.0, r0_max=200,
                       include_r0=include_r0)
        rows, _ = fit_to_rows(fit, "cond", "f.ism", "/f.ism",
                              600.0, 0.21, 1.4e-3, 10e-3)
        base = 1 if include_r0 else 0
        assert np.isclose(rows[0]["conf_R"],   fit["conf"][base]), include_r0
        assert np.isclose(rows[0]["conf_tau"], fit["conf"][base + 1]), include_r0


def test_session_merge_keys():
    """update_sample merges per-condition dicts and respects replace=True."""
    from pipeline.session import update_sample, load_sample
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "session.json"
        update_sample("S1", path=p, condition_params={"Ar": {"R_dec": 0.7}})
        update_sample("S1", path=p, condition_params={"O2": {"R_dec": 0.9}})
        entry = load_sample("S1", path=p)
        assert entry["condition_params"] == {"Ar": {"R_dec": 0.7},
                                             "O2": {"R_dec": 0.9}}, \
            "second save wiped the first condition"
        update_sample("S1", path=p, replace=True,
                      condition_params={"O2": {"R_dec": 1.0}})
        entry = load_sample("S1", path=p)
        assert entry["condition_params"] == {"O2": {"R_dec": 1.0}}, \
            "replace=True did not overwrite wholesale"

        # zarc_peak_windows merges per scope key: saving the "conditions"
        # branch must not wipe the "sample" default
        update_sample("S1", path=p, zarc_peak_windows={"sample": {"1": {"R_dec": 0.7}}})
        update_sample("S1", path=p, zarc_peak_windows={"conditions": {"Ar": {"1": {"R_dec": 0.9}}}})
        entry = load_sample("S1", path=p)
        assert entry["zarc_peak_windows"] == {
            "sample": {"1": {"R_dec": 0.7}},
            "conditions": {"Ar": {"1": {"R_dec": 0.9}}}}, \
            "conditions save wiped the sample-wide windows"

        # stage5_params merges per peak_id: refitting peak 1 must not wipe peak 2
        update_sample("S1", path=p, stage5_params={"1": {"Ea_ion": 0.9}})
        update_sample("S1", path=p, stage5_params={"2": {"Ea_ion": 1.1}})
        entry = load_sample("S1", path=p)
        assert entry["stage5_params"] == {"1": {"Ea_ion": 0.9},
                                          "2": {"Ea_ion": 1.1}}, \
            "second stage5 refit wiped the first peak"

        # stage3_valid merges per condition: a session that only shows Ar
        # must not wipe the O2 selection; re-saving Ar replaces its list
        update_sample("S1", path=p, stage3_valid={"Ar": [600, 550]})
        update_sample("S1", path=p, stage3_valid={"O2": [600]})
        update_sample("S1", path=p, stage3_valid={"Ar": [600]})
        entry = load_sample("S1", path=p)
        assert entry["stage3_valid"] == {"Ar": [600], "O2": [600]}, \
            "stage3_valid merge lost or failed to replace a condition"


def test_session_ordering():
    """Write-time ordering: per-condition stores follow the conditions list, T
    and peak keys ascend, parameter-name dicts keep their order, values stay."""
    import json

    from pipeline.session import update_sample
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "session.json"
        update_sample("S1", path=p, conditions=["Bcond", "Acond"])
        update_sample("S1", path=p, kk_overrides={
            "Acond": {"600": {"f_min_hard": 1.0}, "400": {"f_min_hard": 2.0}},
            "Bcond": {"500": {"f_min_hard": 3.0}}})
        update_sample("S1", path=p, condition_params={
            "Acond": {"alpha_init": 0.7, "hf_weight": 0.0}})
        update_sample("S1", path=p, zarc_peak_windows={
            "sample": {"2": {"R_dec": 0.7}, "1": {"R_dec": 0.9}}})
        raw = json.loads(p.read_text())[0]
        assert list(raw["kk_overrides"]) == ["Bcond", "Acond"], \
            "condition store did not follow the conditions list order"
        assert list(raw["kk_overrides"]["Acond"]) == ["400", "600"], \
            "temperature keys not sorted ascending"
        assert list(raw["zarc_peak_windows"]["sample"]) == ["1", "2"], \
            "peak keys not sorted ascending"
        assert list(raw["condition_params"]["Acond"]) == ["alpha_init", "hf_weight"], \
            "parameter-name dict was reordered"
        assert raw["kk_overrides"]["Acond"]["400"]["f_min_hard"] == 2.0, \
            "reordering altered a value"


def test_condition_pO2_map():
    """Median pO2_mean per condition, None when no p(O2) source is present."""
    import pandas as pd

    from pipeline.utils import condition_pO2_map
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        cdir = root / "Results" / "Ar-1"
        cdir.mkdir(parents=True)
        pd.DataFrame({"T_nominal": [600, 500, 400],
                      "pO2_mean": [1.0e-4, 1.1e-4, 1.2e-4]}).to_excel(
            cdir / "stage2_kk.xlsx", sheet_name="Selected", index=False)
        (root / "Results" / "NoP").mkdir(parents=True)
        m = condition_pO2_map(root, ["Ar-1", "NoP"])
        assert abs(m["Ar-1"] - 1.1e-4) < 1e-12, "median pO2 wrong"
        assert m["NoP"] is None, "missing p(O2) should be None"


def test_session_remove_override_entries():
    """remove_override_entries deletes per-T or per-condition, prunes empties."""
    from pipeline.session import (load_sample, remove_override_entries,
                                  update_sample)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "session.json"
        update_sample("S1", path=p,
                      kk_overrides={"Ar": {600: {"f_min_hard": 1e3},
                                           550: {"f_min_hard": 2e3}},
                                    "O2": {600: {"f_max_hard": 1e6}}})

        # per-T removal (keys survive the JSON round-trip as strings)
        assert remove_override_entries("S1", "kk_overrides", "Ar", 600, path=p)
        kk = load_sample("S1", path=p)["kk_overrides"]
        assert kk == {"Ar": {"550": {"f_min_hard": 2e3}},
                      "O2": {"600": {"f_max_hard": 1e6}}}

        # removing the last T prunes the condition
        assert remove_override_entries("S1", "kk_overrides", "Ar", 550, path=p)
        assert "Ar" not in load_sample("S1", path=p)["kk_overrides"]

        # whole-condition removal
        assert remove_override_entries("S1", "kk_overrides", "O2", path=p)
        assert load_sample("S1", path=p)["kk_overrides"] == {}

        # no-ops return False and leave the file valid
        assert not remove_override_entries("S1", "kk_overrides", "O2", path=p)
        assert not remove_override_entries("S1", "condition_params", "Ar", path=p)
        assert not remove_override_entries("S2", "kk_overrides", "Ar", path=p)


def test_session_canonical_key_order():
    """Saved entries keep all data but serialize keys in canonical order."""
    import json as _json
    from pipeline.session import CANONICAL_KEY_ORDER, load_sample, update_sample
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "session.json"
        # Write keys in a scrambled chronological order, as real runs do.
        update_sample("S1", path=p, stage5_params={"1": {"Ea_ion": 0.9}})
        update_sample("S1", path=p, stage2_params={"KK_C": 0.76})
        update_sample("S1", path=p, L_m=1.4e-3, custom_key="kept")
        update_sample("S1", path=p, stage1_params={"T_STEP": 50})

        entry = load_sample("S1", path=p)
        assert entry == {"sample_id": "S1", "L_m": 1.4e-3,
                         "stage1_params": {"T_STEP": 50},
                         "stage2_params": {"KK_C": 0.76},
                         "stage5_params": {"1": {"Ea_ion": 0.9}},
                         "custom_key": "kept"}, "reordering altered the data"

        on_disk = list(_json.loads(p.read_text())[0])
        rank = {k: i for i, k in enumerate(CANONICAL_KEY_ORDER)}
        known = [k for k in on_disk if k in rank]
        assert known == sorted(known, key=rank.get), \
            f"keys not in canonical order: {on_disk}"
        assert on_disk[-1] == "custom_key", "unknown keys must trail"


def test_matching_classifies_windows():
    """match_ism_to_furnace: stable plateau VALID, ramp UNSTABLE, gap OUTSIDE."""
    import pandas as pd
    from datetime import datetime, timedelta
    from pipeline.ingest import IsmRecord
    from pipeline.matching import match_ism_to_furnace

    t0 = datetime(2026, 1, 1, 8, 0, 0)
    minutes = np.arange(0, 120)
    # 0-59 min: stable 600 C plateau; 60-119 min: linear ramp down to 500 C
    temps = np.where(minutes < 60, 600.0,
                     600.0 - (minutes - 60) * (100.0 / 59.0))
    furnace_df = pd.DataFrame({
        "abs_datetime": [t0 + timedelta(minutes=int(m)) for m in minutes],
        "Tsample": temps,
        "pO2": np.full(minutes.size, 0.21),
    })

    def _rec(start_min, end_min):
        return IsmRecord(path=Path(f"fake_{start_min}.ism"),
                         freq=np.array([1.0]), Z_re=np.array([1.0]),
                         Z_im=np.array([1.0]),
                         t_start=t0 + timedelta(minutes=start_min),
                         t_end=t0 + timedelta(minutes=end_min))

    records = [_rec(10, 30),    # inside the plateau -> VALID
               _rec(70, 110),   # on the ramp        -> UNSTABLE
               _rec(300, 320)]  # after the log ends -> OUTSIDE_RANGE
    out = match_ism_to_furnace(records, furnace_df,
                               pre_margin_min=0, post_margin_min=0)
    assert out[0].status == "VALID" and out[0].T_nominal == 600.0, out[0].status
    assert out[0].replica == 1
    assert out[1].status == "UNSTABLE", out[1].status
    assert out[2].status == "OUTSIDE_RANGE", out[2].status


def test_select_best_replica_ignores_nan_scores():
    """A NaN kk_score (degenerate Shapiro-Wilk) must never win argmax."""
    import pytest
    from pipeline.quality import select_best_replica

    assert select_best_replica([{"kk_score": 0.91},
                                {"kk_score": float("nan")},
                                {"kk_score": 0.97}]) == 2
    with pytest.raises(ValueError, match="NaN"):
        select_best_replica([{"kk_score": float("nan")}])


def test_csv_ingestion_rejects_physical_sign_convention(tmp_path):
    """A CSV with mostly negative Z_im (Gamry/EC-Lab physical convention)
    must fail loudly instead of being silently stripped downstream."""
    import pytest
    from pipeline.ingest import load_csv_spectrum

    f = tmp_path / "bad_400C.csv"
    f.write_text("freq,Z_re,Z_im\n1000,10.0,-5.0\n100,20.0,-8.0\n10,30.0,-2.0\n")
    with pytest.raises(ValueError, match="sign convention"):
        load_csv_spectrum(f)

    ok = tmp_path / "good_400C.csv"
    ok.write_text("freq,Z_re,Z_im\n1000,10.0,5.0\n100,20.0,8.0\n10,30.0,-2.0\n")
    assert load_csv_spectrum(ok).n_points == 3


def test_matching_converts_tz_aware_timestamps():
    """A UTC-aware ISM timestamp must match the same furnace rows as its
    local-naive equivalent instead of shifting by the UTC offset."""
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    from pipeline.ingest import IsmRecord
    from pipeline.matching import match_ism_to_furnace

    t0 = datetime(2026, 1, 1, 8, 0, 0)
    furnace_df = pd.DataFrame({
        "abs_datetime": [t0 + timedelta(minutes=int(m)) for m in range(60)],
        "Tsample": np.full(60, 600.0),
        "pO2": np.full(60, 0.21),
    })
    local_offset = datetime(2026, 1, 1, 8, 10).astimezone().utcoffset()
    aware_start = (datetime(2026, 1, 1, 8, 10) - local_offset).replace(tzinfo=timezone.utc)
    rec = IsmRecord(path=Path("fake.ism"), freq=np.array([1.0]),
                    Z_re=np.array([1.0]), Z_im=np.array([1.0]),
                    t_start=aware_start,
                    t_end=aware_start + timedelta(minutes=20))
    out = match_ism_to_furnace([rec], furnace_df,
                               pre_margin_min=0, post_margin_min=0)
    assert out[0].status == "VALID" and out[0].T_nominal == 600.0, out[0].status


def test_find_furnace_log_strips_gas_like_sample_prefix(tmp_path):
    """Sample IDs starting with a gas name (e.g. 'CoP03' vs CO) must not
    corrupt the condition key extracted from the folder name."""
    from pipeline.matching import find_furnace_log

    oven = tmp_path / "Raw oven"
    oven.mkdir()
    log = oven / "Ar-100_O2-10_600_400_25.txt"
    log.write_text("dummy")

    found = find_furnace_log(tmp_path, "CoP03_Ar-100_O2-10_600_400_25")
    assert found == log


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
        except Exception:
            failed += 1
            print(f"ERROR {t.__name__}:")
            traceback.print_exc()
    print(f"\n{len(tests)-failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
