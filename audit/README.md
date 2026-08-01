# Methodological audit scripts

This directory documents how the methodological decisions behind the pipeline
defaults were made, and lets any user repeat those decisions on their own data.

The pact is simple:

- Every script here is a **procedure**, not a result. It encodes the method
  (which grid to sweep, which criterion to rank by, how to interpret the
  output) and takes the sample to analyze from the command line.
- Scripts read your local data and write to `audit/output/`, which is
  gitignored. No measurement, fitted parameter, or ranking derived from real
  data ever enters this repository.
- The default values shipped in the notebooks (regularization strength, peak
  cap, KK settings) were calibrated with these procedures on the authors'
  datasets. They are documented starting points, not universal constants:
  different materials, geometries, or frequency windows can prefer different
  values. Rerun the relevant script on your own spectra before trusting a
  default.
- Every script is validated against a synthetic case with a known answer; the
  synthetic case ships with the script as both usage example and test.

## Scripts

- `calibrate_drt.py`: sweeps the stage-3 DRT grid (RBF derivative,
  regularization lambda, HF weight, peak cap) and ranks each combination by
  Arrhenius consistency of the tracked peaks instead of raw residual, so that
  overfitting noise into extra peaks is penalized rather than rewarded.
  Validated in `tests/test_audit_calibrate_drt.py`: on the bundled synthetic
  sample the ranking must prefer the peak cap equal to the true process count
  and recover both known activation energies within 0.05 eV.
- `calibrate_fit.py`: with the DRT frozen at the set chosen above, sweeps the
  Zarc-fit knobs (HF weight, R/tau seed windows) and ranks by high-frequency
  fidelity under a physics guard, reporting the fraction of alpha exponents
  pinned at their bounds as an over-constraint stress signal. Validated in
  `tests/test_audit_calibrate_fit.py`: on the synthetic sample the standard
  alpha window must leave every exponent unpinned, and a window squeezed
  below the true exponents must pin them all and cost HF fidelity.
- `kk_mode_comparison.py`: runs the Lin-KK validity test with the RelaxIS
  Percentage modes and the Schoenleber automatic mu = 0.5 mode on every
  spectrum, tabulating KK score, M, mu and the adaptive frequency cuts so the
  M-selection mode can be chosen per instrument and material. Validated in
  `tests/test_audit_kk_mode_comparison.py`: on the causal synthetic spectra
  every mode must retain at least 80% of the frequency window and honor the
  M = round(c * N) and mu >= 0.5 contracts.
- `zarc_window_check.py`: recomputes the R/tau fit windows from each DRT seed
  and flags fitted parameters sitting on a window boundary, separating
  bound-limited fits (widen the window) from seed-limited ones (fix peak
  detection). Validated in `tests/test_audit_zarc_window_check.py` against
  analytically constructed boundary geometries and a fabricated xlsx round
  trip.
- `fitting_v2/`: v2 Zarc engine migration record (design, validation gates, synthetic results; see its README)
  (log-space parametrization, analytic Jacobian, robust loss) with the
  acceptance gates G1-G5 fixed before any result was produced. The prototype
  lives on the `fitting-v2-prototype` branch until the gates decide its fate.
- `fitting_v2/synthetic_gate.py` (branch only): ground-truth recovery gate,
  gate G3. Both engines fit the same known-parameter spectra with shared
  seeds; recovery error is measured against the truth. Smoke-tested in
  `tests/test_fitting_v2_gate.py`.
- `fitting_v2/ab_harness.py` (branch only): paired refit of the production
  dataset with both engines under the exact saved inputs (stage-2 selected
  spectra, stage-3 DRT seeds, session knobs, warm-start chain), reporting
  the G1/G2/G4 verdicts; stdout carries only paths and counts. Integration
  test on a fabricated sample in `tests/test_fitting_v2_ab.py`.

## Running

All scripts run from the repository root with the project virtualenv:

```bash
.venv/bin/python audit/<script>.py --help
```
