# Methodological audit scripts

This directory documents how the methodological decisions behind the pipeline
defaults were made, and lets any user repeat those decisions on their own data.

The pact is simple:

- Every script here is a **procedure**, not a result. It encodes the method
  (which grid to sweep, which criterion to rank by, how to interpret the
  output) and takes the sample to analyse from the command line.
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

## Running

All scripts run from the repository root with the project virtualenv:

```bash
.venv/bin/python audit/<script>.py --help
```
