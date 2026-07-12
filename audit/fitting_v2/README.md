# Zarc engine migration: validation record

This folder is the complete, reproducible record of why the Zarc fitting
engine was replaced and of the evidence that the replacement does not
change the science. It is written for a reviewer who wants to check the
claim, not take it on faith.

## Why the engine changed

The original engine (v1) minimized the weighted least-squares objective in
the linear parameters (R, tau, alpha) through `impedance.py`'s
`CustomCircuit.fit`, with a finite-difference Jacobian. R and tau span
decades, so the least-squares valley in linear space is a long,
ill-conditioned trench: convergence is slow and, in the overlapping-tau
regime, occasionally lands in poor local minima.

The v2 engine keeps the **same model, objective, weighting, constraint
windows, warm-start chain, restart policy and output schema**, and changes
only the optimizer path: log-space parametrization (ln R, ln tau, alpha),
an analytic Jacobian, and a direct `scipy.optimize.least_squares` (TRF)
call. The mathematics is stated in `docs/MATHEMATICS.md` (section 3); the
full design rationale and the pre-registered acceptance gates are in
[`design.md`](design.md).

## Pre-registered gates

The five gates were fixed in the design document **before any comparison
number was produced**:

| gate | statement | where tested |
|---|---|---|
| G1 | fit quality (rmse) never worse than v1 | `ab_harness.py` (user data) |
| G2 | parameters within v1's 68% CI on clean fits | `ab_harness.py` (user data) |
| G3 | synthetic ground-truth recovery not worse at any noise level | `synthetic_gate.py` (this repo) |
| G4 | number of bound-pinned fits does not increase | `ab_harness.py` (user data) |
| G5 | no new dependencies | by construction (numpy + scipy only) |

## G3: the synthetic ground-truth gate (fully public)

Synthetic data is the only place where accuracy is measurable absolutely:
spectra are generated from known parameters, so recovery error is a fact.
The gate covers 1 to 4 Zarc elements with decade-spread parameters, two
overlapping-tau stress cases (0.8 and 0.5 decades of separation), three
multiplicative noise levels (0.1%, 0.5%, 2%), five seeded replicates,
DRT-like seed displacement (up to +-0.3 decades), and the standard
production knobs a user actually runs (R_dec = tau_dec = 0.7,
alpha in [0.5, 1], five restarts). Both engines see identical spectra and
identical seeds.

Committed outcome (also reproducible in seconds, see below):
[`results/synthetic_gate.csv`](results/synthetic_gate.csv), one row per
(case, noise, replicate, engine).

* Median recovery error per noise level: statistically indistinguishable
  (differences of 1e-6 to 1e-5 on medians of 1e-3 to 2e-2, i.e. well
  inside a 1% equivalence band).
* On the objective both engines minimize, v2 is never worse (0 of 90
  paired fits).
* v2 recovers three catastrophic v1 local minima in the overlapping-tau
  stress cases (recovery-error gains of 0.65 to 0.73).
* Convergence 180/180 for both engines; total fit wall time ~15x smaller
  for v2.

The literal G3 criterion ("median v2 <= v1 at every noise level") fails by
these epsilon margins at two noise levels; the recorded, operator-approved
reading (design.md, section 4b, dated before the migration decision) is
paired non-inferiority within a 1% band, which passes. The script prints
both verdicts so nothing is hidden.

## G1/G2/G4: the real-data A/B harness

`ab_harness.py` refits every production spectrum of a sample with both
engines under exactly the saved production inputs (stage-2 frequency cuts,
stage-3 DRT seeds, session-stored windows including per-peak windows, same
warm-start chain and deterministic restart seeds), and writes one CSV per
sample with rmse, convergence, parameter shifts in CI units, and the
bound-distance diagnostics.

Per this project's privacy policy (see the repository README), numbers
derived from measured data never enter the repository: the harness writes
to the gitignored `audit/output/fitting_v2/` and the operator's verdicts
are recorded qualitatively in `design.md`. Any user can produce the same
record for their own data with one command.

## Reproduce everything

```bash
# synthetic ground-truth gate (~10 s, deterministic, no data needed)
.venv/bin/python audit/fitting_v2/synthetic_gate.py

# real-data A/B on your own sample(s)
.venv/bin/python audit/fitting_v2/ab_harness.py --samples MY_SAMPLE
```

Engine unit tests (analytic Jacobian vs central finite differences,
schema equality, robust-loss plumbing): `pytest tests/test_zarc_v2.py
tests/test_fitting_v2_gate.py tests/test_fitting_v2_ab.py`.
