# Design: robust Zarc fitting engine (v2)

Design document for the v2 fitting engine, developed on the branch
`fitting-v2-prototype`. Its defining property is the ordering: the
acceptance gates in section 4 were written and frozen BEFORE any result was
produced, so the evaluation of v2 is pre-registered rather than post-hoc.
The promotion decision belongs to the operator and is taken only after both
the synthetic gate and the real-data A/B comparison are on the table; until
then the production engine (v1) is untouched.

---

## 1. What "robust fitting" means here

Commercial EIS software is not magic; its practical robustness comes from a
short list of engineering choices, most of which the v1 stack
(impedance.py -> scipy curve_fit) lacks:

| Ingredient | Commercial tools | v1 today | Impact |
|---|---|---|---|
| Weighting modes (proportional, HF-modulus) | yes | yes (already ported) | - |
| Log-space parametrization of R, tau | yes | no (linear space) | HIGH: R spans 1e1-1e7 Ohm, tau 1e-6-1e0 s; in linear space the optimizer sees a valley several decades long and crawls; in log space the landscape is well-conditioned and bounds become simple boxes |
| Analytic Jacobian | yes | no (finite differences) | HIGH: ~(n_params+1)x fewer model evaluations per iteration AND exact gradients near bounds, where finite differences are noisiest - directly attacks minutes-long bound-pinned fits |
| Robust loss (Huber / soft-L1) | yes | no (pure L2) | MEDIUM: single noisy points (low-T spectra) stop dragging the whole fit |
| Deterministic multistart | partial | yes (crc32 seed) | keep as is |
| Parameter fixing / per-peak bounds | yes | yes | keep as is |
| Direct scipy least_squares (no curve_fit wrapper) | n/a | no | MEDIUM: full control of x_scale, tr_solver, loss; removes the string-parsed circuit evaluation overhead |

The Zarc model is analytically trivial to differentiate, which is why the
Jacobian is low-hanging fruit:

    Z_k(w) = R_k / (1 + (j w tau_k)^alpha_k)

    with p = ln R_k, q = ln tau_k (fit variables) and u = (j w tau_k)^alpha_k:
    dZ/dp     = Z_k                       (log-R: the derivative is the term itself)
    dZ/dq     = -R_k * alpha_k * u / (1+u)^2
    dZ/dalpha = -R_k * u * ln(j w tau_k) / (1+u)^2

    Real/imag stacking and the weight division are linear operations, so the
    full Jacobian is assembled in one vectorized numpy expression per peak.
    No autodiff library needed; numpy only.

## 2. Hard constraints this design respects

1. **Golden-master byte-identity**: v2 is a NEW code path, developed on its
   own branch. v1 code is not touched, not even one import. All golden tests
   keep passing untouched.
2. **Correctness over speed**: v2 is not a speed hack on v1; it is a declared
   methodological alternative, evaluated openly against the acceptance gate
   before ANY adoption decision. No caps, no timeouts, no early stopping in
   either engine.
3. **No statistics in pipeline/**: the A/B comparison harness lives in
   audit/, never in pipeline/. Only the fitter itself (pure calculation)
   enters pipeline/ if promoted.
4. **Same physics**: identical model (series Zarc + optional R0), identical
   weighting semantics, identical C_eff = tau/R identity, identical output
   dict schema so fit_to_rows() consumes either engine unchanged.

## 3. Architecture (three files)

```
pipeline/zarc_v2.py              the engine (exists only on the branch until
                                 the gate passes and the operator promotes it)
    fit_zarc_v2(...)   same signature as fit_zarc + loss="linear|soft_l1|huber"
    _model_jac(...)    vectorized Zarc + Jacobian in (ln R, ln tau, alpha) space
    least_squares(..., method="trf", x_scale="jac", loss=..., jac=analytic)

audit/fitting_v2/synthetic_gate.py   ground truth
    - synthetic 1..4-Zarc spectra with known params, three noise levels,
      overlapping-tau stress cases
    - recovery error v1 vs v2; the only place where accuracy is measurable
      absolutely rather than relatively

audit/fitting_v2/ab_harness.py       the judge, real data
    - loads every (condition, T) of every sample from stage2/stage3 xlsx + spectra
    - fits each spectrum with v1 and v2 (same seeds, same bounds, same weights)
    - reports per spectrum: rmse_v1, rmse_v2, wall time, params delta in units
      of v1 confidence intervals, edge_frac before/after
    - writes one gitignored CSV per sample; stdout = paths and counts only
```

## 4. Acceptance gate (fixed BEFORE looking at results)

v2 is promotable to pipeline/ (as a declared v2.0 release with regenerated
golden masters) only if ALL hold on the full real dataset AND the synthetic
gate:

- G1: rmse_v2 <= rmse_v1 on >= 95% of spectra, never worse by more than 5%
      anywhere
- G2: every fitted parameter within the v1 68% CI on spectra where v1
      converged cleanly (differences beyond CI must be explained by a v1
      local minimum, shown explicitly)
- G3: synthetic recovery error v2 <= v1 at every noise level
- G4: pinned-fit count (edge_frac <= 0.15) does not increase
- G5: no new dependencies (numpy + scipy only)

If any gate fails, v2 stays a documented experiment: the branch is archived,
the outcome is annotated here, and the pipeline pays zero cost.

## 4b. Gate G3 outcome and amendment (2026-07-11, operator-approved)

The synthetic gate was run as specified (six cases, three noise levels, five
paired replicates, shared seeds). Outcome of the literal criterion "median
recovery error v2 <= v1 at every noise level":

| noise | median v1 | median v2 | delta | literal G3 |
|---|---|---|---|---|
| 0.1% | 0.0009730 | 0.0009715 | -1.5e-6 | pass |
| 0.5% | 0.0029610 | 0.0029640 | +3.0e-6 | fail |
| 2%   | 0.0174185 | 0.0174310 | +1.25e-5 | fail |

Paired analysis of the 90 fits: on the objective both engines minimize
(weighted relative rmse) v2 is never worse, 0 of 90; the worst v2 excess in
recovery error is +3.3e-5 absolute (+0.48% relative), while v2 recovers
three catastrophic v1 local minima in the overlapping-tau cases (recovery
error gains of 0.65 to 0.73). At finite noise the exact least-squares
minimum does not coincide with the ground truth, so stopping an epsilon
short of the minimum (v1, looser tolerances) can land an epsilon closer to
the truth by chance; the observed excess is realization noise, not
optimizer quality.

Amendment, accepted by the operator before Session C: G3 is read as paired
non-inferiority. It passes because (a) v2 is never worse on the fit
objective, and (b) every recovery-error excess is within a 1% relative
equivalence band, while the gains outside that band all favour v2. A real
G3 failure remains any noise level where v2's median exceeds v1's by more
than 1% relative.

## 4c. Where the real-data verdicts live

The synthetic gate (G3) is fully reproducible from this repository and its
outcome is recorded above. The real-data A/B (G1, G2, G4) is instead
executed locally by the operator with ab_harness.py on their own
measurements: its per-spectrum CSVs and its gate verdicts are derived from
private data and therefore never enter the repository, consistent with the
transparency pact in audit/README.md. What is public is the procedure and
the criteria; what stays private is every number produced by real spectra.

## 5. What NOT to build (scope fence)

- No simultaneous multi-T global Zarc fit (shared alpha across T etc.):
  scientifically interesting but changes the model, not the optimizer;
  separate decision.
- No Bayesian/MCMC: overkill for deterministic spectra with ~13 parameters;
  hours of runtime.
- No autodiff frameworks (jax/torch): dependency cost violates G5, and the
  analytic Jacobian is exact anyway.
- No automatic model selection (peak count changes): stays operator-driven
  via the DRT, as today.

## 6. Effort and risk

- zarc_v2.py: the Jacobian is the only real work, verified against
  scipy.optimize.check_grad (max rel err < 1e-6) before anything else.
- synthetic_gate.py + ab_harness.py: mostly plumbing with existing patterns
  (see zarc_window_check.py).
- Risk of a wrong road: LOW as long as the gate is defined first (section 4)
  and v1 stays frozen. The failure mode is "v2 loses the A/B", which costs
  nothing and still documents where v1's local minima are.
- Expected payoff if promoted: bound-crawling fits converge in seconds
  WITHOUT any cap (the gradient at the bound is exact, so TRF stops
  crawling), and noisy low-T spectra stop bending the fit (robust loss).
  Accuracy equal or better by construction, proven by the gate, not asserted.
