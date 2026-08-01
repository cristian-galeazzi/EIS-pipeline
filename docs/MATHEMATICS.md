# Mathematical reference: how the engine works

This document states, stage by stage, the mathematical problem each engine
module solves, how it is discretized, which algorithm minimizes what, and
where the implementation lives. It is written to be checkable by a
mathematician or statistician who has never opened the code: every symbol
below maps to a named parameter in the source, and every section starts
with a pointer of the form `pipeline/<module>.py::<function>`. No numeric
values from any specific dataset appear here; defaults quoted are the
public notebook defaults.

Notation used throughout: a spectrum is a set of complex impedances
Z(ω_i) = Z'(ω_i) + j Z''(ω_i) measured at angular frequencies ω_i = 2π f_i,
i = 1..N. The pipeline stores Z'' positive in the capacitive region and
flips the sign where the physical convention is needed (see the sign note
in `CLAUDE.md`).

---

## 1. Kramers-Kronig validity test (stage 2)

**Code:** `pipeline/quality.py::run_linkk`, `::_find_optimal_M`
**Reference:** Schönleber et al., Electrochimica Acta 131 (2014) 20-27.

A causal, linear, time-invariant system must satisfy the Kramers-Kronig
relations. Testing them directly requires integrating over all frequencies,
so the linearized test fits the measured spectrum with a basis that
satisfies KK **by construction**: a series of M RC relaxations with fixed,
log-spaced time constants τ_k spanning the measured window,

    Z_KK(ω) = R_ohm + Σ_{k=1..M}  R_k / (1 + j ω τ_k) .

Only the weights R_k (and R_ohm) are unknown, and they enter linearly, so
the fit is a plain linear least-squares solve; the R_k may be negative
(they are basis coefficients, not physical resistors). If the best KK-
consistent model of this form cannot reproduce the data, the residual that
remains is the KK-violating part of the measurement (drift, nonlinearity,
instrument artifacts).

Residuals are magnitude-normalized so that both arcs of very different
size count equally:

    r_re(ω_i) = (Z'_meas − Z'_KK)(ω_i) / |Z(ω_i)| ,
    r_im(ω_i) = (Z''_meas − Z''_KK)(ω_i) / |Z(ω_i)| .

**Choice of M.** Too few RC elements underfit (structure left in the
residual); too many overfit and start reproducing the noise, which shows up
as adjacent weights R_k, R_{k+1} of alternating sign. The sign-change
fraction

    μ(M) = (# adjacent pairs with opposite sign) / (M − 1)

runs from ~0 (underfit) to ~1 (overfit). The automatic mode scans M upward
from 3 and stops at the smallest M with μ ≥ 0.50 (`mu_target`); the scan is
linear because μ(M) is not monotonic, so bisection could skip valid M. The
fixed mode uses M = round(c · N) with the density c (`KK_C` in the stage-2
notebook) calibrated once per instrument/dataset class. When no M in [3, N−1]
reaches μ ≥ mu_target, the scan falls back to the fixed mode with c = 0.85
rather than failing, so a spectrum that never reaches the target is still
scored instead of being dropped silently.

**Pass criterion.** A KK-consistent spectrum leaves residuals that are pure
noise, so both r_re and r_im must pass a Shapiro-Wilk normality test with
W ≥ 0.95, evaluated after edge trimming.

**Edge trimming.** Measurement artifacts concentrate at the frequency
extremes. Pass 1 fits the full spectrum and walks inward from each edge; a
point is cut while its |r| exceeds an adaptive fence

    fence = Q3_interior + k_IQR · IQR_interior

(Q3 and IQR of the interior residuals; `KK_IQR_FENCE`, default 2.0) until
`KK_IQR_WINDOW` consecutive clean points confirm the boundary. Pass 2
refits on the trimmed window and produces the final score; the cut
frequencies f_min/f_max are stored and propagated to stage 3 so every later
fit sees the same validated window.

---

## 2. Distribution of relaxation times (stage 3, step 1)

**Code:** `pipeline/drt.py::compute_drt`, `::find_drt_peaks`
(Tikhonov solver: `pyDRTtools.runs.simple_run`)

The DRT model writes the polarization part of the impedance as a continuous
superposition of ideal RC relaxations with distribution γ(τ):

    Z(ω) = R_∞ + ∫_{−∞}^{+∞} γ(ln τ) / (1 + j ω τ) d(ln τ) .

Recovering γ from Z sampled at N frequencies is a Fredholm integral
equation of the first kind: ill-posed, so plain least squares amplifies
noise into spurious oscillations of γ. Tikhonov regularization restores
well-posedness by penalizing rough solutions:

    min_γ  || Z_meas − Z_model[γ] ||²  +  λ || L γ ||² ,

where L is a derivative operator (the `DRT_RBF_DER` order; default second
derivative) applied to the radial-basis-function discretization of γ
(shape factor `DRT_SHAPE_S`). The regularization parameter λ
(`DRT_LAMBDA`) sets the bias-variance compromise directly: small λ gives
sharp peaks and noise sensitivity, large λ gives smooth, merged peaks.
λ is therefore a **calibration** input, chosen per dataset class with the
public procedure `audit/calibrate_drt.py`, not a fitted quantity.

**Peak extraction.** Peaks of γ(ln τ) are located with a prominence
criterion in log-τ space (`PEAK_MIN_PROM_DECADES`, `PEAK_MIN_DIST_DECADES`)
and each peak is integrated between its flanking minima:

    R_approx = ∫_{ln τ_left}^{ln τ_right} γ(ln τ) d(ln τ) .

The integration variable matters: integrating over log₁₀ τ instead of ln τ
would underestimate R by a factor ln 10 ≈ 2.303. `find_drt_peaks`
integrates over `np.log(tau)` for this reason. The pair (R_approx, τ_peak)
of every detected peak is the **seed** of the circuit fit below; peaks are
numbered `peak_id` = 1, 2, ... by ascending τ (descending frequency).

---

## 3. Zarc equivalent-circuit fit (stage 3, step 2)

**Code:** `pipeline/fitting.py::fit_zarc`, `::build_bounds`,
`::resolve_peak_windows`, `::fit_condition_batch`

### 3.1 Model

The circuit is R0 − Zarc₁ − ... − Zarc_N (R0 optional), each Zarc being a
resistor in parallel with a constant-phase element, parametrized as

    Z_k(ω) = R_k / (1 + (j ω τ_k)^{α_k}) ,      k = 1..N ,

with R_k > 0 the process resistance, τ_k > 0 its relaxation time and
α_k ∈ [0.5, 1] the depression exponent (α = 1 is an ideal semicircle). The
parameter vector is θ = (R0?, R₁, τ₁, α₁, ..., R_N, τ_N, α_N).

### 3.2 Effective capacitance is exactly τ/R

For a CPE with admittance Y = Q (jω)^α in parallel with R, the standard
Brug effective capacitance is C_eff = Q^{1/α} R^{(1−α)/α}. In the Zarc
parametrization Q = τ^α / R, hence

    C_eff = (τ^α/R)^{1/α} · R^{(1−α)/α}
          = τ · R^{−1/α} · R^{1/α − 1}
          = τ / R ,

exactly, for every α. This is an identity of the parametrization, not an
approximation, and it is enforced by a golden test
(`tests/test_engine_golden.py::test_ceff_identity_exact`).

### 3.3 Objective function and weighting

The residual vector stacks weighted real and imaginary parts over the N
frequencies:

    r(θ) = [ (Z'_model − Z'_meas)/s ;  (Z''_model − Z''_meas)/s ] ∈ R^{2N} ,

and the optimizer minimizes ||r(θ)||² subject to box bounds (next
subsection). The per-frequency scale s_i implements the weighting mode:

* proportional (default, `weight_by_modulus=True`): s_i = |Z_meas(ω_i)|,
  i.e. relative residuals, so the small high-frequency arc counts as much
  as the large low-frequency one;
* high-frequency emphasis (`hf_weight` = h > 0):
  s_i = |Z(ω_i)| / (1 + h · ν_i) with ν_i the log₁₀ f normalized to [0, 1],
  which shrinks s at high frequency and pins the HF arc harder;
* unit weighting (legacy): s_i = 1.

The reported quality figure `rmse_rel` is the root mean square of the
proportional residuals, i.e. a dimensionless relative misfit.

### 3.4 Constraint windows (decade boxes around the seeds)

`build_bounds` converts each DRT seed into a box constraint per parameter:

    R_k   ∈ [ R_approx,k · 10^{−d_R,k} ,  R_approx,k · 10^{+d_R,k} ] ,
    τ_k   ∈ [ τ_seed,k  · 10^{−d_τ,k} ,  τ_seed,k  · 10^{+d_τ,k} ] ,
    α_k   ∈ [ α_min, α_max ] ,

where d_R,k and d_τ,k are half-widths **in decades** (`R_dec`, `tau_dec`,
default 0.7). A window is a statement of trust in that peak's DRT seed, not
a physical prior on the parameter, which is why the half-widths may differ
per peak: `resolve_peak_windows` resolves them per `peak_id` from
session-stored maps (sample-wide default, then per-condition override),
falling back to the scalar defaults. Windows are held constant along the
temperature series of a condition by design; varying them per spectrum
would imprint operator choices onto the Arrhenius trends extracted later.

The **pinning diagnostic** measures, for each fitted R and τ, the distance
to the nearer window edge as a fraction of the log₁₀ half-width (0 = on the
bound, 1 = window center); a parameter with edge fraction ≤ 0.15 is flagged
PINNED, meaning the optimizer pushed against the constraint and the seed or
the window, not the data, is limiting the result.

### 3.5 Seeding: cold starts, warm starts, restarts

Within one condition the temperatures are fitted in descending order
(`fit_condition_batch`). The hottest spectrum starts **cold**, i.e. from
the raw DRT seeds. Every following temperature starts **warm** when the
previous (hotter) fit converged with the same peak count: its fitted
(R_k, τ_k) replace the DRT seeds, and the constraint boxes are rebuilt
around these warm seeds. Physically this encodes continuity of each process
along T; numerically it keeps the optimizer in the same valley across the
series, which is what makes the per-temperature parameters comparable in an
Arrhenius plot.

On top of the seeded start, `n_restarts` additional starts are drawn with R
and τ log-uniform inside their windows (α uniform). Restart draws use a
deterministic seed crc32(condition|T), so a re-run of the notebook
reproduces the identical fit bit for bit. Restarts stop early once
`rmse_rel` < `rmse_tol` (default 0.02): further polish would refine digits
below the noise floor.

### 3.6 Optimizer

The engine minimizes ||r(θ)||² with scipy's trust-region reflective (TRF)
bounded least squares in a **logarithmic parametrization**
x = (ln R0?, ln R_k, ln τ_k, α_k). R and τ span decades, so in linear space
the least-squares valley is a long ill-conditioned trench, while in log
space the decade boxes of §3.4 become plain symmetric intervals and the
conditioning improves by orders of magnitude. The Jacobian is analytic:
with p = ln R, q = ln τ, a = α and u = (jωτ)^a,

    ∂Z/∂p = Z_k ,
    ∂Z/∂q = − R u a / (1 + u)² ,
    ∂Z/∂a = − R u ln(jωτ) / (1 + u)² ,   ln(jωτ) = ln(ωτ) + jπ/2 ,

each column stacked as real/imaginary parts and divided by the same s_i as
the residual (weighting commutes with differentiation). Parameters held via
`ZARC_FIX_PARAMS` are removed from the free vector and substituted exactly.
The 1σ confidence intervals come from the Gauss-Newton covariance
s² (JᵀJ)⁻¹ at the optimum, mapped back to linear units by the delta method
(σ_R = σ_lnR · R). The analytic Jacobian is verified against central finite
differences to relative error < 10⁻⁶ (`tests/test_zarc_v2.py`), and the
migration from the previous linear-space engine is recorded, with its
pre-registered synthetic ground-truth gates and the frozen v1 reference
implementation, in `audit/fitting_v2/`.

### 3.7 Derived quantities

With the pellet geometry (thickness L, diameter D, area A = π(D/2)²), each
process resistance maps to a conductivity

    σ_k = L / (R_k · A)      [S/m] ,

and C_eff,k = τ_k / R_k (§3.2) classifies the process (bulk, grain
boundary, electrode) by magnitude.

---

## 4. Arrhenius analysis (stage 4)

**Code:** `pipeline/plots.py::build_arrhenius_results`

Thermally activated transport with mobility ∝ 1/T follows
σT = A · exp(−Ea / k_B T), so the pipeline fits straight lines by ordinary
least squares in the linearized coordinates:

    ln(σ_k T) = ln A − (Ea_cond / k_B) · (1/T)        (conduction) ,
    ln(τ_k)   = const + (Ea_pol  / k_B) · (1/T)       (relaxation) ,

with the slope errors propagated to ±ΔEa and the R² of each regression
reported. Ea_C (from ln C_eff) equals Ea_pol − Ea_cond by the identity of
§3.2, and serves as an internal consistency check: it is derived from the
other two, not an independent measurement.

---

## 5. Ionic/electronic decomposition per isotherm (stage 4, NNLS)

**Code:** `pipeline/plots.py::fit_transference`

At a fixed temperature T, the Patterson model in the dilute defect regime
decomposes the measured conductivity of one process along the pO₂ axis:

    σ(pO₂) = σ_ion + σ_p · pO₂^{+x} + σ_n · pO₂^{−x} ,     x = Brouwer exponent .

The three coefficients enter **linearly**. Writing the m measured points of
the isotherm as y ∈ R^m, the design matrix is

    A = [ 1   pO₂^{+x}   pO₂^{−x} ] ∈ R^{m×3} ,

and the coefficients are the solution of the non-negative least squares
problem

    min_{c ≥ 0} || A c − y ||² ,       c = (σ_ion, σ_p, σ_n) ,

solved by `scipy.optimize.nnls` (active-set method; no initial guess, global
optimum guaranteed because the problem is convex). Non-negativity is the
physics: a partial conductivity cannot be negative, and an unconstrained
solve on noisy plateau data routinely returns small negative σ_n that would
corrupt the transference numbers. From the solution,

    t_ion(pO₂) = σ_ion / σ(pO₂) ,     t_el = 1 − t_ion ,

and the local Brouwer slope obeys d(log σ)/d(log pO₂) = x (t_p − t_n): a
plateau is purely ionic, a +x slope purely p-type. Isotherms with fewer
than 4 pO₂ points are skipped (3 unknowns need redundancy for a meaningful
R²). The per-isotherm σ_ion(T), σ_p(T) then feed an Arrhenius plot
(`plot_transference_arrhenius`) whose slopes are the **first estimates** of
the channel activation energies; they motivate and initialize the global
model of stage 5, which replaces this per-isotherm view with one
temperature-coupled fit.

---

## 6. Global MIEC conductivity model (stage 5, VARPRO)

**Code:** `pipeline/model.py::fit_global_conductivity`,
`::_design_matrix`, `::_solve_sigma0`, `::global_transference_table`

### 6.1 Model

The whole surface σ(pO₂, T) of one process is described by three parallel,
Arrhenius-activated channels with mobility ∝ 1/T:

    σ(pO₂, T) = (σ0_ion / T) e^{−Ea_ion / k_B T}
              + (σ0_p   / T) e^{−Ea_p   / k_B T} · pO₂^{+x}
              + (σ0_n   / T) e^{−Ea_n   / k_B T} · pO₂^{−x} .

Six parameters: three prefactors σ0 (with the 1/T pulled out) and three
activation energies. One parameter set must fit **all** points at once;
that constancy across the surface is the physical-validity test of the
defect-chemical picture. Which channels are included (`MODEL_CHANNELS`) is
an operator decision from defect chemistry, never a fit outcome: an
excluded channel is stored as σ0 = 0, Ea = NaN.

### 6.2 Variable projection (VARPRO)

At **fixed** activation energies the model is linear in the prefactors:
with points (pO₂_i, T_i, σ_i), i = 1..m, and selected channels
c = 1..n_ch, define the design matrix

    A_ic(Ea) = (1 / T_i) · e^{−Ea_c / k_B T_i} · pO₂_i^{e_c x} ,
    e_c = 0 (ion), +1 (p), −1 (n) ,

so σ_model = A(Ea) s0 with s0 the prefactor vector. The fit is weighted by
w_i = 1/σ_i (relative residuals, so a weak channel is not drowned by a
strong one). The **inner problem**

    s0*(Ea) = argmin_{s0 ≥ 0} || diag(w) (A(Ea) s0 − σ) ||²

is convex and solved exactly by weighted NNLS at every step. The **outer
problem** optimizes only the activation energies,

    min_{Ea ∈ [0, 3] eV}  || diag(w) (A(Ea) s0*(Ea) − σ) ||² ,

with bounded least squares (TRF). Eliminating the linear parameters this
way removes the σ0/Ea seesaw direction that traps a naive 6-parameter fit
in false minima, guarantees non-negative prefactors by construction, and
reduces the nonlinear search space from 6 to n_ch dimensions.

A fit is refused outright below `MIN_POINTS` = 8 usable (pO₂, T) points
(`pipeline/model.py::fit_global_conductivity`): with up to 6 free parameters,
fewer points leave no redundancy, and the covariance of 6.3 would be
meaningless rather than merely wide.

### 6.3 Covariance and reported uncertainties

The outer VARPRO solution is used to seed one **full polish**: a bounded
least squares over all 2·n_ch parameters θ = (s0, Ea) starting at the
VARPRO optimum. Its Jacobian J at convergence gives the Gauss-Newton
covariance estimate

    Cov(θ) ≈ (JᵀJ)⁻¹ · 2 · SSR / dof ,     dof = m − dim θ ,

computed through the SVD of J for rank safety
(`pipeline/model.py::_covariance_errors`); the square roots of the diagonal
are the ±1σ errors quoted on each Ea. When a selected channel is driven to
σ0 = 0 by the NNLS, its Ea is reported as "not active" instead of a number:
the energy of an absent channel is not identifiable and its CI would be
meaningless.

### 6.4 Derived diagnostics

* Global R² over the whole surface (all conditions, all T).
* Structureless residual map over (pO₂, T)
  (`pipeline/plots.py::plot_fit_residuals`): any remaining pattern flags a
  wrong exponent x, a missing channel, or process misassignment.
* Conductivity minimum (n = p crossover), meaningful only when both
  electronic channels are present:

      pO₂_min(T) = [ (σ0_n / σ0_p) · e^{(Ea_p − Ea_n)/k_B T} ]^{1/2x} ,

  from setting the p and n terms equal
  (`pipeline/model.py::stoichiometric_pO2`).
* Model-based transference table
  (`pipeline/model.py::global_transference_table`): the
  stage-4 figure redrawn from the single global parameter set instead of
  the per-isotherm NNLS.

---

## 7. Determinism and reproducibility

Every stochastic element of the chain is seeded deterministically:

* Zarc restart draws: crc32(condition|T) (`pipeline/fitting.py`), so the
  same notebook re-run reproduces every fit exactly;
* warm-start chains are strictly ordered (descending T within a condition;
  conditions independent, hence parallelizable without changing results);
* all calibrated knobs (λ, windows, weighting) live in `session.json` and
  are re-applied by the batch cells, so a fresh kernel reproduces the
  exported sheets byte for byte, which is what the golden-master test suite
  (`tests/`) asserts.
