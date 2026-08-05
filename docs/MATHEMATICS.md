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
$`Z(\omega_i) = Z'(\omega_i) + j Z''(\omega_i)`$ measured at angular frequencies
$`\omega_i = 2\pi f_i`$, $`i = 1 \dots N`$. Boltzmann's constant is written $`k_B`$
throughout, since a bare $`k`$ already indexes channels and RC elements below.
The pipeline stores $`Z''`$ positive in the capacitive region, the convention
required of CSV input ([docs/INPUT_FORMAT.md](INPUT_FORMAT.md)), and flips the
sign only where a step expects the physical one.

---

## 1. Kramers-Kronig validity test (stage 2)

**Code:** `pipeline/quality.py::run_linkk`, `::_find_optimal_M`
**Reference:** Schönleber et al., Electrochimica Acta 131 (2014) 20-27.

A causal, linear, time-invariant system must satisfy the Kramers-Kronig
relations. Testing them directly requires integrating over all frequencies,
so the linearized test fits the measured spectrum with a basis that
satisfies KK **by construction**: a series of $`M`$ RC relaxations with fixed,
log-spaced time constants $`\tau_k`$ spanning the measured window,

```math
Z_\text{KK}(\omega) = R_\text{ohm} + \sum_{k=1}^{M} \frac{R_k}{1 + j\omega\tau_k}
```

Only the weights $`R_k`$ (and $`R_\text{ohm}`$) are unknown, and they enter
linearly, so the fit is a plain linear least-squares solve; the $`R_k`$ may be negative
(they are basis coefficients, not physical resistors). If the best KK-
consistent model of this form cannot reproduce the data, the residual that
remains is the KK-violating part of the measurement (drift, nonlinearity,
instrument artifacts).

Residuals are magnitude-normalized so that both arcs of very different
size count equally:

```math
r_\text{re}(\omega_i) = \frac{Z'_\text{meas}(\omega_i) - Z'_\text{KK}(\omega_i)}{|Z(\omega_i)|}, \qquad r_\text{im}(\omega_i) = \frac{Z''_\text{meas}(\omega_i) - Z''_\text{KK}(\omega_i)}{|Z(\omega_i)|}
```

**Choice of M.** Too few RC elements underfit (structure left in the
residual); too many overfit and start reproducing the noise, which shows up
as adjacent weights $`R_k`$, $`R_{k+1}`$ of alternating sign. The sign-change
fraction

```math
\mu(M) = \frac{\text{number of adjacent pairs with opposite sign}}{M - 1}
```

runs from about 0 (underfit) to about 1 (overfit). The automatic mode scans $`M`$
upward from 3 and stops at the smallest $`M`$ whose $`\mu`$ is at least 0.50
(`_find_optimal_M(mu_target=0.50)`); the scan is linear because $`\mu(M)`$ is not
monotonic, so bisection could skip valid $`M`$. The Percentage mode uses
$`M = \text{round}(c N)`$ with the density $`c`$ (`KK_C` in the stage-2
notebook) calibrated once per instrument/dataset class. When no $`M`$ in
$`[3, N-1]`$ reaches $`\mu \geq`$ `mu_target`, the scan falls back to the
Percentage mode at the configured density rather than failing, so a spectrum that never
reaches the target is still scored instead of being dropped silently.

**Pass criterion.** A KK-consistent spectrum leaves residuals that are pure
noise, so $`r_\text{re}`$ and $`r_\text{im}`$ are each scored with a
Shapiro-Wilk normality test after edge trimming and averaged into `kk_score`;
the classification thresholds are in [STAGES.md](STAGES.md).

**Edge trimming.** Measurement artifacts concentrate at the frequency
extremes. Pass 1 fits the full spectrum and walks inward from each edge; a
point is cut while its $`|r|`$ exceeds an adaptive fence

```math
\text{fence} = Q_3^\text{interior} + k_\text{IQR}\,\text{IQR}^\text{interior}
```

($`Q_3`$ and IQR of the interior residuals; `KK_IQR_FENCE`, default 2.0) until
`KK_IQR_WINDOW` consecutive clean points confirm the boundary. Pass 2
refits on the trimmed window and produces the final score; the cut
frequencies `f_min`/`f_max` are stored and propagated to stage 3 so every later
fit sees the same validated window.

---

## 2. Distribution of relaxation times (stage 3, step 1)

**Code:** `pipeline/drt.py::compute_drt`, `::find_drt_peaks`
(Tikhonov solver: `pyDRTtools.runs.simple_run`)

The DRT model writes the polarization part of the impedance as a continuous
superposition of ideal RC relaxations with distribution $`\gamma(\ln\tau)`$:

```math
Z(\omega) = R_\infty + \int_{-\infty}^{+\infty} \frac{\gamma(\ln\tau)}{1 + j\omega\tau}\, d(\ln\tau)
```

Recovering $`\gamma`$ from $`Z`$ sampled at $`N`$ frequencies is a Fredholm integral
equation of the first kind: ill-posed, so plain least squares amplifies
noise into spurious oscillations of $`\gamma`$. Tikhonov regularization restores
well-posedness by penalizing rough solutions:

```math
\min_{\gamma}\; \lVert Z_\text{meas} - Z_\text{model}[\gamma] \rVert^2 + \lambda \lVert L\gamma \rVert^2
```

where $`L`$ is a derivative operator (the `DRT_RBF_DER` order; default second
derivative) applied to the radial-basis-function discretization of $`\gamma`$
(shape factor `DRT_SHAPE_S`). The regularization parameter $`\lambda`$
(`DRT_LAMBDA`) sets the bias-variance compromise directly: small $`\lambda`$ gives
sharp peaks and noise sensitivity, large $`\lambda`$ gives smooth, merged peaks.
$`\lambda`$ is therefore a **calibration** input, chosen per dataset class with the
public procedure `audit/calibrate_drt.py`, not a fitted quantity.

**Peak extraction.** Peaks of $`\gamma(\ln\tau)`$ are located with a prominence
criterion in $`\log\tau`$ space (`PEAK_MIN_PROM_DECADES`, `PEAK_MIN_DIST_DECADES`)
and each peak is integrated between its flanking minima:

```math
R_\text{approx} = \int_{\ln\tau_\text{left}}^{\ln\tau_\text{right}} \gamma(\ln\tau)\, d(\ln\tau)
```

The integration variable matters: integrating over $`\log_{10}\tau`$ instead of
$`\ln\tau`$ would underestimate $`R`$ by a factor $`\ln 10 \approx 2.303`$.
`find_drt_peaks` integrates over `np.log(tau)` for this reason. The pair
$`(R_\text{approx}, \tau_\text{peak})`$ of every detected peak is the **seed** of
the circuit fit below; peaks are numbered `peak_id` $`= 1, 2, \dots`$ by ascending
$`\tau`$ (descending frequency).

---

## 3. Zarc equivalent-circuit fit (stage 3, step 2)

**Code:** `pipeline/fitting.py::fit_zarc`, `::build_bounds`,
`::resolve_peak_windows`, `::fit_condition_batch`

### 3.1 Model

The circuit is $`R_0 + \text{Zarc}_1 + \dots + \text{Zarc}_N`$ ($`R_0`$ optional),
each Zarc being a resistor in parallel with a constant-phase element,
parametrized as

```math
Z_k(\omega) = \frac{R_k}{1 + (j\omega\tau_k)^{\alpha_k}}, \qquad k = 1 \dots N
```

with $`R_k > 0`$ the process resistance, $`\tau_k > 0`$ its relaxation time and
$`\alpha_k \in [0.5, 1]`$ the depression exponent ($`\alpha = 1`$ is an ideal
semicircle). The parameter vector is
$`\theta = (R_0, R_1, \tau_1, \alpha_1, \dots, R_N, \tau_N, \alpha_N)`$, with the
leading $`R_0`$ present only when `ZARC_INCLUDE_R0` selects it.

### 3.2 Effective capacitance is exactly τ/R

For a CPE with admittance $`Y = Q(j\omega)^\alpha`$ in parallel with $`R`$, the
standard Brug effective capacitance is
$`C_\text{eff} = Q^{1/\alpha} R^{(1-\alpha)/\alpha}`$. In the Zarc
parametrization $`Q = \tau^\alpha / R`$, hence

```math
C_\text{eff} = \left(\frac{\tau^\alpha}{R}\right)^{1/\alpha} R^{(1-\alpha)/\alpha} = \tau\, R^{-1/\alpha} R^{1/\alpha - 1} = \frac{\tau}{R}
```

exactly, for every $`\alpha`$. This is an identity of the parametrization, not an
approximation, and it is enforced by a golden test
(`tests/test_engine_golden.py::test_ceff_identity_exact`).

### 3.3 Objective function and weighting

The residual vector stacks weighted real and imaginary parts over the $`N`$
frequencies:

```math
r(\theta) = \left[\frac{Z'_\text{model} - Z'_\text{meas}}{s};\ \frac{Z''_\text{model} - Z''_\text{meas}}{s}\right] \in \mathbb{R}^{2N}
```

and the optimizer minimizes $`\lVert r(\theta) \rVert^2`$ subject to box bounds
(next subsection). The per-frequency scale $`s_i`$ implements the weighting mode:

* proportional (default, `weight_by_modulus=True`): $`s_i = |Z_\text{meas}(\omega_i)|`$,
  that is relative residuals, so the small high-frequency arc counts as much
  as the large low-frequency one;
* high-frequency emphasis (`hf_weight` $`= h > 0`$):
  $`s_i = |Z(\omega_i)| / (1 + h\nu_i)`$ with $`\nu_i`$ the $`\log_{10} f`$ normalized
  to $`[0, 1]`$, which shrinks $`s`$ at high frequency and pins the HF arc harder;
* unit weighting (legacy): $`s_i = 1`$.

The reported quality figure `rmse_rel` is the root mean square of the
proportional residuals, i.e. a dimensionless relative misfit.

### 3.4 Constraint windows (decade boxes around the seeds)

`build_bounds` converts each DRT seed into a box constraint per parameter:

```math
R_k \in \left[R_{\text{approx},k}\,10^{-d_{R,k}},\; R_{\text{approx},k}\,10^{+d_{R,k}}\right], \quad \tau_k \in \left[\tau_{\text{seed},k}\,10^{-d_{\tau,k}},\; \tau_{\text{seed},k}\,10^{+d_{\tau,k}}\right], \quad \alpha_k \in [\alpha_\text{min}, \alpha_\text{max}]
```

where $`d_{R,k}`$ and $`d_{\tau,k}`$ are half-widths **in decades**
(`ZARC_R_DEC`, `ZARC_TAU_DEC`, 0.70 in the notebooks; the `R_dec` and `tau_dec`
arguments of `fit_zarc`). A window is a statement of trust in that peak's DRT seed, not
a physical prior on the parameter, which is why the half-widths may differ
per peak: `resolve_peak_windows` resolves them per `peak_id` from
session-stored maps (sample-wide default, then per-condition override),
falling back to the scalar defaults. Windows are held constant along the
temperature series of a condition by design; varying them per spectrum
would imprint operator choices onto the Arrhenius trends extracted later.

The **pinning diagnostic** measures, for each fitted $`R`$ and $`\tau`$, the distance
to the nearer window edge as a fraction of the $`\log_{10}`$ half-width (0 on the
bound, 1 at the window center); a parameter with edge fraction at most 0.15 is flagged
PINNED, meaning the optimizer pushed against the constraint and the seed or
the window, not the data, is limiting the result.

### 3.5 Seeding: cold starts, warm starts, restarts

Within one condition the temperatures are fitted in descending order
(`fit_condition_batch`). The hottest spectrum starts **cold**, i.e. from
the raw DRT seeds. Every following temperature starts **warm** when the
previous (hotter) fit converged with the same peak count: its fitted
$`(R_k, \tau_k)`$ replace the DRT seeds, and the constraint boxes are rebuilt
around these warm seeds. Physically this encodes continuity of each process
along $`T`$; numerically it keeps the optimizer in the same valley across the
series, which is what makes the per-temperature parameters comparable in an
Arrhenius plot.

On top of the seeded start, `n_restarts` additional starts are drawn with $`R`$
and $`\tau`$ log-uniform inside their windows ($`\alpha`$ uniform). Restart
draws use a deterministic seed: the CRC-32 checksum of the condition name and
the nominal temperature joined by a literal `|`, so a re-run of the notebook
reproduces the identical fit bit for bit. Restarts stop early once
`rmse_rel` < `rmse_tol` (default 0.02): further polish would refine digits
below the noise floor.

### 3.6 Optimizer

The engine minimizes $`\lVert r(\theta) \rVert^2`$ with scipy's trust-region
reflective (TRF) bounded least squares in a **logarithmic parametrization**
$`x = (\ln R_0, \ln R_k, \ln \tau_k, \alpha_k)`$, again with $`R_0`$ only when it
is part of the circuit. $`R`$ and $`\tau`$ span decades, so
in linear space the least-squares valley is a long ill-conditioned trench, while
in log space the decade boxes of 3.4 become plain symmetric intervals and the
conditioning improves by orders of magnitude. The Jacobian is analytic:
with $`p = \ln R`$, $`q = \ln\tau`$, $`a = \alpha`$ and $`u = (j\omega\tau)^a`$,

```math
\frac{\partial Z}{\partial p} = Z_k, \qquad \frac{\partial Z}{\partial q} = -\frac{R u a}{(1 + u)^2}, \qquad \frac{\partial Z}{\partial a} = -\frac{R u \ln(j\omega\tau)}{(1 + u)^2}
```

with $`\ln(j\omega\tau) = \ln(\omega\tau) + j\pi/2`$, each column stacked as
real and imaginary parts and divided by the same $`s_i`$ as
the residual (weighting commutes with differentiation). Parameters held via
`ZARC_FIX_PARAMS` are removed from the free vector and substituted exactly.
The $`1\sigma`$ confidence intervals come from the Gauss-Newton covariance
$`s^2 (J^\mathsf{T} J)^{-1}`$ at the optimum, mapped back to linear units by the
delta method ($`\sigma_R = \sigma_{\ln R} R`$). The analytic Jacobian is verified
against central finite differences to relative error below $`10^{-6}`$
(`tests/test_zarc_v2.py`), and the
migration from the previous linear-space engine is recorded, with its
pre-registered synthetic ground-truth gates and the frozen v1 reference
implementation, in `audit/fitting_v2/`.

### 3.7 Derived quantities

With the pellet geometry (thickness $`L`$, diameter $`D`$, area $`A = \pi(D/2)^2`$),
each process resistance maps to a conductivity

```math
\sigma_k = \frac{L}{R_k A} \quad [\text{S/m}]
```

kept in that unit in the `sigma_Sm_i` column every later stage reads, and
converted to S/cm at the reporting boundary, so every figure and every derived
table is in S/cm. The effective capacitance $`C_{\text{eff},k} = \tau_k / R_k`$
(3.2) is the quantity compared across temperatures to check that a peak keeps
its character along the series. The same geometry turns that capacitance into
a relative permittivity

```math
\varepsilon_{\text{r},k} = \frac{C_{\text{eff},k}\, L}{\varepsilon_0 A}
```

with $`\varepsilon_0`$ the vacuum permittivity (`EPS_0` in
`pipeline/plots.py`). Assigning a peak to a named physical process is an
operator decision from independent evidence, never a conclusion drawn from a
capacitance magnitude alone.

---

## 4. Arrhenius analysis (stage 4)

**Code:** `pipeline/plots.py::build_arrhenius_results`

Thermally activated transport with mobility $`\propto 1/T`$ follows
$`\sigma T = A \exp\left(-E_\text{a}/k_B T\right)`$, so the pipeline fits
straight lines by ordinary least squares in the linearized coordinates:

```math
\ln(\sigma_k T) = \ln A - \frac{E_\text{a}^\text{cond}}{k_B}\cdot\frac{1}{T} \quad \text{(conduction)}, \qquad \ln(\tau_k) = \text{const} + \frac{E_\text{a}^\text{pol}}{k_B}\cdot\frac{1}{T} \quad \text{(relaxation)}
```

with the slope errors propagated to $`\pm\Delta E_\text{a}`$ and the $`R^2`$ of each
regression reported. `Ea_C` (from $`\ln C_\text{eff}`$) equals
$`E_\text{a}^\text{pol} - E_\text{a}^\text{cond}`$ by the identity of 3.2, and
serves as an internal consistency check: it is derived from the other two, not
an independent measurement.

---

## 5. Ionic/electronic decomposition per isotherm (stage 4, NNLS)

**Code:** `pipeline/plots.py::fit_transference`

At a fixed temperature $`T`$, the Patterson model in the dilute defect regime
decomposes the measured conductivity of one process along the $`p_{\text{O}_2}`$ axis:

```math
\sigma(p_{\text{O}_2}) = \sigma_{\text{ion}} + \sigma_{p\text{-type}}\, p_{\text{O}_2}^{+x} + \sigma_{n\text{-type}}\, p_{\text{O}_2}^{-x}
```

with $`x`$ the Brouwer exponent. The three coefficients enter **linearly**, so
over the $`N`$ measured pressures $`p_1 \dots p_N`$ of the isotherm the design
matrix and the unknown vector are

```math
A =
\begin{bmatrix}
1      & p_1^{+x} & p_1^{-x} \\
\vdots & \vdots   & \vdots   \\
1      & p_N^{+x} & p_N^{-x}
\end{bmatrix}
\in \mathbb{R}^{N \times 3},
\qquad
\mathbf{s} =
\begin{bmatrix}
\sigma_{\text{ion}} \\
\sigma_{p\text{-type}} \\
\sigma_{n\text{-type}}
\end{bmatrix}
```

the three columns being the ionic, p-type and n-type channels. The
coefficients solve the non-negative least squares problem

```math
\mathbf{s} = \arg\min_{\mathbf{s} \geq 0} \lVert A\mathbf{s} - \boldsymbol{\sigma}^\text{meas} \rVert^2
```

by `scipy.optimize.nnls` (active-set method; no initial guess, global optimum
guaranteed because the problem is convex). Non-negativity is the physics: a
partial conductivity cannot be negative, and an unconstrained solve on noisy
plateau data routinely returns a small negative $`\sigma_{n\text{-type}}`$ that
would corrupt the transference numbers. From the solution,

```math
t_{\text{ion}}(p_{\text{O}_2}) = \frac{\sigma_{\text{ion}}}{\sigma(p_{\text{O}_2})}, \qquad t_{\text{elect}} = 1 - t_{\text{ion}}
```

and the local Brouwer slope obeys
$`d\log\sigma / d\log p_{\text{O}_2} = x\,(t_{p\text{-type}} - t_{n\text{-type}})`$:
a plateau is purely ionic, a $`+x`$ slope purely p-type. Isotherms with fewer
than 4 $`p_{\text{O}_2}`$ points are skipped (3 unknowns need redundancy for a
meaningful $`R^2`$). The per-isotherm $`\sigma_{\text{ion}}(T)`$ and
$`\sigma_{p\text{-type}}(T)`$ then feed an Arrhenius plot
(`plot_transference_arrhenius`) whose slopes are the **first estimates** of
the channel activation energies; they motivate and initialize the global
model of stage 5, which replaces this per-isotherm view with one
temperature-coupled fit.

---

## 6. Global MIEC conductivity model (stage 5, VARPRO)

**Code:** `pipeline/model.py::fit_global_conductivity`,
`::_design_matrix`, `::_solve_sigma0`, `::global_transference_table`

### 6.1 Model

The whole surface $`\sigma(p_{\text{O}_2}, T)`$ of one process is described by
three parallel, Arrhenius-activated channels with mobility $`\propto 1/T`$:

```math
\sigma(p_{\text{O}_2}, T) = \frac{\sigma_0^{\text{ion}}}{T} \exp\left(-\frac{E_\text{a}^{\text{ion}}}{k_B T}\right) + \frac{\sigma_0^{p\text{-type}}}{T} \exp\left(-\frac{E_\text{a}^{p\text{-type}}}{k_B T}\right) p_{\text{O}_2}^{+x} + \frac{\sigma_0^{n\text{-type}}}{T} \exp\left(-\frac{E_\text{a}^{n\text{-type}}}{k_B T}\right) p_{\text{O}_2}^{-x}
```

Six parameters: three prefactors $`\sigma_0`$ (with the $`1/T`$ pulled out,
reported in $`\text{S K cm}^{-1}`$) and three activation energies. One parameter
set must fit **all** points at once; that constancy across the surface is the
physical-validity test of the defect-chemical picture. Which channels are
included (`MODEL_CHANNELS`) is an operator decision from defect chemistry,
never a fit outcome: an excluded channel is stored as $`\sigma_0 = 0`$,
$`E_\text{a} = \text{NaN}`$.

### 6.2 Variable projection (VARPRO)

At **fixed** activation energies the model is linear in the prefactors. With
points $`(p_i, T_i, \sigma_i)`$, $`i = 1 \dots N`$, and the selected channels as
columns, the design matrix is

```math
A(E_\text{a}) =
\begin{bmatrix}
\frac{1}{T_1} \exp\left(-\frac{E_\text{a}^{\text{ion}}}{k_B T_1}\right) &
\frac{1}{T_1} \exp\left(-\frac{E_\text{a}^{p\text{-type}}}{k_B T_1}\right) p_1^{+x} &
\frac{1}{T_1} \exp\left(-\frac{E_\text{a}^{n\text{-type}}}{k_B T_1}\right) p_1^{-x} \\
\vdots & \vdots & \vdots \\
\frac{1}{T_N} \exp\left(-\frac{E_\text{a}^{\text{ion}}}{k_B T_N}\right) &
\frac{1}{T_N} \exp\left(-\frac{E_\text{a}^{p\text{-type}}}{k_B T_N}\right) p_N^{+x} &
\frac{1}{T_N} \exp\left(-\frac{E_\text{a}^{n\text{-type}}}{k_B T_N}\right) p_N^{-x}
\end{bmatrix}
\in \mathbb{R}^{N \times n_\text{ch}}
```

that is
$`A_{ic}(E_\text{a}) = T_i^{-1} \exp(-E_{\text{a},c}/k_B T_i)\, p_i^{e_c x}`$
with $`e_c = 0`$ (ionic), $`+1`$ (p-type), $`-1`$ (n-type), so
$`\boldsymbol{\sigma}^\text{model} = A(E_\text{a})\,\mathbf{s}_0`$ with
$`\mathbf{s}_0`$ the prefactor vector. The fit is weighted by $`w_i = 1/\sigma_i`$
(relative residuals, so a weak channel is not drowned by a strong one).

The **inner problem** is convex and solved exactly by weighted NNLS at every
step:

```math
\mathbf{s}_0(E_\text{a}) = \arg\min_{\mathbf{s}_0 \geq 0} \lVert \text{diag}(w)\left(A(E_\text{a})\,\mathbf{s}_0 - \boldsymbol{\sigma}\right) \rVert^2
```

The **outer problem** optimizes only the activation energies, with bounded
least squares (TRF) from a seed of 1 eV:

```math
E_\text{a} = \arg\min_{E_\text{a} \in [0, 3]\,\text{eV}} \lVert \text{diag}(w)\left(A(E_\text{a})\,\mathbf{s}_0(E_\text{a}) - \boldsymbol{\sigma}\right) \rVert^2
```

Eliminating the linear parameters this way removes the
$`\sigma_0`$ versus $`E_\text{a}`$ seesaw direction that traps a naive
six-parameter fit in false minima, guarantees non-negative prefactors by
construction, and reduces the nonlinear search space from 6 to
$`n_\text{ch}`$ dimensions.

A fit is refused outright below `MIN_POINTS` = 8 usable $`(p_{\text{O}_2}, T)`$ points
(`pipeline/model.py::fit_global_conductivity`): with up to 6 free parameters,
fewer points leave no redundancy, and the covariance of 6.3 would be
meaningless rather than merely wide.

### 6.3 Covariance and reported uncertainties

The outer VARPRO solution is used to seed one **full polish**: a bounded
least squares over all $`2 n_\text{ch}`$ parameters
$`\theta = (\mathbf{s}_0, E_\text{a})`$ starting at the VARPRO optimum. Its
Jacobian $`J`$ at convergence gives the Gauss-Newton covariance estimate

```math
\text{Cov}(\theta) \approx (J^\mathsf{T} J)^{-1}\,\frac{2\,\text{SSR}}{\text{dof}}, \qquad \text{dof} = N - \dim\theta
```

computed through the SVD of $`J`$ for rank safety
(`pipeline/model.py::_covariance_errors`); the square roots of the diagonal
are the $`\pm 1\sigma`$ errors quoted on each $`E_\text{a}`$. When a selected
channel is driven to $`\sigma_0 = 0`$ by the NNLS, its $`E_\text{a}`$ is reported
as "not active" instead of a number:
the energy of an absent channel is not identifiable and its CI would be
meaningless.

### 6.4 Derived diagnostics

* Global $`R^2`$ over the whole surface (all conditions, all $`T`$).
* Structureless residual map over $`(p_{\text{O}_2}, T)`$
  (`pipeline/plots.py::plot_fit_residuals`): any remaining pattern flags a
  wrong exponent $`x`$, a missing channel, or process misassignment.
* Conductivity minimum ($`n = p`$ crossover), meaningful only when both
  electronic channels are present:

```math
p_{\text{O}_2}^\text{min}(T) = \left[\frac{\sigma_0^{n\text{-type}}}{\sigma_0^{p\text{-type}}} \exp\left(\frac{E_\text{a}^{p\text{-type}} - E_\text{a}^{n\text{-type}}}{k_B T}\right)\right]^{1/2x}
```

  from setting the p and n terms equal
  (`pipeline/model.py::stoichiometric_pO2`).
* Model-based transference table
  (`pipeline/model.py::global_transference_table`): the
  stage-4 figure redrawn from the single global parameter set instead of
  the per-isotherm NNLS.

---

## 7. Determinism and reproducibility

**Code:** `pipeline/fitting.py::fit_condition_batch`, `pipeline/session.py::update_sample`

Every stochastic element of the chain is seeded deterministically:

* Zarc restart draws: the CRC-32 checksum of the condition name and the
  nominal temperature (`pipeline/fitting.py`), so the same notebook re-run
  reproduces every fit exactly;
* warm-start chains are strictly ordered (descending T within a condition;
  conditions independent, hence parallelizable without changing results);
* all calibrated knobs ($`\lambda`$, windows, weighting) live in `session.json` and
  are re-applied by the batch cells, so a fresh kernel reproduces the
  exported sheets byte for byte, which is what the golden-master test suite
  (`tests/`) asserts.
