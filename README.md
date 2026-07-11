# EIS Analysis Pipeline

Python pipeline for electrochemical impedance spectroscopy of high-resistance
ceramic electrolytes and mixed conductors. It goes from raw Zahner `.ism`
files (or plain CSV spectra) to publication figures and a global conductivity
model in six sequential Jupyter notebooks, with every methodological choice
documented, tested, and reproducible.

| Stage | Notebook | What it does |
| ----- | -------- | ------------ |
| 0 | `stage0_oven.ipynb` | Check that furnace plateaus match the measurement schedule |
| 1 | `stage1_labeling.ipynb` | Match `.ism` files to furnace windows, label temperatures, copy valid files |
| 2 | `stage2_kk.ipynb` | Lin-KK validity test; select the best replica per (condition, T) |
| 3 | `stage3_drt.ipynb` | Tikhonov DRT, peak detection, Zarc equivalent-circuit fit |
| 4 | `stage4_plots.ipynb` | Nyquist, Bode, DRT, Arrhenius, Brouwer p(O₂) figures |
| 5 | `stage5_model.ipynb` | Global MIEC model: one six-parameter fit of σ(p(O₂), T) per process |

EIS spectra of oxide ceramics at 400-600 °C contain overlapping arcs from
bulk conduction, grain boundaries and electrode processes. Separating them
requires validation, deconvolution and circuit fitting in sequence. Each
stage writes its output to disk as an Excel file, so any stage can be re-run
independently, and all shared configuration lives in `session.json`.

---

## Quickstart

```bash
git clone https://github.com/cristian-galeazzi/eis-pipeline.git
cd eis-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab          # or open the folder in VS Code
```

Copy `sample_template/` to `{SAMPLE_ID}/`, drop your data into `Raw data/`
and `Raw oven/`, then run the notebooks in order, stage 0 through stage 5
(stage 5 is optional: it needs several p(O₂) conditions). The first cell of
each notebook shows a numbered sample list, prompts for settings via
`input()` and saves them to `session.json`, so later stages pick up where
the previous one left off. Conditions and temperatures are discovered
automatically from the folder names.

**Try it without data:** the repository bundles `EXAMPLE_SAMPLE/`, a
synthetic sample (two gas conditions, five temperatures, two-Zarc spectra
with realistic noise, regenerable with
`python tools/generate_example_sample.py`). It uses the CSV entry path, so
start from `stage2_kk.ipynb`, type `EXAMPLE_SAMPLE` at the sample prompt,
and continue through stages 3 and 4. Any plausible pellet geometry works
(e.g. thickness 1.4 mm, diameter 10 mm). Without an oven log the Brouwer
p(O₂) figures are skipped; everything else runs end to end.

**Changing parameters (`USE_SAVED_PARAMS`):** once a sample has been
processed, stages 2 to 4 resume their parameters from `session.json`, so
editing a value in the notebook cell has no effect on its own. Set
`USE_SAVED_PARAMS = False` in the configuration cell to write the cell
values to `session.json`, run once, then set it back to `True`. A printout
confirms which source is active. Per-spectrum decisions (frequency cuts in
`kk_overrides`, manual replica `overrides`, per-condition fit tweaks in
`condition_params`) live in separate `session.json` keys and are not
affected by the switch.

All notebooks support temperature-by-temperature processing via `FOCUS_T`;
export cells are merge-aware, so rows for other temperatures are preserved
when you process one at a time.

---

## Stage 0 and Stage 1: from furnace log to labelled spectra

Stage 0 parses the furnace logs and plots temperature against time with
plateau annotations, so schedule mistakes are caught before any analysis.
Stage 1 matches each `.ism` file to its furnace window, verifies the
temperature was stable while the spectrum was measured, assigns the
temperature label and copies valid files to `ISM validation/`. Gas-flow
numbers in folder names are valve setpoints; the actual p(O₂) is always
read from the lambda-probe signal in the furnace log.

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `TABLE_INTERVAL_S` | 300 s | Stage-0 plateau table sampling interval |
| `T_STABILITY_STD` | 1 °C | Max std(T) during measurement |
| `T_PRE_MARGIN_MIN` | 25 min | Stability window before measurement start |
| `T_POST_MARGIN_MIN` | 5 min | Stability window after measurement end |
| `T_ROUND_STEP` | 25 °C | Temperature rounding step |
| `T_PLATEAU_RANGE` | (395, 605) | Valid plateau range [°C]; widen for other ranges |

Status codes: `VALID`, `UNSTABLE`, `NEAR_TRANSITION`, `OUT_OF_RANGE`,
`OUTSIDE_RANGE`. The matching cell prints one summary row per condition and
expands to per-file tables when a mismatch is detected (or with
`SHOW_DETAILS = True`).

---

## Stage 2: Lin-KK validity test

A spectrum is worth fitting only if it describes a causal, linear,
time-invariant system. The Lin-KK test (Schönleber et al. 2014) checks this
by fitting the spectrum with a chain of M RC elements whose time constants
are fixed log-spaced across the measured range, leaving only the weights
free:

```
Z_KK(ω) = R_∞ + Σₖ rₖ / (1 + jωτₖ),   k = 1 … M
```

The chain is Kramers-Kronig consistent by construction, so any systematic
misfit flags a KK violation (drift, nonlinearity, artifacts). Compliance is
judged on the magnitude-normalised residuals: a compliant spectrum leaves
only noise in Δ_re and Δ_im, so each residual vector is tested for normality
with the Shapiro-Wilk statistic (W → 1 for structure-free residuals) and
`kk_score = (W_re + W_im) / 2`. Classification: GREEN (`kk_score ≥ 0.97`),
YELLOW (`≥ 0.90`), RED (excluded downstream). A ceramic-aware dual criterion
(`KK_USE_W_CRITERIA = True`) is looser on Z″, which is intrinsically noisier
in high-impedance ceramics; enable it only when the strict score rejects
spectra that look valid by eye.

Two M-selection modes exist (`KK_USE_BINARY_M`):

- **Automatic** (`True`): smallest M whose residual sign-change fraction μ
  reaches `KK_MU_TARGET` (under-fitting leaves correlated residuals with
  μ ≪ 0.5, over-fitting chases noise with μ → 1). M is scanned linearly
  because μ(M) is not monotonic.
- **Percentage** (default): M = round(`KK_C` × N). KK_C = 0.76 is calibrated
  on the authors' dataset (see `audit/kk_mode_comparison.py` to recalibrate
  on yours); the RelaxIS KK-View default is 0.50.

Edge frequencies whose residuals sit outside an adaptive IQR fence are cut,
and the surviving `f_min`/`f_max` window is stored per (condition, T) and
applied by every later stage. μ is only used to select M; spectrum quality
is always judged by the W statistics.

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `KK_C` | 0.76 | M = round(KK_C × N) in Percentage mode |
| `KK_MU_TARGET` | 0.50 | Sign-change target in automatic mode |
| `KK_F_MIN_HARD` | 80 Hz | Hard lower frequency cutoff |
| `KK_F_MAX_HARD` | None | Hard upper frequency cutoff (None = off) |
| `KK_IQR_FENCE` | 2.0 | IQR fence for the adaptive residual cut |
| `KK_IQR_WINDOW` | 5 | Consecutive clean points anchoring the cut |
| `KK_USE_W_CRITERIA` | False | Ceramic dual criterion |
| `KK_OVERRIDES`, `OVERRIDES` | `{}` | Per-(condition, T) frequency and replica overrides |

---

## Stage 3: DRT deconvolution and Zarc fit

The distribution of relaxation times decomposes the impedance response as

```
Z(ω) = R_∞ + ∫ γ(τ) / (1 + jωτ) d(ln τ)
```

and is obtained by Tikhonov regularisation,
`minimise ‖Aγ − Z‖² + λ‖Lγ‖²` with a 2nd-order derivative operator L
(pyDRTtools, Wan et al. 2015). Each peak of γ(τ) is one relaxation process;
its area over d(ln τ) estimates the process resistance and seeds the circuit
fit (integrating over d(log₁₀τ) instead would underestimate areas by
ln 10 ≈ 2.303):

```
Rᵢ ≈ ∫_peak γ(τ) d(ln τ)
```

The detected peaks then seed a non-linear least-squares fit to a series-Zarc
circuit,

```
Z(ω) = R₀ + Σᵢ Rᵢ / (1 + (jωτᵢ)^αᵢ)
```

where Rᵢ is the arc resistance (the diameter of the i-th depressed
semicircle, i.e. the DC resistance of process i), τᵢ its relaxation time and
αᵢ ∈ (0, 1] the CPE exponent. R and τ are bounded to `ZARC_R_DEC` /
`ZARC_TAU_DEC` decades around their DRT seeds; fits run with deterministic
multi-start (`ZARC_N_RESTARTS`, fixed per-(condition, T) seed) and a
warm-start chain down the temperature ladder. Independent conditions are
fitted in parallel processes; the warm-start chain stays sequential within a
condition, so parallelism does not change the numbers.

For the Zarc parametrization the effective capacitance is exactly
`C_eff = τ/R` (the identity holds for any α; derivation in
`pipeline/fitting.py`), and the conductivity follows from the pellet
geometry, σᵢ = L/(Rᵢ·A) with A = π(D/2)². The relative permittivity is
εᵣ = C_eff·L/(ε₀·A).

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `L_m`, `D_m` | required | Pellet thickness and diameter [m] |
| `DRT_CV_TYPE` | custom | λ selection: `custom` or cross-validation |
| `DRT_RBF_DER` | 2nd order | RBF derivative order (RelaxIS: Derivative) |
| `DRT_SHAPE_S` | 0.5 | RBF shape factor S |
| `DRT_LAMBDA` | 6.5e-6 | Regularisation λ (custom mode) |
| `PEAK_HEIGHT_FRAC` | 0.05 | Height floor as fraction of γ_max |
| `PEAK_MIN_DIST_DECADES` | 0.3 | Minimum peak separation in log τ |
| `PEAK_MIN_PROM_DECADES` | 0 | Log-prominence threshold (0 = off) |
| `N_PEAKS_CAP` | 4 | Keep at most N peaks (largest R first) |
| `N_PEAKS_OVERRIDE` | `{}` | Force N peaks for specific (condition, T) |
| `ZARC_INCLUDE_R0` | False | Include series R₀ in the circuit |
| `ZARC_R0_MAX` | 200 Ω | Upper bound on R₀ |
| `ZARC_R_DEC`, `ZARC_TAU_DEC` | 0.70 | Search window around DRT seeds [decades] |
| `ZARC_ALPHA_INIT` | 0.70 | Initial α per Zarc |
| `ZARC_HF_WEIGHT` | 0 | Extra high-frequency weighting (0 = off) |
| `ZARC_FIX_PARAMS` | `{}` | Pin individual R/τ/α values per (condition, T) |
| `ZARC_N_RESTARTS` | 4 | Random restarts until `ZARC_RMSE_TOL` (0.02) |
| `ZARC_N_JOBS` | 0 | Parallel fit processes (0 = one per core) |

Per-condition overrides tuned in the live panel (`condition_params`,
`zarc_peak_bounds`) persist in `session.json` and are re-applied by the
batch fit, so a fresh kernel reproduces the exported sheets exactly.

**Process identification** is never automatic. Use the C_eff magnitude plot
and the Arrhenius behaviour; starting-point thresholds from Vendrell & West
2018 (YSZ):

| C_eff | Typical process |
| ----- | --------------- |
| < 10⁻¹¹ F | bulk |
| 10⁻¹¹ – 10⁻⁸ F | grain boundary |
| 10⁻⁸ – 10⁻⁶ F | near-electrode |
| > 10⁻⁶ F | electrode |

---

## Stage 4: figures and per-isotherm analysis

Reads the stage-3 outputs and generates publication figures (PNG + PDF):
per condition the DRT stack, Nyquist, Bode and a 2×2 Arrhenius panel; across
conditions the Brouwer p(O₂) diagram and its ionic/electronic decomposition.
Nyquist and Bode figures keep only physically valid points (rows with
Z′ < 0 or Z″ < 0 are high-frequency instrument artifacts; no passive circuit
can produce them).

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `DRT_TAU_MAX` | 0.1 s | x-axis upper limit on the DRT stacked plot |
| `BROUWER_PEAK_ID` | 1 | Peak index for the Brouwer diagram |
| `BROUWER_TEMPS` | None | Temperatures shown in Brouwer (None = all) |
| `ARRHENIUS_T_MIN` | None | Exclude T below this [°C] from Arrhenius fits |
| `ARRHENIUS_SUM_PEAKS` | None | Peaks summed into the HF-block σ Arrhenius |
| `TRANSF_EXPONENT` | 0.25 | Brouwer exponent x for the σ_ion/σ_el split |
| `TRANSF_PEAK_IDS` | None | Peaks shown in transference figures (None = all) |
| `PLOT_WINDOWS` | `{}` | Per-(condition, T) axis crop, kept in `session.json` |

### Arrhenius analysis

Two independent fits per Zarc peak,

```
ln(σT) = ln(A₀) − Eₐᶜᵒⁿᵈ / (k_B T)      conductivity   (slope < 0)
ln(τ)  = ln(τ₀) + Eₐᵖᵒˡ  / (k_B T)      relaxation time (slope > 0)
```

plus a third on ln(C_eff), whose slope must satisfy
Eₐᶜ = Eₐᵖᵒˡ − Eₐᶜᵒⁿᵈ (from ln C = ln τ − ln R): an internal consistency
check, not a new measurement. Energies are reported in eV.

### Ionic/electronic decomposition (per isotherm)

The conductivity of a mixed conductor is the sum of three parallel channels:
ionic (oxygen vacancies, p(O₂)-independent), p-type (holes or small
polarons, increasing with p(O₂)) and n-type (reduction electrons, increasing
as p(O₂) falls). In the dilute defect regime mass action fixes the slopes,
so at each temperature

```
σ(pO₂) = σ_ion + σ_p · pO₂^(+x) + σ_n · pO₂^(−x),      x = TRANSF_EXPONENT
```

is **linear** in the three unknown partial conductivities. The fit is a
single linear least-squares solve under the physical constraint σᵢ ≥ 0,
which is exactly the NNLS algorithm. The constraint matters: where a channel
is genuinely absent, NNLS returns exactly zero instead of a small negative
value chasing noise, so a zero means "not present in the data", not a
measured value. Each isotherm is solved independently and needs at least
four p(O₂) points; the local Brouwer slope equals x·t_el, and the ionic
transference number t_ion = σ_ion/σ_tot is tabulated per peak in
`Results/pO2/stage4_transference.xlsx`. The default x = 1/4 holds in the
dilute regime; use 1/6 where that regime applies.

A second fit, ln(σT) vs 1/T on the per-isotherm σ_ion(T) and σ_p(T), gives
each channel its activation energy. Two straight lines with distinct Eₐ are
the rigorous proof that the decomposition separated two physically different
channels. Temperatures where NNLS returned zero carry no information about
that channel and are skipped, so the point count can differ between σ_ion
and σ_p for the same peak. Transference figures are drawn only for transport
processes (`TRANSF_PEAK_IDS`): t_ion is meaningful for bulk and grain
boundary, not for electrode arcs whose p(O₂) dependence comes from
oxygen-exchange kinetics at the interface.

### HF-block sum (`ARRHENIUS_SUM_PEAKS`)

Below some temperature two close peaks may no longer separate reliably.
Their series resistances still add, so σ = L/((R₁+R₂)·A) stays well defined
at every T. The `Arrhenius_sigma_HF_*` figure draws the separated branches
only for T ≥ `ARRHENIUS_T_MIN` and the series sum over the full range. The
sum mixes processes with different Eₐ, so its line may curve slightly and
its Eₐ is an effective value for the block.

---

## Stage 5: global MIEC model

The stage-4 decomposition treats each isotherm in isolation and drops
temperatures with too few p(O₂) points. Stage 5 removes both limitations by
fitting the whole conductivity surface of each process at once with a
six-parameter model, three prefactors and three activation energies:

```
σ(pO₂,T) = (σ₀_ion/T)·e^(−Eₐ_ion/kT)
         + (σ₀_p  /T)·e^(−Eₐ_p  /kT)·pO₂^(+x)
         + (σ₀_n  /T)·e^(−Eₐ_n  /kT)·pO₂^(−x),      x = MODEL_EXPONENT
```

The six parameters are material constants over the whole (p(O₂), T) surface;
that constancy is itself the validity test. The fit is solved by variable
projection: at fixed activation energies the model is linear in the three
prefactors (exact NNLS, inner problem), so only the three Eₐ are optimised
non-linearly (outer problem), and a final six-parameter polish yields the
uncertainties. Validation: the global R², a structureless residual map over
(p(O₂), T), and the predicted electronic conductivity minimum, which follows
in closed form,

```
ln pO₂_min(T) = (1/2x)·[ ln(σ₀_n/σ₀_p) − (Eₐ_n − Eₐ_p)/(k_B T) ]
```

and whose migration with T (slope set by Eₐ_n − Eₐ_p) is an independent
check against the data.

| Parameter | Default | Purpose |
| --------- | ------- | ------- |
| `MODEL_PEAK_IDS` | `[]` | Peaks to fit (`[]` = all) |
| `MODEL_EXPONENT` | 0.25 | Brouwer exponent x (1/4 dilute, 1/6 otherwise) |
| `MODEL_CONDITIONS` | `[]` | Conditions (pressures) to include (`[]` = all) |
| `MODEL_T_MIN`, `MODEL_T_MAX` | None | Temperature window for the fit [°C] |
| `MODEL_CHANNELS` | `["ion","p","n"]` | Channels of the model to fit (subset) |

Which channels exist (`MODEL_CHANNELS`) is an operator decision based on
defect-chemistry reasoning, not a fit outcome: drop `n` when the measured
p(O₂) window is never reducing enough to create electrons. The fit never
adds or removes channels on its own; an excluded channel is stored with
σ₀ = 0 and Eₐ = NaN (all nine parameter columns are kept for schema
stability) and appears in no figure. A selected channel that NNLS drives to
σ₀ = 0 is reported as `not active (s0 = 0)` instead of a meaningless
activation energy, and the conductivity-minimum prediction is reported only
when both electronic channels are present. Outputs:
`Results/pO2/stage5_model.xlsx`, the `Stage5_*` figures, and
`session.json → stage5_params`.

---

## Non-Zahner instruments

Skip stages 0 and 1: place your spectra as CSV in
`{sample_id}/input_spectra/{condition}/` and start from stage 2. A template
with the expected structure is in `sample_template/input_spectra/`.

```
{sample_id}/input_spectra/Ar_200_O2_10_700_300_50/
├── SampleID_Ar_200_O2_10_300C.csv
├── SampleID_Ar_200_O2_10_400C.csv
├── SampleID_Ar_200_O2_10_400C_1.csv   ← replica 2
└── SampleID_Ar_200_O2_10_700C.csv
```

The condition folder follows the same naming convention as `Raw data/`
(`{prefix}_{gas_flows}_{T_max}_{T_min}_{T_step}`; the gas section is
recognised from the first `Ar`/`O2`/`N2`/`H2` token). The file prefix before
`_{T}C` is free. CSV format, separator comma/semicolon/tab:

```
freq,Z_re,Z_im,temperature
100000,5.3,0.2,400
```

`Z_im` must be **positive** in the capacitive region (BioLogic EC-Lab
exports −Im(Z): multiply by −1 before saving). Temperature in the filename
(`_{T}C`) and a `temperature` column are required for Arrhenius analysis
(≥ 3 temperatures); Nyquist, Bode, DRT and the Zarc fit work without them.
The Brouwer p(O₂) analysis always requires lambda-probe data from
stages 0-1.

---

## Folder layout

```
eis-pipeline/
│
├── pipeline/               ← calculation engine and helpers
│   ├── ingest.py           ← .ism reader + CSV entry point
│   ├── matching.py         ← furnace log parser and T / p(O₂) matcher
│   ├── quality.py          ← Lin-KK implementation
│   ├── drt.py              ← DRT computation and peak extraction
│   ├── fitting.py          ← Zarc circuit fitting, conductivity, C_eff
│   ├── model.py            ← stage-5 global MIEC conductivity model
│   ├── plots.py            ← publication figures
│   ├── interactive.py      ← ipywidgets helpers (UI only, no physics)
│   ├── session.py          ← session.json persistence (atomic, merge-safe)
│   ├── _worker.py          ← process-pool init (caps BLAS threads)
│   └── utils.py            ← Excel helpers
│
├── stage0_oven.ipynb … stage5_model.ipynb   ← the six pipeline stages
│
├── audit/                  ← runnable calibration procedures (see below)
├── tests/                  ← golden-master + known-answer test suites
├── tools/                  ← synthetic example-sample generator
├── EXAMPLE_SAMPLE/         ← bundled synthetic sample (CSV entry path)
├── sample_template/        ← copy and rename to {SAMPLE_ID}/ to start
│
└── {SAMPLE_ID}/            ← your data, never committed
    ├── Raw data/           ← never modified
    ├── Raw oven/           ← never modified
    ├── ISM validation/     ← created by stage 1
    └── Results/
        ├── {condition}/    ← stage1_labeling / stage2_kk / stage3_drt /
        │                     stage3_fit .xlsx + DRT, Nyquist-Bode,
        │                     Arrhenius figure folders
        └── pO2/            ← Brouwer + transference + stage5_model.xlsx
```

**Requirements:** Python ≥ 3.10 and the pinned `requirements.txt`:
`zahner_analysis` (.ism reader), `pyDRTtools` (DRT), `impedance` (circuit
fit), `scipy`/`numpy`, `pandas`/`openpyxl` (Excel I/O), `matplotlib`,
`ipywidgets` (interactive panels, stages 0-5).

---

## Reproducibility and transparency

Dependencies are pinned in `requirements.txt`. Every `.xlsx` output includes
a `Metadata` sheet with all parameter values and installed library versions,
so any result can be traced back to the exact configuration that produced
it. Fits are deterministic: restart guesses use a fixed per-(condition, T)
seed, and parallelism does not change the numbers.

The test suite (`pytest tests/`) covers three layers: the calculation engine
(`test_engine_golden.py`: C_eff = τ/R identity, synthetic Zarc recovery,
Lin-KK, DRT areas, CSV ingestion, session merge), the stage-5 global model
(`test_model_golden.py`: synthetic-surface recovery, degenerate and
reduced-channel cases), and every calibration procedure in `audit/`
(`test_audit_*.py`: known-answer tests on the bundled synthetic sample).

The default parameter values shipped in the notebooks (DRT regularization
strength, peak cap, Lin-KK M-selection mode, fit seed windows) were not
guessed: each was chosen by a documented calibration procedure, and those
procedures live in [`audit/`](audit/README.md) as runnable scripts. They are
dataset-calibrated starting points, not universal constants: a different
material, geometry or frequency window can legitimately prefer different
values, and the audit scripts exist precisely to find them on your own data.
No measured spectrum, fitted parameter or ranking derived from real data
enters the repository: the audit scripts write to the gitignored
`audit/output/`.

---

## Authorship, data and privacy

Scientific work, experimental design, sample preparation, calibration
choices and methodological decisions are the author's own. Python
implementation developed with AI coding assistance (Anthropic Claude) from
the author's specifications; all code reviewed and validated against
experimental data.

This repository contains code only. No measurement data, fitted parameters
or sample identifiers are committed. Sample names are entered at runtime via
`input()` and stored locally in `session.json` (gitignored; saves are atomic
with a `.bak` copy, and per-condition parameters are merged rather than
overwritten, so re-running one stage cannot erase calibrations from
another). Raw `.ism` files and furnace logs are never modified; results are
written to `{sample_id}/Results/`. No network calls are made.

---

## References

**Lin-KK**

- M. Schönleber, D. Klotz, E. Ivers-Tiffée, *A method for improving the robustness of linear Kramers-Kronig validity tests*, Electrochim. Acta 131 (2014) 20-27.
- B. A. Boukamp, *A linear Kronig-Kramers transform test for immittance data validation*, J. Electrochem. Soc. 142 (1995) 1885-1894.

**DRT**

- T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, *Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools*, Electrochim. Acta 184 (2015) 483-499.
- A. Maradesa, B. Py, T. H. Wan, M. B. Effat, F. Ciucci, *Selecting the regularization parameter in the distribution of relaxation times*, J. Electrochem. Soc. 170 (2023) 030502. (Cited for the automatic λ-selection methods exposed by `DRT_CV_TYPE`; the validated default uses a fixed λ.)

**Equivalent circuits**

- J. T. S. Irvine, D. C. Sinclair, A. R. West, *Electroceramics: characterization by impedance spectroscopy*, Adv. Mater. 2 (1990) 132-138.
- X. Vendrell, A. R. West, *Electrical properties of yttria-stabilized zirconia, YSZ single crystal: local AC and long range DC conduction*, J. Electrochem. Soc. 165 (2018) F966-F975.

**Software used by the engine**

- M. D. Murbach, B. Gerwe, N. Dawson-Elli, L.-k. Tsui, *impedance.py: A Python package for electrochemical impedance analysis*, J. Open Source Softw. 5(52) (2020) 2349.
- pyDRTtools (Ciucci group), https://github.com/ciuccislab/pyDRTtools — implementation of the DRT method of Wan et al. (2015).

**Scientific computing**

- G. Wilson et al., *Best Practices for Scientific Computing*, PLoS Biology 12(1) (2014) e1001745.
- G. Wilson et al., *Good Enough Practices in Scientific Computing*, PLoS Computational Biology 13(6) (2017) e1005510.
- A. Scopatz, K. D. Huff, *Effective Computation in Physics*, O'Reilly Media (2015).

---

## License

MIT, see [LICENSE](LICENSE).

## How to cite

Citation metadata are in [CITATION.cff](CITATION.cff); GitHub shows a "Cite
this repository" button generated from it. A Zenodo DOI will be added with
the first public release.
