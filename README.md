# EIS Analysis Pipeline

Python pipeline for EIS analysis of high-resistance ceramic electrolytes. Goes from raw
Zahner `.ism` files to publication figures in six sequential Jupyter notebooks.

---

## Authorship

Scientific work, experimental design, sample preparation, calibration choices and
methodological decisions are the author's own. Python implementation developed with
AI coding assistance (Anthropic Claude) from the author's specifications; all code
reviewed and validated against experimental data.

---

## Data and privacy

This repository contains code only. No measurement data, fitted parameters or sample
identifiers are committed. Sample names are entered at runtime via `input()` and stored
locally in `session.json` (gitignored). Saves are atomic, the previous version is
kept as `session.json.bak`, and per-condition parameters are merged rather than
overwritten, so re-running one stage cannot erase calibrations from another.
Raw `.ism` files and furnace logs are never
modified; results are written to `{sample_id}/Results/`. No network calls are made.

---

## Quickstart

```bash
git clone https://github.com/cristian-galeazzi/eis-pipeline.git
cd eis-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab          # or open the folder in VS Code
```

Copy `sample_template/` to `{SAMPLE_ID}/`, drop your data into `Raw data/` and
`Raw oven/`, then run the notebooks in order: stage0 → stage1 → stage2 → stage3 → stage4.
Each notebook starts with a numbered sample list and saves its settings to
`session.json`, so later stages pick up where the previous one left off.

### Try it without data

The repository bundles `EXAMPLE_SAMPLE/`, a synthetic sample (two gas
conditions, five temperatures, two-Zarc spectra with realistic noise,
regenerable with `python tools/generate_example_sample.py`). It uses the CSV
entry path, so start from `stage2_kk.ipynb`, type `EXAMPLE_SAMPLE` at the
sample prompt, and continue through stage 3 and 4. When stage 3 asks for the
pellet geometry, any plausible values work (e.g. thickness 1.4 mm, diameter
10 mm). No oven log is included, so the Brouwer p(O₂) figures are skipped;
everything else runs end to end.

### Changing parameters: `USE_SAVED_PARAMS`

Once a sample has been processed, Stages 2, 3 and 4 resume their parameters
from `session.json`; editing a value in the notebook cell has no effect on its
own. The `USE_SAVED_PARAMS` switch at the top of each configuration cell makes
this explicit:

- `True` (default): resume the parameters saved in `session.json`. Normal use:
  your last calibration is picked up automatically.
- `False`: use the values written in the notebook cell and save them to
  `session.json`. To change parameters: edit the values, set `False`, run once,
  then set it back to `True`.

A printout in the configuration cell confirms which source is active.
Per-spectrum decisions (frequency cuts in `kk_overrides`, manual replica
`overrides`, per-condition fit tweaks in `condition_params`) live in separate
`session.json` keys and are not affected by the switch.

---

## Requirements

Python ≥ 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

| Package                  | Purpose                           |
| ------------------------ | --------------------------------- |
| `zahner_analysis`      | Read Zahner `.ism` binary files |
| `pyDRTtools`           | DRT via Tikhonov regularisation   |
| `impedance`            | Zarc equivalent-circuit fitting   |
| `pandas`, `openpyxl` | Excel I/O                         |
| `scipy`, `numpy`     | Numerics                          |
| `matplotlib`           | Figures                           |
| `ipywidgets`           | Interactive panels (stages 0–4)  |

---

## Folder layout

```
eis-pipeline/
│
├── pipeline/               ← Python modules
│   ├── ingest.py           ← ISM file reader
│   ├── matching.py         ← Furnace log parser and T / pO₂ matcher
│   ├── quality.py          ← Lin-KK implementation
│   ├── drt.py              ← DRT computation and peak extraction
│   ├── fitting.py          ← Zarc circuit fitting
│   ├── plots.py            ← Publication figures
│   ├── interactive.py      ← ipywidgets helpers (UI only)
│   ├── session.py          ← session.json persistence (atomic, merge-safe)
│   ├── _worker.py          ← process-pool init (caps BLAS threads per worker)
│   └── utils.py            ← Excel helpers
│
├── stage0_oven.ipynb       ← Stage 0: furnace log diagnostics
├── stage1_labeling.ipynb   ← Stage 1: measurement identification
├── stage2_kk.ipynb         ← Stage 2: Lin-KK quality assessment
├── stage3_drt.ipynb        ← Stage 3: DRT and Zarc fitting
├── stage4_plots.ipynb      ← Stage 4: publication figures
│
├── sample_template/        ← copy and rename to {SAMPLE_ID}/ to start
│   ├── Raw data/
│   │   └── Ar_200_O2_10_700_300_50/   ← rename to your condition
│   ├── Raw oven/
│   └── input_spectra/      ← non-Zahner entry point (see section below)
│       └── Ar_200_O2_10_700_300_50/
│
├── tests/
├── README.md
├── requirements.txt
│
└── {SAMPLE_ID}/
    ├── Raw data/           ← never modified
    ├── Raw oven/           ← never modified
    ├── ISM validation/     ← created by Stage 1
    └── Results/
        ├── {condition}/
        │   ├── stage1_labeling.xlsx
        │   ├── stage2_kk.xlsx
        │   ├── stage3_drt.xlsx
        │   ├── stage3_fit.xlsx
        │   ├── DRT/
        │   ├── Nyquist-Bode/
        │   └── Arrhenius/
        └── pO2/
```

**Condition folder naming:** `{prefix}_{gas_flows}_{T_max}_{T_min}_{T_step}`

The `{prefix}` can be anything (e.g. `MyOxide_A`). The pipeline identifies the gas section automatically from the first recognised gas token (`Ar`, `O2`, `N2`, `H2`). Gas-flow numbers are valve setpoints; actual p(O₂) is always read from the lambda-probe signal in the furnace log.

---

## How it works

EIS spectra of oxide ceramics at 400–600 °C contain overlapping arcs from bulk conduction, grain boundaries and electrode processes. Separating them requires validation, deconvolution and circuit fitting in sequence. Each stage writes its output to disk as an Excel file so any stage can be re-run independently.

| Stage | Notebook                  | What it does                                                                  |
| ----- | ------------------------- | ----------------------------------------------------------------------------- |
| 0     | `stage0_oven.ipynb`     | Check that furnace plateaus match the measurement schedule                    |
| 1     | `stage1_labeling.ipynb` | Match `.ism` files to furnace windows, label temperatures, copy valid files |
| 2     | `stage2_kk.ipynb`       | Lin-KK validation; select best replica per (condition, T)                     |
| 3     | `stage3_drt.ipynb`      | Tikhonov DRT, peak detection, Zarc circuit fit                                |
| 4     | `stage4_plots.ipynb`    | Nyquist, Bode, DRT stacked, Arrhenius, Brouwer p(O₂)                         |
| 5     | `stage5_model.ipynb`    | Global MIEC model: fit σ(p(O₂),T) per peak with 6 parameters; validate       |

---

## Usage

1. Open each notebook in order in JupyterLab or VS Code.
2. Run **Kernel → Restart and Run All Cells**. The first cell prompts for `sample_id`
   and other settings and saves them to `session.json` for the next stages.
3. Check the inline output; adjust overrides if needed and re-run.

All notebooks support temperature-by-temperature processing via `FOCUS_T`. The export cells are merge-aware: rows for other temperatures are preserved when you process one at a time.

---

## Non-Zahner instruments

Skip Stage 0 and Stage 1. Place your spectra in `{sample_id}/input_spectra/`
and start from Stage 2. A template with the expected structure is in
`sample_template/input_spectra/`.

**Minimum requirement:** temperature in the filename (`_{T}C`) and a
`temperature` column in the CSV. Without temperature, Arrhenius analysis
is not possible and the pipeline produces only Nyquist, Bode, DRT and Zarc fit.

### Folder and file naming

```
{sample_id}/
└── input_spectra/
    └── Ar_200_O2_10_700_300_50/
        ├── SampleID_Ar_200_O2_10_300C.csv
        ├── SampleID_Ar_200_O2_10_400C.csv
        ├── SampleID_Ar_200_O2_10_400C_1.csv   ← replica 2
        └── SampleID_Ar_200_O2_10_700C.csv
```

The folder name follows the same convention as Raw data conditions.
The file prefix before `_{T}C` is free — include SampleID and condition
for traceability.

### CSV format

```
freq,Z_re,Z_im,temperature
100000,5.3,0.2,400
10000,7.5,2.4,400
1000,30.2,44.8,400
```

Separator: comma, semicolon, or tab. `Z_im` must be **positive** in the
capacitive region. BioLogic EC-Lab exports −Im(Z): multiply by −1 before saving.

| Feature                      | Available                                             |
| ---------------------------- | ----------------------------------------------------- |
| Nyquist, Bode, DRT, Zarc fit | always                                                |
| Arrhenius plots              | requires `temperature` column and ≥ 3 temperatures |
| Brouwer p(O₂)               | never (requires lambda-probe data from Stage 0–1)    |

---

## Stage 0 - `stage0_oven.ipynb`

Parses furnace logs and plots T vs time with plateau annotations.

| Parameter            | Default | Purpose                         |
| -------------------- | ------- | ------------------------------- |
| `TABLE_INTERVAL_S` | 300 s   | Plateau table sampling interval |

`sample_id` is picked from a numbered list via `input()`. `TABLE_INTERVAL_S` is set
with a small widget in the config cell and saved to `session.json` on every change.
Conditions are selected with the checkbox panel below the import cell; the selection
is saved to `session.json` and reused by Stage 1.

---

## Stage 1 - `stage1_labeling.ipynb`

Matches each `.ism` file to its furnace window, assigns a temperature label and copies valid files to `ISM validation/`.

| Parameter             | Default    | Purpose                                   |
| --------------------- | ---------- | ----------------------------------------- |
| `T_STABILITY_STD`   | 1 °C      | Max std(T) during measurement             |
| `T_PRE_MARGIN_MIN`  | 25 min     | Stability window before measurement start |
| `T_POST_MARGIN_MIN` | 5 min      | Stability window after measurement end    |
| `T_ROUND_STEP`      | 25 °C     | Temperature rounding step                 |
| `T_PLATEAU_RANGE`   | (395, 605) | Valid plateau range [°C]                 |

`sample_id` and `conditions` are read from `session.json` (set in stage 0): this
stage has no condition selector of its own. The matching cell prints one summary
row per condition; the full per-file tables appear automatically when a temperature
or ordering mismatch is detected, or on demand with `SHOW_DETAILS = True`.

Status codes: `VALID`, `UNSTABLE`, `NEAR_TRANSITION`, `OUT_OF_RANGE`, `OUTSIDE_RANGE`.

> **Non-standard temperature ranges:** `T_PLATEAU_RANGE` defaults to (395, 605) °C.
> Measurements outside this range are classified `OUTSIDE_RANGE` and excluded.
> Adjust both `T_PLATEAU_RANGE` and `T_ROUND_STEP` in the config cell if working
> at different temperatures (e.g. room temperature or above 600 °C).

---

## Stage 2 - `stage2_kk.ipynb`

Applies the Lin-KK test [Schönleber et al., 2014] to each spectrum and selects the best replica per (condition, T).

| Parameter             | Default | Purpose                                    |
| --------------------- | ------- | ------------------------------------------ |
| `KK_C`              | 0.76    | M = round(KK_C × N)                       |
| `KK_MU_TARGET`      | 0.50    | Sign-change fraction target                |
| `KK_F_MIN_HARD`     | 80 Hz   | Hard lower frequency cutoff                |
| `KK_F_MAX_HARD`     | None    | Hard upper frequency cutoff (None = off)   |
| `KK_IQR_FENCE`      | 2.0     | IQR fence for the adaptive residual cut    |
| `KK_IQR_WINDOW`     | 5       | Consecutive clean points anchoring the cut |
| `KK_USE_W_CRITERIA` | False   | Ceramic-aware dual criterion (W_re + W_im) |
| `KK_OVERRIDES`      | `{}`  | Per-(condition, T) frequency overrides     |
| `OVERRIDES`         | `{}`  | Manual replica selection                   |

Classification: GREEN (`kk_score >= 0.97`), YELLOW (`>= 0.90`), RED.

---

## Stage 3 - `stage3_drt.ipynb`

Computes the DRT γ(τ) via Tikhonov regularisation, detects peaks and fits a Zarc equivalent circuit seeded by the DRT peaks.

```
Z(ω) = R₀ + Σᵢ Rᵢ / (1 + (j ω τᵢ)^αᵢ)
```

| Parameter                 | Default  | Purpose                                     |
| ------------------------- | -------- | ------------------------------------------- |
| `L_m`, `D_m`          | required | Pellet thickness and diameter [m]           |
| `DRT_CV_TYPE`           | custom   | λ selection: `custom` or cross-validation  |
| `DRT_RBF_DER`           | 2nd order | RBF derivative order (RelaxIS: Derivative) |
| `DRT_SHAPE_S`           | 0.5      | RBF shape factor S                          |
| `DRT_LAMBDA`            | 6.5e-6   | Regularisation λ (custom mode)             |
| `PEAK_MIN_PROM_DECADES` | 0        | Log-prominence threshold (0 = off)          |
| `PEAK_HEIGHT_FRAC`      | 0.05     | Height floor as fraction of γ_max          |
| `PEAK_MIN_DIST_DECADES` | 0.3      | Minimum peak separation in log τ           |
| `N_PEAKS_OVERRIDE`      | `{}`   | Force N peaks for specific (condition, T)   |
| `N_PEAKS_CAP`           | None     | Keep at most N peaks (tallest by γ)        |
| `ZARC_INCLUDE_R0`       | False    | Include series R₀ in circuit               |
| `ZARC_R0_MAX`           | 200 Ω   | Upper bound on R₀                          |
| `ZARC_R_DEC`, `ZARC_TAU_DEC` | 0.70 | Search window around DRT seeds [decades] |
| `ZARC_ALPHA_INIT`       | 0.70     | Initial α per Zarc                         |
| `ZARC_HF_WEIGHT`        | 0        | Extra high-frequency weighting (0 = off)    |
| `ZARC_FIX_PARAMS`       | `{}`   | Pin individual R/τ/α values per (cond, T) |
| `ZARC_N_RESTARTS`       | 4        | Random restarts until `ZARC_RMSE_TOL`      |
| `ZARC_N_JOBS`           | 0        | Parallel fit processes (0 = auto)           |

Per-condition overrides tuned in the live panel (`condition_params`,
`zarc_peak_bounds`) persist in `session.json` and are re-applied by the
batch fit, so a fresh kernel reproduces the exported sheets exactly.

**Process identification:** the pipeline assigns no process label automatically. Use the C_eff magnitude plot (log₁₀(C_eff) vs 1000/T) and Arrhenius behaviour to identify each peak. Starting-point thresholds from Vendrell & West 2018 (YSZ):

| C_eff                | Typical process |
| -------------------- | --------------- |
| < 10⁻¹¹ F         | bulk            |
| 10⁻¹¹ – 10⁻⁸ F | grain boundary  |
| 10⁻⁸ – 10⁻⁶ F   | near-electrode  |
| > 10⁻⁶ F           | electrode       |

---

## Stage 4 - `stage4_plots.ipynb`

Reads Stage 3 outputs and generates publication figures (PNG + PDF).

| Parameter            | Default  | Purpose                                     |
| -------------------- | -------- | ------------------------------------------- |
| `L_m`, `D_m`     | required | Pellet thickness and diameter [m]           |
| `DRT_TAU_MAX`      | 0.1 s    | x-axis upper limit on DRT stacked plot      |
| `BROUWER_PEAK_ID`  | 1        | Peak index for Brouwer diagram              |
| `BROUWER_TEMPS`    | None     | Temperatures shown in Brouwer (None = all)  |
| `ARRHENIUS_T_MIN`  | None     | Exclude T below this [°C] from Arrhenius fits |
| `ARRHENIUS_SUM_PEAKS` | None  | Peaks summed into the HF-block σ Arrhenius (e.g. `[1, 2]`) |
| `TRANSF_EXPONENT`  | 0.25     | Brouwer exponent x for the σ_ion/σ_el split |
| `TRANSF_PEAK_IDS`  | None     | Peaks shown in transference figures (None = all) |
| `PLOT_WINDOWS`     | `{}`   | Per-(condition, T) axis crop                |

### What the figures show

Per condition: DRT stacked, Nyquist, Bode, and the 2×2 Arrhenius panel with
all fitted peaks. The R²(τ) of each Arrhenius fit appears in the
activation-energy summary table. Across conditions: the Brouwer p(O₂)
diagram and its ionic/electronic decomposition (Step 3).

Nyquist and Bode figures keep only physically valid points. Rows with
Z′ < 0 or Z″ < 0 are high-frequency instrument artifacts, since no passive
circuit can produce them. They are removed with the same criterion used
before the Lin-KK test, and removing them was verified not to change the
fitted parameters.

Transference figures are drawn only for the peaks in `TRANSF_PEAK_IDS`.
The reason is physical. t_ion makes sense for transport processes such as
bulk and grain boundary. It does not for electrode arcs, whose pO₂
dependence comes from the oxygen-exchange kinetics at the interface. The
exported table still covers every peak.

Axis crops live in `PLOT_WINDOWS` (`session.json → stage4_params`), one
entry per (condition, T), and survive notebook restarts. Stage 4 needs
stage 3 first: it reads the pellet geometry and the fit results from there,
and says so explicitly if they are missing.

### Ionic/electronic decomposition (Step 3)

The conductivity of a mixed conductor is the sum of three parallel channels:
an ionic one (oxygen vacancies, p(O₂)-independent), an n-type electronic one
(reduction electrons, increasing as p(O₂) falls) and a p-type one (holes /
small polarons, increasing as p(O₂) rises). In the dilute defect regime mass
action fixes the slopes, so at each fixed temperature:

```
σ(pO₂) = σ_ion + σ_p · pO₂^(+x) + σ_n · pO₂^(−x)        x = TRANSF_EXPONENT
```

with the three partial conductivities σ_ion, σ_p, σ_n the unknowns. Because
the right-hand side is **linear** in those three unknowns, the fit is not a
non-linear optimisation but a single linear least-squares solve, with one
physical constraint: a conductivity cannot be negative (σᵢ ≥ 0). The algorithm
that solves a linear least-squares problem under a non-negativity constraint is
**NNLS** (non-negative least squares). Two consequences of that constraint
matter physically:

- where a channel is genuinely absent (e.g. no n-type contribution at high
  p(O₂)), NNLS sets its σᵢ to **exactly zero** instead of fitting a small
  negative value to chase noise. A zero therefore means "this channel is not
  present in the data at this temperature", not a measured value;
- the fit is done **independently at each temperature** (one isotherm at a
  time), which needs at least four p(O₂) points per temperature; temperatures
  with fewer points are skipped.

The local Brouwer slope equals `x · t_el`: a plateau means purely ionic
conduction, a +x slope means purely p-type (polaron) conduction. The ionic
transference number t_ion = σ_ion / σ_tot is tabulated for every peak in
`Results/pO2/stage4_transference.xlsx`. The default x = 1/4 holds in the dilute
defect regime; use 1/6 where that regime applies.

Step 3 then takes the per-temperature σ_ion(T) and σ_p(T) from the NNLS solves
and runs a second fit, the Arrhenius plot ln(σT) vs 1000/T, one figure per peak
in `TRANSF_PEAK_IDS`. The slope of each line gives that channel's activation
energy (Eₐ = −slope·k_B): the ionic Eₐ from σ_ion, the hole Eₐ from σ_p. Two
straight lines with distinct Eₐ are the rigorous proof that the decomposition
separated two physically different channels. Temperatures where NNLS returned
exactly zero for a channel carry no information about it and are skipped, so the
`n` in the legend (the number of temperatures actually used) can differ between
σ_ion and σ_p for the same peak. σ_n is not drawn (in p-type samples it is a
noise floor) but stays in the exported table.

This two-step route (NNLS per isotherm, then Arrhenius per channel) drops
temperatures with too few p(O₂) points and treats each isotherm in isolation.
Stage 5 removes both limitations by fitting the whole σ(p(O₂),T) surface at
once with a single six-parameter model.

### HF-block sum (`ARRHENIUS_SUM_PEAKS`)

Below some temperature two close peaks may no longer separate reliably.
Their series resistances still add, so σ = L/((R₁+R₂)·A) stays well defined
at every T. The figure `Arrhenius_sigma_HF_*` draws the separated branches
only for T ≥ `ARRHENIUS_T_MIN` and the series sum over the full range, and
it declares the threshold on the plot. One caveat: the sum mixes processes
with different Eₐ, so its line may curve slightly and its Eₐ is an
effective value for the block.

### Stage 5 - global MIEC model

Reads the fitted σ per peak from `stage3_fit.xlsx` across conditions and fits
the whole conductivity surface σ(p(O₂), T) of each process with one global
six-parameter model (the "backward" approach: from data to parameters).

| Parameter          | Default | Purpose                                          |
| ------------------ | ------- | ------------------------------------------------ |
| `MODEL_PEAK_IDS`   | `[]`    | Peaks to fit (`[]` = all)                        |
| `MODEL_EXPONENT`   | 0.25    | Brouwer exponent x (1/4 dilute, 1/6 otherwise)   |
| `MODEL_CONDITIONS` | `[]`    | Conditions (pressures) to include (`[]` = all)   |
| `MODEL_T_MIN`      | None    | Exclude T below this [°C] from the fit           |
| `MODEL_T_MAX`      | None    | Exclude T above this [°C] from the fit           |
| `MODEL_CHANNELS`   | `["ion","p","n"]` | Channels of the model to fit (subset)  |

Unlike the two-step Stage 4 decomposition (per-isotherm NNLS, then per-channel
Arrhenius), which drops temperatures with too few p(O₂) points, the global fit
uses every selected point at once and shares the six parameters over the whole
surface. The fit is solved by variable projection: at fixed activation energies
the model is linear in the three prefactors (exact non-negative NNLS, inner
problem), and only the three Eₐ are optimised non-linearly (outer problem); a
final six-parameter polish yields the uncertainties. Validation: the global R²,
a structureless residual map over (p(O₂), T), and the predicted conductivity
minimum p(O₂)_min(T). Use `MODEL_T_MIN`/`MODEL_CONDITIONS` to restrict the fit
to the window where the ionic/electronic separation is physically reliable.
Outputs: `Results/pO2/stage5_model.xlsx` (Parameters, Residuals, Metadata),
the `Stage5_*` figures, and `session.json → stage5_params` (per peak).

Which channels exist (`MODEL_CHANNELS`) is an operator decision based on
Brouwer/defect-chemistry reasoning, not a fit outcome: e.g. drop `n` when the
measured p(O₂) window is never reducing enough to create electrons. The fit
never adds or removes channels on its own; an excluded channel is stored with
σ₀ = 0 and Eₐ = NaN (all nine parameter columns are kept for schema
stability), contributes nothing to the model surface, and appears in no
figure. A selected channel that NNLS drives to σ₀ = 0 is reported as
`not active (s0 = 0)` instead of a meaningless activation energy, and the
conductivity-minimum prediction is reported only when both electronic
channels are present.

---

## Physical formulae

The equations below follow the pipeline execution order.

### Stage 2 - Lin-KK validity test

Each spectrum is tested for Kramers-Kronig compliance using M RC elements
(Schönleber et al. 2014):

```
Z_KK(ω) = R_∞ + Σₖ rₖ / (1 + jωτₖ),   k = 1 … M
```

τₖ are log-spaced across the measured frequency range; rₖ are fitted by
least squares. Two M-selection modes are available via `KK_USE_BINARY_M`:

- **Automatic scan** (`KK_USE_BINARY_M = True`): finds the smallest M such
  that the sign-change fraction μ of adjacent RC weights satisfies
  μ ≥ `KK_MU_TARGET` (default 0.50). M is scanned linearly from 3 upward
  because μ(M) is not monotonic, so a bisection could skip valid values.
  Adaptive per spectrum.
- **Fixed ratio** (default): M = round(`KK_C` × N), with KK_C = 0.76
  calibrated on the author's dataset. Faster but spectrum-independent.

Compliance is assessed on the magnitude-normalised residuals:

```
Δ_re(ω) = (Z′_meas − Z′_KK) / |Z_meas|
Δ_im(ω) = (Z″_meas − Z″_KK) / |Z_meas|
```

A KK-compliant spectrum leaves only measurement noise in Δ, so each residual
vector is tested for normality with the Shapiro-Wilk test. Its statistics
W_re and W_im approach 1 for gaussian (structure-free) residuals, and the
overall score is their average: `kk_score = (W_re + W_im) / 2`.

Two classification criteria are available via `KK_USE_W_CRITERIA`:

- **Strict** (default, `False`): GREEN if `kk_score >= 0.97`, YELLOW if
  `>= 0.90`, RED otherwise. Best for clean spectra.
- **Ceramic dual criterion** (`True`): GREEN if `W_re >= 0.95` AND
  `W_im >= 0.93` with at most 20% of points removed by the edge cutoffs
  (YELLOW: `W_re >= 0.90`, `W_im >= 0.88`, 40%). Looser on Z″ because
  high-impedance ceramics have intrinsically noisier imaginary parts;
  enable it only when the strict score rejects spectra that look valid by eye.

Note that μ (the sign-change fraction) is used only to select M; spectrum
quality is always judged by the W statistics. RED spectra are excluded from
downstream analysis.

### Stage 3 - Distribution of Relaxation Times

The DRT γ(τ) decomposes the impedance response as:

```
Z(ω) = R_∞ + ∫ γ(τ) / (1 + jωτ) d(ln τ)
```

γ(τ) is normalised with respect to d(ln τ). Peak areas used as resistance
seeds for the Zarc fit are therefore:

```
Rᵢ ≈ ∫_peak γ(τ) d(ln τ)
```

integrated numerically over ln(τ), consistent with the pyDRTtools kernel
(Wan et al. 2015). Using d(log₁₀τ) instead would underestimate areas by
a factor of ln(10) ≈ 2.303.

γ(τ) is obtained by Tikhonov regularisation with 2nd-order derivative operator L:

```
minimise  ‖Aγ − Z‖² + λ ‖Lγ‖²
```

λ is set via `DRT_LAMBDA` (custom mode) or selected automatically by GCV.

### Stage 3 - Zarc equivalent circuit

DRT peaks seed a non-linear least-squares fit to:

```
Z(ω) = R₀ + Σᵢ Rᵢ / (1 + (jωτᵢ)^αᵢ)
```

where R₀ is the series resistance, Rᵢ the arc resistance — the diameter of
the i-th depressed semicircle in the Nyquist plot, i.e. the DC resistance
contribution of process i, from which σᵢ = L/(Rᵢ·A) and C_eff,ᵢ = τᵢ/Rᵢ
follow — τᵢ the relaxation time, and αᵢ ∈ (0,1] the CPE exponent of the
i-th process.

Fits run with multi-start (`ZARC_N_RESTARTS`) and a warm-start chain down
the temperature ladder of each condition. Restart guesses use a fixed
per-(condition, T) seed, so results are reproducible run to run.
Independent conditions are fitted in parallel processes (`ZARC_N_JOBS`,
0 = one per CPU core); the warm-start chain stays sequential within each
condition, so parallelism does not change the numbers.

### Stage 4 - Derived quantities

| Quantity          | Formula                             | Notes                                                       |
| ----------------- | ----------------------------------- | ----------------------------------------------------------- |
| Area              | `A = π(D/2)²`                   | D = pellet diameter                                         |
| Conductivity      | `σ = L / (R · A)`               | L = pellet thickness                                        |
| Eff. capacitance  | `C_eff = τ / R`                  | equivalent to Q^(1/α)·R^((1-α)/α) when τ = (RQ)^(1/α) |
| Rel. permittivity | `εᵣ = C_eff · L / (ε₀ · A)` | ε₀ = 8.854 × 10⁻¹² F/m                                |

### Stage 4 - Arrhenius analysis

Two independent fits are performed per Zarc peak:

```
ln(σT) = ln(A₀) − Eₐᶜᵒⁿᵈ / (k_B T)     conductivity   (slope < 0)
ln(τ)  = ln(τ₀) + Eₐᵖᵒˡ  / (k_B T)     relaxation time (slope > 0)
```

A third fit on ln(C_eff) yields Eₐᶜ = Eₐᵖᵒˡ − Eₐᶜᵒⁿᵈ (from ln C = ln τ − ln R).
Activation energies are reported in eV (k_B = 8.617 × 10⁻⁵ eV/K).

### Stage 5 - global MIEC model

One global fit per process to the whole conductivity surface:

```
σ(pO₂,T) = (σ₀_ion/T)·e^(−Eₐ_ion/kT)
         + (σ₀_p  /T)·e^(−Eₐ_p  /kT)·pO₂^(+x)
         + (σ₀_n  /T)·e^(−Eₐ_n  /kT)·pO₂^(−x)        x = MODEL_EXPONENT
```

The three prefactors σ₀ are the intercepts of ln(σT) vs 1/T; the six parameters
are material constants, constant over the whole (pO₂,T) surface (the validity
test). The electronic conductivity minimum, where the n- and p-type branches
cross, is predicted in closed form:

```
ln pO₂_min(T) = (1/2x)·[ ln(σ₀_n/σ₀_p) − (Eₐ_n − Eₐ_p)/(k_B T) ]
```

and its migration with T (slope set by Eₐ_n − Eₐ_p) is an independent check
against the data.

---

## Reproducibility

Dependencies are pinned in `requirements.txt`. Each `.xlsx` output includes a `Metadata` sheet with all parameter values and installed library versions, so any result can be traced back to the exact configuration that produced it.

A regression test suite verifies the engine (`tests/test_engine_golden.py`: C_eff = τ/R identity, synthetic Zarc recovery, saved results satisfying the identity to floating-point precision) and the Stage-5 global model (`tests/test_model_golden.py`: synthetic-surface parameter recovery, degenerate and reduced-channel cases).

```bash
pytest tests/
```

---

## Methodological transparency

The default parameter values shipped in the notebooks (DRT regularization
strength, peak cap, Lin-KK M-selection mode, fit seed windows) were not
guessed: each was chosen by a documented calibration procedure run on the
authors' datasets. Those procedures live in [`audit/`](audit/README.md) as
runnable scripts, so any user can repeat the same decisions on their own
data instead of trusting values calibrated on someone else's material.

Two properties are worth stating explicitly. First, the shipped defaults are
dataset-calibrated starting points, not universal constants: a different
material, geometry, or frequency window can legitimately prefer different
values, and the audit scripts exist precisely to find them. Second, every
audit script is validated against a synthetic case with a known answer
(bundled in `EXAMPLE_SAMPLE/` and exercised by the test suite), so the
procedures themselves are under the same regression protection as the
calculation engine. No measured spectrum, fitted parameter, or ranking
derived from real data enters the repository: the scripts write to the
gitignored `audit/output/`.

---

## Processing a new sample

1. Copy `sample_template/` to `{sample_id}/` and rename the condition folder(s)
   inside `Raw data/` following the naming convention above.
2. Run stage0 through stage4 in order. The first cell of each notebook prompts for settings or reads them from `session.json`.

The pipeline discovers conditions and temperatures automatically.

---

## References

**Lin-KK**

- M. Schönleber, D. Klotz, E. Ivers-Tiffée, *A method for improving the robustness of linear Kramers-Kronig validity tests*, Electrochim. Acta 131 (2014) 20–27.
- B. A. Boukamp, *A linear Kronig-Kramers transform test for immittance data validation*, J. Electrochem. Soc. 142 (1995) 1885–1894.

**DRT**

- T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, *Influence of the discretization methods on the distribution of relaxation times deconvolution: implementing radial basis functions with DRTtools*, Electrochim. Acta 184 (2015) 483–499.
- A. Maradesa, B. Py, T. H. Wan, M. B. Effat, F. Ciucci, *Selecting the regularization parameter in the distribution of relaxation times*, J. Electrochem. Soc. 170 (2023) 030502. (Cited for the automatic λ-selection methods exposed by `DRT_CV_TYPE`; the validated default uses a fixed λ.)

**Equivalent circuits**

- J. T. S. Irvine, D. C. Sinclair, A. R. West, *Electroceramics: characterization by impedance spectroscopy*, Adv. Mater. 2 (1990) 132–138.
- X. Vendrell, A. R. West, *Electrical properties of yttria-stabilized zirconia, YSZ single crystal: local AC and long range DC conduction*, J. Electrochem. Soc. 165 (2018) F966–F975.

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

Citation metadata are in [CITATION.cff](CITATION.cff); GitHub shows a "Cite this
repository" button generated from it. A Zenodo DOI will be added with the first
public release.

---
