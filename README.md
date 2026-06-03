# EIS Analysis Pipeline

Python pipeline for EIS analysis of high-resistance ceramic electrolytes. Goes from raw
Zahner `.ism` files to publication figures in five sequential Jupyter notebooks.

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
locally in `session.json` (gitignored). Raw `.ism` files and furnace logs are never
modified; results are written to `{sample_id}/Results/`. No network calls are made.

---

## Requirements

Python ≥ 3.10. Install dependencies:

```bash
pip install -r requirements.txt
```

| Package                  | Purpose                           |
| ------------------------ | --------------------------------- |
| `zahner_analysis`    | Read Zahner `.ism` binary files     |
| `pyDRTtools`         | DRT via Tikhonov regularisation     |
| `impedance`          | Zarc equivalent-circuit fitting     |
| `pandas`, `openpyxl` | Excel I/O                          |
| `scipy`, `numpy`     | Numerics                            |
| `matplotlib`         | Figures                             |
| `ipywidgets`         | Interactive panels (stages 0–4)     |

---

## Folder layout

```
EIS program/
│
├── pipeline/               ← Python modules
│   ├── ingest.py           ← ISM file reader
│   ├── matching.py         ← Furnace log parser and T / pO₂ matcher
│   ├── quality.py          ← Lin-KK implementation
│   ├── drt.py              ← DRT computation and peak extraction
│   ├── fitting.py          ← Zarc circuit fitting
│   ├── plots.py            ← Publication figures
│   ├── interactive.py      ← ipywidgets helpers (UI only)
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

| Stage | Notebook                   | What it does                                                                  |
| ----- | -------------------------- | ----------------------------------------------------------------------------- |
| 0     | `stage0_oven.ipynb` | Check that furnace plateaus match the measurement schedule                    |
| 1     | `stage1_labeling.ipynb`  | Match `.ism` files to furnace windows, label temperatures, copy valid files |
| 2     | `stage2_kk.ipynb` | Lin-KK validation; select best replica per (condition, T)                     |
| 3     | `stage3_drt.ipynb`      | Tikhonov DRT, peak detection, Zarc circuit fit                                |
| 4     | `stage4_plots.ipynb`         | Nyquist, Bode, DRT stacked, Arrhenius, Brouwer p(O₂)                         |

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

| Feature | Available |
|---------|-----------|
| Nyquist, Bode, DRT, Zarc fit | always |
| Arrhenius plots | requires `temperature` column and ≥ 3 temperatures |
| Brouwer p(O₂) | never (requires lambda-probe data from Stage 0–1) |

---

## Stage 0 - `stage0_oven.ipynb`

Parses furnace logs and plots T vs time with plateau annotations.

| Parameter           | Default | Purpose                         |
| ------------------- | ------- | ------------------------------- |
| `TABLE_INTERVAL_S`  | 300 s   | Plateau table sampling interval |

`sample_id` is entered via `input()`. Conditions are selected with the widget below the import cell and saved to `session.json`.

---

## Stage 1 - `stage1_labeling.ipynb`

Matches each `.ism` file to its furnace window, assigns a temperature label and copies valid files to `ISM validation/`.

| Parameter            | Default    | Purpose                                   |
| -------------------- | ---------- | ----------------------------------------- |
| `T_STABILITY_STD`    | 1 °C       | Max std(T) during measurement             |
| `T_PRE_MARGIN_MIN`   | 25 min     | Stability window before measurement start |
| `T_POST_MARGIN_MIN`  | 5 min      | Stability window after measurement end    |
| `T_ROUND_STEP`       | 25 °C      | Temperature rounding step                 |
| `T_PLATEAU_RANGE`    | (395, 605) | Valid plateau range [°C]                  |

`sample_id` and `conditions` are read from `session.json` (set in stage 0).

Status codes: `VALID`, `UNSTABLE`, `NEAR_TRANSITION`, `OUT_OF_RANGE`, `OUTSIDE_RANGE`.

> **Non-standard temperature ranges:** `T_PLATEAU_RANGE` defaults to (395, 605) °C.
> Measurements outside this range are classified `OUTSIDE_RANGE` and excluded.
> Adjust both `T_PLATEAU_RANGE` and `T_ROUND_STEP` in the config cell if working
> at different temperatures (e.g. room temperature or above 600 °C).

---

## Stage 2 - `stage2_kk.ipynb`

Applies the Lin-KK test [Schönleber et al., 2014] to each spectrum and selects the best replica per (condition, T).

| Parameter            | Default | Purpose                                    |
| -------------------- | ------- | ------------------------------------------ |
| `KK_C`               | 0.76    | M = round(KK_C × N)                       |
| `KK_MU_TARGET`       | 0.50    | Sign-change fraction target                |
| `KK_F_MIN_HARD`      | 50 Hz   | Hard lower frequency cutoff                |
| `KK_USE_W_CRITERIA`  | False   | Ceramic-aware dual criterion (W_re + W_im) |
| `KK_OVERRIDES`       | `{}`    | Per-(condition, T) frequency overrides     |
| `OVERRIDES`          | `{}`    | Manual replica selection                   |

Classification: GREEN (`kk_score >= 0.97`), YELLOW (`>= 0.90`), RED.

---

## Stage 3 - `stage3_drt.ipynb`

Computes the DRT γ(τ) via Tikhonov regularisation, detects peaks and fits a Zarc equivalent circuit seeded by the DRT peaks.

```
Z(ω) = R₀ + Σᵢ Rᵢ / (1 + (j ω τᵢ)^αᵢ)
```

| Parameter               | Default  | Purpose                                     |
| ----------------------- | -------- | ------------------------------------------- |
| `L_m`, `D_m`            | required | Pellet thickness and diameter [m]           |
| `DRT_REG_PARAM`         | 4e-5     | Regularisation λ (custom mode)              |
| `PEAK_MIN_PROM_DECADES` | 0.01     | Log-prominence threshold for peak detection |
| `ZARC_INCLUDE_R0`       | True     | Include series R₀ in circuit                |
| `ZARC_R0_MAX`           | 200 Ω    | Upper bound on R₀                           |
| `N_PEAKS_OVERRIDE`      | `{}`     | Force N peaks for specific (condition, T)   |

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

| Parameter          | Default     | Purpose                                    |
| ------------------ | ----------- | ------------------------------------------ |
| `L_m`, `D_m`       | required    | Pellet thickness and diameter [m]          |
| `DRT_TAU_MAX`      | 0.1 s       | x-axis upper limit on DRT stacked plot     |
| `BROUWER_PEAK_ID`  | 1           | Peak index for Brouwer diagram             |
| `BROUWER_TEMPS`    | None        | Temperatures shown in Brouwer (None = all) |
| `TAU_R2_THRESHOLD` | 0.97        | R² floor to flag a peak as physically real |
| `PLOT_WINDOWS`     | `{}`        | Per-(condition, T) axis crop               |

Figures per condition: DRT stacked, Nyquist, Bode, Arrhenius 2×2, C_eff magnitude, τ consistency. Multi-condition: Brouwer p(O₂) diagram.

---

## Physical formulae

| Quantity              | Formula                                |
| --------------------- | -------------------------------------- |
| Area                  | `A = π (D/2)²`                     |
| Conductivity          | `σ = L / (R · A)`                  |
| Effective capacitance | `C_eff = Q^(1/α) · R^((1-α)/α)`  |
| Permittivity          | `εᵣ = C · L / (ε₀ · A)`        |
| Arrhenius (σT)       | `ln(σT) = ln(A₀) − Eₐ / (k_B T)` |
| Arrhenius (τ)        | `ln(τ) = ln(τ₀) + Eₐ / (k_B T)`  |

---

## Reproducibility

Dependencies are pinned in `requirements.txt`. Each `.xlsx` output includes a `Metadata` sheet with all parameter values and installed library versions, so any result can be traced back to the exact configuration that produced it.

A regression test suite (`tests/test_engine_golden.py`) verifies the C_eff = τ/R identity, synthetic Zarc recovery, and that saved results satisfy the identity to floating-point precision.

```bash
pytest tests/test_engine_golden.py
```

---

## Processing a new sample

1. Create `{sample_id}/` with `Raw data/` and `Raw oven/` sub-folders.
2. Run stage0 through stage4 in order. The first cell of each notebook prompts for settings or reads them from `session.json`.

The pipeline discovers conditions and temperatures automatically.

---

## References

**Lin-KK**

- M. Schönleber, D. Klotz, E. Ivers-Tiffée, *A method for improving the robustness of linear Kramers-Kronig validity tests*, Electrochim. Acta 131 (2014) 20–27.
- B. A. Boukamp, *A linear Kronig-Kramers transform test for immittance data validation*, J. Electrochem. Soc. 142 (1995) 1885–1894.

**DRT**

- T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, *Influence of the discretization methods on the distribution of relaxation times deconvolution*, Electrochim. Acta 184 (2015) 483–499.

**Equivalent circuits**

- J. T. S. Irvine, D. C. Sinclair, A. R. West, *Electroceramics: characterization by impedance spectroscopy*, Adv. Mater. 2 (1990) 132–138.
- C. Vendrell, A. R. West, *Electrical properties of YSZ single crystal*, J. Electrochem. Soc. 165 (2018) F966–F975.

**Scientific computing**

- G. Wilson et al., *Best Practices for Scientific Computing*, PLoS Biology 12(1) (2014) e1001745.
- G. Wilson et al., *Good Enough Practices in Scientific Computing*, PLoS Computational Biology 13(6) (2017) e1005510.
- A. Scopatz, K. D. Huff, *Effective Computation in Physics*, O'Reilly Media (2015).

---
