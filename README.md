# EIS Analysis Pipeline

Python pipeline for Electrochemical Impedance Spectroscopy (EIS) analysis of high-resistance ceramic electrolytes. Replaces a manual RelaxIS + Origin workflow with a fully reproducible environment, from raw Zahner `.ism` files to publication figures, in five sequential Jupyter notebooks.

---

## Authorship

All scientific work, experimental design, sample preparation, calibration choices and methodological decisions in this pipeline are the author's own.

The Python implementation was developed with the assistance of an AI coding assistant (Anthropic Claude), working from the author's scientific specifications. All code was reviewed and validated by the author against experimental data.

---

## Data and privacy

This repository contains code only. No measurement data, fitted parameters or sample-specific values are committed.

The pipeline runs entirely on the user's machine. Raw `.ism` files and furnace logs are never modified; the pipeline only writes copies and results into `{SAMPLE_ID}/Results/`. No network calls are made at runtime.

---

## Requirements

Python ≥ 3.10. Install dependencies with:

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
| `ipywidgets`           | Live control panels (optional)    |

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
├── 00_oven_analysis.ipynb  ← Stage 0: furnace log diagnostics
├── 01_ism_labeling.ipynb   ← Stage 1: measurement identification
├── 02_linkk_quality.ipynb  ← Stage 2: Lin-KK quality assessment
├── 03_drt_zarc.ipynb       ← Stage 3: DRT and Zarc fitting
├── 04_plots.ipynb          ← Stage 4: publication figures
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
| 0     | `00_oven_analysis.ipynb` | Check that furnace plateaus match the measurement schedule                    |
| 1     | `01_ism_labeling.ipynb`  | Match `.ism` files to furnace windows, label temperatures, copy valid files |
| 2     | `02_linkk_quality.ipynb` | Lin-KK validation; select best replica per (condition, T)                     |
| 3     | `03_drt_zarc.ipynb`      | Tikhonov DRT, peak detection, Zarc circuit fit                                |
| 4     | `04_plots.ipynb`         | Nyquist, Bode, DRT stacked, Arrhenius, Brouwer p(O₂)                         |

---

## Usage

1. Open each notebook in JupyterLab or VS Code.
2. Edit only the first code cell (`# CONFIGURATION`). Everything else runs automatically.
3. Run: **Kernel → Restart and Run All Cells**.
4. Check the inline output, adjust overrides if needed, re-run.

All notebooks support temperature-by-temperature processing via `FOCUS_T`. The export cells are merge-aware: rows for other temperatures are preserved when you process one at a time.

---

## Stage 0 - `00_oven_analysis.ipynb`

Parses furnace logs and plots T vs time with plateau annotations.

| Parameter            | Default | Purpose                                    |
| -------------------- | ------- | ------------------------------------------ |
| `SAMPLE_ID`        |         | Sample folder name                         |
| `CONDITION_FILTER` | `[]`  | Limit to specific conditions (empty = all) |
| `TABLE_INTERVAL_S` | 300 s   | Plateau table sampling interval            |

---

## Stage 1 - `01_ism_labeling.ipynb`

Matches each `.ism` file to its furnace window, assigns a temperature label and copies valid files to `ISM validation/`.

| Parameter             | Default    | Purpose                                   |
| --------------------- | ---------- | ----------------------------------------- |
| `T_STABILITY_STD`   | 1 °C      | Max std(T) during measurement             |
| `T_PRE_MARGIN_MIN`  | 25 min     | Stability window before measurement start |
| `T_POST_MARGIN_MIN` | 5 min      | Stability window after measurement end    |
| `T_ROUND_STEP`      | 25 °C     | Temperature rounding step                 |
| `T_PLATEAU_RANGE`   | (390, 610) | Valid plateau range [°C]                 |

Status codes: `VALID`, `UNSTABLE`, `NEAR_TRANSITION`, `OUT_OF_RANGE`, `OUTSIDE_RANGE`.

---

## Stage 2 - `02_linkk_quality.ipynb`

Applies the Lin-KK test [Schönleber et al., 2014] to each spectrum and selects the best replica per (condition, T).

| Parameter         | Default | Purpose                                |
| ----------------- | ------- | -------------------------------------- |
| `KK_C`          | 0.76    | M = round(KK_C × N)                   |
| `KK_MU_TARGET`  | 0.50    | Sign-change fraction target            |
| `KK_F_MIN_HARD` | 50 Hz   | Hard lower frequency cutoff            |
| `KK_OVERRIDES`  | `{}`  | Per-(condition, T) frequency overrides |
| `OVERRIDES`     | `{}`  | Manual replica selection               |

Classification: 🟢 GREEN (`kk_score ≥ 0.97`), 🟡 YELLOW (`≥ 0.90`), 🔴 RED.

---

## Stage 3 - `03_drt_zarc.ipynb`

Computes the DRT γ(τ) via Tikhonov regularisation, detects peaks and fits a Zarc equivalent circuit seeded by the DRT peaks.

```
Z(ω) = R₀ + Σᵢ Rᵢ / (1 + (j ω τᵢ)^αᵢ)
```

| Parameter                 | Default | Purpose                                     |
| ------------------------- | ------- | ------------------------------------------- |
| `DRT_REG_PARAM`         | 4e-5    | Regularisation λ (custom mode)             |
| `PEAK_MIN_PROM_DECADES` | 0.01    | Log-prominence threshold for peak detection |
| `ZARC_R_DEC`            | 1.5     | R fit bounds in log-decades                 |
| `ZARC_TAU_DEC`          | 1.5     | τ fit bounds in log-decades                |
| `ZARC_R0_MAX`           | 200 Ω  | Upper bound on R₀                          |
| `N_PEAKS_OVERRIDE`      | `{}`  | Force N peaks for specific (condition, T)   |

**Process identification:** the pipeline assigns no process label automatically. Use the C_eff magnitude plot (log₁₀(C_eff) vs 1000/T) and Arrhenius behaviour to identify each peak. Starting-point thresholds from Vendrell & West 2018 (YSZ):

| C_eff                | Typical process |
| -------------------- | --------------- |
| < 10⁻¹¹ F         | bulk            |
| 10⁻¹¹ – 10⁻⁸ F | grain boundary  |
| 10⁻⁸ – 10⁻⁶ F   | near-electrode  |
| > 10⁻⁶ F           | electrode       |

---

## Stage 4 - `04_plots.ipynb`

Reads Stage 3 outputs and generates publication figures (PNG + PDF).

| Parameter           | Default     | Purpose                           |
| ------------------- | ----------- | --------------------------------- |
| `L_m`, `D_m`    | set by user | Pellet thickness and diameter [m] |
| `DRT_TAU_MAX`     | 0.1 s       | x-axis limit on DRT stacked plot  |
| `BROUWER_PEAK_ID` | 1           | Peak for Brouwer diagram          |
| `PLOT_WINDOWS`    | `{}`      | Per-(condition, T) axis crop      |

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

1. Create `{SAMPLE_ID}/` with `Raw data/` and `Raw oven/` sub-folders.
2. Set `SAMPLE_ID` in each notebook's configuration cell.
3. Run notebooks 00 → 04 in order.

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
