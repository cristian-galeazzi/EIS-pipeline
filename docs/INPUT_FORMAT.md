# Input format and folder layout

How to feed non-Zahner (CSV) spectra to the pipeline, and where every file
lives on disk. The [README](../README.md) covers the Zahner `.ism` path; this
file is the CSV contract and the directory map. Per-stage detail is in
[STAGES.md](STAGES.md).

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
freq,Z_re,Z_im,temperature,pO2
100000,5.3,0.2,400,0.21
```

`Z_im` must be **positive** in the capacitive region (BioLogic EC-Lab
exports −Im(Z): multiply by −1 before saving); the loader rejects a file
whose `Z_im` column is mostly negative, naming the convention in the error. Temperature in the filename
(`_{T}C`) and a `temperature` column are required for Arrhenius analysis
(≥ 3 temperatures); Nyquist, Bode, DRT and the Zarc fit work without them.
An optional `pO2` column [bar] enables the Brouwer p(O₂), transference and
Stage 5 analyses without an oven log; without it (or stage 0-1 lambda-probe
data) those p(O₂) steps are skipped.

---

## Folder layout

```
EIS-pipeline/
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

**Requirements:** Python ≥ 3.11 and the pinned `requirements.txt`:
`zahner_analysis` (.ism reader), `pyDRTtools` (DRT), `impedance` (circuit
fit), `scipy`/`numpy`, `pandas`/`openpyxl` (Excel I/O), `matplotlib`,
`ipywidgets` (interactive panels, stages 0-5).
