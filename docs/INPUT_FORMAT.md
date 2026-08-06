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
(`{prefix}_{gas_flows}_{T_max}_{T_min}_{T_step}`; the furnace-log matcher
recognizes the gas section from the first `Ar`, `O2`, `N2`, `H2`, `CO2`, `CO`,
`He` or `H2O` token, while the label formatter also accepts `Air`). The file prefix before
`_{T}C` is free. CSV format, separator comma/semicolon/tab:

```
freq,Z_re,Z_im,temperature,pO2
100000,5.3,0.2,400,0.21
```

`Z_im` must be **positive** in the capacitive region (BioLogic EC-Lab
exports $`-\text{Im}(Z)`$: multiply by $`-1`$ before saving); the loader
rejects a file whose `Z_im` column is mostly negative, naming the convention
in the error. Temperature in the filename (`_{T}C`) and a `temperature`
column are required for Arrhenius analysis (at least three temperatures);
Nyquist, Bode, DRT and the Zarc fit work without them.
An optional `pO2` column [bar] enables the Brouwer $`p_{\text{O}_2}`$, transference and
stage 5 analyses without an oven log; without it (or stage 0-1 lambda-probe
data) those $`p_{\text{O}_2}`$ steps are skipped, as they are for a run recorded
with the probe off.

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
│   └── utils.py            ← Excel, p(O2) label and condition-name helpers
│
├── stage0_oven.ipynb … stage5_model.ipynb   ← the six pipeline stages
│
├── audit/                  ← runnable calibration procedures (see below)
├── tests/                  ← golden-master + known-answer test suites
├── tools/                  ← example-sample generator, documentation checks
├── EXAMPLE_SAMPLE/         ← bundled synthetic sample (CSV entry path)
├── sample_template/        ← copy and rename to {SAMPLE_ID}/ to start
│
└── {SAMPLE_ID}/            ← your data, never committed
    ├── Raw data/           ← never modified
    ├── Raw oven/           ← never modified
    ├── ISM validation/     ← copies made by stage 1, renamed to _{T}C when needed
    └── Results/
        ├── {condition}/    ← stage1_labeling / stage2_kk / stage3_drt /
        │                     stage3_fit .xlsx + DRT, Nyquist-Bode,
        │                     Arrhenius figure folders
        └── pO2/            ← Brouwer + transference + stage5_model.xlsx
```

`{SAMPLE_ID}` is both the folder name and the key in `session.json`. A repeated
measurement of the same pellet takes its own id, formed by suffixing the
existing one (`{SAMPLE_ID}_Tvar`), so the two runs stay separate entries; the
suffix must extend the condition-folder prefix, since an unrelated name cannot
be stripped from the condition labels.
