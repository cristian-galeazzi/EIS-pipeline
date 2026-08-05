# input_spectra

Place your EIS spectra here if you are not using a Zahner system or do not have furnace log data.

## Folder structure

Each subfolder is one measurement condition (atmosphere + temperature range).
The folder name becomes the condition label in all outputs.

```
input_spectra/
  condition_name/
    anyname_500C.csv
    anyname_500C_1.csv    <- second replica at 500 C
    anyname_550C.csv
    anyname_600C.txt      <- .txt is accepted too
```

## File format

The file must have a header on the first line with these exact column names:

Minimum (no temperature sensor):
```
freq,Z_re,Z_im
100000,12.3,0.45
50000,14.1,1.20
...
```

With temperature (enables Arrhenius plots):
```
freq,Z_re,Z_im,temperature
100000,12.3,0.45,500
50000,14.1,1.20,500
...
```

An optional `pO2` column in bar adds the Brouwer and transference analyses; the
full contract is in `docs/INPUT_FORMAT.md`.

Separator can be comma, semicolon, or tab. Decimal separator must be a dot.

Z_im must be positive in the capacitive region (standard EIS convention).
BioLogic exports -Im(Z) with inverted sign: rename the column to Z_im and multiply by -1 before saving.

## File naming

The temperature must appear in the filename as `_NNNc` (case insensitive):

```
sample_500C.csv      <- T = 500, replica 1
sample_500C_1.csv    <- T = 500, replica 2
sample_500C_2.csv    <- T = 500, replica 3
sample_25C.csv       <- T = 25, works at any temperature
```

Files without the temperature pattern in the name are ignored.

## What you get

- Always: Nyquist, Bode, DRT stacked, Zarc fit
- With temperature column: Arrhenius plots
- With an optional `pO2` column in bar: Brouwer diagram and transference
- Without it, and with a single condition: both are skipped

Start from stage2_kk.ipynb. Stage 0 and Stage 1 are not needed.
