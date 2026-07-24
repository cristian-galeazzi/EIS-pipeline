"""
Generate the bundled synthetic sample (EXAMPLE_SAMPLE/).

Eight oxygen partial pressures x five temperatures of two-Zarc impedance
spectra, written as CSVs in the non-Zahner input format so anyone can run
stages 2-5 without real measurements. Deterministic: same seed, same files.

The spectra are built from a physically consistent MIEC conductivity model so
that every downstream figure is meaningful, not just well-formed:

  * each Zarc process has a conductivity sigma(pO2, T) that is the sum of
    channels of the exact form Stage 5 fits,
        sigma_channel(pO2, T) = (sigma0 / T) * exp(-Ea / kT) * pO2**expo
    with expo = 0 (ionic), +x (p-type holes) or -x (n-type electrons);
  * the resistance handed to the Zarc is R = (L / A) / sigma, the inverse of
    the pipeline's sigma = L / (R * A), so Stage 3 recovers exactly sigma;
  * C_eff = tau / R is held constant per process (a T-independent geometric
    capacitance), so tau tracks R and the peak walks in frequency with T.

Process 1 (bulk) is a mixed conductor with all three channels (n-type
electrons, ionic, p-type holes), so its Brouwer diagram is a full bathtub: the
p branch rises at high pO2, the ionic plateau is flat in the middle, and the
n branch rises at low pO2, with the electronic minimum near pO2 = 1e-6 bar.
Process 2 (grain boundary) is a pure ionic conductor, so its Brouwer diagram
is flat. The pO2 grid spans 1 bar down to 1e-12 bar to sample both electronic
branches; the four oxidizing conditions still read as a clean ion+p bulk (the
n channel is 0.5% of ionic there), matching what the audit tests expect.

The pO2 value of each condition is written into a `pO2` column of every CSV;
`ingest.load_csv_spectrum` reads it, which is what makes Stage 4/5 testable.

Usage (from repo root):
  python tools/generate_example_sample.py
"""
from __future__ import annotations

import csv
import math
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SAMPLE = REPO / "EXAMPLE_SAMPLE" / "input_spectra"

K_B = 8.617333e-5          # eV/K
T600_K = 873.15            # reference temperature (600 C) in K
TEMPS_C = [600, 550, 500, 450, 400]
N_POINTS = 40
F_MAX, F_MIN = 1.0e6, 0.5

# Sample geometry (must match the L_m / D_m entered in the notebooks): the
# generator converts sigma -> R through the same L / A the pipeline inverts.
L_M, D_M = 0.0014, 0.01
GEOM = L_M / (math.pi * (D_M / 2.0) ** 2)   # L / A [1/m]

BROUWER_X = 0.25           # p(O2) exponent (dilute regime)
R0 = 120.0                 # series resistance [ohm]

# (condition folder name, pO2 [bar]) - a full Brouwer sweep from oxidizing
# (p-type branch) through the ionic plateau to reducing (n-type branch). The
# reducing side is an Ar/H2 forming-gas series with H2 rising as pO2 falls, the
# way a real reducing atmosphere is set (pure Ar/O2 cannot go below ~1e-5 bar).
# pO2 is authoritative from each CSV's pO2 column, as it would come from the
# lambda probe; the gas tokens are valve-setpoint labels. The four oxidizing
# names are fixed: the audit DRT/fit known-answer tests look them up here.
CONDITIONS = [
    ("O2-100_600_400_50",      1.00),
    ("Ar-80_O2-20_600_400_50", 0.20),
    ("Ar-95_O2-5_600_400_50",  0.05),
    ("Ar-99_O2-1_600_400_50",  0.01),
    ("Ar-100_600_400_50",      1.0e-4),
    ("Ar-99_H2-1_600_400_50",  1.0e-6),
    ("Ar-97_H2-3_600_400_50",  1.0e-9),
    ("Ar-95_H2-5_600_400_50",  1.0e-12),
]

# Each process: constant C_eff [F], Zarc depression alpha, and a list of
# conductivity channels (sigma600 [S/m] at 600 C and pO2=1, Ea [eV], pO2 expo).
PROCESSES = [
    {   # bulk: mixed conductor, full Brouwer (n + ionic + p)
        # C_eff set so tau(600 C) ~ 2e-6 s at the Ar-80/O2-20 reference
        "C_eff": 3.2e-10, "alpha": 0.90,
        "channels": [
            (4.7e-6, 1.25, -BROUWER_X),   # n-type electrons (low pO2)
            (1.5e-3, 0.90, 0.0),          # ionic
            (4.7e-3, 1.05, +BROUWER_X),   # p-type holes (high pO2)
        ],
    },
    {   # grain boundary: pure ionic conductor; tau(600 C) ~ 3e-4 s
        "C_eff": 1.0e-8, "alpha": 0.86,
        "channels": [
            (6.0e-4, 1.10, 0.0),          # ionic
        ],
    },
]


def channel_sigma(sigma600: float, ea_ev: float, expo: float,
                  pO2: float, t_c: float) -> float:
    """Conductivity of one channel [S/m]; sigma*T is Arrhenius in T."""
    t_k = t_c + 273.15
    arr = (T600_K / t_k) * math.exp(-(ea_ev / K_B) * (1.0 / t_k - 1.0 / T600_K))
    return sigma600 * arr * pO2 ** expo


def process_sigma(proc: dict, pO2: float, t_c: float) -> float:
    """Total conductivity of a process [S/m]: sum over its channels."""
    return sum(channel_sigma(s600, ea, expo, pO2, t_c)
               for s600, ea, expo in proc["channels"])


def zarc(R: float, tau: float, alpha: float, f: float) -> complex:
    x = (2 * math.pi * f * tau) ** alpha
    return R / complex(1 + x * math.cos(alpha * math.pi / 2),
                       x * math.sin(alpha * math.pi / 2))


def main() -> None:
    random.seed(20260612)
    for cond, pO2 in CONDITIONS:
        out_dir = SAMPLE / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        for t_c in TEMPS_C:
            rows = []
            for k in range(N_POINTS):
                logf = math.log10(F_MAX) - k * (math.log10(F_MAX) - math.log10(F_MIN)) / (N_POINTS - 1)
                f = 10 ** logf
                Z = complex(R0, 0)
                for proc in PROCESSES:
                    sigma = process_sigma(proc, pO2, t_c)
                    R = GEOM / sigma
                    tau = proc["C_eff"] * R   # C_eff = tau / R held constant
                    Z += zarc(R, tau, proc["alpha"], f)
                noise = 1 + random.uniform(-0.003, 0.003)
                rows.append((f, Z.real * noise, -Z.imag * noise))  # Z_im positive
            path = out_dir / f"demo_{t_c}C.csv"
            with path.open("w", newline="") as fh:
                w = csv.writer(fh, lineterminator="\n")   # LF, not the csv default CRLF
                w.writerow(["freq", "Z_re", "Z_im", "temperature", "pO2"])
                for f, zr, zi in rows:
                    w.writerow([f"{f:.6g}", f"{zr:.6g}", f"{zi:.6g}", t_c, f"{pO2:.6g}"])
        print(f"{cond}: {len(TEMPS_C)} spectra written (pO2={pO2:g} bar)")
    print(f"done -> {SAMPLE}")


if __name__ == "__main__":
    main()
