"""
Generate the bundled synthetic sample (EXAMPLE_SAMPLE/).

Two gas conditions x five temperatures of two-Zarc impedance spectra
(bulk + grain boundary, Arrhenius-consistent R(T), 0.3% noise), written as
CSVs in the non-Zahner input format so anyone can run stages 2-4 without
real measurements. Deterministic: same seed, same files.

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
TEMPS_C = [600, 550, 500, 450, 400]
N_POINTS = 40
F_MAX, F_MIN = 1.0e6, 0.5

# (condition name, scale factor on both resistances)
CONDITIONS = [
    ("Ar-80_O2-20_600_400_50", 1.0),
    ("O2-100_600_400_50",      0.8),
]

# (R at 600 C [ohm], Ea [eV], tau at 600 C [s], alpha) per process
PROCESSES = [
    (8.0e3, 0.90, 2.0e-6, 0.92),   # bulk
    (2.5e4, 1.10, 3.0e-4, 0.88),   # grain boundary
]
R0 = 120.0


def zarc(R: float, tau: float, alpha: float, f: float) -> complex:
    x = (2 * math.pi * f * tau) ** alpha
    return R / complex(1 + x * math.cos(alpha * math.pi / 2),
                       x * math.sin(alpha * math.pi / 2))


def arrhenius(value_600: float, ea_ev: float, t_c: float) -> float:
    t_k, t600_k = t_c + 273.15, 873.15
    return value_600 * math.exp(ea_ev / K_B * (1 / t_k - 1 / t600_k))


def main() -> None:
    random.seed(20260612)
    for cond, scale in CONDITIONS:
        out_dir = SAMPLE / cond
        out_dir.mkdir(parents=True, exist_ok=True)
        for t_c in TEMPS_C:
            rows = []
            for k in range(N_POINTS):
                logf = math.log10(F_MAX) - k * (math.log10(F_MAX) - math.log10(F_MIN)) / (N_POINTS - 1)
                f = 10 ** logf
                Z = complex(R0, 0)
                for R600, ea, tau600, alpha in PROCESSES:
                    R = arrhenius(R600, ea, t_c) * scale
                    # tau follows R so C_eff = tau/R stays T-independent-ish
                    tau = arrhenius(tau600, ea, t_c)
                    Z += zarc(R, tau, alpha, f)
                noise = 1 + random.uniform(-0.003, 0.003)
                rows.append((f, Z.real * noise, -Z.imag * noise))  # Z_im positive
            path = out_dir / f"demo_{t_c}C.csv"
            with path.open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["freq", "Z_re", "Z_im", "temperature"])
                for f, zr, zi in rows:
                    w.writerow([f"{f:.6g}", f"{zr:.6g}", f"{zi:.6g}", t_c])
        print(f"{cond}: {len(TEMPS_C)} spectra written")
    print(f"done -> {SAMPLE}")


if __name__ == "__main__":
    main()
