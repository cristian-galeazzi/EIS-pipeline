"""Smoke validation of the synthetic gate harness (reduced grid).

The full gate is a CLI evaluation run; this test keeps its machinery honest
in CI: spectrum generation in the IsmRecord convention, seed displacement,
paired execution of both engines on identical inputs, and the v2 objective
guarantee (rmse_rel never worse than v1, since both minimize the same
weighted least squares and v2 converges at least as deep).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from audit.fitting_v2 import synthetic_gate as gate


def test_reduced_gate_structure_and_objective():
    cases = gate.CASES[:2]   # 1zarc + 2zarc
    rows, medians = gate.run_gate(replicates=2, noise_levels=(0.005,),
                                  cases=cases)
    assert len(rows) == len(cases) * 2 * 2   # cases x replicates x engines
    assert all(r["converged"] for r in rows)
    # paired: v2 must never end on a worse value of the fit objective
    by_key = {}
    for r in rows:
        by_key.setdefault((r["case"], r["rep"]), {})[r["engine"]] = r
    for k, v in by_key.items():
        assert v["v2"]["rmse_rel"] <= v["v1"]["rmse_rel"] + 1e-12, k
    assert ("v1", 0.005) in medians and ("v2", 0.005) in medians


def test_recovery_error_zero_on_truth():
    fake = {"R": np.array([1e4, 2e4]), "tau": np.array([1e-4, 1e-2]),
            "alpha": np.array([0.9, 0.8])}
    true = [(1e4, 1e-4, 0.9), (2e4, 1e-2, 0.8)]
    assert gate.recovery_error(fake, true) == 0.0
    # one decade off in R1 contributes 0.5 (mean over two peaks)
    fake["R"] = np.array([1e5, 2e4])
    assert gate.recovery_error(fake, true) == 0.5
