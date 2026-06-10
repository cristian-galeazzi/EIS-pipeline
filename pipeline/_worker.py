"""
Process-pool worker initializer. Deliberately free of heavy imports:
with the spawn start method the worker imports this module before numpy,
which is the only moment the BLAS thread caps below can still take effect.
"""
from __future__ import annotations

import os


def limit_blas_threads() -> None:
    """
    Cap BLAS at one thread per worker process.

    The Zarc fits factor tiny matrices, so per-call multithreading gains
    nothing; with several worker processes it only oversubscribes the CPU
    (n_workers x n_blas_threads runnable threads) and slows everyone down.
    """
    for var in ("VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
