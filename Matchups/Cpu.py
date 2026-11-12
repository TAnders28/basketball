"""
CPU cap helpers. Tim's rule for this work: use no more than about half of Tomputer's processing power.

- MaxWorkers(): floor(logical cores / 2) worker processes, never more.
- LimitThreads(): one BLAS/OpenMP thread per process (call before importing numpy).
- LowerPriority(): BelowNormal priority on Windows (nice 10 elsewhere).
"""

import os

THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
BELOW_NORMAL_PRIORITY_CLASS = 0x00004000


def MaxWorkers(requested: int = None) -> int:
    cap = max(1, (os.cpu_count() or 2) // 2)
    return cap if requested is None else max(1, min(requested, cap))


def LimitThreads() -> None:
    for var in THREAD_VARS:
        os.environ[var] = "1"


def LowerPriority() -> None:
    try:
        if os.name == "nt":
            import ctypes
            kernel = ctypes.windll.kernel32
            kernel.SetPriorityClass(kernel.GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS)
        else:
            os.nice(10)
    except Exception:
        pass


def WorkerInit() -> None:
    LimitThreads()
    LowerPriority()
