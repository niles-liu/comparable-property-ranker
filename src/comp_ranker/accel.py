"""GPU acceleration backend with automatic fallback to CPU.

Provides a numpy-compatible array module (``xp``) that resolves to CuPy when
a CUDA-capable GPU is available, and plain NumPy otherwise.  All downstream
code can write ``from comp_ranker.accel import xp`` and use the same API
regardless of hardware.

Helper utilities handle the CuPy <-> NumPy boundary (e.g. moving data back to
CPU for pandas / scikit-learn interop).
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Detect CuPy / CUDA availability once at import time
# ---------------------------------------------------------------------------
_GPU_AVAILABLE: bool = False

try:
    import cupy as cp

    # Verify an actual GPU is reachable (driver installed, device visible)
    cp.cuda.runtime.getDeviceCount()
    _GPU_AVAILABLE = True
except Exception:
    cp = None  # type: ignore[assignment]

#: Array module -- ``cupy`` when a GPU is present, else ``numpy``.
xp = cp if _GPU_AVAILABLE else np


def gpu_available() -> bool:
    """Return True if CuPy + a CUDA GPU are available."""
    return _GPU_AVAILABLE


def to_cpu(arr) -> np.ndarray:
    """Move an array to CPU (no-op if already a numpy array)."""
    if _GPU_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def to_gpu(arr):
    """Move a numpy array to GPU.  Returns the array unchanged if no GPU."""
    if _GPU_AVAILABLE:
        return cp.asarray(arr)
    return arr
