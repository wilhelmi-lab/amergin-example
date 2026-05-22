from __future__ import annotations

import numpy as np

import amergin


@amergin.original("normalize", atol=1e-6)
def normalize(data: np.ndarray) -> np.ndarray:
    """Normalize a vector or array to unit norm (L2)."""
    return data / np.linalg.norm(data)


@amergin.input_generator(for_function="normalize")
def make_normalize_inputs(size: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {"data": rng.random(size).astype(np.float64)}


# Tier 2a: dot-product normalization — always available (numpy).
# Used to exercise verify on CPU-only runners.
@amergin.alternative(amergin.jit_backend, name="numpy", replaces="normalize")
def normalize_dotprod(data: np.ndarray) -> np.ndarray:
    """Normalize via explicit dot product (equivalent, different code path)."""
    norm = np.sqrt(data @ data)
    return data / norm


# Tier 2b: GPU alternative — skipped on CPU-only runners.
@amergin.alternative(amergin.gpu_backend, name="cupy", replaces="normalize")
def normalize_cupy(data: np.ndarray) -> np.ndarray:
    import cupy as cp  # type: ignore[import-untyped]

    data_gpu = cp.asarray(data)
    return cp.asnumpy(data_gpu / cp.linalg.norm(data_gpu))
