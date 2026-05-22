from __future__ import annotations

import numpy as np

import amergin

from numops.aggregation import moving_average
from numops.processing import normalize


@amergin.original
def normalize_smoothed(data: np.ndarray, window: int) -> np.ndarray:
    """Smooth a signal with a moving average then normalize to unit norm.

    Composes two registered functions; has no direct alternatives. The
    benchmark runner discovers the inner calls and enumerates combinations
    of their backends.
    """
    smoothed = moving_average(data, window=window)
    return normalize(smoothed)


@amergin.input_generator(for_function="normalize_smoothed")
def make_normalize_smoothed_inputs(
    size: int, window: int
) -> dict[str, np.ndarray | int]:
    rng = np.random.default_rng(0)
    return {"data": rng.random(size).astype(np.float64), "window": window}


@amergin.parameter_filter(for_function="normalize_smoothed")
def filter_normalize_smoothed(params: dict[str, int]) -> bool:
    size = params.get("size", 1)
    window = params.get("window", 0)
    return bool(window < size // 4)
