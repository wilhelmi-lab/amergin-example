from __future__ import annotations

from typing import Any

import numpy as np

import amergin


@amergin.capability("numpy_convolve")
def _detect_numpy_convolve() -> tuple[bool, dict[str, Any]]:
    return True, {}


@amergin.capability("numpy_cumsum")
def _detect_numpy_cumsum() -> tuple[bool, dict[str, Any]]:
    return True, {}


@amergin.original("moving_average", atol=1e-10)
def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """Naive loop-based moving average (intentionally slow baseline)."""
    n = len(data) - window + 1
    result = np.empty(n)
    for i in range(n):
        result[i] = np.mean(data[i:i + window])
    return result


@amergin.input_generator(for_function="moving_average")
def make_moving_average_inputs(
    size: int, window: int
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    return {"data": rng.random(size).astype(np.float64), "window": window}


@amergin.parameter_filter(for_function="moving_average")
def filter_moving_average(params: dict[str, Any]) -> bool:
    size = params.get("size", 1)
    window = params.get("window", 0)
    return bool(window < size // 4)


@amergin.alternative(
    amergin.jit_backend, name="numpy_convolve", replaces="moving_average"
)
def moving_average_convolve(data: np.ndarray, window: int) -> np.ndarray:
    """np.convolve-based moving average."""
    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode="valid")


@amergin.alternative(
    amergin.jit_backend, name="numpy_cumsum", replaces="moving_average"
)
def moving_average_cumsum(data: np.ndarray, window: int) -> np.ndarray:
    """O(n) cumsum-based moving average."""
    cs = np.cumsum(data)
    cs[window:] = cs[window:] - cs[:-window]
    return cs[window - 1:] / window
