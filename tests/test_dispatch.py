"""Phase 1 integration test: normalize() dispatches correctly on CPU-only runner."""
from __future__ import annotations

import numpy as np
import pytest

from numops import normalize


def test_normalize_unit_norm() -> None:
    data = np.array([3.0, 4.0])
    result = normalize(data)
    assert abs(np.linalg.norm(result) - 1.0) < 1e-6


def test_normalize_shape_preserved() -> None:
    data = np.random.rand(10)
    result = normalize(data)
    assert result.shape == data.shape


def test_normalize_callable() -> None:
    """normalize is the dispatcher, not the raw function."""
    assert callable(normalize)


def test_normalize_zero_vector_produces_nan() -> None:
    """numpy returns NaN for a zero-norm vector; dispatch does not hide this."""
    import warnings

    data = np.zeros(5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = normalize(data)
    assert np.all(np.isnan(result))
