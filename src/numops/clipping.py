from __future__ import annotations

from typing import Any

import numpy as np

import amergin


@amergin.capability("fast_clip")
def _detect_fast_clip() -> tuple[bool, dict[str, Any]]:
    """A capability that is always available — exercises adaptive re-runs.

    In a real package this might check for a SIMD library or a hardware
    flag. Here we just return True so the benchmark always probes it.
    """
    return True, {}


@amergin.original("clip_normalize", atol=1e-6)
def clip_normalize(data: np.ndarray, clip: float) -> np.ndarray:
    """Clip then normalize to unit norm. Has an is_available() branch.

    The fast_clip path uses np.clip (vectorised); the fallback path runs
    a Python loop. The fast path is not registered as a formal
    @amergin.alternative — so the benchmark runner should schedule an
    adaptive re-run with fast_clip force-disabled to measure the gap.
    """
    if amergin.is_available("fast_clip"):
        clipped = np.clip(data, -clip, clip)
    else:
        clipped = np.empty_like(data)
        for i in range(len(data)):
            clipped[i] = max(-clip, min(clip, float(data[i])))
    return clipped / np.linalg.norm(clipped)


@amergin.input_generator(for_function="clip_normalize")
def make_clip_inputs(size: int) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    return {
        "data": rng.standard_normal(size).astype(np.float64),
        "clip": 1.0,
    }
