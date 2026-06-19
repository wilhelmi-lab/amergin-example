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


# Tier 2b: NVIDIA-only GPU alternative. `device="nvidia"` means it is
# cleanly skipped (not crashed) on AMD/Intel/CPU machines, and the report
# files it under the nvidia device.
@amergin.alternative(
    amergin.gpu_backend, name="cupy", replaces="normalize",
    implementation="cupy", device="nvidia",
)
def normalize_cupy(data: np.ndarray) -> np.ndarray:
    import cupy as cp  # type: ignore[import-untyped]

    data_gpu = cp.asarray(data)
    return cp.asnumpy(data_gpu / cp.linalg.norm(data_gpu))


# Tier 2c: device-agnostic GPU alternative. ONE torch code path runs on
# every GPU vendor — CUDA (NVIDIA), HIP/ROCm (AMD, which torch also calls
# "cuda"), and XPU (Intel) — falling back to CPU where no GPU is present.
# No `device=` requirement, so it runs everywhere torch imports; the
# machine bucket + recorded device distinguish where it ran. This is what
# makes "the same benchmark runs on all three GPUs" true with one call.
@amergin.alternative(
    amergin.gpu_backend, name="torch", replaces="normalize",
    implementation="torch",
)
def normalize_torch(data: np.ndarray) -> np.ndarray:
    import torch  # type: ignore[import-untyped]

    if torch.cuda.is_available():            # NVIDIA CUDA or AMD ROCm/HIP
        device = "cuda"
    elif getattr(torch, "xpu", None) is not None and torch.xpu.is_available():
        device = "xpu"                        # Intel GPU
    else:
        device = "cpu"
    t = torch.as_tensor(data, device=device)
    out = t / torch.linalg.norm(t)
    return out.detach().cpu().numpy()
