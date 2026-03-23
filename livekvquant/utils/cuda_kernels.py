"""
cuda_kernels.py

Lazy-loading wrapper for the custom CUDA extension (dequant_restore).
Falls back to pure PyTorch if CUDA is unavailable or compilation fails.
"""

import os
import torch
from typing import Optional

_ext = None          # cached compiled extension
_load_failed = False  # avoid retrying after a failed load


def _load_extension():
    """JIT-compile and cache the CUDA extension on first call."""
    global _ext, _load_failed
    if _ext is not None or _load_failed:
        return _ext

    if not torch.cuda.is_available():
        _load_failed = True
        return None

    try:
        from torch.utils.cpp_extension import load

        csrc_dir = os.path.join(os.path.dirname(__file__), "..", "..", "csrc")
        csrc_dir = os.path.abspath(csrc_dir)
        cu_file  = os.path.join(csrc_dir, "dequant_restore.cu")

        if not os.path.isfile(cu_file):
            _load_failed = True
            return None

        _ext = load(
            name="dequant_restore_ext",
            sources=[cu_file],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=False,
        )
    except Exception as e:
        import warnings
        warnings.warn(
            f"[cuda_kernels] Failed to compile CUDA extension, "
            f"falling back to PyTorch: {e}"
        )
        _load_failed = True
        _ext = None

    return _ext


# ---------------------------------------------------------------------------
# PyTorch fallback (exact same semantics as the CUDA kernel)
# ---------------------------------------------------------------------------

def _pytorch_fused_dequant_restore(
    packed:       torch.Tensor,
    scales:       torch.Tensor,
    sparse_vals:  torch.Tensor,
    sparse_idx:   torch.Tensor,
) -> torch.Tensor:
    """Pure-PyTorch reference implementation (used when CUDA ext is unavailable)."""
    from .quant_utils import unpack_int4, dequantize_symmetric
    from .outliers import restore_outliers

    unpacked     = unpack_int4(packed)
    dequantized  = dequantize_symmetric(unpacked, scales)
    reconstructed = restore_outliers(dequantized, sparse_vals, sparse_idx)
    return reconstructed


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fused_dequant_restore(
    packed:      torch.Tensor,
    scales:      torch.Tensor,
    sparse_vals: torch.Tensor,
    sparse_idx:  torch.Tensor,
    target_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Fused INT4 unpack + symmetric dequantize + outlier scatter.

    Args:
        packed       : int8 tensor (B, H, S, D//2), nibble-packed signed INT4
        scales       : fp16/bf16 tensor, broadcast-compatible with (B, H, S, D)
                       e.g. shape (1, H, 1, D) for K-cache
                            shape (B, H, S, 1) for V-cache
        sparse_vals  : fp16/bf16 tensor (N,) – outlier values
        sparse_idx   : int32  tensor (N,) – flat indices into (B, H, S, D)
        target_dtype : optional output dtype override (default: same as scales)

    Returns:
        Reconstructed tensor of shape (B, H, S, D) in target_dtype.
    """
    ext = _load_extension()

    if ext is not None and packed.is_cuda():
        # Ensure sparse tensors are on the right device
        if sparse_vals.numel() == 0:
            sparse_vals = sparse_vals.to(device=packed.device, dtype=scales.dtype)
            sparse_idx  = sparse_idx.to(device=packed.device, dtype=torch.int32)

        out = ext.fused_dequant_restore(
            packed.contiguous(),
            scales.contiguous(),
            sparse_vals.contiguous(),
            sparse_idx.contiguous(),
        )
    else:
        out = _pytorch_fused_dequant_restore(packed, scales, sparse_vals, sparse_idx)

    if target_dtype is not None and out.dtype != target_dtype:
        out = out.to(dtype=target_dtype)

    return out


def is_cuda_available() -> bool:
    """Return True if the custom CUDA extension was loaded successfully."""
    return _load_extension() is not None
