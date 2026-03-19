"""
test_cuda_kernel.py

Tests for the fused INT4 dequant + outlier scatter kernel.
Verifies that the CUDA kernel (or its PyTorch fallback) produces results
numerically equivalent to the reference PyTorch pipeline.

Run with:
    python -m pytest tests/test_cuda_kernel.py -v
or:
    python tests/test_cuda_kernel.py
"""

import sys
import os
import torch
import pytest

# Add project root to path so we can import submodules directly
# (avoids livekvquant/__init__.py which requires transformers)
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

# Import submodules directly to bypass the top-level __init__.py
import importlib
_quant_utils   = importlib.import_module("livekvquant.utils.quant_utils")
_outliers_mod  = importlib.import_module("livekvquant.utils.outliers")
_cuda_kernels  = importlib.import_module("livekvquant.utils.cuda_kernels")

pack_int4            = _quant_utils.pack_int4
unpack_int4          = _quant_utils.unpack_int4
dequantize_symmetric = _quant_utils.dequantize_symmetric
isolate_outliers     = _outliers_mod.isolate_outliers
restore_outliers     = _outliers_mod.restore_outliers
fused_dequant_restore = _cuda_kernels.fused_dequant_restore
is_cuda_available    = _cuda_kernels.is_cuda_available


def _reference_reconstruct(packed, scale, sparse_vals, sparse_idx, dtype):
    """Exact same logic as the original PyTorch path in chunk.py."""
    unpacked = unpack_int4(packed)
    dequantized = dequantize_symmetric(unpacked, scale)
    restored = restore_outliers(dequantized, sparse_vals, sparse_idx)
    return restored.to(dtype)


def _make_quantized_chunk(B, H, S, D, scale_shape, dtype, device, outlier_ratio=0.01):
    """
    Helper: build a synthetic quantized KV chunk and return all necessary tensors.

    Returns:
        packed       : int8 (B, H, S, D//2)
        scale        : dtype, shape=scale_shape
        sparse_vals  : dtype (N,)
        sparse_idx   : int32 (N,)
        original_fp  : dtype (B, H, S, D)  – the original FP tensor before quantization
    """
    torch.manual_seed(42)

    # Synthetic original tensor in FP
    original = torch.randn(B, H, S, D, dtype=dtype, device=device) * 0.5

    # Compute scale matching scale_shape
    if scale_shape == (1, H, 1, D):
        absmax = torch.amax(original.abs(), dim=2, keepdim=True)  # (B, H, 1, D) but B=1
        absmax = torch.amax(absmax, dim=0, keepdim=True)           # (1, H, 1, D)
    elif scale_shape == (B, H, S, 1):
        absmax = torch.amax(original.abs(), dim=3, keepdim=True)   # (B, H, S, 1)
    else:
        raise ValueError(f"Unsupported scale_shape: {scale_shape}")

    max_int4 = 7.0
    scale = (absmax + 1e-6) / max_int4
    scale = scale.to(dtype)

    # Quantize dense part
    quantized = torch.round(original / scale).clamp(-max_int4, max_int4).to(torch.int8)

    # Isolate outliers
    dense, sparse_vals, sparse_idx = isolate_outliers(original, ratio=outlier_ratio)

    # Pack INT4
    packed = pack_int4(quantized)

    return packed, scale, sparse_vals, sparse_idx, original


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Skip all tests gracefully on CPU without CUDA
NEEDS_GPU = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA device not available"
)


class TestFusedDequantRestoreCorrectness:
    """Correctness: fused output == reference output within tolerance."""

    def _run_test(self, B, H, S, D, scale_shape, dtype, outlier_ratio=0.01):
        packed, scale, sparse_vals, sparse_idx, _ = _make_quantized_chunk(
            B, H, S, D, scale_shape, dtype, DEVICE, outlier_ratio
        )

        ref = _reference_reconstruct(packed, scale, sparse_vals, sparse_idx, dtype)
        out = fused_dequant_restore(packed, scale, sparse_vals, sparse_idx, target_dtype=dtype)

        assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
        assert out.dtype == ref.dtype, f"Dtype mismatch: {out.dtype} vs {ref.dtype}"

        max_diff = (out.float() - ref.float()).abs().max().item()
        # Allow tiny floating-point rounding differences
        assert max_diff < 1e-3, (
            f"max_diff={max_diff:.6f} exceeds threshold for "
            f"B={B} H={H} S={S} D={D} scale={scale_shape} dtype={dtype}"
        )

    # ── K-scale: (1, H, 1, D) ──

    def test_k_scale_fp16_small(self):
        self._run_test(B=1, H=4, S=8, D=16, scale_shape=(1, 4, 1, 16), dtype=torch.float16)

    def test_k_scale_fp16_typical(self):
        self._run_test(B=1, H=32, S=512, D=128, scale_shape=(1, 32, 1, 128), dtype=torch.float16)

    def test_k_scale_bf16(self):
        self._run_test(B=1, H=8, S=64, D=64, scale_shape=(1, 8, 1, 64), dtype=torch.bfloat16)

    # ── V-scale: (B, H, S, 1) ──

    def test_v_scale_fp16_small(self):
        self._run_test(B=1, H=4, S=8, D=16, scale_shape=(1, 4, 8, 1), dtype=torch.float16)

    def test_v_scale_fp16_typical(self):
        self._run_test(B=1, H=32, S=512, D=128, scale_shape=(1, 32, 512, 1), dtype=torch.float16)

    def test_v_scale_bf16(self):
        self._run_test(B=1, H=8, S=64, D=64, scale_shape=(1, 8, 64, 1), dtype=torch.bfloat16)

    # ── Edge cases ──

    def test_zero_outliers(self):
        """No outliers: sparse_vals and sparse_idx are empty tensors."""
        B, H, S, D = 1, 4, 8, 16
        packed, scale, _, _, _ = _make_quantized_chunk(
            B, H, S, D, (1, 4, 1, 16), torch.float16, DEVICE, outlier_ratio=0.0
        )
        empty_vals = torch.empty(0, dtype=torch.float16, device=DEVICE)
        empty_idx  = torch.empty(0, dtype=torch.int32,  device=DEVICE)

        ref = _reference_reconstruct(packed, scale, empty_vals, empty_idx, torch.float16)
        out = fused_dequant_restore(packed, scale, empty_vals, empty_idx,
                                    target_dtype=torch.float16)

        max_diff = (out.float() - ref.float()).abs().max().item()
        assert max_diff < 1e-3

    def test_high_outlier_ratio(self):
        """5% outliers – more scatter operations."""
        self._run_test(B=1, H=4, S=32, D=32, scale_shape=(1, 4, 1, 32),
                       dtype=torch.float16, outlier_ratio=0.05)

    def test_single_token(self):
        """S=1 edge case (decode step)."""
        self._run_test(B=1, H=8, S=1, D=128, scale_shape=(1, 8, 1, 128), dtype=torch.float16)

    def test_target_dtype_cast(self):
        """target_dtype different from scale dtype should still work."""
        B, H, S, D = 1, 4, 8, 16
        packed, scale, sv, si, _ = _make_quantized_chunk(
            B, H, S, D, (1, 4, 1, 16), torch.float16, DEVICE
        )
        out = fused_dequant_restore(packed, scale, sv, si, target_dtype=torch.float32)
        assert out.dtype == torch.float32


class TestKVChunkIntegration:
    """End-to-end: KVChunk.reconstruct() uses fused kernel after pack()."""

    def test_chunk_reconstruct_symmetric_packed(self):
        KVChunk = importlib.import_module("livekvquant.modules.chunk").KVChunk

        B, H, S, D = 1, 8, 32, 64
        packed, scale, sparse_vals, sparse_idx, _ = _make_quantized_chunk(
            B, H, S, D, (1, 8, 1, 64), torch.float16, DEVICE
        )

        # Build a pre-packed chunk (simulating post pack_all_chunks state)
        chunk = KVChunk(
            chunk_type="quantized",
            quantized_data=packed,
            scale=scale,
            sparse_values=sparse_vals,
            sparse_indices=sparse_idx,
            _unpacked_last_dim=D,
            _is_packed=True,
        )

        ref = _reference_reconstruct(packed, scale, sparse_vals, sparse_idx, torch.float16)
        out = chunk.reconstruct(target_dtype=torch.float16)

        assert out.shape == (B, H, S, D)
        max_diff = (out.float() - ref.float()).abs().max().item()
        assert max_diff < 1e-3, f"KVChunk.reconstruct max_diff={max_diff:.6f}"

    def test_chunk_reconstruct_symmetric_unpacked(self):
        """Unpacked quantized chunk (prefill phase) still uses PyTorch path."""
        KVChunk = importlib.import_module("livekvquant.modules.chunk").KVChunk
        quantize_symmetric = _quant_utils.quantize_symmetric

        B, H, S, D = 1, 4, 16, 32
        scale = torch.ones(1, H, 1, D, dtype=torch.float16, device=DEVICE) * 0.1
        x = torch.randn(B, H, S, D, dtype=torch.float16, device=DEVICE)
        q = quantize_symmetric(x, scale)

        dense, sv, si = isolate_outliers(x, ratio=0.01)

        chunk = KVChunk(
            chunk_type="quantized",
            quantized_data=q,
            scale=scale,
            sparse_values=sv,
            sparse_indices=si,
            _is_packed=False,
        )

        # Should not crash; result should be close to x
        out = chunk.reconstruct(target_dtype=torch.float16)
        assert out.shape == (B, H, S, D)


# ──────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    cuda_status = "YES (custom ext)" if is_cuda_available() else "NO (PyTorch fallback)"
    print(f"CUDA extension loaded: {cuda_status}")
    print(f"Running on device: {DEVICE}\n")

    suite = TestFusedDequantRestoreCorrectness()
    tests = [
        ("K-scale FP16 small",    suite.test_k_scale_fp16_small),
        ("K-scale FP16 typical",  suite.test_k_scale_fp16_typical),
        ("K-scale BF16",          suite.test_k_scale_bf16),
        ("V-scale FP16 small",    suite.test_v_scale_fp16_small),
        ("V-scale FP16 typical",  suite.test_v_scale_fp16_typical),
        ("V-scale BF16",          suite.test_v_scale_bf16),
        ("Zero outliers",         suite.test_zero_outliers),
        ("High outlier ratio",    suite.test_high_outlier_ratio),
        ("Single token (S=1)",    suite.test_single_token),
        ("target_dtype cast",     suite.test_target_dtype_cast),
    ]

    integration = TestKVChunkIntegration()
    tests += [
        ("KVChunk packed",        integration.test_chunk_reconstruct_symmetric_packed),
        ("KVChunk unpacked",      integration.test_chunk_reconstruct_symmetric_unpacked),
    ]

    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1

    print(f"\n{passed}/{passed+failed} tests passed.")
    sys.exit(0 if failed == 0 else 1)
