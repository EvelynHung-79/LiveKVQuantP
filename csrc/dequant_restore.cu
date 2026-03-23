/*
 * dequant_restore.cu
 *
 * Fused CUDA kernels for KV cache INT4 dequantization + outlier restoration.
 *
 * Operations fused (replaces 3 separate PyTorch ops):
 *   1. INT4 nibble unpack  (int8 packed → int8 unpacked)
 *   2. Symmetric dequant   (int8 × scale → fp16/bf16)
 *   3. Outlier scatter     (sparse fp16/bf16 values → output)
 *
 * Supports both K-scale shape (1, H, 1, D) and V-scale shape (B, H, S, 1)
 * via stride-based indexing passed from PyTorch.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Sign-extend a 4-bit value to int8 using 32-bit arithmetic (safe on all archs)
__device__ __forceinline__ int8_t sign_extend_lo4(uint8_t p) {
    int32_t v = (int32_t)(p & 0xF);
    return (int8_t)(((v << 28) >> 28));
}

__device__ __forceinline__ int8_t sign_extend_hi4(uint8_t p) {
    int32_t v = (int32_t)((p >> 4) & 0xF);
    return (int8_t)(((v << 28) >> 28));
}

// ---------------------------------------------------------------------------
// Kernel 1: unpack INT4 + symmetric dequantize
//
// Thread mapping: 1 thread per packed byte (= 2 output elements)
// Total threads : B * H * S * (D/2)
//
// Scale stride parameters support arbitrary broadcast shapes, e.g.:
//   K-scale  (1, H, 1, D) -> sB=H*D, sH=D, sS=D, sD=1
//   V-scale  (B, H, S, 1) -> sB=H*S, sH=S, sS=1,  sD=1
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void unpack_dequant_sym_kernel(
    const int8_t*   __restrict__ packed,      // [B, H, S, D/2]  int8
    const scalar_t* __restrict__ scales,      // broadcast-able scale
    scalar_t*       __restrict__ output,      // [B, H, S, D]    scalar_t
    int B, int H, int S, int D,
    int64_t sB, int64_t sH, int64_t sS, int64_t sD   // scale strides
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int D2  = D / 2;
    int total = B * H * S * D2;
    if (tid >= total) return;

    // Decode (b, h, s, d2) from flat tid in (B, H, S, D/2) row-major layout
    int d2 = tid % D2;
    int s  = (tid / D2) % S;
    int h  = (tid / D2 / S) % H;
    int b  =  tid / D2 / S / H;

    int d_lo = d2 * 2;
    int d_hi = d2 * 2 + 1;

    // Unpack: low nibble = even channel, high nibble = odd channel
    uint8_t p = (uint8_t)packed[tid];
    int8_t lo = sign_extend_lo4(p);
    int8_t hi = sign_extend_hi4(p);

    // Scale lookup via strides (handles any broadcast shape)
    int64_t scale_base = (int64_t)b * sB + (int64_t)h * sH + (int64_t)s * sS;
    float scale_lo = (float)scales[scale_base + (int64_t)d_lo * sD];
    float scale_hi = (float)scales[scale_base + (int64_t)d_hi * sD];

    // Write output in (B, H, S, D) row-major layout
    int out_base = b * (H * S * D) + h * (S * D) + s * D;
    output[out_base + d_lo] = (scalar_t)((float)lo * scale_lo);
    output[out_base + d_hi] = (scalar_t)((float)hi * scale_hi);
}

// ---------------------------------------------------------------------------
// Kernel 2: scatter outlier values back into the output tensor
//
// Thread mapping: 1 thread per outlier
// sparse_indices: flat indices into the flattened output tensor
// No atomics needed because outlier positions are guaranteed non-overlapping.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void scatter_outliers_kernel(
    scalar_t*       __restrict__ output,       // flattened [B*H*S*D]
    const scalar_t* __restrict__ sparse_vals,  // [num_outliers]
    const int32_t*  __restrict__ sparse_idx,   // [num_outliers]  flat indices
    int num_outliers
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_outliers) return;
    output[sparse_idx[tid]] = sparse_vals[tid];
}

// ---------------------------------------------------------------------------
// C++ dispatch helpers
// ---------------------------------------------------------------------------

#define BLOCK_SIZE 256

static void launch_unpack_dequant(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor output
) {
    int B = packed.size(0);
    int H = packed.size(1);
    int S = packed.size(2);
    int D = output.size(3);   // unpacked dim
    int total_threads = B * H * S * (D / 2);

    int64_t sB = scales.stride(0);
    int64_t sH = scales.stride(1);
    int64_t sS = scales.stride(2);
    int64_t sD = scales.stride(3);

    int blocks = (total_threads + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(scales.scalar_type(), "unpack_dequant_sym", [&]() {
        unpack_dequant_sym_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
            packed.data_ptr<int8_t>(),
            scales.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            B, H, S, D,
            sB, sH, sS, sD
        );
    });
}

static void launch_scatter_outliers(
    torch::Tensor output,
    torch::Tensor sparse_vals,
    torch::Tensor sparse_idx
) {
    int num_outliers = sparse_idx.size(0);
    if (num_outliers == 0) return;

    int blocks = (num_outliers + BLOCK_SIZE - 1) / BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(sparse_vals.scalar_type(), "scatter_outliers", [&]() {
        scatter_outliers_kernel<scalar_t><<<blocks, BLOCK_SIZE>>>(
            output.data_ptr<scalar_t>(),
            sparse_vals.data_ptr<scalar_t>(),
            sparse_idx.data_ptr<int32_t>(),
            num_outliers
        );
    });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/*
 * fused_dequant_restore
 *
 * Inputs:
 *   packed       : int8  tensor of shape (B, H, S, D//2), nibble-packed INT4
 *   scales       : fp16/bf16 tensor, broadcast-compatible with (B, H, S, D)
 *                  e.g. (1, H, 1, D) for K-cache or (B, H, S, 1) for V-cache
 *   sparse_vals  : fp16/bf16 tensor of shape (N,)  – outlier values
 *   sparse_idx   : int32 tensor of shape (N,)       – flat indices into (B,H,S,D)
 *
 * Returns:
 *   output       : fp16/bf16 tensor of shape (B, H, S, D)
 */
torch::Tensor fused_dequant_restore(
    torch::Tensor packed,
    torch::Tensor scales,
    torch::Tensor sparse_vals,
    torch::Tensor sparse_idx
) {
    TORCH_CHECK(packed.is_cuda(),      "packed must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(),      "scales must be a CUDA tensor");
    TORCH_CHECK(packed.dtype() == torch::kInt8, "packed must be int8");
    TORCH_CHECK(packed.dim() == 4,     "packed must be 4-D (B, H, S, D//2)");
    TORCH_CHECK(scales.dim() == 4,     "scales must be 4-D");

    int B = packed.size(0);
    int H = packed.size(1);
    int S = packed.size(2);
    int D = packed.size(3) * 2;  // unpacked head_dim

    // Allocate output tensor (same dtype as scales, same device)
    auto output = torch::empty({B, H, S, D},
                               torch::TensorOptions()
                                   .dtype(scales.dtype())
                                   .device(packed.device()));

    // Step 1+2: unpack INT4 + dequantize
    launch_unpack_dequant(packed, scales, output);

    // Step 3: scatter outliers (if any)
    if (sparse_vals.numel() > 0) {
        TORCH_CHECK(sparse_idx.dtype() == torch::kInt32, "sparse_idx must be int32");
        launch_scatter_outliers(output, sparse_vals, sparse_idx);
    }

    return output;
}

// ---------------------------------------------------------------------------
// pybind11 module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "fused_dequant_restore",
        &fused_dequant_restore,
        "Fused INT4 dequant + outlier scatter (CUDA)",
        py::arg("packed"),
        py::arg("scales"),
        py::arg("sparse_vals"),
        py::arg("sparse_idx")
    );
}
