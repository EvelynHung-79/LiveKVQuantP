# CUDA Kernel 設計計劃：Fused Dequant + Outlier Restore

## 動機

目前 `KVChunk.reconstruct()` 在 decode 每一步都需要重建壓縮的 KV cache。
這個重建流程包含三個 **獨立的 PyTorch 操作**，每個都是獨立的 CUDA kernel launch 加上 global memory roundtrip：

```
步驟 1: unpack_int4(packed)              → INT8 tensor        (kernel #1)
步驟 2: dequantize_symmetric(int8, scale) → FP16 tensor        (kernel #2)
步驟 3: restore_outliers(fp16, vals, idx) → FP16 tensor (final) (kernel #3)
```

對於長序列（LongBench，典型 prefill length 8K-32K），每次 decode step 需要對
**所有 layers × chunks** 重複這個三步流程，造成大量不必要的 kernel launch overhead
與 DRAM 讀寫放大。

**目標**：把步驟 1-3 合成一個 fused CUDA kernel，大幅降低 decode 的 memory bandwidth 消耗與 kernel launch overhead。

---

## 張量 Shape 分析

### K Cache
| Tensor | Shape | Dtype |
|--------|-------|-------|
| `packed` | `(B, H, S, D//2)` | `int8` |
| `scale` | `(1, H, 1, D)` | `fp16/bf16` |
| `sparse_values` | `(N_outliers,)` | `fp16/bf16` |
| `sparse_indices` | `(N_outliers,)` | `int32` |
| output | `(B, H, S, D)` | `fp16/bf16` |

Scale broadcast 方向：per-channel（D 維度），每個 head 的每個 channel 共用同一個 scale。

### V Cache
| Tensor | Shape | Dtype |
|--------|-------|-------|
| `packed` | `(B, H, S, D//2)` | `int8` |
| `scale` | `(B, H, S, 1)` | `fp16/bf16` |
| `sparse_values` | `(N_outliers,)` | `fp16/bf16` |
| `sparse_indices` | `(N_outliers,)` | `int32` |
| output | `(B, H, S, D)` | `fp16/bf16` |

Scale broadcast 方向：per-token（S 維度），每個 token 的所有 channel 共用同一個 scale。

---

## INT4 Packing 格式

沿用現有 `pack_int4` 的格式：
```
packed_byte = (lo & 0xF) | ((hi & 0xF) << 4)
```
- `lo` = even index (d = 2k)，存於低 4 bits
- `hi` = odd index (d = 2k+1)，存於高 4 bits
- 兩者均為 signed INT4，值域 [-7, 7]

Unpack 時需要做 sign extension（4-bit → 8-bit）：
```c
int32_t p32 = (int32_t)(uint8_t)packed_byte;
int8_t lo = (int8_t)(((p32 & 0xF) << 28) >> 28);
int8_t hi = (int8_t)(((p32 >> 4)  << 28) >> 28);
```

---

## Kernel 設計

### Kernel 1：`unpack_dequant_sym_kernel`

**功能**：fuse INT4 unpack + symmetric dequantize。

**Thread mapping**：
- 每個 thread 處理 **1 個 packed byte**（= 2 個 output elements）
- 總 thread 數 = `B × H × S × (D/2)`

**Index 計算**：
```
tid = blockIdx.x * blockDim.x + threadIdx.x
d2  = tid % (D/2)
s   = (tid / (D/2)) % S
h   = (tid / (D/2) / S) % H
b   = (tid / (D/2) / S / H)

d_lo = 2 * d2
d_hi = 2 * d2 + 1

// Scale 用 stride-based 索引，同時支援 (1,H,1,D) 和 (B,H,S,1)
scale_idx_lo = b*sB + h*sH + s*sS + d_lo*sD
scale_idx_hi = b*sB + h*sH + s*sS + d_hi*sD

// Output 為 row-major (B,H,S,D)
out_lo = b*(H*S*D) + h*(S*D) + s*D + d_lo
out_hi = b*(H*S*D) + h*(S*D) + s*D + d_hi

output[out_lo] = (float)lo * (float)scales[scale_idx_lo]
output[out_hi] = (float)hi * (float)scales[scale_idx_hi]
```

Scale 用 **PyTorch stride**（`tensor.stride(i)`）傳入，自動相容兩種 broadcast 模式。

### Kernel 2：`scatter_outliers_kernel`

**功能**：將 sparse outlier values scatter 回 output tensor。

**Thread mapping**：
- 每個 thread 處理 **1 個 outlier**
- 總 thread 數 = `num_outliers`
- `sparse_indices` 是 flat index（對應 output 的 flatten view）

```c
output[sparse_indices[tid]] = sparse_values[tid];
```

Outlier 之間沒有重疊（by construction），不需要 atomic。

---

## Python 整合

### 檔案結構

```
csrc/
  dequant_restore.cu          ← CUDA kernels
livekvquant/utils/
  cuda_kernels.py             ← 懶加載 wrapper，含 PyTorch fallback
livekvquant/modules/
  chunk.py                    ← 使用 try_fused_dequant_restore()
```

### `cuda_kernels.py` 設計

```python
# 懶加載：第一次 import 時 JIT 編譯，之後 cache
_ext = None

def _load_ext():
    global _ext
    if _ext is None:
        # 用 torch.utils.cpp_extension.load() JIT 編譯
        ...
    return _ext

def fused_dequant_restore(packed, scale, sparse_vals, sparse_idx) -> torch.Tensor:
    """
    若 CUDA 可用且編譯成功，使用 fused kernel；
    否則 fallback 到純 PyTorch 實作。
    """
    ...
```

### `chunk.py` 修改

在 `KVChunk.reconstruct()` 的 symmetric quantized path 加上 fused kernel：

```python
# 舊：
unpacked    = unpack_int4(self.quantized_data)
dequantized = dequantize_symmetric(unpacked, self.scale)
reconstructed = restore_outliers(dequantized, self.sparse_values, self.sparse_indices)

# 新：
reconstructed = fused_dequant_restore(
    self.quantized_data, self.scale,
    self.sparse_values, self.sparse_indices
)
```

---

## 預期效益

| 指標 | Before | After |
|------|--------|-------|
| Kernel launches / chunk | 3 | 1 (dequant) + 1 (scatter) = 2 |
| Global memory reads | packed×1 + int8×1 + fp16×1 | packed×1 |
| Global memory writes | int8×1 + fp16×1 + fp16×1 | fp16×1 |
| Decode latency (long-ctx) | baseline | 預期降低 30–50% |

主要節省來自：
1. **消除 INT8 中間 tensor** 的 allocate + write + read roundtrip
2. **減少 kernel launch overhead**（對短序列 / 小 batch 尤其顯著）
3. 提升 **L2 cache 命中率**（packed data 更小，更容易 cache）

---

## 測試計劃

1. **Correctness test**：對比 fused kernel 與 PyTorch reference 的輸出（relative error < 1e-3）
2. **K scale test**：scale shape `(1, H, 1, D)`
3. **V scale test**：scale shape `(B, H, S, 1)`
4. **Zero outliers test**：`num_outliers = 0` 的 edge case
5. **dtype test**：FP16 和 BF16 兩種

---

## 未來擴展（Out of Scope）

- Asymmetric (uint4) 路徑：架構相同，只需修改 unpack 與 scale 計算
- Fused quantize kernel（prefill 加速）：可另立 kernel，fuse outlier isolation + EMA update + quantize
- FlashAttention 整合：直接在 attention kernel 內 dequant，省掉 output 寫回（最大優化空間）
