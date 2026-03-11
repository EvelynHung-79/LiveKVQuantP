# LiveKVQuant-P Memory Optimization Analysis

## 1. 現況：目前的方法做了什麼

### Chunk-wise Prefill 流程

將 input 切成 chunk_size=512 的片段，逐 chunk 處理：

```
Input: [chunk_0] [chunk_1] [chunk_2] ... [chunk_N]

For each chunk_i, for each of 32 layers:
  1. store_raw(k, v)                    → 暫存當前 chunk 的 FP16 KV
  2. get_full_kv() → reconstruct ALL    → 把 chunk 0..i 全部解壓成 FP16
  3. GQA expand (8→32 heads)            → 4x 放大 KV (repeat_interleave)
  4. SDPA(q[512], k_full[0..i], v_full) → 算 attention
  5. finalize() → 量化 chunk_i          → 壓縮回 INT8 + sparse outliers
```

### 為什麼 chunk_i 需要歷史 chunk 的 FP16 KV？

這是 Transformer causal attention 的本質，不是 bug：

- chunk_2 的 token 1024 需要 attend to token 0~1024
- 所以 SDPA 需要完整的 K, V tensor（chunk_0 到 chunk_i 的所有 KV）
- chunk-wise 省的是 **Q 的大小**（從 N 降到 chunk_size），不是 KV 的大小

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
  Q: 來自 chunk_i (512 tokens)
  K, V: 來自 chunk_0 到 chunk_i (所有歷史 + 當前)
```

---

## 2. Peak Memory 瓶頸分析

以最長輸入（narrativeqa, ~128K tokens）為例：

| 項目 | 大小 | 性質 | 說明 |
|---|---|---|---|
| 模型權重 | ~14,960 MB | 永久 | BF16, 不可壓縮 |
| 32 層壓縮 chunks (INT8+sparse) | ~8,000 MB | 永久（跨層累積） | 所有層的量化 KV |
| 1 層 FP16 reconstruct | ~500 MB | 暫時（每層各做） | get_full_kv() 解壓 |
| GQA expand (8→32 heads) | ~2,000 MB | 暫時 | repeat_interleave 真實複製 |
| Attention matrix (SDPA) | ~8,000 MB (math) 或 ~0 (flash) | 暫時 | **最大隱藏瓶頸** |

### 三個核心瓶頸

**瓶頸 1：Attention matrix（最容易被忽視，但最大）**
- 第 2 個 chunk 開始，q_len < kv_len，需要 attn_mask
- 傳入 attn_mask 導致 SDPA fallback 到 math backend
- Math backend 分配完整 attention matrix: `(1, 32_heads, 512, kv_len) * float32`
- 128K tokens 時 = **~8,000 MB**

**瓶頸 2：壓縮 chunks 仍佔 8 GB**
- INT8 比 FP16 省一半，但 32 層 × 128K tokens 仍然很大
- 且 reconstruct 時壓縮格式 + FP16 **同時存在**（雙倍持有）

**瓶頸 3：GQA expand**
- `repeat_interleave` 把 8 KV heads 複製成 32 heads
- 產生真實的記憶體複製，128K 時 ~2,000 MB

### 實際量測 vs 理論

```
理論 peak (flash attn): ~25,460 MB
理論 peak (math attn):  ~33,460 MB
實際量測:               ~29,508 MB
FullKV 實際量測:        ~30,944 MB
```

---

## 3. 優化方案

### 方案 A：修 SDPA 讓 FlashAttention 接手（最高優先）

**問題**：非第一個 chunk 時 q_len ≠ kv_len，目前用 `attn_mask` → SDPA fallback 到 math backend，分配巨大的 attention matrix。

**方案**：
- PyTorch 2.5+ 的 `F.scaled_dot_product_attention` 支持 `is_causal=True` 配合不等長 Q/K
- 它會自動做 bottom-right aligned causal mask，語意上等價於目前的手動 attn_mask
- 只需要修改 `attention_core.py` 的 SDPA 呼叫方式

**代碼位置**：`livekvquant/modules/attention_core.py` 第 51-62 行

**效果**：128K 時省 **~8,000 MB**（最大單點改善）

**風險**：低，不影響 score（數學上等價）

---

### 方案 B：GQA 用 expand 取代 repeat_interleave

**問題**：`k_full.repeat_interleave(n_rep, dim=1)` 會真的分配新 tensor，複製資料。

**方案**：
```python
# 現在（真實複製）：
k_full = k_full.repeat_interleave(n_rep, dim=1)  # 分配新 tensor

# 改為（虛擬展開，不複製）：
# (1, 8, seq, 128) → (1, 8, 1, seq, 128) → (1, 8, 4, seq, 128) → (1, 32, seq, 128)
k_full = k_full[:, :, None, :, :].expand(b, n_kv, n_rep, seq, hd).reshape(b, n_kv * n_rep, seq, hd)
```

**代碼位置**：`livekvquant/modules/attention_core.py` 第 38-41 行

**效果**：128K 時省 **~2,000 MB**

**風險**：低，不影響 score。但需確認 SDPA 是否接受 non-contiguous tensor（可能需要 `.contiguous()`，但通常 SDPA 內部會處理）

---

### 方案 C：INT4 Packing

**問題**：目前用 INT8 tensor 存 4-bit 量化值，每個元素浪費 4 bits。

**方案**：將兩個 4-bit 值 pack 到一個 INT8 裡：
```python
# Pack: 兩個 int4 → 一個 int8
packed = (high_nibble << 4) | (low_nibble & 0xF)

# Unpack:
high = packed >> 4
low = (packed << 4) >> 4  # sign-extend
```

**代碼位置**：`livekvquant/utils/quantization.py` 的 quantize/dequantize 函數

**效果**：壓縮 chunks 從 ~8,000 MB 降到 ~4,000 MB（128K tokens），省 **~4,000 MB**

**風險**：中等。pack/unpack 增加計算開銷（影響 latency），但不影響 score（數學上等價）

---

### 方案 D：Chunk-wise Attention（逐 chunk reconstruct + Online Softmax）

**問題**：`get_full_kv()` 一次 reconstruct 所有歷史 chunks 成完整 FP16 tensor，導致壓縮 + FP16 雙倍持有。

**方案**：不一次 reconstruct 全部，而是：
1. 逐 chunk 從壓縮格式 reconstruct 成 FP16（每次只有 1 chunk ~2 MB）
2. 計算該 chunk 對 Q 的 partial attention score
3. 用 Online Softmax (log-sum-exp trick) 累積跨 chunk 的 attention output
4. 釋放該 chunk 的 FP16，處理下一個

```python
# Pseudocode
running_max = -inf
running_sum = 0
running_output = 0

for chunk in compressed_chunks:
    k_i, v_i = chunk.reconstruct()        # 只解壓 1 chunk
    scores_i = Q @ k_i.T / sqrt(d)        # partial scores

    new_max = max(running_max, scores_i.max())
    correction = exp(running_max - new_max)

    running_output = running_output * correction + exp(scores_i - new_max) @ v_i
    running_sum = running_sum * correction + exp(scores_i - new_max).sum()
    running_max = new_max

    del k_i, v_i                           # 立即釋放

output = running_output / running_sum
```

**效果**：
- FP16 recon 從 ~500 MB 降到 ~2 MB（只持有 1 chunk）
- 同時也解決了 GQA expand 的問題（每次只 expand 1 chunk）
- 也不需要傳 attn_mask（逐 chunk 計算本身就是 causal 的）
- 128K 時省 **~2,500 MB**（recon + GQA）

**風險**：
- 高難度：需要自己實作 online softmax，不能用 SDPA
- 失去 FlashAttention 硬體加速，latency 可能變慢
- 但如果方案 A 成功，這個方案的邊際收益就小了

---

## 4. 優先建議排序

| 優先級 | 方案 | 省多少 (128K) | 難度 | 影響 score？ | 影響 latency？ |
|---|---|---|---|---|---|
| 1 | A: 修 SDPA 用 Flash | **~8,000 MB** | 低 | 不影響 | 可能更快 |
| 2 | B: GQA expand 不 copy | **~2,000 MB** | 低 | 不影響 | 不影響 |
| 3 | C: INT4 packing | **~4,000 MB** | 中 | 不影響 | 略慢 |
| 4 | D: Chunk-wise attention | **~2,500 MB** | 高 | 可能微小 | 可能慢 |

### 建議策略

1. **先做 A + B**（改動最小、效果最大、風險最低）：預計 128K 省 ~10 GB
2. **再做 C**（中等改動、不錯效果）：再省 ~4 GB
3. **D 視情況**：如果 A+B+C 之後 memory 已可接受，D 不需要做

### A+B 做完後的預估 peak（128K tokens）

```
模型權重:           14,960 MB
32 層壓縮 chunks:    8,000 MB (INT8)
1 層 FP16 recon:       500 MB
GQA (expand, 不 copy):   0 MB
Attention (flash):       ~0 MB
────────────────────────────
預估 peak:          ~23,460 MB (vs 現在 ~29,508 MB, 省 ~6 GB)
```

### A+B+C 做完後的預估 peak（128K tokens）

```
模型權重:           14,960 MB
32 層壓縮 chunks:    4,000 MB (packed INT4)
1 層 FP16 recon:       500 MB
GQA (expand, 不 copy):   0 MB
Attention (flash):       ~0 MB
────────────────────────────
預估 peak:          ~19,460 MB (vs FullKV ~30,944 MB, 省 ~37%)
```

---

## 5. 目前優化（v1）的成果回顧

基於 0306-0307 的最新結果 vs 優化前結果 vs FullKV baseline：

### Score（15 個共同 task）
- 新 LiveKV 平均: **0.4497** vs 舊: 0.4475 → **+0.0022 提升**
- 8/15 個 task 改善或持平
- 新版 output 與 FullKV 的一致性更高（multifieldqa_en: 85/150 vs 63/150）

### Memory
- 新 LiveKV 平均: **21,154 MB** vs 舊: 22,379 MB → **-1,225 MB (-5.5%)**
- narrativeqa 反而增加（+3,830 MB），因 store_raw 的雙倍持有問題
- 長 output task 改善明顯（gov_report -6,395 MB, repobench-p -4,824 MB）

### Latency
- 新 LiveKV 平均: **9,513 ms** vs 舊: 11,157 ms → **-1,644 ms (-14.7%)**
- QA 類 task 改善最大（narrativeqa -13s, musique -3s）
- 仍比 FullKV 慢 ~1.9x（量化/解壓開銷 + SDPA fallback）
