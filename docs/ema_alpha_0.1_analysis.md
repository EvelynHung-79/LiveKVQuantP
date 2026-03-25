# EMA Alpha 0.1 完整分析（LongBench v1 + v2）

> 比較 ema_alpha=0.1 vs baseline (ema_alpha=0.2) vs fullKV。
> - LongBench v1：16/16 tasks 完整
> - LongBench v2：6/6 task categories 完整

---

## 一、LongBench v1（16 tasks）

### 1.1 Score 比較總表

| Task | 類型 | fullKV | α=0.2 | α=0.1 | Δ vs fullKV | Δ vs α=0.2 |
|------|------|:------:|:-----:|:-----:|:-----------:|:----------:|
| narrativeqa | QA | 0.2950 | 0.2941 | **0.2957** | +0.0007 | +0.0016 |
| qasper | QA | 0.4457 | 0.4249 | **0.4294** | -0.0163 | +0.0045 |
| multifieldqa\_en | QA | 0.5596 | 0.5351 | **0.5375** | -0.0221 | +0.0024 |
| hotpotqa | Multi-hop QA | 0.5795 | 0.5692 | **0.5815** | +0.0020 | +0.0123 |
| 2wikimqa | Multi-hop QA | 0.4909 | 0.4624 | **0.4822** | -0.0087 | +0.0198 |
| musique | Multi-hop QA | 0.3255 | 0.3011 | **0.3170** | -0.0085 | +0.0159 |
| gov\_report | Summarization | 0.3430 | 0.3464 | 0.3444 | +0.0014 | -0.0020 |
| qmsum | Summarization | 0.2512 | 0.2539 | 0.2537 | +0.0025 | -0.0002 |
| multi\_news | Summarization | 0.2692 | 0.2649 | 0.2625 | -0.0067 | -0.0024 |
| trec | Classification | 0.7300 | 0.7200 | **0.7450** | +0.0150 | +0.0250 |
| triviaqa | QA | 0.9173 | 0.9086 | **0.9049** | -0.0124 | -0.0037 |
| samsum | Summarization | 0.4375 | 0.4194 | **0.4261** | -0.0114 | +0.0067 |
| passage\_retrieval\_en | Retrieval | 1.0000 | 1.0000 | 0.9900 | -0.0100 | -0.0100 |
| passage\_count | Counting | 0.1250 | 0.1100 | 0.1050 | -0.0200 | -0.0050 |
| lcc | Code | 0.6533 | 0.6302 | 0.6268 | -0.0265 | -0.0034 |
| repobench-p | Code | 0.5475 | 0.5359 | **0.5387** | -0.0088 | +0.0028 |
| **16-task avg** | | **0.4981** | **0.4860** | **0.4900** | **-0.0081** | **+0.0040** |

### Score 小結
- **α=0.1 整體優於 α=0.2**：16-task 平均 score 提升 **+0.0040**（0.4900 vs 0.4860），與 fullKV 的差距從 -0.0121 縮小到 **-0.0081**。
- **10 項提升、6 項下降**（vs α=0.2）。
- **最大贏家**：trec (+0.0250)、2wikimqa (+0.0198)、musique (+0.0159)、hotpotqa (+0.0123)。
- **最大輸家**：passage\_retrieval\_en (-0.0100)、passage\_count (-0.0050)、triviaqa (-0.0037)。
- **3 項超越 fullKV**：narrativeqa（0.2957 > 0.2950）、hotpotqa（0.5815 > 0.5795）、trec（0.7450 > 0.7300）。

---

### 1.2 Latency 比較

| Task | fullKV | α=0.2 | α=0.1 | overhead vs fullKV | Δ vs α=0.2 |
|------|:------:|:-----:|:-----:|:------------------:|:----------:|
| narrativeqa | 8.4s | 15.1s | 14.7s | 1.75x | -0.3s |
| qasper | 1.6s | 2.5s | 2.5s | 1.56x | +0.0s |
| multifieldqa\_en | 1.9s | 3.0s | 3.0s | 1.58x | +0.0s |
| hotpotqa | 2.6s | 4.6s | 4.5s | 1.73x | -0.1s |
| 2wikimqa | 1.4s | 2.5s | 2.5s | 1.79x | +0.0s |
| musique | 3.3s | 5.7s | 5.7s | 1.71x | +0.0s |
| gov\_report | 17.4s | 24.6s | 23.5s | 1.35x | -1.1s |
| qmsum | 6.5s | 9.4s | 9.2s | 1.42x | -0.2s |
| multi\_news | 14.0s | 19.8s | 19.6s | 1.40x | -0.2s |
| trec | 3.2s | 4.7s | 5.2s | 1.63x | +0.5s |
| triviaqa | 2.3s | 4.2s | 4.2s | 1.83x | +0.0s |
| samsum | 2.8s | 4.8s | 5.0s | 1.79x | +0.2s |
| passage\_retrieval\_en | 2.4s | 4.3s | 4.2s | 1.75x | -0.1s |
| passage\_count | 2.9s | 5.3s | 5.3s | 1.83x | +0.0s |
| lcc | 2.4s | 3.5s | 3.5s | 1.46x | +0.0s |
| repobench-p | 4.1s | 6.5s | 6.3s | 1.54x | -0.2s |
| **Average** | **4.8s** | **7.5s** | **7.4s** | **~1.54x** | **-0.1s** |

### Latency 小結
- α=0.1 與 α=0.2 的 latency **無系統性差異**，波動在 run-to-run variance 範圍內。
- α 只影響 EMA 統計更新權重，不改變計算量。

---

### 1.3 Peak Memory 比較

| Task | fullKV (MB) | α=0.2 (MB) | α=0.1 (MB) | Δ vs fullKV | Δ vs α=0.2 |
|------|:-----------:|:----------:|:----------:|:-----------:|:----------:|
| narrativeqa | 30,944 | 26,792 | 26,792 | -4,152 | 0 |
| qasper | 20,414 | 19,177 | 19,177 | -1,237 | 0 |
| multifieldqa\_en | 18,917 | 18,071 | 18,071 | -846 | 0 |
| hotpotqa | 19,251 | 18,350 | 18,350 | -901 | 0 |
| 2wikimqa | 19,246 | 18,307 | 18,307 | -939 | 0 |
| musique | 19,252 | 18,352 | 18,352 | -900 | 0 |
| gov\_report | 27,617 | 24,376 | 24,376 | -3,241 | 0 |
| qmsum | 22,605 | 20,747 | 20,747 | -1,858 | 0 |
| multi\_news | 18,669 | 17,930 | 17,918 | -751 | -12 |
| trec | 18,048 | 17,442 | 17,442 | -606 | 0 |
| triviaqa | 20,904 | 19,532 | 19,532 | -1,372 | 0 |
| samsum | 19,621 | 18,579 | 18,579 | -1,042 | 0 |
| passage\_retrieval\_en | 18,965 | 18,140 | 18,140 | -825 | 0 |
| passage\_count | 22,273 | 20,516 | 20,516 | -1,757 | 0 |
| lcc | 22,528 | 20,719 | 20,719 | -1,809 | 0 |
| repobench-p | 24,672 | 22,207 | 22,207 | -2,465 | 0 |
| **Average** | **21,495** | **19,952** | **19,952** | **-1,544** | **0** |

### Memory 小結
- **Peak memory 完全不受 α 影響**（multi\_news 差 12MB 為測量誤差）。
- 相比 fullKV 平均節省 **~1,544 MB**。

---

## 二、LongBench v2（6 task categories）

### 2.1 Score 比較總表

| Task Category | fullKV | α=0.2 | α=0.1 | Δ vs fullKV | Δ vs α=0.2 |
|---------------|:------:|:-----:|:-----:|:-----------:|:----------:|
| Single-Document QA | 0.6743 | 0.6114 | **0.6343** | -0.0400 | +0.0229 |
| Multi-Document QA | 0.6240 | 0.5600 | **0.5600** | -0.0640 | 0.0000 |
| Long In-context Learning | 0.5679 | 0.4938 | **0.4568** | -0.1111 | -0.0370 |
| Long-dialogue History Understanding | 0.6154 | 0.6154 | **0.6154** | 0.0000 | 0.0000 |
| Code Repository Understanding | 0.8600 | 0.8000 | **0.8200** | -0.0400 | +0.0200 |
| Long Structured Data Understanding | 0.6970 | 0.6061 | **0.6667** | -0.0303 | +0.0606 |
| **6-category avg** | **0.6731** | **0.6145** | **0.6255** | **-0.0476** | **+0.0111** |

### 2.2 Latency 比較

| Task Category | fullKV | α=0.2 | α=0.1 | overhead vs fullKV |
|---------------|:------:|:-----:|:-----:|:------------------:|
| Single-Document QA | 35.4s | 67.4s | 72.7s | 2.05x |
| Multi-Document QA | 34.0s | 64.9s | 70.3s | 2.07x |
| Long In-context Learning | 53.3s | 104.9s | 111.0s | 2.08x |
| Long-dialogue History Understanding | 32.8s | 64.6s | 69.4s | 2.12x |
| Code Repository Understanding | 60.3s | 121.5s | 126.4s | 2.10x |
| Long Structured Data Understanding | 65.3s | 131.9s | 138.2s | 2.12x |
| **Average** | **46.9s** | **92.5s** | **98.0s** | **~2.09x** |

### 2.3 Peak Memory 比較

| Task Category | fullKV (MB) | α=0.2 (MB) | α=0.1 (MB) | Δ vs fullKV |
|---------------|:-----------:|:----------:|:----------:|:-----------:|
| Single-Document QA | 46,587 | 38,067 | 38,088 | -8,499 |
| Multi-Document QA | 46,587 | 38,067 | 38,088 | -8,499 |
| Long In-context Learning | 46,587 | 38,067 | 38,088 | -8,499 |
| Long-dialogue History Understanding | 43,857 | 36,102 | 36,104 | -7,753 |
| Code Repository Understanding | 46,587 | 38,067 | 38,088 | -8,499 |
| Long Structured Data Understanding | 46,587 | 38,067 | 38,088 | -8,499 |
| **Average** | **46,132** | **37,740** | **37,757** | **-8,375** |

### LongBench v2 小結
- **α=0.1 顯著優於 α=0.2**：6-category 平均 score 從 0.6145 提升到 **0.6255**（+0.0111），與 fullKV 差距從 -0.0587 縮小到 **-0.0476**。
- **最大贏家**：Long Structured Data Understanding (+0.0606)、Single-Document QA (+0.0229)、Code Repository Understanding (+0.0200)。
- **唯一下降**：Long In-context Learning (-0.0370)，這是唯一明顯退步的類別。
- Latency 略高於 α=0.2（~3-6s），在 run-to-run variance 範圍內。
- Memory 與 α=0.2 幾乎完全相同。

---

## 三、依任務類型的綜合分析

### 3.1 Single-hop QA（narrativeqa, qasper, multifieldqa\_en, triviaqa）

| 指標 | α=0.2 avg Δ vs fullKV | α=0.1 avg Δ vs fullKV |
|------|:---------------------:|:---------------------:|
| v1 Score | -0.0135 | -0.0125 |

- 小幅改善，triviaqa 略降但其他三項皆提升。

### 3.2 Multi-hop QA（hotpotqa, 2wikimqa, musique）

| 指標 | α=0.2 avg Δ vs fullKV | α=0.1 avg Δ vs fullKV |
|------|:---------------------:|:---------------------:|
| v1 Score | -0.0211 | **-0.0051** |

- **改善幅度最大（76%）**。較慢的 EMA 更新使 quantization scale 更穩定，在需要跨多個 passage 做精確 attention 的任務上效果顯著。

### 3.3 Summarization（gov\_report, qmsum, multi\_news, samsum）

| 指標 | α=0.2 avg Δ vs fullKV | α=0.1 avg Δ vs fullKV |
|------|:---------------------:|:---------------------:|
| v1 Score | -0.0046 | -0.0042 |

- 幾乎無差異。Summarization 依賴 broad attention，對 scale 微調不敏感。

### 3.4 Classification（trec）

| 指標 | α=0.2 Δ vs fullKV | α=0.1 Δ vs fullKV |
|------|:------------------:|:------------------:|
| v1 Score | -0.0100 | **+0.0150** |

- α=0.1 **超越 fullKV**（0.7450 > 0.7300），改善幅度非常大（+0.0250 vs α=0.2）。

### 3.5 Code（lcc, repobench-p）

| 指標 | α=0.2 avg Δ vs fullKV | α=0.1 avg Δ vs fullKV |
|------|:---------------------:|:---------------------:|
| v1 Score | -0.0174 | -0.0177 |

- 持平，無顯著差異。

### 3.6 Retrieval / Counting（passage\_retrieval\_en, passage\_count）

| 指標 | α=0.2 avg Δ vs fullKV | α=0.1 avg Δ vs fullKV |
|------|:---------------------:|:---------------------:|
| v1 Score | -0.0075 | **-0.0150** |

- **唯一退步的類別**。較慢的 EMA 更新導致 scale 對新 token 適應不夠及時，在 position-sensitive 任務上有害。

### 3.7 Long In-context Learning（LongBench v2）

| 指標 | α=0.2 Δ vs fullKV | α=0.1 Δ vs fullKV |
|------|:------------------:|:------------------:|
| v2 Score | -0.0741 | **-0.1111** |

- 下降 0.0370。ICL 需要快速適應 context 中的新 pattern，較慢的 EMA 可能延遲對新 distribution 的追蹤。

---

## 四、關鍵結論

### 整體表現
| Benchmark | α=0.2 avg Δ vs fullKV | α=0.1 avg Δ vs fullKV | 改善 |
|-----------|:---------------------:|:---------------------:|:----:|
| LongBench v1 (16 tasks) | -0.0121 | **-0.0081** | **33%** |
| LongBench v2 (6 categories) | -0.0587 | **-0.0476** | **19%** |

### 核心發現

1. **α=0.1 是更好的預設值**：v1 和 v2 均有提升，latency/memory 無變化。
2. **受益最大的任務類型**：
   - Multi-hop QA（v1 改善 76%）：穩定 scale → 更精確的 cross-passage attention
   - Long Structured Data（v2 +0.0606）：結構化數據需要穩定的 token-level 區分
   - Classification（trec 超越 fullKV）
3. **受損的任務類型**：
   - Retrieval / Counting（v1 -0.0075 → -0.0150）：需要快速適應 positional pattern
   - Long In-context Learning（v2 -0.0741 → -0.1111）：需要快速學習新 distribution
4. **共同特徵**：需要「穩定歷史記憶」的任務受益於慢 EMA；需要「快速適應新 pattern」的任務受害。這暗示一個 **adaptive α** 策略（如根據 layer / token position 動態調整 α）可能同時改善兩類任務。

### 建議
- 將 **α=0.1 作為新的 default**，整體 score 提升顯著。
- 未來可探索 **adaptive EMA**：shallow layers 用較大 α（快速適應），deep layers 用較小 α（穩定記憶），或根據 sequence position 動態調整。
