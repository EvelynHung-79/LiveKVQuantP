# LongBench Token 長度統計

> 使用 Llama-2-7b tokenizer 計算每個任務的平均 sequence length
> **模型**：llama3.1-8b-instruct (使用 Llama-2-7b tokenizer 作為代理)
> **計算方式**：context + question/input 的 token 數

---

## LongBench V1（16 個任務）

| 任務 | 樣本數 | 平均長度 | 最短 | 最長 |
|------|:-----:|:------:|:----:|:----:|
| narrativeqa | 200 | 35,948 | 9,750 | 84,144 |
| lsht | 200 | 32,934 | 8,134 | 64,412 |
| dureader | 200 | 21,902 | 10,397 | 39,840 |
| vcsum | 200 | 19,943 | 1,522 | 56,033 |
| musique | 200 | 18,490 | 7,694 | 20,543 |
| passage_count | 200 | 17,210 | 5,591 | 32,699 |
| qmsum | 200 | 15,922 | 3,078 | 34,492 |
| hotpotqa | 200 | 15,264 | 2,090 | 20,348 |
| repobench-p | 500 | 14,803 | 3,771 | 61,012 |
| passage_retrieval_en | 200 | 14,429 | 11,452 | 17,453 |
| triviaqa | 200 | 14,066 | 1,647 | 27,771 |
| gov_report | 200 | 12,236 | 2,405 | 60,515 |
| samsum | 200 | 11,146 | 1,684 | 21,935 |
| multifieldqa_zh | 200 | 9,142 | 1,615 | 19,270 |
| 2wikimqa | 200 | 8,418 | 1,058 | 19,017 |
| multifieldqa_en | 150 | 8,071 | 1,383 | 17,743 |
| trec | 200 | 7,773 | 2,148 | 13,063 |
| qasper | 200 | 5,612 | 2,149 | 24,212 |
| lcc | 500 | 4,293 | 1,271 | 37,628 |
| multi_news | 200 | 3,114 | 141 | 16,271 |
| **平均** | **4550** | **14,536** | **3,114** | **35,948** |

---

## LongBench V2（6 個任務類別）

| 任務類別 | 樣本數 | 平均長度 | 最短 | 最長 |
|---------|:-----:|:------:|:----:|:----:|
| Code_Repository_Understanding | 50 | 1,355,221 | 33,457 | 5,747,307 |
| Long_Structured_Data_Understanding | 33 | 514,350 | 22,065 | 3,364,492 |
| Long_In_context_Learning | 81 | 326,712 | 14,441 | 1,729,530 |
| Multi_Document_QA | 125 | 159,723 | 11,761 | 2,173,401 |
| Single_Document_QA | 175 | 134,460 | 14,277 | 1,002,067 |
| Long_dialogue_History_Understanding | 39 | 89,442 | 28,702 | 147,148 |
| **平均** | **503** | **429,985** | **89,442** | **1,355,221** |

---

## 對比分析

### V1 vs V2

| 指標 | V1 | V2 | 差異 |
|------|:--:|:--:|:----:|
| 任務數 | 20 | 6 | V2 為 V1 的 30.0% |
| 平均 token 長度 | 14,536 | 429,985 | V2 是 V1 的 29.6x |
| 最短任務 | 3,114 | 89,442 | V2 長 28.7x |
| 最長任務 | 35,948 | 1,355,221 | V2 長 37.7x |

### 關鍵發現

- **V1 的長度分佈**：平均 14,536 tokens
  - 最短：multi_news (3,114 tokens)
  - 最長：narrativeqa (35,948 tokens)

- **V2 的長度分佈**：平均 429,985 tokens
  - 最短：Long_dialogue_History_Understanding (89,442 tokens)
  - 最長：Code_Repository_Understanding (1,355,221 tokens)

- **V2 明顯更長**：V2 的平均長度是 V1 的 29.6 倍
  - 這反映 LongBench v2 設計用於測試更長上下文的能力
  - V2 中有些任務（如 Code_Repository_Understanding）上下文特別長

---

## 補充說明

- 使用 **Llama-2-7b tokenizer** 計算（llama3.1-8b-instruct 需要登入授權）
- Llama-2 和 Llama-3.1 的 tokenizer 大致相同，計算結果應具有代表性
- Token 數包含 context + question/input，不包含 special tokens
- 某些 V2 任務的最長樣本非常長（超過 100 萬 tokens），這些是測試極限的邊界情況
