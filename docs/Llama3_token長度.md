# LongBench Token 長度統計

> 使用 **llama3.1-8b-instruct tokenizer** 計算每個任務的平均 sequence length
> **模型**：llama3.1-8b-instruct
> **計算方式**：context + question/input 的 token 數

---

## LongBench V1（16 個任務）

| 任務 | 樣本數 | 平均長度 | 最短 | 最長 |
|------|:-----:|:------:|:----:|:----:|
| narrativeqa | 200 | 29,786 | 7,969 | 65,287 |
| lsht | 200 | 18,326 | 4,628 | 36,144 |
| musique | 200 | 15,560 | 6,496 | 16,349 |
| passage_count | 200 | 14,880 | 4,741 | 28,966 |
| qmsum | 200 | 13,867 | 2,584 | 30,389 |
| hotpotqa | 200 | 12,798 | 1,749 | 16,340 |
| dureader | 200 | 12,709 | 6,203 | 22,685 |
| passage_retrieval_en | 200 | 12,457 | 9,952 | 15,145 |
| triviaqa | 200 | 11,750 | 1,476 | 23,298 |
| vcsum | 200 | 10,956 | 830 | 31,568 |
| repobench-p | 500 | 10,805 | 2,540 | 39,125 |
| gov_report | 200 | 10,241 | 2,020 | 51,393 |
| samsum | 200 | 9,149 | 1,386 | 17,974 |
| 2wikimqa | 200 | 7,112 | 932 | 16,331 |
| multifieldqa_en | 150 | 6,900 | 1,298 | 14,960 |
| trec | 200 | 6,767 | 1,866 | 11,378 |
| multifieldqa_zh | 200 | 5,289 | 909 | 10,776 |
| qasper | 200 | 4,930 | 1,855 | 21,118 |
| lcc | 500 | 3,166 | 990 | 30,151 |
| multi_news | 200 | 2,609 | 131 | 13,936 |
| **平均** | **4550** | **11,003** | **2,609** | **29,786** |

---

## LongBench V2（6 個任務類別）

| 任務類別 | 樣本數 | 平均長度 | 最短 | 最長 |
|---------|:-----:|:------:|:----:|:----:|
| Code_Repository_Understanding | 50 | 1,005,115 | 23,775 | 4,144,398 |
| Long_Structured_Data_Understanding | 33 | 389,392 | 16,412 | 2,944,629 |
| Long_In_context_Learning | 81 | 258,629 | 12,469 | 1,462,030 |
| Multi_Document_QA | 125 | 126,574 | 9,929 | 1,665,287 |
| Single_Document_QA | 175 | 110,528 | 12,447 | 860,227 |
| Long_dialogue_History_Understanding | 39 | 73,816 | 22,971 | 119,363 |
| **平均** | **503** | **327,342** | **73,816** | **1,005,115** |

---

## 對比分析

### V1 vs V2

| 指標 | V1 | V2 | 差異 |
|------|:--:|:--:|:----:|
| 任務數 | 20 | 6 | V2 為 V1 的 30.0% |
| 平均 token 長度 | 11,003 | 327,342 | V2 是 V1 的 29.7x |
| 最短任務 | 2,609 | 73,816 | V2 長 28.3x |
| 最長任務 | 29,786 | 1,005,115 | V2 長 33.7x |

### 關鍵發現

- **V1 的長度分佈**：平均 11,003 tokens
  - 最短：multi_news (2,609 tokens)
  - 最長：narrativeqa (29,786 tokens)

- **V2 的長度分佈**：平均 327,342 tokens
  - 最短：Long_dialogue_History_Understanding (73,816 tokens)
  - 最長：Code_Repository_Understanding (1,005,115 tokens)

- **V2 明顯更長**：V2 的平均長度是 V1 的 29.7 倍
  - 這反映 LongBench v2 設計用於測試更長上下文的能力
  - V2 中有些任務（如 Code_Repository_Understanding）上下文特別長

---

## 補充說明

- 使用 **llama3.1-8b-instruct tokenizer** 計算（來自官方 Meta 模型）
- Token 數包含 context + question/input，不包含 special tokens
- 某些 V2 任務的最長樣本非常長（超過 100 萬 tokens），這些是測試極限的邊界情況
- V1 任務相對較短（平均 11K tokens），適合測試中等長度上下文
- V2 任務明顯更長（平均 327K tokens），設計用於測試極長上下文的能力
