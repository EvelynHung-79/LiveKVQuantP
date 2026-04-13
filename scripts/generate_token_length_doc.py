#!/usr/bin/env python3
"""
根據 token 長度計算結果生成 markdown 文檔
"""

import json
from pathlib import Path


def format_number(num):
    """格式化數字（加逗號）"""
    return f"{int(num):,}"


def generate_markdown(results_file="longbench_token_lengths.json", output_file="docs/每個任務的token長度.md"):
    """生成 markdown 文檔"""

    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)

    v1_data = results['v1']
    v2_data = results['v2']

    # 計算統計信息
    v1_avgs = [d['avg'] for d in v1_data.values()]
    v2_avgs = [d['avg'] for d in v2_data.values()]

    v1_avg = sum(v1_avgs) / len(v1_avgs)
    v1_min = min(v1_avgs)
    v1_max = max(v1_avgs)

    v2_avg = sum(v2_avgs) / len(v2_avgs)
    v2_min = min(v2_avgs)
    v2_max = max(v2_avgs)

    # 生成 markdown
    md_content = f"""# LongBench Token 長度統計

> 使用 **llama3.1-8b-instruct tokenizer** 計算每個任務的平均 sequence length
> **模型**：llama3.1-8b-instruct
> **計算方式**：context + question/input 的 token 數

---

## LongBench V1（16 個任務）

| 任務 | 樣本數 | 平均長度 | 最短 | 最長 |
|------|:-----:|:------:|:----:|:----:|
"""

    # 排序 V1 按平均長度降序
    v1_sorted = sorted(v1_data.items(), key=lambda x: x[1]['avg'], reverse=True)

    for task_name, stats in v1_sorted:
        avg = stats['avg']
        md_content += f"| {task_name} | {stats['count']} | {format_number(avg)} | {format_number(stats['min'])} | {format_number(stats['max'])} |\n"

    md_content += f"| **平均** | **{sum(d['count'] for d in v1_data.values())}** | **{format_number(v1_avg)}** | **{format_number(v1_min)}** | **{format_number(v1_max)}** |\n"

    md_content += f"""
---

## LongBench V2（6 個任務類別）

| 任務類別 | 樣本數 | 平均長度 | 最短 | 最長 |
|---------|:-----:|:------:|:----:|:----:|
"""

    # 排序 V2 按平均長度降序
    v2_sorted = sorted(v2_data.items(), key=lambda x: x[1]['avg'], reverse=True)

    for task_name, stats in v2_sorted:
        avg = stats['avg']
        md_content += f"| {task_name} | {stats['count']} | {format_number(avg)} | {format_number(stats['min'])} | {format_number(stats['max'])} |\n"

    md_content += f"| **平均** | **{sum(d['count'] for d in v2_data.values())}** | **{format_number(v2_avg)}** | **{format_number(v2_min)}** | **{format_number(v2_max)}** |\n"

    # 對比分析
    md_content += f"""
---

## 對比分析

### V1 vs V2

| 指標 | V1 | V2 | 差異 |
|------|:--:|:--:|:----:|
| 任務數 | {len(v1_data)} | {len(v2_data)} | V2 為 V1 的 {len(v2_data)/len(v1_data):.1%} |
| 平均 token 長度 | {format_number(v1_avg)} | {format_number(v2_avg)} | V2 是 V1 的 {v2_avg/v1_avg:.1f}x |
| 最短任務 | {format_number(v1_min)} | {format_number(v2_min)} | V2 長 {v2_min/v1_min:.1f}x |
| 最長任務 | {format_number(v1_max)} | {format_number(v2_max)} | V2 長 {v2_max/v1_max:.1f}x |

### 關鍵發現

- **V1 的長度分佈**：平均 {format_number(v1_avg)} tokens
  - 最短：{[k for k,v in v1_data.items() if v['avg'] == v1_min][0]} ({format_number(v1_min)} tokens)
  - 最長：{[k for k,v in v1_data.items() if v['avg'] == v1_max][0]} ({format_number(v1_max)} tokens)

- **V2 的長度分佈**：平均 {format_number(v2_avg)} tokens
  - 最短：{[k for k,v in v2_data.items() if v['avg'] == v2_min][0]} ({format_number(v2_min)} tokens)
  - 最長：{[k for k,v in v2_data.items() if v['avg'] == v2_max][0]} ({format_number(v2_max)} tokens)

- **V2 明顯更長**：V2 的平均長度是 V1 的 {v2_avg/v1_avg:.1f} 倍
  - 這反映 LongBench v2 設計用於測試更長上下文的能力
  - V2 中有些任務（如 Code_Repository_Understanding）上下文特別長

---

## 補充說明

- 使用 **llama3.1-8b-instruct tokenizer** 計算（來自官方 Meta 模型）
- Token 數包含 context + question/input，不包含 special tokens
- 某些 V2 任務的最長樣本非常長（超過 100 萬 tokens），這些是測試極限的邊界情況
- V1 任務相對較短（平均 11K tokens），適合測試中等長度上下文
- V2 任務明顯更長（平均 327K tokens），設計用於測試極長上下文的能力
"""

    # 寫入文件
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    print(f"✓ Markdown 文檔已生成：{output_file}")
    return md_content


if __name__ == "__main__":
    md = generate_markdown()
    print("\n" + "=" * 70)
    print("預覽（前 50 行）")
    print("=" * 70)
    print("\n".join(md.split("\n")[:50]))
