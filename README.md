# LiveKVQuant-P

LiveKVQuant-P 是一個針對大型語言模型（LLMs）在 **Prefill 階段** 進行 **即時 KV Cache 量化與壓縮** 的研究型實作專案。目標是在不丟棄任何 token 的前提下，透過 **Chunk-wise Real-time Quantization**、**Clipped EMA 統計穩定** 與 **Outlier 分離儲存**，在長上下文推論中大幅降低 KV Cache 記憶體用量，同時維持模型的推論品質。

## 目錄

- [專案目標](#專案目標)
- [核心概念](#核心概念)
- [快速開始 (Quick Start)](#快速開始-quick-start)
- [安裝與環境設定](#安裝與環境設定)
- [測試 (Testing)](#測試-testing)
- [使用方式 (Usage)](#使用方式-usage)
- [目錄結構說明](#目錄結構說明)
- [Citation](#citation)

## 專案目標

- 在 **Prefill 階段** 即時壓縮 KV Cache，避免長序列產生 OOM。
- 採用 **INT4 + FP16 混合精度**：
  - 大部分 KV 以 INT4 儲存。
  - 數值極端的 outliers 以 FP16 單獨保存。
- 使用 **Chunking 機制**：
  - 目前預設 `chunk_size = 512`。
  - Warm-up 階段使用前 `2` 個 chunks 先穩定 EMA 統計，再開始正式量化。
- 提供可重現的推論、評估與 ablation study pipeline。

## 核心概念

### KV Cache 量化 (KV Cache Quantization)
Key-Value Cache 是 Transformer 模型在推論時儲存的中間狀態，用於加速 decode 階段。在長上下文推論中，KV Cache 會成為瓶頸，佔據大量記憶體。LiveKVQuant-P 在 Prefill 階段（處理整個輸入上下文時）即時量化 KV 數值，將精度從 FP16 降低到 INT4，達到 **4 倍的記憶體壓縮**。

### Chunk-wise Real-time Quantization
量化過程以 **固定大小的 Chunk**（預設 512 tokens）為單位進行，而非全局操作。這樣做的優點：
- **即時性 (Real-time)**：無需等待整個上下文完成，逐 chunk 量化可減少延遲。
- **穩定性**：每個 chunk 獨立計算量化參數，避免數值分佈失衡。

### Clipped EMA 統計穩定化 (Clipped EMA)
採用 **指數移動平均 (EMA)** 追蹤每層的數值範圍（最大值、最小值），並加入 **Clipping 機制** 防止極端值扭曲統計。
- 計算公式見論文 Eq. 3-2～3-4
- 透過 `ema_alpha` 參數（預設 0.2）控制平滑度

### Outlier 分離儲存 (Outlier Isolation)
模型輸出中少數極端數值會顯著影響量化品質。LiveKVQuant-P 將這些 **Outliers** 單獨以 FP16 儲存，其餘數值用 INT4 表示：
- **自動偵測**：基於量化噪聲大小動態識別 outliers
- **可配置比例**：`outlier_ratio` 參數控制 outlier 數量（預設 1%）

### Warm-up 與 Phase Transition
- **Warm-up 階段**：前 2 個 chunks 用於穩定 EMA 統計，不進行量化
- **Phase Transition**：從 warm-up 轉換到正式量化時自動激活

### 評估真實資料集（LongBench）
如果要測試實際效能，執行完整評估：

```bash
# 下載 LongBench 資料集（首次運行會自動下載，或手動複製到 Remote Server）
# scp -r ./longbench_v1/ pod:/root/LiveKVQuantP/data/

# 執行單個任務的評估
python scripts/run_liveKVQuantP.py \
  --task_type single-doc \
  --ema_alpha 0.2 \
  --clip_factor_n 4.0 \
  --outlier_ratio 0.01 \
  --num_samples -1
```

### 執行消融實驗 (Ablation Studies)
測試不同的量化策略對效能的影響：

```bash
# 停用 Warm-up 觀察效果
python scripts/run_liveKVQuantP.py --task_type narrativeqa --use_warmup false

# 停用 Outlier 分離（純 INT4 量化）
python scripts/run_liveKVQuantP.py --task_type narrativeqa --use_outlier_isolation false

# 改用不同的統計方法
python scripts/run_liveKVQuantP.py --task_type narrativeqa --stats_method ema_minmax
```

## 安裝與環境設定

### 1. 建立虛擬環境

```bash
python3.11 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或者在 Windows: venv\Scripts\activate
```

### 2. 安裝依賴

```bash
pip install --upgrade pip

# 安裝 PyTorch（GPU 版本）
pip3 install torch --index-url https://download.pytorch.org/whl/cu124

# 安裝專案依賴
pip install -r requirements.txt
pip install wheel
pip install flash-attn==2.7.4.post1 --no-build-isolation

pip install ipykernel
python -m ipykernel install --user --name=livekv --display-name "Python (LiveKVQuantP)"
```

### 3. 下載資料集（選用）

如需使用 LongBench 資料集進行評估，可透過以下方式取得：

```bash
# 從遠端伺服器複製
scp -r ./longbench_v1/ pod:/root/LiveKVQuantP/data/
scp -r ./longbench_v2/ pod:/root/LiveKVQuantP/data/

# 或自動下載（首次執行時）
python scripts/run_liveKVQuantP.py --task_type single-doc --num_samples 1
```

### 4. 驗證安裝

運行單元測試確認環境正確設定：

```bash
python -m pytest tests/ -v
```

### 配置參數

- **主要設定檔**：`config.py`（如 `CHUNK_SIZE`, `WARMUP_CHUNKS`）
- **執行時覆寫**：命令列參數可覆寫配置檔案中的設定
- **YAML 設定**（選用）：使用 `--config configs/default.yaml` 指定自訂配置

## 測試 (Testing)

為確保環境與核心模組運作正常，請執行單元測試。
**注意：** 請使用 `python -m pytest` 以確保模組路徑正確。

```bash
python -m pytest tests/
```

測試涵蓋範圍：

  - `test_ema_utils.py`：驗證 Clipped EMA 數學公式 (Eq. 3-2 \~ 3-4)。
  - `test_quant_utils.py`：驗證量化計算與 Outlier 分離邏輯。
  - `test_integration.py`：驗證端到端 (End-to-End) 的 Warm-up 與 Phase Transition 流程。

## 使用方式 (Usage)

### 推論腳本總覽

| 腳本 | 用途 | 適用場景 |
|-----|------|---------|
| `run_liveKVQuantP.py` | 主推論程式 | 測試量化版本的推論品質 |
| `run_fullKV.py` | 基準測試 | 無量化版本，作為對比基準 |
| `run_ablation.py` | 消融實驗 | 掃描參數空間，分析超參數影響 |

### 1. 執行主要推論 (Main Inference)

#### 模式 A：Dummy Test（快速驗證，無需資料集）

```bash
python scripts/run_liveKVQuantP.py \
  --model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --input_mode dummy \
  --chunk_size 512
```

#### 模式 B：LongBench 單任務評估

```bash
python scripts/run_liveKVQuantP.py \
  --task_type single-doc \
  --ema_alpha 0.2 \
  --clip_factor_n 4.0 \
  --outlier_ratio 0.01 \
  --num_samples -1  # -1 表示評估全部樣本
```

#### 模式 C：消融實驗（停用特定功能）

```bash
# 停用 Warm-up 階段
python scripts/run_liveKVQuantP.py \
  --task_type narrativeqa \
  --num_samples 2 \
  --use_warmup false

# 停用 Outlier 分離（純 INT4 量化）
python scripts/run_liveKVQuantP.py \
  --task_type narrativeqa \
  --num_samples 2 \
  --use_outlier_isolation false

# 改用 EMA MinMax 統計（不使用 Clipping）
python scripts/run_liveKVQuantP.py \
  --task_type narrativeqa \
  --num_samples 2 \
  --stats_method ema_minmax
```

### 2. 基準測試 (`run_fullKV.py`)

執行未量化版本作為對比基準：

```bash
python scripts/run_fullKV.py \
  --bench_version v1 \
  --task_type passage_count \
  --num_samples -1
```

### 3. 批量評估與消融實驗

#### 單任務批量評估 (`run_liveKVQuant.py`)

執行特定任務的完整評估：

```bash
# 預設：NarrativeQA，前 5 筆資料
python scripts/run_liveKVQuant.py

# 自訂參數
python scripts/run_liveKVQuant.py \
  --task narrativeqa \
  --num_samples 20 \
  --chunk_size 1024
```

**輸出**：`results/inference_results_*.json`（包含 F1 Score、記憶體使用、推論時間等）

#### 全任務批量評估（後台執行）

```bash
nohup bash scripts/run_fullKV_all_tasks.sh > run_log.txt 2>&1 &
```

#### 參數掃描 (`run_ablation.py`)

自動掃描參數空間，觀察效能變化：

```bash
# 預設：測試 EMA Alpha 的影響 [0.01, 0.1, 0.3, 0.5, 0.9]
python scripts/run_ablation.py

# 自訂參數與掃描值
python scripts/run_ablation.py \
  --param_name ema_alpha \
  --values "0.01,0.1,0.5,0.9"
```

**輸出**：`results/ablation_ema_alpha.csv`（CSV 格式報表）

### 常用命令列參數

| 參數 | 說明 | 預設值 | 範例 |
|-----|------|--------|------|
| `--model_id` | 模型 ID（Hugging Face） | `meta-llama/Meta-Llama-3-8B-Instruct` | `meta-llama/Meta-Llama-2-7B` |
| `--task_type` | 任務類型 | `single-doc` | `narrative_qa`, `passage_count` |
| `--chunk_size` | KV Cache Chunk 大小 | `512` | `256`, `1024` |
| `--ema_alpha` | EMA 平滑係數 | `0.2` | `0.1`, `0.5` |
| `--clip_factor_n` | Clipping 因子 | `4.0` | `2.0`, `6.0` |
| `--outlier_ratio` | Outlier 佔比 | `0.01` | `0.005`, `0.02` |
| `--num_samples` | 評估樣本數 | `-1`（全部） | `5`, `20` |
| `--bits` | 量化位元數 | `4` | `2`, `8` |
| `--config` | YAML 配置檔案 | `configs/default.yaml` | 自訂路徑 |

### 腳本參數編輯（進階）

若需更改腳本內的固定參數，直接編輯 `__main__` 區塊：

```python
# scripts/run_liveKVQuant.py
if __name__ == "__main__":
    args = {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "task": "narrativeqa",
        "num_samples": 5,
        # ... 其他參數
    }
```

## 目錄結構說明

```
LiveKVQuantP/
├── README.md                          # 本檔案
├── config.py                          # 全局配置參數
├── requirements.txt                   # Python 依賴清單
├── tests/                             # 單元測試目錄
│   ├── test_ema_utils.py             # EMA 統計算法測試
│   ├── test_quant_utils.py           # 量化與 Outlier 分離測試
│   └── test_integration.py           # 端到端流程測試
├── scripts/                           # 實驗與推論腳本
│   ├── run_liveKVQuantP.py           # 主程式：即時量化推論
│   ├── run_fullKV.py                 # 基準測試：未量化版本
│   ├── run_ablation.py               # 消融實驗（掃描參數空間）
│   ├── run_liveKVQuant.py            # 批量評估單個任務
│   └── run_fullKV_all_tasks.sh       # 批量評估所有任務（基準）
├── src/                               # 核心實作模組
│   ├── quantization/                 # 量化相關
│   │   ├── quant_utils.py           # INT4 量化、Outlier 偵測
│   │   └── outlier_handler.py       # Outlier 分離與儲存
│   ├── stats/                        # 統計與監控
│   │   ├── ema_utils.py             # Clipped EMA 實作
│   │   └── stat_tracker.py          # 數值範圍追蹤
│   ├── inference/                    # 推論管線
│   │   ├── kvcache_manager.py       # KV Cache 管理與量化調度
│   │   └── prefill_handler.py       # Prefill 階段量化邏輯
│   └── utils/                        # 通用工具
│       ├── config_loader.py         # 配置檔案解析
│       └── metrics.py                # 效能指標計算（F1、PPL 等）
├── data/                              # 資料集目錄（需手動或自動下載）
│   ├── longbench_v1/                # LongBench v1 資料集
│   └── longbench_v2/                # LongBench v2 資料集
├── configs/                           # 配置檔案（YAML 格式）
│   └── default.yaml                 # 預設配置範本
└── results/                           # 實驗結果輸出目錄
    ├── inference_results_*.json      # 推論結果與 metrics
    └── ablation_*.csv                # 消融實驗報表
```

### 關鍵檔案說明

| 檔案/模組 | 功能 |
|---------|------|
| `run_liveKVQuantP.py` | 主推論腳本，支援 Dummy、LongBench、Ablation 三種模式 |
| `quant_utils.py` | 實作 INT4 量化與 Outlier 偵測演算法 |
| `ema_utils.py` | 實作 Clipped EMA 統計穩定化（Eq. 3-2～3-4） |
| `kvcache_manager.py` | 協調 Warm-up、Phase Transition 與量化調度 |
| `test_integration.py` | 驗證完整工作流程（建議優先執行） |

## Citation

如果你在研究中使用了 LiveKVQuant-P，請引用以下論文：

```bibtex
@article{livekvquantp2026,
  title={LiveKVQuant-P: Real-Time Chunk-wise KV Cache Compression for Long-Context LLM Inference},
  author={Tzu-Chia Hung},
  journal={Preprint},
  year={2026}
}
```
