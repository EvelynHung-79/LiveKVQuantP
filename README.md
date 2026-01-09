# LiveKVQuant-P

LiveKVQuant-P 是一個針對大型語言模型（LLMs）在 **Prefill 階段** 進行 **即時 KV Cache 量化與壓縮** 的研究型實作專案。目標是在不丟棄任何 token 的前提下，透過 **Chunk-wise Real-time Quantization**、**Clipped EMA 統計穩定** 與 **Outlier 分離儲存**，在長上下文推論中大幅降低 KV Cache 記憶體用量，同時維持模型的推論品質。

## 專案目標

- 在 **Prefill 階段** 即時壓縮 KV Cache，避免長序列產生 OOM。
- 採用 **INT4 + FP16 混合精度**：
  - 大部分 KV 以 INT4 儲存。
  - 數值極端的 outliers 以 FP16 單獨保存。
- 使用 **Chunking 機制**：
  - 目前預設 `chunk_size = 512`。
  - Warm-up 階段使用前 `2` 個 chunks 先穩定 EMA 統計，再開始正式量化。
- 提供可重現的推論、評估與 ablation study pipeline。

## 安裝與環境設定

1. **建立並啟用虛擬環境（強烈建議）：**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   # venv\Scripts\activate       # Windows
   ```

2.  **安裝相依套件：**

    ```bash
    pip install --upgrade pip

    # 1. 先安裝 PyTorch (建議依照您的 CUDA 版本選擇，以下為 CUDA 12.4 範例)
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    # 2. 安裝其他專案依賴
    pip install -r requirements.txt
    ```

3.  **配置設定：**

      - 主要參數位於 `config.py`（如 `CHUNK_SIZE`, `WARMUP_CHUNKS`）。
      - 執行時亦可透過命令列參數覆寫部分設定。

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

### 1\. 執行主要推論 (Main Inference)

`run_inference.py` 是專案的主要入口，支援多種輸入模式。

**模式 A：Dummy Test (快速驗證)**
使用生成的假資料進行快速測試，確認 Pipeline 無誤。

```bash
python scripts/run_inference.py \
  --model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --input_mode dummy \
  --chunk_size 512
```

**模式 B：LongBench 資料集評估**
自動下載並載入 LongBench 資料集進行推論（需指定 `task`）。

```bash
python scripts/run_inference.py \
  --model_id meta-llama/Meta-Llama-3-8B-Instruct \
  --input_mode longbench \
  --task narrativeqa

python scripts/run_inference.py \
  --bench_version v1 \
  --task_type single-doc \
  --num_samples -1 \
  --ema_alpha 1.0 \
  --clip_factor_n 1.5 \
  --outlier_ratio 0.05
```

**模式 C：互動模式 (Interactive)**
手動輸入 Prompt 進行測試。

```bash
python scripts/run_inference.py \
  --input_mode interactive
```

**模式 D：基準模型 (Baseline)**

```bash
python scripts/run_baseline.py \
  --bench_version v1 \
  --task_type single-doc \
  --num_samples 5
python scripts/run_baseline.py \
  --bench_version v1 \
  --task_type narrativeqa \
  --num_samples 10
python scripts/run_baseline.py \
  --bench_version v2 \
  --task_type single-doc \
  --num_samples 5

```

**可選參數：**

  - `--config`: 指定 YAML 設定檔路徑 (預設 `configs/default.yaml`)。
  - `--chunk_size`: 覆寫 chunk 大小 (e.g., 512, 1024)。
  - `--bits`: 覆寫量化位元數 (預設 4)。
  - `--warmup`: 覆寫 Warm-up chunk 數量。

### 2\. 執行實驗腳本 (Scripts)

`scripts/` 目錄下包含特定實驗的執行腳本。這些腳本目前的參數是在 `__main__` 區塊中直接定義的，若需更改設定（如測試不同模型或資料筆數），請直接編輯檔案內容。

#### 批量推論評估 (`scripts/run_inference.py`)

執行 LongBench 特定任務的批量評估，並計算 F1 Score 與記憶體使用量。

```bash
# 預設執行 NarrativeQA 的前 5 筆資料
python scripts/run_inference.py

python scripts/run_inference.py --task narrativeqa --num_samples 20 --chunk_size 1024
```

*輸出結果將儲存為 JSON 檔案，包含詳細的 Metric 與推論結果。*

#### Ablation Study (`scripts/run_ablation.py`)

針對特定參數（如 EMA Alpha, Warmup Steps）進行掃描，觀察 Perplexity (PPL) 的變化。

```bash
# 預設測試不同的 ema_alpha 值 [0.01, 0.1, 0.3, 0.5, 0.9]
python scripts/run_ablation.py

python scripts/run_ablation.py --param_name ema_alpha --values "0.01,0.1,0.5,0.9"
```

*執行後會生成 `ablation_ema_alpha.csv` 報表。*