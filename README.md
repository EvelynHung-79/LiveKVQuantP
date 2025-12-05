***

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
- 提供可重現的推論、評估與 ablation study pipeline，便於撰寫與驗證論文實驗。

## 專案結構概覽
參考 repo_structure.txt 檔案

## 安裝與環境設定

1. 建立並啟用虛擬環境（選擇性但建議）：
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows 使用 .venv\Scripts\activate
   ```

2. 安裝相依套件：
   ```bash
   # 這是 PyTorch 官方針對 CUDA 12.4 的安裝指令
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   # 再跑 requirements.txt
   pip install -r requirements.txt
   ```

3. 在 `config.py` 中確認關鍵參數，例如：
   - `CHUNK_SIZE = 512`
   - `WARMUP_CHUNKS = 2`
   - 模型名稱、權重路徑、是否啟用 quantization 等。

## 基本使用方式

### 1. 執行主要推論入口

`main_inference.py` 會載入模型、建立 LiveKVQuant-P pipeline，並在指定資料集上執行推論與量化：

```bash
python main_inference.py \
  --model_name meta-llama/Llama-3-8B \
  --dataset wikitext \
  --enable_quantization true
```

常見參數（實際依你實作為準）：

- `--model_name`：HuggingFace 模型名稱或本地路徑。
- `--dataset`：`wikitext` 或 `longbench`。
- `--enable_quantization`：是否啟用 LiveKVQuant-P。
- `--max_length`：最大 context 長度。

### 2. 使用 scripts 進行實驗

#### 單次推論測試

```bash
python scripts/run_inference.py \
  --model_name meta-llama/Llama-3-8B \
  --input_file examples/long_context.txt \
  --enable_quantization true
```

此腳本適合用來觀察單一長輸入在啟用/未啟用量化下的行為與記憶體使用。

#### Ablation Study

```bash
python scripts/run_ablation.py \
  --model_name meta-llama/Llama-3-8B \
  --dataset longbench \
  --vary_chunk_size "256,512,1024" \
  --vary_warmup_chunks "0,1,2"
```

此腳本會自動掃描不同的 `chunk_size`、`warmup_chunks`、或其他超參數設定，並呼叫 `evaluation/metrics.py` 與 `evaluation/profiler.py` 收集指標。

## 測試

在本地執行單元測試與整合測試：

```bash
pytest tests/
```

推薦先確保：
- `test_ema_utils.py` 通過（確認數學式 Eq. 3-2 ~ 3-4 實作正確）。
- `test_quant_utils.py` 通過（量化與 outlier 切分邏輯正確）。
- `test_integration.py` 通過（端到端管線在小模型/短序列上可正常跑完）。

***