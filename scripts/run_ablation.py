import sys
import os
import torch
import pandas as pd
import gc
import logging
from typing import List

# 設定 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.wikitext_loader import WikitextLoader
from evaluation.metrics import calculate_perplexity
from evaluation.profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_ppl(model, loader, stride=512, max_length=2048):
    """
    計算 Perplexity (PPL) 的輔助函式。
    使用 Sliding Window 方法計算。
    """
    encodings = loader.get_tokenized_stream(model.tokenizer)
    input_ids = encodings[:, :max_length].to(model.device) # 限制長度以節省時間
    
    nlls = []
    prev_end_loc = 0
    
    # Sliding Window Loop
    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + max_length, input_ids.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100 # Mask 掉 context 部分的 loss
        
        with torch.no_grad():
            outputs = model.model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == input_ids.size(1):
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def run_ablation_study(param_name: str, param_values: List):
    """
    針對特定參數執行 Ablation Study。
    
    Args:
        param_name: 要測試的參數名稱 (e.g., 'ema_alpha', 'n_warmup')
        param_values: 測試數值列表 (e.g., [0.1, 0.5, 0.9])
    """
    logger.info(f"=== Starting Ablation Study on {param_name} ===")
    logger.info(f"Values to test: {param_values}")
    
    results = []
    
    # 準備數據集 (只載入一次)
    # 使用 Wikitext-2 測 PPL，因為它對量化誤差最敏感
    data_loader = WikitextLoader(split="test")
    
    for val in param_values:
        logger.info(f"Running experiment with {param_name} = {val}...")
        
        # 1. 設定 Config
        config_kwargs = {
            "chunk_size": 512,
            "n_warmup": 2,
            "bits": 4,
            "ema_alpha": 0.1
        }
        # 覆寫測試參數
        config_kwargs[param_name] = val
        config = LiveKVQuantConfig(**config_kwargs)
        
        # 2. 載入模型
        # 注意：每次都要重新載入模型以重置狀態
        model = LiveKVQuantModel("meta-llama/Meta-Llama-3-8B-Instruct", config)
        
        # 3. 執行 PPL 評估
        profiler = MemoryProfiler()
        profiler.start()
        
        try:
            ppl = evaluate_ppl(model, data_loader, max_length=4096) # 測試 4k tokens
        except Exception as e:
            logger.error(f"Failed at {val}: {e}")
            ppl = float('nan')
            
        metrics = profiler.stop()
        
        # 4. 記錄結果
        results.append({
            param_name: val,
            "PPL": ppl,
            "Peak_Memory_MB": metrics.peak_memory_mb,
            "Latency_s": metrics.total_latency_ms / 1000
        })
        
        # 5. 清理記憶體 (非常重要！)
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 6. 輸出報告
    df = pd.DataFrame(results)
    csv_filename = f"ablation_{param_name}.csv"
    df.to_csv(csv_filename, index=False)
    
    logger.info(f"Ablation study completed. Results saved to {csv_filename}")
    print("\n=== Ablation Results Summary ===")
    print(df)

if __name__ == "__main__":
    # 範例 1: 測試不同的 EMA Alpha (驗證論文 3.3.2)
    # 預期：Alpha 太小 (反應慢) 或太大 (不穩定) 都會導致 PPL 升高
    run_ablation_study("ema_alpha", [0.01, 0.1, 0.3, 0.5, 0.9])
    
    # 範例 2: 測試不同的 Warm-up Chunks (驗證論文 3.2.2)
    # run_ablation_study("n_warmup", [0, 1, 2, 4])