import sys
import os
import torch
import pandas as pd
import gc
import logging
import argparse # [新增]
from typing import List

# 設定 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.wikitext_loader import WikitextLoader
from evaluation.profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="Run Ablation Study for LiveKVQuant-P")
    
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Model ID")
    
    # 指定要進行 Ablation 的參數名稱，例如 "ema_alpha" 或 "n_warmup"
    parser.add_argument("--param_name", type=str, required=True, help="Parameter name to vary (e.g., ema_alpha, n_warmup, chunk_size)")
    
    # 指定要測試的數值列表，以逗號分隔，例如 "0.1,0.5,0.9"
    parser.add_argument("--values", type=str, required=True, help="Comma-separated list of values to test (e.g., '0.1,0.5,0.9' or '1,2,4')")
    
    return parser.parse_args()

def parse_values_string(val_str: str) -> List:
    """將逗號分隔的字串轉換為適當型別的列表 (int 或 float)"""
    str_values = val_str.split(',')
    parsed_values = []
    for v in str_values:
        v = v.strip()
        try:
            # 嘗試轉為 int
            if '.' not in v:
                parsed_values.append(int(v))
            else:
                parsed_values.append(float(v))
        except ValueError:
            # 如果不是數字，保持字串
            parsed_values.append(v)
    return parsed_values

def evaluate_ppl(model, loader, stride=512, max_length=2048):
    """計算 PPL (保持不變)"""
    encodings = loader.get_tokenized_stream(model.tokenizer)
    input_ids = encodings[:, :max_length].to(model.device)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, input_ids.size(1), stride):
        end_loc = min(begin_loc + max_length, input_ids.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_chunk = input_chunk.clone()
        target_chunk[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model.model(input_chunk, labels=target_chunk)
            neg_log_likelihood = outputs.loss
        
        nlls.append(neg_log_likelihood)
        prev_end_loc = end_loc
        if end_loc == input_ids.size(1):
            break

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def run_ablation_study(args):
    """執行 Ablation"""
    param_name = args.param_name
    param_values = parse_values_string(args.values)
    
    logger.info(f"=== Starting Ablation Study on {param_name} ===")
    logger.info(f"Values to test: {param_values}")
    
    results = []
    
    # 準備資料集
    try:
        data_loader = WikitextLoader(split="test")
    except Exception as e:
        logger.error(f"Failed to load Wikitext: {e}")
        return
    
    for val in param_values:
        logger.info(f"Running experiment with {param_name} = {val}...")
        
        # 1. 設定 Config (基礎值)
        config_kwargs = {
            "chunk_size": 512,
            "n_warmup": 2,
            "bits": 4,
            "ema_alpha": 0.1
        }
        
        # 覆寫測試參數
        # 注意：這裡假設 param_name 是 LiveKVQuantConfig 的有效屬性
        if param_name not in config_kwargs:
            logger.warning(f"Warning: {param_name} is not a default config key. Make sure it exists in LiveKVQuantConfig.")
        
        config_kwargs[param_name] = val
        config = LiveKVQuantConfig(**config_kwargs)
        
        # 2. 載入模型
        try:
            model = LiveKVQuantModel(args.model_id, config)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            break
        
        # 3. 執行 PPL 評估
        profiler = MemoryProfiler()
        profiler.start()
        
        try:
            ppl = evaluate_ppl(model, data_loader, max_length=4096)
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
        
        # 5. 清理記憶體
        del model
        torch.cuda.empty_cache()
        gc.collect()

    # 6. 輸出報告
    if results:
        df = pd.DataFrame(results)
        csv_filename = f"ablation_{param_name}.csv"
        df.to_csv(csv_filename, index=False)
        
        logger.info(f"Ablation study completed. Results saved to {csv_filename}")
        print("\n=== Ablation Results Summary ===")
        print(df)
    else:
        logger.warning("No results generated.")

if __name__ == "__main__":
    args = parse_args()
    run_ablation_study(args)