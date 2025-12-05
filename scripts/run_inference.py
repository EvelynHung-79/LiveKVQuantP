import sys
import os
import torch
import logging
import json
from tqdm import tqdm
from datetime import datetime

# 將專案根目錄加入 Path，確保能 Import livekvquant 等模組
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.longbench_loader import LongBenchLoader
from evaluation.metrics import calculate_f1_score, calculate_accuracy
from evaluation.profiler import MemoryProfiler

# 設定 Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_experiment(task_name="narrativeqa", num_samples=10):
    """
    執行單一任務的完整評估。
    """
    # 1. 初始化配置與模型
    # 這裡可以讀取 configs/default.yaml，這裡示範直接用 Dataclass
    config = LiveKVQuantConfig(
        chunk_size=512, 
        n_warmup=2, 
        bits=4,
        ema_alpha=0.1
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    logger.info(f"Initializing Model: {model_id} with config: {config}")
    model = LiveKVQuantModel(model_id, config, device)
    
    # 2. 載入資料集
    loader = LongBenchLoader(task_name=task_name)
    logger.info(f"Loaded {len(loader)} samples from {task_name}. Running eval on first {num_samples} samples.")

    # 3. 準備記錄器
    results = []
    total_f1 = 0.0
    profiler = MemoryProfiler()

    # 4. 評估迴圈
    for i in tqdm(range(min(num_samples, len(loader))), desc="Evaluating"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        # 開始測量資源
        profiler.start()
        
        try:
            # 執行推論
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=64)
            
            # 停止測量
            perf_metrics = profiler.stop()
            
            # 計算分數
            f1 = calculate_f1_score(output, ground_truths)
            total_f1 += f1
            
            # 記錄單筆結果
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "f1": f1,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "output": output,
                "ground_truth": ground_truths[0] # 僅記錄第一個 GT 供參考
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM at sample {i}")
                torch.cuda.empty_cache()
            else:
                logger.error(f"Error at sample {i}: {e}")

    # 5. 計算平均並存檔
    avg_f1 = total_f1 / len(results) if results else 0.0
    logger.info(f"Experiment Finished. Average F1: {avg_f1:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{task_name}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "config": config.__dict__,
            "avg_f1": avg_f1,
            "details": results
        }, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    # 執行實驗：跑 narrativeqa 的前 5 筆資料作為測試
    run_experiment(task_name="narrativeqa", num_samples=5)