import sys
import os
import torch
import logging
import json
import argparse  # [新增] 引入 argparse
from tqdm import tqdm
from datetime import datetime

# 將專案根目錄加入 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.longbench_loader import LongBenchLoader
from evaluation.metrics import calculate_f1_score
from evaluation.profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令列參數"""
    parser = argparse.ArgumentParser(description="Run LiveKVQuant-P Inference Experiment")
    
    # 基礎設定
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    # 量化配置 (覆寫 Config)
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--n_warmup", type=int, default=2, help="Number of warmup chunks")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor")
    
    return parser.parse_args()

def run_experiment(args):
    """
    執行單一任務的完整評估。
    """
    # 1. 初始化配置
    config = LiveKVQuantConfig(
        chunk_size=args.chunk_size, 
        n_warmup=args.n_warmup, 
        bits=args.bits,
        ema_alpha=args.ema_alpha
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Initializing Model: {args.model_id}")
    logger.info(f"Config: {config}")
    
    # 載入模型
    try:
        model = LiveKVQuantModel(args.model_id, config, device)
    except Exception as e:
        logger.error(f"Failed to load model {args.model_id}: {e}")
        return

    # 2. 載入資料集
    try:
        loader = LongBenchLoader(task_name=args.task)
    except Exception as e:
        logger.error(f"Failed to load task {args.task}: {e}")
        return

    logger.info(f"Loaded {len(loader)} samples from {args.task}. Running eval on first {args.num_samples} samples.")

    # 3. 準備記錄器
    results = []
    total_f1 = 0.0
    profiler = MemoryProfiler()

    # 4. 評估迴圈
    for i in tqdm(range(min(args.num_samples, len(loader))), desc="Evaluating"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        # 開始測量資源
        profiler.start()
        
        try:
            # 執行推論
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=args.output_len)
            
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
                "ground_truth": ground_truths[0]
            })
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error(f"OOM at sample {i}")
                torch.cuda.empty_cache()
            else:
                logger.error(f"Error at sample {i}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error at sample {i}: {e}")

    # 5. 計算平均並存檔
    avg_f1 = total_f1 / len(results) if results else 0.0
    logger.info(f"Experiment Finished. Average F1: {avg_f1:.4f}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{args.task}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "config": config.__dict__,
            "avg_f1": avg_f1,
            "details": results
        }, f, indent=4, ensure_ascii=False)
        
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)