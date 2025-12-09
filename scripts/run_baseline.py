import sys
import os
import torch
import logging
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# 將專案根目錄加入 Path，確保能 import data 和 evaluation
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.longbench_loader import LongBenchLoader
from evaluation.metrics import calculate_f1_score
from evaluation.profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Baseline Inference Script (No Chunking, No Compression)")
    
    # 任務設定
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    return parser.parse_args()

def print_metrics(metrics):
    """
    只印出效能數據，不印出生成的文字 (應使用者要求)
    """
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    print(f"Peak Memory Usage : {metrics.peak_memory_mb:.2f} MB")
    print(f"Total Latency     : {metrics.total_latency_ms:.2f} ms")
    print(f"Throughput        : {metrics.throughput_tokens_per_sec:.2f} tokens/sec")
    print("="*40)

def run_baseline_longbench(model, tokenizer, args, profiler):
    """
    執行標準的 LongBench 評估流程 (使用原生 HF generate)
    """
    try:
        loader = LongBenchLoader(task_name=args.task)
    except Exception as e:
        logger.error(f"Failed to load task {args.task}: {e}")
        return

    logger.info(f"Loaded {len(loader)} samples. Evaluating first {args.num_samples} samples...")
    logger.info("Running in PURE BASELINE mode (Standard HF Generate, FP16 KV Cache)...")
    
    results = []
    total_f1 = 0.0
    
    # 確保 Pad Token 設定正確
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(min(args.num_samples, len(loader))), desc="Evaluating Baseline"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        # 加入 Truncation 機制，避免超過模型長度上限
        # 計算模型最大長度 (保留 output_len 給生成用)
        # 若 config 中沒有 max_position_embeddings，預設為 128k (Llama 3.1)
        max_pos = getattr(model.config, "max_position_embeddings", 131072)
        max_context_len = max_pos - args.output_len
        
        inputs = tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,             # 啟用截斷
            max_length=max_context_len   # 設定上限
        ).to(model.device)
        
        input_len = inputs.input_ids.size(1)
        
        profiler.start()
        try:
            with torch.no_grad():
                # 標準 HF Generate
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.output_len,
                    use_cache=True, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 停止 Profiler
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            # 解碼輸出
            generated_ids = outputs[0][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # [修改] 只印出 Metrics，不印 Text
            print_metrics(perf_metrics)
            
            # 計算 F1
            f1 = calculate_f1_score(output_text, ground_truths)
            total_f1 += f1
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "actual_input_len": input_len,
                "f1": f1,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "throughput": perf_metrics.throughput_tokens_per_sec,
                "output": output_text,
                "ground_truth": ground_truths[0]
            })
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM at sample {i} (Length: {input_len})")
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "error": "OOM",
                "f1": 0.0,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2)
            })
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")

    avg_f1 = total_f1 / len(results) if results else 0.0
    
    # [修改] 計算 Max Peak Memory
    max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0

    logger.info(f"Baseline Average F1: {avg_f1:.4f}")
    logger.info(f"Max Peak Memory: {max_peak_memory:.2f} MB")
    
    # 存檔
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/baseline/results_{args.task}_pure_baseline_{timestamp}.json"
    
    # 準備 Config 字典
    config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else str(model.config)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "config": config_dict,
            "avg_f1": avg_f1,
            "max_peak_memory_mb": max_peak_memory, # [修改] 插入在此處
            "details": results
        }, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved baseline results to {output_file}")

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    
    # 1. 載入標準模型與 Tokenizer
    logger.info(f"Loading standard model: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16, 
            device_map=device
        )
        model.eval()
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    # 2. 執行評估
    profiler = MemoryProfiler()
    run_baseline_longbench(model, tokenizer, args, profiler)

if __name__ == "__main__":
    main()