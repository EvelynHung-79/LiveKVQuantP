import sys
import os
import torch
import logging
import json
import argparse
from tqdm import tqdm
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# 將專案根目錄加入 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.longbench_loader import LongBenchLoader
from data.longbench_v2_loader import LongBenchV2Loader  # [新增] 引入 v2 loader
from evaluation.metrics import calculate_f1_score, calculate_accuracy
from evaluation.profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Baseline Inference Script")
    
    # 任務設定
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name or 'v2-all' for LongBench v2")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    return parser.parse_args()

def print_metrics(metrics):
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    print(f"Peak Memory Usage : {metrics.peak_memory_mb:.2f} MB")
    print(f"Total Latency     : {metrics.total_latency_ms:.2f} ms")
    print(f"Throughput        : {metrics.throughput_tokens_per_sec:.2f} tokens/sec")
    print("="*40)

def run_baseline_longbench(model, tokenizer, args, profiler):
    """
    執行標準的 LongBench / LongBench v2 評估流程
    """
    # [修改] 自動判斷要使用哪個 Loader
    # 判斷邏輯：如果 task 名稱包含 "v2" 或者是 v2 特有的 domain 名稱，就用 v2 loader
    v2_domains = ["single-document", "multi-document", "long in-context", "long dialogue", "code repository"]
    is_v2 = "v2" in args.task.lower() or any(d in args.task.lower() for d in v2_domains)

    try:
        if is_v2:
            logger.info(f"Detected LongBench v2 task: {args.task}")
            # 如果 args.task 是 "v2-all" 或類似，我們可以傳入 "all"，否則傳入 args.task 做過濾
            task_filter = "all" if "v2" in args.task.lower() else args.task
            loader = LongBenchV2Loader(task_name=task_filter)
        else:
            logger.info(f"Detected LongBench v1 task: {args.task}")
            loader = LongBenchLoader(task_name=args.task)
    except Exception as e:
        logger.error(f"Failed to load task {args.task}: {e}")
        return

    logger.info(f"Loaded {len(loader)} samples. Evaluating first {args.num_samples} samples...")
    logger.info("Running in PURE BASELINE mode (Standard HF Generate, FP16 KV Cache)...")
    
    results = []
    total_score = 0.0 # 改名為 score，因為 v2 可能是 accuracy
    
    # 確保 Pad Token 設定正確
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(min(args.num_samples, len(loader))), desc="Evaluating Baseline"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        # [關鍵] 加入 Truncation 機制
        # 這是 Baseline 為了避免 OOM 或長度報錯的標準做法
        # 對於 LongBench v2 (超長文本)，這裡會把重要的後段截斷，導致 Baseline 分數低
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
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.output_len,
                    use_cache=True, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            generated_ids = outputs[0][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            print_metrics(perf_metrics)
            
            # [修改] 針對 v2 計算 Accuracy，針對 v1 計算 F1
            if is_v2:
                # v2 是選擇題，我們可以用 Accuracy (Exact Match)
                # 簡單處理：檢查 output_text 是否包含正確選項 (A/B/C/D)
                # 這裡使用原本 metrics.py 的 calculate_accuracy (Exact Match)
                # 建議: 實際使用時可能需要正則表達式提取 "The answer is A" 中的 "A"
                score = calculate_accuracy(output_text, ground_truths)
                metric_name = "Accuracy"
            else:
                score = calculate_f1_score(output_text, ground_truths)
                metric_name = "F1"

            total_score += score
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "actual_input_len": input_len,
                "score": score,
                "metric": metric_name,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "throughput": perf_metrics.throughput_tokens_per_sec,
                "output": output_text,
                "ground_truth": ground_truths[0],
                "truncated": input_len == max_context_len # 標記是否被截斷
            })
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM at sample {i} (Length: {input_len})")
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "error": "OOM",
                "score": 0.0,
                "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024**2)
            })
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")

    avg_score = total_score / len(results) if results else 0.0
    max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0

    logger.info(f"Baseline Average Score: {avg_score:.4f}")
    logger.info(f"Max Peak Memory: {max_peak_memory:.2f} MB")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/baseline/results_{args.task}_baseline_{timestamp}.json"
    
    config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else str(model.config)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "config": config_dict,
            "avg_score": avg_score,
            "max_peak_memory_mb": max_peak_memory,
            "details": results
        }, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved baseline results to {output_file}")

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading standard model: {args.model_id}")
    try:
        # [選項] 如果想讓 Baseline 支援 Llama-2 長文本 (避免長度報錯但引發 OOM)，可在此加入 RoPE Scaling
        # config = AutoConfig.from_pretrained(args.model_id)
        # config.rope_scaling = {"type": "dynamic", "factor": 8.0}
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

    profiler = MemoryProfiler()
    run_baseline_longbench(model, tokenizer, args, profiler)

if __name__ == "__main__":
    main()