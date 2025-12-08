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
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    # 移除所有量化參數 (chunk_size, bits, alpha 等)，因為這是純 Baseline
    
    return parser.parse_args()

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
        
        # Tokenize (一次性處理全部 Context，不分塊)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.size(1)
        
        # 如果 Context 超過模型上限，可能需要截斷 (視模型而定，Llama-3 支援 8k+)
        # 這裡假設 GPU 記憶體足夠跑完 Baseline (這也是我們要測的重點：Baseline 會不會 OOM)
        
        profiler.start()
        try:
            with torch.no_grad():
                # 標準 HF Generate
                # use_cache=True 是預設值，這會啟用標準 FP16 KV Cache
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.output_len,
                    use_cache=True, 
                    pad_token_id=tokenizer.pad_token_id
                )
            
            # 停止 Profiler
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            # 解碼輸出 (只取生成的部份)
            # HF generate 回傳的是 [input + generated]，我們只解碼新增的部分
            generated_ids = outputs[0][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # 計算 F1
            f1 = calculate_f1_score(output_text, ground_truths)
            total_f1 += f1
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "f1": f1,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "throughput": perf_metrics.throughput_tokens_per_sec,
                "output": output_text,
                "ground_truth": ground_truths[0]
            })
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM at sample {i} (Length: {input_len})")
            # 記錄失敗
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "error": "OOM",
                "f1": 0.0
            })
            torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")

    avg_f1 = total_f1 / len(results) if results else 0.0
    logger.info(f"Baseline Average F1: {avg_f1:.4f}")
    
    # 存檔 (保持與 run_inference.py 相同的格式)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{args.task}_pure_baseline_{timestamp}.json"
    
    # 準備 Config 字典 (處理某些物件無法序列化的問題)
    config_dict = model.config.to_dict() if hasattr(model.config, "to_dict") else str(model.config)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "config": config_dict,
            "avg_f1": avg_f1,
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
            torch_dtype=torch.float16, # 保持 FP16 以進行公平比較 (也是 LiveKVQuant 的基礎型別)
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