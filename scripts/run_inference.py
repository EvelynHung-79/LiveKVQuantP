import sys
import os
import torch
import logging
import json
import argparse
from tqdm import tqdm
from datetime import datetime

# 將專案根目錄加入 Path，確保能 import config 和 livekvquant
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.longbench_loader import LongBenchLoader
from evaluation.metrics import calculate_f1_score
from evaluation.profiler import MemoryProfiler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LiveKVQuant-P Unified Inference Script")
    
    # 模式選擇
    parser.add_argument("--input_mode", type=str, choices=["longbench", "interactive", "dummy"], default="longbench", 
                        help="Execution mode: 'longbench' (batch eval), 'interactive' (demo), 'dummy' (debug)")
    
    # 模型與任務
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name (only for longbench mode)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (only for longbench mode)")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    # 量化配置
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--n_warmup", type=int, default=2, help="Number of warmup chunks")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor")
    parser.add_argument("--baseline", action="store_true", help="Enable Baseline mode (Full FP16, No Quantization)")
    
    return parser.parse_args()

def get_dummy_input():
    return "This is a test prompt to verify the chunking mechanism. " * 500

def print_metrics(output_text, metrics):
    """統一的輸出格式函式"""
    print("\n" + "="*40)
    print("FINAL OUTPUT TEXT")
    print("="*40)
    # 僅印出前 500 字避免洗版
    print(output_text[:500] + "...(truncated)" if len(output_text) > 500 else output_text)
    
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    print(f"Peak Memory Usage : {metrics.peak_memory_mb:.2f} MB")
    print(f"Total Latency     : {metrics.total_latency_ms:.2f} ms")
    print(f"Throughput        : {metrics.throughput_tokens_per_sec:.2f} tokens/sec")
    print("="*40)

def run_interactive(model, args, profiler):
    """互動模式：手動輸入"""
    print("\n=== Interactive Mode (Press Ctrl+D or Ctrl+C to exit) ===")
    try:
        while True:
            print("\nEnter your prompt:")
            lines = []
            try:
                while True:
                    line = input()
                    if not line: break
                    lines.append(line)
            except EOFError:
                pass
            
            prompt = "\n".join(lines)
            if not prompt.strip():
                try:
                    prompt = input("Enter prompt: ")
                except EOFError:
                    break

            if not prompt.strip(): break

            logger.info(f"Generating response... (Baseline: {args.baseline})")
            
            profiler.start()
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=args.output_len)
            metrics = profiler.stop(num_output_tokens=args.output_len) # [修正] 傳入 token 數以計算 Throughput

            print_metrics(output, metrics)

    except KeyboardInterrupt:
        print("\nExiting...")

def run_dummy(model, args, profiler):
    """Dummy 模式：快速測試"""
    logger.info("Running Dummy Test...")
    prompt = get_dummy_input()
    
    profiler.start()
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=args.output_len)
    # [修正] 傳入 num_output_tokens 讓 Profiler 能計算 tokens/sec
    metrics = profiler.stop(num_output_tokens=args.output_len)
    
    logger.info("Dummy Test Completed.")
    print_metrics(output, metrics)

def run_longbench(model, args, profiler):
    """LongBench 模式：批量評估與存檔"""
    try:
        loader = LongBenchLoader(task_name=args.task)
    except Exception as e:
        logger.error(f"Failed to load task {args.task}: {e}")
        return

    logger.info(f"Loaded {len(loader)} samples. Evaluating first {args.num_samples}...")
    
    results = []
    total_f1 = 0.0
    
    for i in tqdm(range(min(args.num_samples, len(loader))), desc="Evaluating"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        profiler.start()
        try:
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=args.output_len)
            
            # [修正] 傳入 output_len 計算 Throughput
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            f1 = calculate_f1_score(output, ground_truths)
            total_f1 += f1
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "f1": f1,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "throughput": perf_metrics.throughput_tokens_per_sec, # [新增] 記錄 Throughput
                "output": output,
                "ground_truth": ground_truths[0]
            })
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()

    avg_f1 = total_f1 / len(results) if results else 0.0
    logger.info(f"Average F1: {avg_f1:.4f}")
    
    # 存檔
    mode_str = "baseline" if args.baseline else f"quant_w{args.n_warmup}_b{args.bits}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results_{args.task}_{mode_str}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "config": model.config.__dict__,
            "avg_f1": avg_f1,
            "details": results
        }, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved to {output_file}")

def main():
    args = parse_args()
    
    # 1. 初始化 Config
    config = LiveKVQuantConfig(
        chunk_size=args.chunk_size, 
        n_warmup=args.n_warmup, 
        bits=args.bits,
        ema_alpha=args.ema_alpha,
        baseline=args.baseline
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Mode: {args.input_mode.upper()} | Baseline: {args.baseline} | Device: {device}")
    
    # 2. 載入模型
    try:
        model = LiveKVQuantModel(args.model_id, config, device)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    # 3. 根據模式執行
    profiler = MemoryProfiler()
    
    if args.input_mode == "interactive":
        run_interactive(model, args, profiler)
    elif args.input_mode == "dummy":
        run_dummy(model, args, profiler)
    else:
        run_longbench(model, args, profiler)

if __name__ == "__main__":
    main()