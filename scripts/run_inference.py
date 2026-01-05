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
from data.longbench_v2_loader import LongBenchV2Loader
from evaluation.metrics import calculate_f1_score, calculate_accuracy
from evaluation.profiler import MemoryProfiler
from data.constants import get_task_list

# 引入繪圖函式所需的套件
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LiveKVQuant-P Unified Inference Script")
    
    # 模式選擇
    parser.add_argument("--input_mode", type=str, choices=["longbench", "interactive", "dummy"], default="longbench", 
                        help="Execution mode: 'longbench' (batch eval), 'interactive' (demo), 'dummy' (debug)")
    
    # 模型與任務
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--bench_version", type=str, choices=["v1", "v2"], default="v1", help="LongBench version (v1 or v2)")
    
    # [修改] 參數名稱改為 --task_type 以對齊 run_baseline.py，並支援 single-doc 等群組
    parser.add_argument("--task_type", type=str, default="narrativeqa", help="Specific task name or category (e.g., 'single-doc', 'summarization', 'all')")
    
    # [修改] 預設 -1 代表跑全部，與 baseline 邏輯一致
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples per task. Set -1 to run ALL.")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    # 量化配置
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--n_warmup", type=int, default=2, help="Number of warmup chunks")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor")
    parser.add_argument("--clip_factor_n", type=float, default=1.5, help="Clipped EMA factor N")
    parser.add_argument("--outlier_ratio", type=float, default=0.01, help="Ratio of outliers to keep in FP16")
    
    return parser.parse_args()

def visualize_ema_tracking(model, save_dir, sample_idx, task_name):
    """
    將 StatisticsManager 紀錄的 EMA 變化畫成圖表。
    """
    target_layers = [0, 16, 31]
    # 依照 Task 分類資料夾
    task_save_dir = os.path.join(save_dir, task_name)
    os.makedirs(task_save_dir, exist_ok=True)
    
    for layer_idx in target_layers:
        if layer_idx >= len(model.controllers): break
        
        controller = model.controllers[layer_idx]
        stats_mgr = controller.stats_manager
        
        if not stats_mgr.history:
            continue
            
        # 整理數據 [Batch, Heads, Chunks, Dim] -> [Heads, Chunks, Dim]
        try:
            raw_history = torch.cat([x[0] for x in stats_mgr.history], dim=-2).squeeze(0) 
            ema_history = torch.cat([x[1] for x in stats_mgr.history], dim=-2).squeeze(0)
        except Exception:
            continue
        
        head_idx = 0
        raw_data = raw_history[head_idx].cpu().numpy() # [Chunks, Dim]
        ema_data = ema_history[head_idx].cpu().numpy()
        
        # 繪圖: Line Plot
        avg_magnitude = np.mean(raw_data, axis=0)
        top_channels = np.argsort(avg_magnitude)[-3:] 
        
        plt.figure(figsize=(12, 6))
        for ch in top_channels:
            plt.plot(raw_data[:, ch], linestyle='--', alpha=0.6, label=f'Ch {ch} Raw ($m_t$)')
            plt.plot(ema_data[:, ch], linestyle='-', linewidth=2, label=f'Ch {ch} EMA ($\mu_t$)')
            
        plt.title(f'[{task_name}] Layer {layer_idx} Head {head_idx} - Scale Evolution')
        plt.xlabel('Chunk Index')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(task_save_dir, f"sample_{sample_idx}_layer_{layer_idx}_lines.png"))
        plt.close()

def get_dummy_input():
    return "This is a test prompt to verify the chunking mechanism. " * 500

def print_metrics(output_text, metrics):
    """統一的輸出格式函式"""
    print("\n" + "="*40)
    print("FINAL OUTPUT TEXT")
    print("="*40)
    print(output_text[:500] + "...(truncated)" if len(output_text) > 500 else output_text)
    
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    print(f"Peak Memory Usage : {metrics.peak_memory_mb:.2f} MB")
    print(f"Total Latency     : {metrics.total_latency_ms:.2f} ms")
    print(f"Throughput        : {metrics.throughput_tokens_per_sec:.2f} tokens/sec")
    print("="*40)

def run_interactive(model, args, profiler):
    """互動模式"""
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

            logger.info(f"Generating response...")
            
            profiler.start()
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=args.output_len)
            metrics = profiler.stop(num_output_tokens=args.output_len)

            print_metrics(output, metrics)

    except KeyboardInterrupt:
        print("\nExiting...")

def run_dummy(model, args, profiler):
    """Dummy 模式"""
    logger.info("Running Dummy Test...")
    prompt = get_dummy_input()
    
    profiler.start()
    with torch.no_grad():
        output = model.generate(prompt, max_new_tokens=args.output_len)
    metrics = profiler.stop(num_output_tokens=args.output_len)
    
    logger.info("Dummy Test Completed.")
    print_metrics(output, metrics)

def run_longbench(model, args, profiler, task_name):
    """LongBench 模式 - 執行單一任務"""
    # 1. 初始化 Loader
    try:
        if args.bench_version == "v1":
            logger.info(f"Loading LongBench V1 task: {task_name}")
            loader = LongBenchLoader(task_name=task_name)
        else:
            logger.info(f"Loading LongBench V2 domain: {task_name}")
            loader = LongBenchV2Loader(task_name=task_name, split="train") 
    except Exception as e:
        logger.error(f"Skipping task '{task_name}' due to load error: {e}")
        return

    # 處理 num_samples = -1 的情況
    if args.num_samples < 0:
        sample_count = len(loader)
    else:
        sample_count = min(args.num_samples, len(loader))

    logger.info(f"Evaluating {sample_count} samples for task: {task_name}")
    
    results = []
    total_score = 0.0
    
    # 建立 EMA 圖表資料夾
    plots_dir = "results/ema_plots"

    for i in tqdm(range(sample_count), desc=f"Eval {task_name}"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        # 清除快取以避免累積
        torch.cuda.empty_cache()
        
        profiler.start()
        try:
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=args.output_len)
            
            # 繪製 EMA 圖表
            visualize_ema_tracking(model, plots_dir, i, task_name)
            
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            if args.bench_version == "v2":
                score = calculate_accuracy(output, ground_truths)
                metric_name = "Accuracy"
            else:
                score = calculate_f1_score(output, ground_truths)
                metric_name = "F1"

            total_score += score
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "score": score,
                "metric": metric_name,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "throughput": perf_metrics.throughput_tokens_per_sec,
                "output": output,
                "ground_truth": ground_truths[0]
            })
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                results.append({"index": i, "error": "OOM", "score": 0.0})

    avg_score = total_score / len(results) if results else 0.0
    max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0

    logger.info(f"Task: {task_name} | Avg Score: {avg_score:.4f} | Max Memory: {max_peak_memory:.2f} MB")
    
    # 存檔
    mode_str = f"quant_w{args.n_warmup}_b{args.bits}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_results_{task_name}_{mode_str}.json"
    output_path = os.path.join("./results/compression", output_filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task_name,
            "version": args.bench_version,
            "args": vars(args),
            "config": model.config.__dict__,
            "avg_score": avg_score,
            "max_peak_memory_mb": max_peak_memory,
            "details": results
        }, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved results to {output_path}")

def main():
    args = parse_args()
    
    # 1. 初始化 Config
    config = LiveKVQuantConfig(
        chunk_size=args.chunk_size, 
        n_warmup=args.n_warmup, 
        bits=args.bits,
        ema_alpha=args.ema_alpha,
        clip_factor_n=args.clip_factor_n,
        outlier_ratio=args.outlier_ratio
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Mode: {args.input_mode.upper()} | Device: {device}")
    
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
        # LongBench 模式：支援多任務迴圈
        # 現在使用 args.task_type
        tasks_to_run = get_task_list(args.bench_version, args.task_type)
        logger.info(f"Tasks scheduled for execution: {tasks_to_run}")
        
        for task_name in tasks_to_run:
            run_longbench(model, args, profiler, task_name)
            torch.cuda.empty_cache() # 任務間清除快取

if __name__ == "__main__":
    main()