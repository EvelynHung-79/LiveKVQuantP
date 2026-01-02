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
from evaluation.metrics import calculate_f1_score
from evaluation.profiler import MemoryProfiler

# [新增] 引入繪圖函式所需的套件
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
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name (only for longbench mode)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate (only for longbench mode)")
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    
    # 量化配置
    parser.add_argument("--chunk_size", type=int, default=512, help="Chunk size")
    parser.add_argument("--n_warmup", type=int, default=2, help="Number of warmup chunks")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits")
    parser.add_argument("--ema_alpha", type=float, default=0.1, help="EMA smoothing factor")
    parser.add_argument("--clip_factor_n", type=float, default=1.5, help="Clipped EMA factor N")
    parser.add_argument("--outlier_ratio", type=float, default=0.01, help="Ratio of outliers to keep in FP16")
    # [移除] --baseline 參數已刪除
    
    return parser.parse_args()

def visualize_ema_tracking(model, save_dir, sample_idx):
    """
    將 StatisticsManager 紀錄的 EMA 變化畫成圖表。
    """
    target_layers = [0, 16, 31]
    os.makedirs(save_dir, exist_ok=True)
    
    # print(f"Generating EMA tracking plots for Sample {sample_idx}...")

    for layer_idx in target_layers:
        if layer_idx >= len(model.controllers): break
        
        controller = model.controllers[layer_idx]
        stats_mgr = controller.stats_manager
        
        if not stats_mgr.history:
            continue
            
        # 整理數據 [Batch, Heads, Chunks, Dim] -> [Heads, Chunks, Dim]
        raw_history = torch.cat([x[0] for x in stats_mgr.history], dim=-2).squeeze(0) 
        ema_history = torch.cat([x[1] for x in stats_mgr.history], dim=-2).squeeze(0)
        
        head_idx = 0
        raw_data = raw_history[head_idx].numpy() # [Chunks, Dim]
        ema_data = ema_history[head_idx].numpy()
        
        # 繪圖 1: Heatmap
        # fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # im1 = axes[0].imshow(raw_data.T, aspect='auto', cmap='viridis', origin='lower')
        # axes[0].set_title(f'Layer {layer_idx} Head {head_idx} - Raw Absmax ($m_t$)')
        # axes[0].set_ylabel('Channels')
        # plt.colorbar(im1, ax=axes[0], label='Magnitude')
        
        # im2 = axes[1].imshow(ema_data.T, aspect='auto', cmap='viridis', origin='lower')
        # axes[1].set_title(f'Layer {layer_idx} Head {head_idx} - Stabilized EMA ($\mu_t$)')
        # axes[1].set_ylabel('Channels')
        # axes[1].set_xlabel('Chunk Index')
        # plt.colorbar(im2, ax=axes[1], label='Scale')
        
        # plt.tight_layout()
        # plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_layer_{layer_idx}_heatmap.png"))
        # plt.close()

        # 繪圖 2: Line Plot
        avg_magnitude = np.mean(raw_data, axis=0)
        top_channels = np.argsort(avg_magnitude)[-3:] 
        
        plt.figure(figsize=(12, 6))
        for ch in top_channels:
            plt.plot(raw_data[:, ch], linestyle='--', alpha=0.6, label=f'Ch {ch} Raw ($m_t$)')
            plt.plot(ema_data[:, ch], linestyle='-', linewidth=2, label=f'Ch {ch} EMA ($\mu_t$)')
            
        plt.title(f'Layer {layer_idx} Head {head_idx} - Scale Evolution (Top Active Channels)')
        plt.xlabel('Chunk Index')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"sample_{sample_idx}_layer_{layer_idx}_lines.png"))
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

            # [移除] 這裡的 Baseline 顯示
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

def run_longbench(model, args, profiler):
    """LongBench 模式"""
    try:
        if "v2" in args.task.lower() or args.input_mode == "longbench_v2":
            # 假設你想跑 LongBench v2
            # task 可以傳入 "all" 或是特定領域如 "Code"
            loader = LongBenchV2Loader(task_name=args.task, split="train")
        else:
            loader = LongBenchLoader(task_name=args.task)
    except Exception as e:
        logger.error(f"Failed to load task {args.task}: {e}")
        return

    logger.info(f"Loaded {len(loader)} samples. Evaluating first {args.num_samples}...")
    
    results = []
    total_f1 = 0.0
    
    # 建立 EMA 圖表資料夾
    plots_dir = "results/ema_plots"

    for i in tqdm(range(min(args.num_samples, len(loader))), desc="Evaluating"):
        sample = loader.get_sample(i)
        prompt = sample["prompt"]
        ground_truths = sample["answers"]
        
        profiler.start()
        try:
            with torch.no_grad():
                output = model.generate(prompt, max_new_tokens=args.output_len)
            
            # [新增] 繪製 EMA 圖表
            visualize_ema_tracking(model, plots_dir, i)
            
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            f1 = calculate_f1_score(output, ground_truths)
            total_f1 += f1
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "f1": f1,
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

    avg_f1 = total_f1 / len(results) if results else 0.0
    # [新增] 計算平均 Peak Memory
    max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0

    logger.info(f"Average F1: {avg_f1:.4f}")
    
    # [修改] 檔名只保留量化配置，移除 baseline 選項
    mode_str = f"quant_w{args.n_warmup}_b{args.bits}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"./results/compression/results_{args.task}_{mode_str}_{timestamp}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "config": model.config.__dict__,
            "avg_f1": avg_f1,
            "max_peak_memory_mb": max_peak_memory, # [新增] 寫入 JSON
            "details": results
        }, f, indent=4, ensure_ascii=False)
    logger.info(f"Saved to {output_file}")

def main():
    args = parse_args()
    
    # 1. 初始化 Config (baseline 預設為 False，不需傳入參數)
    config = LiveKVQuantConfig(
        chunk_size=args.chunk_size, 
        n_warmup=args.n_warmup, 
        bits=args.bits,
        ema_alpha=args.ema_alpha,
        clip_factor_n=args.clip_factor_n,
        outlier_ratio=args.outlier_ratio
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # [移除] 這裡的 Baseline 顯示
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
        run_longbench(model, args, profiler)

if __name__ == "__main__":
    main()