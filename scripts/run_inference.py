import sys
import os
import torch
import logging
import json
import argparse
from tqdm import tqdm
from datetime import datetime

# 將專案根目錄加入 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.longbench_loader import LongBenchLoader
from data.longbench_v2_loader import LongBenchV2Loader
from evaluation.metrics import calculate_f1_score, calculate_accuracy
from evaluation.profiler import MemoryProfiler
# [修改] 從 constants 引入
from data.constants import get_task_list, LONGBENCH_PROMPTS, TASK_OUTPUT_LEN

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LiveKVQuant-P Unified Inference Script")
    parser.add_argument("--input_mode", type=str, choices=["longbench", "interactive", "dummy"], default="longbench")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--bench_version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--task_type", type=str, default="narrativeqa")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--output_len", type=int, default=64)
    
    # Quantization Args
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--n_warmup", type=int, default=2)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--ema_alpha", type=float, default=0.1)
    parser.add_argument("--clip_factor_n", type=float, default=1.5)
    parser.add_argument("--outlier_ratio", type=float, default=0.01)
    
    return parser.parse_args()

def visualize_ema_tracking(model, save_dir, sample_idx, task_name):
    """
    將 StatisticsManager 紀錄的 EMA 變化畫成圖表。
    保留 .float() 轉型以修正 BFloat16 錯誤。
    """
    target_layers = [0, 16, 31]
    task_save_dir = os.path.join(save_dir, task_name)
    os.makedirs(task_save_dir, exist_ok=True)
    
    for layer_idx in target_layers:
        if layer_idx >= len(model.controllers): break
        
        controller = model.controllers[layer_idx]
        stats_mgr = controller.stats_manager
        
        if not stats_mgr.history:
            continue
            
        try:
            raw_history = torch.cat([x[0] for x in stats_mgr.history], dim=-2).squeeze(0) 
            ema_history = torch.cat([x[1] for x in stats_mgr.history], dim=-2).squeeze(0)
        except Exception:
            continue
        
        head_idx = 0
        raw_data = raw_history[head_idx].float().cpu().numpy() 
        ema_data = ema_history[head_idx].float().cpu().numpy()
        
        avg_magnitude = np.mean(raw_data, axis=0)
        top_channels = np.argsort(avg_magnitude)[-3:] 
        
        plt.figure(figsize=(12, 6))
        for ch in top_channels:
            plt.plot(raw_data[:, ch], linestyle='--', alpha=0.6, label=f'Ch {ch} Raw')
            plt.plot(ema_data[:, ch], linestyle='-', linewidth=2, label=f'Ch {ch} EMA')
            
        plt.title(f'[{task_name}] Layer {layer_idx} Scale Evolution')
        plt.savefig(os.path.join(task_save_dir, f"sample_{sample_idx}_layer_{layer_idx}.png"))
        plt.close()
        plt.clf() 

def run_longbench(model, args, profiler, task_name):
    # 1. Loader
    try:
        if args.bench_version == "v1":
            loader = LongBenchLoader(task_name=task_name)
        else:
            loader = LongBenchV2Loader(task_name=task_name, split="train") 
    except Exception as e:
        logger.error(f"Skipping task {task_name}: {e}")
        return

    if args.num_samples < 0:
        sample_count = len(loader)
    else:
        sample_count = min(args.num_samples, len(loader))

    # [修改] 從 constants 取得長度
    current_output_len = TASK_OUTPUT_LEN.get(task_name, args.output_len)
    logger.info(f"Evaluating {sample_count} samples for {task_name} | Max New Tokens: {current_output_len}")

    results = []
    total_score = 0.0
    plots_dir = "results/ema_plots"

    max_pos = getattr(model.model.config, "max_position_embeddings", 131072)
    max_input_len = max_pos - current_output_len - 100 

    for i in tqdm(range(sample_count), desc=f"Eval {task_name}"):
        sample = loader.get_sample(i)
        ground_truths = sample["answers"]
        
        # [修改] 從 constants 取得 Prompt
        raw_context = sample["raw_context"]
        raw_input = sample["raw_input"]
        
        if task_name in LONGBENCH_PROMPTS:
            prompt_template = LONGBENCH_PROMPTS[task_name]
            final_prompt = prompt_template.format(context=raw_context, input=raw_input)
        else:
            final_prompt = sample["prompt"]

        messages = [{"role": "user", "content": final_prompt}]
        try:
            final_input_text = model.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            final_input_text = final_prompt

        inputs = model.tokenizer(final_input_text, return_tensors="pt", add_special_tokens=False)
        input_ids = inputs.input_ids
        
        # Head + Tail Truncation
        if input_ids.shape[1] > max_input_len:
            half = int(max_input_len / 2)
            front_ids = input_ids[0, :half]
            back_ids = input_ids[0, -half:]
            
            final_input_text = model.tokenizer.decode(front_ids, skip_special_tokens=True) + \
                               model.tokenizer.decode(back_ids, skip_special_tokens=True)
            
        torch.cuda.empty_cache()
        profiler.start()
        try:
            with torch.no_grad():
                # [修改] 傳入正確的 current_output_len
                output = model.generate(final_input_text, max_new_tokens=current_output_len)
            
            # visualize_ema_tracking(model, plots_dir, i, task_name)
            
            perf_metrics = profiler.stop(num_output_tokens=current_output_len)
            
            if args.bench_version == "v2":
                score = calculate_accuracy(output, ground_truths)
                metric_name = "Accuracy"
            else:
                score = calculate_f1_score(output, ground_truths)
                metric_name = "F1"

            total_score += score
            
            results.append({
                "index": i,
                "score": score,
                "metric": metric_name,
                "output": output,
                "ground_truth": ground_truths[0],
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms
            })
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                results.append({"index": i, "error": "OOM", "score": 0.0})

    avg_score = total_score / len(results) if results else 0.0
    avg_latency = sum([r.get("latency_ms", 0.0) for r in results]) / len(results) if results else 0.0
    max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0

    logger.info(f"Task: {task_name} | Avg Score: {avg_score:.4f} | Avg Latency: {avg_latency:.2f} ms | Max Memory: {max_peak_memory:.2f} MB")
    
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
            "avg_score": avg_score,
            "avg_latency_ms": avg_latency,
            "max_peak_memory_mb": max_peak_memory,
            "details": results
        }, f, indent=4, ensure_ascii=False)

def main():
    args = parse_args()
    
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
    
    try:
        model = LiveKVQuantModel(args.model_id, config, device)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return

    profiler = MemoryProfiler()
    
    if args.input_mode == "interactive":
        # (略)
        pass
    elif args.input_mode == "dummy":
        # (略)
        pass
    else:
        tasks_to_run = get_task_list(args.bench_version, args.task_type)
        logger.info(f"Tasks scheduled: {tasks_to_run}")
        
        for task_name in tasks_to_run:
            run_longbench(model, args, profiler, task_name)
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()