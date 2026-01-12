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
from data.longbench_v2_loader import LongBenchV2Loader
from evaluation.metrics import calculate_f1_score, calculate_accuracy
from evaluation.profiler import MemoryProfiler
# [修改] 從 constants 引入 prompt 和 output_len
from data.constants import get_task_list, LONGBENCH_PROMPTS, TASK_OUTPUT_LEN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Baseline Inference Script")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--bench_version", type=str, choices=["v1", "v2"], default="v1")
    parser.add_argument("--task_type", type=str, default="narrativeqa")
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--output_len", type=int, default=64)
    return parser.parse_args()

def evaluate_single_task(model, tokenizer, args, task_name, profiler):
    # 1. 初始化 Loader
    try:
        if args.bench_version == "v1":
            logger.info(f"Loading LongBench V1 task: {task_name}")
            loader = LongBenchLoader(task_name=task_name)
        else:
            logger.info(f"Loading LongBench V2 domain: {task_name}")
            loader = LongBenchV2Loader(task_name=task_name)
    except Exception as e:
        logger.error(f"Skipping task '{task_name}' due to load error: {e}")
        return

    if args.num_samples < 0:
        sample_count = len(loader)
    else:
        sample_count = min(args.num_samples, len(loader))
        
    # [修改] 從 constants 取得正確的長度
    current_output_len = TASK_OUTPUT_LEN.get(task_name, args.output_len)
    logger.info(f"Evaluating {sample_count} samples for {task_name} | Max New Tokens: {current_output_len}")
    
    results = []
    total_score = 0.0
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # 計算最大輸入長度
    max_pos = getattr(model.config, "max_position_embeddings", 131072)
    max_input_len = max_pos - current_output_len - 100 

    for i in tqdm(range(sample_count), desc=f"Eval {task_name}"):
        sample = loader.get_sample(i)
        ground_truths = sample["answers"]
        
        # [修改] 從 constants 取得正確的 Prompt Template
        raw_context = sample["raw_context"]
        raw_input = sample["raw_input"]
        
        if task_name in LONGBENCH_PROMPTS:
            prompt_template = LONGBENCH_PROMPTS[task_name]
            final_prompt = prompt_template.format(context=raw_context, input=raw_input)
        else:
            final_prompt = sample["prompt"]

        # 只使用 User Role，模擬 FastKV
        messages = [{"role": "user", "content": final_prompt}]
        try:
            final_input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            final_input_text = final_prompt

        inputs = tokenizer(final_input_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        input_ids = inputs.input_ids
        
        # Head + Tail Truncation
        if input_ids.shape[1] > max_input_len:
            half = int(max_input_len / 2)
            front_ids = input_ids[0, :half]
            back_ids = input_ids[0, -half:]
            
            new_text = tokenizer.decode(front_ids, skip_special_tokens=True) + \
                       tokenizer.decode(back_ids, skip_special_tokens=True)
            
            inputs = tokenizer(
                new_text, 
                return_tensors="pt", 
                add_special_tokens=True,
                truncation=True,
                max_length=max_input_len
            ).to(model.device)
            input_len = inputs.input_ids.shape[1]
        else:
            input_len = input_ids.shape[1]
        
        # Inference
        profiler.start()
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=current_output_len, # [修改] 使用正確長度
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1.0 
                )
            
            perf_metrics = profiler.stop(num_output_tokens=current_output_len)
            generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            
            if args.bench_version == "v2":
                score = calculate_accuracy(generated_text, ground_truths)
                metric_name = "Accuracy"
            else:
                score = calculate_f1_score(generated_text, ground_truths)
                metric_name = "F1"

            total_score += score
            
            results.append({
                "index": i,
                "score": score,
                "metric": metric_name,
                "output": generated_text,
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_{args.bench_version}_{task_name}.json"
    output_path = os.path.join("./results/baselines/fullKV/", output_filename)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # [保持] 使用 bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    profiler = MemoryProfiler()
    tasks_to_run = get_task_list(args.bench_version, args.task_type)
    logger.info(f"Tasks scheduled: {tasks_to_run}")

    for task_name in tasks_to_run:
        evaluate_single_task(model, tokenizer, args, task_name, profiler)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()