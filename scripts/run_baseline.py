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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === 任務定義 ===
V1_TASK_GROUPS = {
    "single-doc": ["narrativeqa", "qasper", "multifieldqa_en"],
    "multi-doc": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "summarization": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "few-shot": ["trec", "triviaqa", "samsum", "lsht"],
    "synthetic": ["passage_retrieval_en", "passage_count"],
    "code": ["lcc", "repobench-p"],
    "all": [
        "narrativeqa", "qasper", "multifieldqa_en", 
        "hotpotqa", "2wikimqa", "musique", "dureader", 
        "gov_report", "qmsum", "multi_news", "vcsum",
        "trec", "triviaqa", "samsum", "lsht",
        "passage_retrieval_en", "passage_count",
        "lcc", "repobench-p"
    ]
}

V2_DOMAIN_MAP = {
    "single-doc": "single-document",
    "multi-doc": "multi-document",
    "summarization": "summarization",
    "long-context": "long in-context",
    "dialogue": "long dialogue",
    "code": "code repository",
    "all": "all"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Baseline Inference Script with Batch Support")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--bench_version", type=str, choices=["v1", "v2"], default="v1", help="LongBench version (v1 or v2)")
    parser.add_argument("--task_type", type=str, default="narrativeqa", help="Specific task name or category")
    
    # [修改] 您可以把 default 改成 -1，這樣不打參數就會跑全部。目前設為 10 是為了防呆。
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples per task. Set -1 to run ALL.")
    
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    return parser.parse_args()

def get_task_list(version, task_type):
    task_type_lower = task_type.lower()
    if version == "v1":
        if task_type_lower in V1_TASK_GROUPS:
            return V1_TASK_GROUPS[task_type_lower]
        return [task_type]
    elif version == "v2":
        domain = V2_DOMAIN_MAP.get(task_type_lower, task_type)
        return [domain]

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

    # [修改] 處理 num_samples = -1 的情況 (跑全部)
    if args.num_samples < 0:
        sample_count = len(loader)
    else:
        sample_count = min(args.num_samples, len(loader))
        
    logger.info(f"Evaluating {sample_count} samples for task: {task_name}")
    
    results = []
    total_score = 0.0
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    for i in tqdm(range(sample_count), desc=f"Eval {task_name}"):
        sample = loader.get_sample(i)
        
        # === Chat Template ===
        user_content = sample["prompt"]
        system_prompt = "You are a helpful assistant. Please answer the question based on the context provided."
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            final_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
            final_prompt = user_content
            
        ground_truths = sample["answers"]
        
        max_pos = getattr(model.config, "max_position_embeddings", 131072)
        max_input_len = max_pos - args.output_len
        
        inputs = tokenizer(
            final_prompt, 
            return_tensors="pt",
            truncation=True,             
            max_length=max_input_len   
        ).to(model.device)
        
        input_len = inputs.input_ids.size(1)
        
        # Inference
        profiler.start()
        try:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.output_len,
                    use_cache=True, 
                    pad_token_id=tokenizer.pad_token_id,
                    temperature=0.0,
                    do_sample=False
                )
            
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            generated_ids = outputs[0][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            if args.bench_version == "v2":
                score = calculate_accuracy(output_text, ground_truths)
                metric_name = "Accuracy"
            else:
                score = calculate_f1_score(output_text, ground_truths)
                metric_name = "F1"

            total_score += score
            
            results.append({
                "index": i,
                "input_length": sample["context_length"],
                "score": score,
                "metric": metric_name,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "output": output_text,
                "ground_truth": ground_truths[0]
            })
            
        except torch.cuda.OutOfMemoryError:
            logger.error(f"OOM at sample {i}")
            torch.cuda.empty_cache()
            results.append({"index": i, "error": "OOM", "score": 0.0})
        except Exception as e:
            logger.error(f"Error at sample {i}: {e}")

    avg_score = total_score / len(results) if results else 0.0
    max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0
    
    logger.info(f"Task: {task_name} | Avg Score: {avg_score:.4f} | Max Memory: {max_peak_memory:.2f} MB")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{timestamp}_baseline_{args.bench_version}_{task_name}.json"
    output_path = os.path.join("./results/baseline", output_filename)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "task": task_name,
            "version": args.bench_version,
            "args": vars(args),
            "avg_score": avg_score,
            "max_peak_memory_mb": max_peak_memory,
            "details": results
        }, f, indent=4, ensure_ascii=False)
    
    logger.info(f"Saved results to {output_path}")

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Loading Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=torch.float16, device_map=device)
    model.eval()

    profiler = MemoryProfiler()
    tasks_to_run = get_task_list(args.bench_version, args.task_type)
    logger.info(f"Tasks scheduled for execution: {tasks_to_run}")

    for task_name in tasks_to_run:
        evaluate_single_task(model, tokenizer, args, task_name, profiler)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()