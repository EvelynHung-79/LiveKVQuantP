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
from data.constants import get_task_list

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Standard Baseline Inference Script with Batch Support")
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HuggingFace Model ID")
    parser.add_argument("--bench_version", type=str, choices=["v1", "v2"], default="v1", help="LongBench version (v1 or v2)")
    parser.add_argument("--task_type", type=str, default="narrativeqa", help="Specific task name or category")
    
    # [修改] 您可以把 default 改成 -1，這樣不打參數就會跑全部。目前設為 10 是為了防呆。
    parser.add_argument("--num_samples", type=int, default=-1, help="Number of samples per task. Set -1 to run ALL.")
    
    parser.add_argument("--output_len", type=int, default=64, help="Max new tokens to generate")
    return parser.parse_args()

def evaluate_single_task(model, tokenizer, args, task_name, profiler):
    """
    執行單一任務的評估。
    修正了參數順序以匹配 main() 的呼叫: (model, tokenizer, args, task_name, profiler)
    包含 FastKV 風格的 Head+Tail Truncation 策略。
    """
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

    # 決定測試樣本數
    if args.num_samples < 0:
        sample_count = len(loader)
    else:
        sample_count = min(args.num_samples, len(loader))
    
    logger.info(f"Evaluating {sample_count} samples for task: {task_name}")
    
    results = []
    total_score = 0.0
    
    # 確保 pad_token 存在
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id
    
    # 計算最大輸入長度 (Llama-3.1 支援 128k，預留 output_len)
    max_pos = getattr(model.config, "max_position_embeddings", 131072)
    max_input_len = max_pos - args.output_len

    for i in tqdm(range(sample_count), desc=f"Eval {task_name}"):
        sample = loader.get_sample(i)
        ground_truths = sample["answers"]
        raw_prompt = sample["prompt"]
        
        # =================================================================
        # [修改 1] 移除 System Prompt，只使用 User Role
        # FastKV 的做法是不加 system prompt，避免干擾模型的格式遵循能力
        # =================================================================
        messages = [{"role": "user", "content": raw_prompt}]
        
        try:
            # 不 tokenize，先拿回字串，因為我們要手動處理 truncation
            final_prompt_text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception:
            # 如果 template 失敗，退回 raw prompt
            final_prompt_text = raw_prompt

        # 先轉成 Token ID
        inputs = tokenizer(final_prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        input_ids = inputs.input_ids
        
        # =================================================================
        # [修改 2] 實作 FastKV 的 "Head + Tail" Truncation
        # 如果超過長度，保留最前面一半與最後面一半，捨棄中間
        # =================================================================
        if input_ids.shape[1] > max_input_len:
            half = int(max_input_len / 2)
            
            # 取出頭尾 Token
            front_ids = input_ids[0, :half]
            back_ids = input_ids[0, -half:]
            
            # 解碼回文字並接合 (模擬 FastKV 的處理方式)
            new_text = tokenizer.decode(front_ids, skip_special_tokens=True) + \
                       tokenizer.decode(back_ids, skip_special_tokens=True)
            
            # 重新 Tokenize 處理過的文本 (add_special_tokens=True 會補回 BOS)
            inputs = tokenizer(
                new_text, 
                return_tensors="pt", 
                add_special_tokens=True,
                truncation=True,
                max_length=max_input_len
            ).to(model.device)
            
            # logger.debug(f"Sample {i} truncated: {input_ids.shape[1]} -> {inputs.input_ids.shape[1]}")
            input_len = inputs.input_ids.shape[1]
        else:
            input_len = input_ids.shape[1]
        
        # Inference
        profiler.start()
        try:
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.output_len,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                    do_sample=False,
                    temperature=1.0 # Greedy decoding usually implies do_sample=False, temp doesn't matter much but keeping consistent
                )
            
            perf_metrics = profiler.stop(num_output_tokens=args.output_len)
            
            # Decode output (只取新生成的 tokens)
            generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            
            # 計算分數
            if args.bench_version == "v2":
                score = calculate_accuracy(generated_text, ground_truths)
                metric_name = "Accuracy"
            else:
                score = calculate_f1_score(generated_text, ground_truths)
                metric_name = "F1"

            total_score += score
            
            results.append({
                "index": i,
                "input_length": sample["context_length"], 
                "score": score,
                "metric": metric_name,
                "peak_memory_mb": perf_metrics.peak_memory_mb,
                "latency_ms": perf_metrics.total_latency_ms,
                "output": generated_text,
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