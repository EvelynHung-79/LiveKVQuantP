import torch
import logging
import os
import json
from tqdm import tqdm
from datetime import datetime
from evaluation.metrics import calculate_f1_score, calculate_accuracy
from livekvquant.utils.data_utils import truncate_input_ids
from data.constants import TASK_OUTPUT_LEN

logger = logging.getLogger(__name__)

class LongBenchEvaluator:
    def __init__(self, model, tokenizer, bench_version="v1", output_dir="./results/"):
        self.model = model
        self.tokenizer = tokenizer
        self.bench_version = bench_version
        self.output_dir = output_dir

    def run_task(self, task_name: str, loader, profiler, args):
        """
        執行單一任務的完整評估流程。
        """
        # 1. 準備參數
        num_samples = len(loader) if args.num_samples < 0 else min(args.num_samples, len(loader))
        current_output_len = TASK_OUTPUT_LEN.get(task_name, args.output_len)
        
        # 計算最大輸入長度
        max_pos = getattr(self.model.config, "max_position_embeddings", 131072)
        max_input_len = max_pos - current_output_len - 100
        
        logger.info(f"Evaluating {task_name} | Samples: {num_samples} | Max New Tokens: {current_output_len}")
        
        results = []
        total_score = 0.0
        
        # 2. Evaluation Loop
        for i in tqdm(range(num_samples), desc=f"Eval {task_name}"):
            sample = loader.get_sample(i)
            ground_truths = sample["answers"]
            
            # A. Format Prompt
            final_prompt = sample["prompt"]

            # B. Apply Chat Template
            messages = [{"role": "user", "content": final_prompt}]
            try:
                final_input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except:
                final_input_text = final_prompt

            inputs = self.tokenizer(final_input_text, return_tensors="pt", add_special_tokens=False)
            input_ids = inputs.input_ids
            
            # C. Truncation (現在這個函數只做 Tensor 操作，很快)
            input_ids = truncate_input_ids(input_ids, max_input_len, self.tokenizer)
            input_ids = input_ids.to(self.model.device)

            # D. Inference
            profiler.start()
            try:
                with torch.no_grad():
                    if hasattr(self.model, "controllers"): 
                        # Case 1: LiveKVQuantModel
                        output_text = self.model.generate(input_ids=input_ids, max_new_tokens=current_output_len)
                    else: 
                        # Case 2: 原生 HuggingFace Model 
                        output_ids = self.model.generate(
                            input_ids, 
                            max_new_tokens=current_output_len,
                            use_cache=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                            do_sample=False
                        )
                        # 記得要把 prompt 部分截掉，只留生成的 tokens
                        output_text = self.tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
                
                # E. Metrics
                perf_metrics = profiler.stop(num_output_tokens=current_output_len)
                
                if self.bench_version == "v2":
                    score = calculate_accuracy(output_text, ground_truths)
                    metric_name = "Accuracy"
                else:
                    score = calculate_f1_score(output_text, ground_truths)
                    metric_name = "F1"
                
                total_score += score
                results.append({
                    "index": i,
                    "score": score,
                    "metric": metric_name,
                    "output": output_text,
                    "ground_truth": ground_truths[0],
                    "peak_memory_mb": perf_metrics.peak_memory_mb,
                    "latency_ms": perf_metrics.total_latency_ms
                })

            except Exception as e:
                logger.error(f"Error at sample {i}: {e}")
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    results.append({"index": i, "error": "OOM", "score": 0.0})

        # 3. Save Results
        self._save_results(task_name, results, total_score, args)

    def _save_results(self, task_name, results, total_score, args):
        avg_score = total_score / len(results) if results else 0.0
        avg_latency = sum([r.get("latency_ms", 0.0) for r in results]) / len(results) if results else 0.0
        max_peak_memory = max([r.get("peak_memory_mb", 0.0) for r in results]) if results else 0.0

        logger.info(f"Task: {task_name} | Avg Score: {avg_score:.4f} | Avg Latency: {avg_latency:.2f} ms | Max Memory: {max_peak_memory:.2f} MB")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        subdir = "liveKVQuant" if hasattr(self.model, "controllers") else "baselines/fullKV"
        output_filename = f"{timestamp}_{task_name}.json"
        output_path = os.path.join(self.output_dir, subdir, output_filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "task": task_name,
                "version": self.bench_version,
                "args": vars(args),
                "avg_score": avg_score,
                "avg_latency_ms": avg_latency,
                "max_peak_memory_mb": max_peak_memory,
                "details": results
            }, f, indent=4, ensure_ascii=False)