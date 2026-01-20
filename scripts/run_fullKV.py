import sys
import os
import torch
import logging
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# 將專案根目錄加入 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.longbench_loader import LongBenchLoader
from data.longbench_v2_loader import LongBenchV2Loader
from data.constants import get_task_list
from evaluation.profiler import MemoryProfiler
from evaluation.engine import LongBenchEvaluator

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

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading Model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    # 1. 載入原生 HF 模型
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    # 2. 初始化共用元件
    profiler = MemoryProfiler()
    evaluator = LongBenchEvaluator(
        model=model, 
        tokenizer=tokenizer, 
        bench_version=args.bench_version
    )

    tasks_to_run = get_task_list(args.bench_version, args.task_type)
    logger.info(f"Tasks scheduled: {tasks_to_run}")

    # 3. 執行任務迴圈
    for task_name in tasks_to_run:
        try:
            # 根據版本選擇 Loader
            if args.bench_version == "v1":
                loader = LongBenchLoader(task_name=task_name)
            else:
                loader = LongBenchV2Loader(task_name=task_name)
        except Exception as e:
            logger.error(f"Failed to load task {task_name}: {e}")
            continue

        evaluator.run_task(task_name, loader, profiler, args)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()