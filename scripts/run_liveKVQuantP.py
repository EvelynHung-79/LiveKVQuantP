import sys
import os
import torch
import logging
import argparse

# 將專案根目錄加入 Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.longbench_loader import LongBenchLoader
from data.longbench_v2_loader import LongBenchV2Loader
from data.constants import get_task_list
from evaluation.profiler import MemoryProfiler
from evaluation.engine import LongBenchEvaluator

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
    parser.add_argument("--quant_start_layer", type=int, default=3, help="Layer index to start quantization from (default: 3)")

    # Ablation flags
    parser.add_argument("--use_warmup", type=lambda x: x.lower() != 'false', default=True,
                        help="Warmup chunk strategy (default: True). Pass 'false' to disable.")
    parser.add_argument("--use_outlier_isolation", type=lambda x: x.lower() != 'false', default=True,
                        help="Outlier isolation (default: True). Pass 'false' for pure dense quantization.")
    parser.add_argument("--stats_method", type=str, choices=["ema_absmax", "ema_minmax"], default="ema_absmax",
                        help="Statistics method for quantization scale (default: ema_absmax).")

    # Memory optimization flags
    parser.add_argument("--use_chunked_attn", type=lambda x: x.lower() != 'false', default=False,
                        help="Per-chunk dequant + online softmax to reduce attention peak memory (default: False).")

    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Config & Model
    config = LiveKVQuantConfig(
        chunk_size=args.chunk_size,
        n_warmup=args.n_warmup,
        bits=args.bits,
        ema_alpha=args.ema_alpha,
        clip_factor_n=args.clip_factor_n,
        outlier_ratio=args.outlier_ratio,
        quant_start_layer=args.quant_start_layer,
        use_warmup=args.use_warmup,
        use_outlier_isolation=args.use_outlier_isolation,
        stats_method=args.stats_method,
        use_chunked_attn=args.use_chunked_attn,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Mode: {args.input_mode.upper()} | Device: {device}")
    
    try:
        model = LiveKVQuantModel(args.model_id, config, device)
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        return
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        allocated_mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        logger.info(f"📊 [Memory] GPU Allocated (Model + Context): {allocated_mem_gb:.2f} GB")

    # 2. Components
    profiler = MemoryProfiler()
    evaluator = LongBenchEvaluator(
        model=model, 
        tokenizer=model.tokenizer, 
        bench_version=args.bench_version
    )
    
    # 3. Task Scheduling
    if args.input_mode == "longbench":
        tasks_to_run = get_task_list(args.bench_version, args.task_type)
        logger.info(f"Tasks scheduled: {tasks_to_run}")
        
        for task_name in tasks_to_run:
            # 載入 Data Loader
            try:
                if args.bench_version == "v1":
                    loader = LongBenchLoader(task_name=task_name)
                else:
                    loader = LongBenchV2Loader(task_name=task_name, split="train")
            except Exception as e:
                logger.error(f"Failed to load task {task_name}: {e}")
                continue
            
            # 執行評估 (一行搞定！)
            evaluator.run_task(task_name, loader, profiler, args)
            
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()