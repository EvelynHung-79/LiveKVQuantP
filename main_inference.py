import os
import argparse
import time
import torch
import yaml
import logging
from typing import Optional

# 引入專案內部的模組
from config import LiveKVQuantConfig
from livekvquant.model_wrapper import LiveKVQuantModel
from data.longbench_loader import LongBenchLoader
from evaluation.profiler import MemoryProfiler

# 設定 Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LiveKVQuant-P Inference Entry Point")
    
    # 配置文件路徑
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to experiment config file")
    
    # 覆寫特定參數 (方便快速測試)
    parser.add_argument("--chunk_size", type=int, default=None, help="Override chunk size (e.g., 512)")
    parser.add_argument("--bits", type=int, default=None, help="Override quantization bits (e.g., 4)")
    parser.add_argument("--warmup", type=int, default=None, help="Override warmup steps (N_warmup)")
    
    # 輸入模式
    parser.add_argument("--input_mode", type=str, choices=["dummy", "longbench", "interactive"], default="dummy", 
                        help="Input source: 'dummy' (test prompt), 'longbench' (dataset), or 'interactive' (user input)")
    parser.add_argument("--task", type=str, default="narrativeqa", help="LongBench task name (if input_mode is longbench)")
    
    # 模型設定
    parser.add_argument("--model_id", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="HuggingFace Model ID")
    
    return parser.parse_args()

def set_seed(seed: int = 42):
    """固定隨機種子以確保實驗可重現 (Reproducibility)"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(args) -> LiveKVQuantConfig:
    """載入 YAML 配置並根據 argparse 進行覆寫"""
    # 1. 載入預設配置
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        config = LiveKVQuantConfig(**config_dict)
    else:
        logger.warning(f"Config file {args.config} not found. Using default dataclass values.")
        config = LiveKVQuantConfig()

    # 2. 命令行參數優先 (Command-line overrides)
    if args.chunk_size: config.chunk_size = args.chunk_size
    if args.bits: config.bits = args.bits
    if args.warmup: config.n_warmup = args.warmup
    
    logger.info(f"Active Configuration: {config}")
    return config

def get_input_text(args) -> str:
    """根據模式獲取輸入 Prompt"""
    if args.input_mode == "dummy":
        # 用於快速 Debug 的長文本
        return "This is a test prompt to verify the chunking mechanism. " * 500  # 約 4000 tokens
    
    elif args.input_mode == "longbench":
        try:
            loader = LongBenchLoader(task_name=args.task)
            prompt, _ = loader.get_formatted_input(index=0) # 預設拿第一筆測試
            logger.info(f"Loaded sample from LongBench task: {args.task}, Length: {len(prompt)}")
            return prompt
        except Exception as e:
            logger.error(f"Failed to load LongBench: {e}")
            raise e
            
    elif args.input_mode == "interactive":
        print("\n=== Please enter your prompt (Press Ctrl+D to finish) ===")
        import sys
        return sys.stdin.read()
    
    return ""

def main():
    args = parse_args()
    set_seed(42)
    
    # 1. 環境檢查
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA is not available. Running on CPU will be extremely slow.")
    else:
        logger.info(f"Running on GPU: {torch.cuda.get_device_name(0)}")

    # 2. 準備配置與模型
    config = load_config(args)
    
    logger.info(f"Loading Model: {args.model_id}...")
    # LiveKVQuantModel 是我們自定義的 Wrapper，內部會注入 Controller/Quantizer
    model_wrapper = LiveKVQuantModel(model_id=args.model_id, config=config, device=device)
    
    # 3. 準備輸入數據
    input_text = get_input_text(args)
    
    # 4. 初始化 Profiler (測量 Memory & Latency)
    profiler = MemoryProfiler()
    
    logger.info("Starting Inference...")
    logger.info(f"Strategy: Chunk Size={config.chunk_size}, Warmup={config.n_warmup}, Bits={config.bits}")

    # --- INFERENCE START ---
    profiler.start()
    
    try:
        # 使用 torch.no_grad() 減少不必要的梯度記憶體開銷 (Inference Only)
        with torch.no_grad():
            # generate 函式內部會處理:
            # Tokenize -> Prefill (Chunking Loop) -> Decoding (Token-by-token)
            output_text = model_wrapper.generate(
                input_text, 
                max_new_tokens=128,  # 生成長度
                temperature=0.7
            )
            
    except torch.cuda.OutOfMemoryError:
        logger.error("OOM Error occurred during inference!")
        return
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        traceback.print_exc()
        return

    metrics = profiler.stop()
    # --- INFERENCE END ---

    # 5. 輸出結果與指標
    print("\n" + "="*40)
    print("FINAL OUTPUT TEXT")
    print("="*40)
    # 僅印出前 500 字避免洗版，實際可存檔
    print(output_text[:500] + "...(truncated)" if len(output_text) > 500 else output_text)
    
    print("\n" + "="*40)
    print("PERFORMANCE METRICS")
    print("="*40)
    print(f"Peak Memory Usage : {metrics['peak_memory_mb']:.2f} MB")
    print(f"Total Latency     : {metrics['latency_ms']:.2f} ms")
    print(f"Throughput        : {metrics['latency_ms'] / len(input_text.split()):.2f} ms/token (approx)")
    print("="*40)

    # (Optional) Save metrics to CSV for plotting
    # save_metrics_to_csv(metrics, args)

if __name__ == "__main__":
    main()