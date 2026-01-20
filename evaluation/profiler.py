import torch
import time
from dataclasses import dataclass

@dataclass
class ProfilingResult:
    peak_memory_mb: float
    prefill_latency_ms: float
    total_latency_ms: float
    throughput_tokens_per_sec: float

class MemoryProfiler:
    """
    負責測量記憶體使用量與延遲。
    對應論文 Page 45 Evaluation Metrics: 
    - Peak Memory Usage = max(memUsed_t) 
    - Latency & Throughput 
    """
    
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.prefill_end_time = 0.0
        self.peak_memory_bytes = 0
        
        # 確保在 CUDA 環境下
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA is not available for profiling.")

    def start(self):
        """開始測量：重置記憶體統計，記錄開始時間"""
        # torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() # 確保之前的運算都結束
        self.start_time = time.perf_counter()

    def mark_prefill_end(self):
        """(選用) 標記 Prefill 階段結束的時間點"""
        torch.cuda.synchronize()
        self.prefill_end_time = time.perf_counter()

    def stop(self, num_output_tokens: int = 1) -> ProfilingResult:
        """
        結束測量並回傳結果。
        
        Args:
            num_output_tokens (int): 生成的 token 總數，用於計算 Throughput
        """
        torch.cuda.synchronize()
        self.end_time = time.perf_counter()
        
        # 獲取 Peak Memory (bytes)
        self.peak_memory_bytes = torch.cuda.max_memory_allocated()
        
        # 計算指標
        total_latency_s = self.end_time - self.start_time
        prefill_latency_s = (self.prefill_end_time - self.start_time) if self.prefill_end_time > 0 else 0
        
        # 轉換單位
        peak_memory_mb = self.peak_memory_bytes / (1024 ** 2)
        total_latency_ms = total_latency_s * 1000
        prefill_latency_ms = prefill_latency_s * 1000
        
        # Throughput = Output Tokens / Time (s) 
        # 注意：如果只測 Prefill，throughput 定義可能不同，但在生成任務通常指 decoding throughput
        throughput = num_output_tokens / total_latency_s if total_latency_s > 0 else 0

        return ProfilingResult(
            peak_memory_mb=peak_memory_mb,
            prefill_latency_ms=prefill_latency_ms,
            total_latency_ms=total_latency_ms,
            throughput_tokens_per_sec=throughput
        )

    @staticmethod
    def calculate_compression_ratio(baseline_fullKV_mem_mb: float, compressed_mem_mb: float) -> float:
        """
        計算記憶體壓縮率。
        對應公式: Compress Ratio = Original Memory / Compressed Memory 
        
        Args:
            baseline_fullKV_mem_mb: FP16 原始模型的記憶體佔用
            compressed_mem_mb: LiveKVQuant-P 的記憶體佔用
        """
        if compressed_mem_mb == 0: return 0.0
        return baseline_fullKV_mem_mb / compressed_mem_mb