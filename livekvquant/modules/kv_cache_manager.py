import torch

class KVCacheManager:
    """
    混合精度 KV Cache 儲存管理器。
    對應論文 3.5 KV Cache Manager Module.
    支援兩種儲存格式：
    1. Warm-up Chunks: 純 FP16
    2. Quantized Chunks: Dense(INT4) + Sparse(FP16)
    """
    def __init__(self, config):
        self.config = config
        # 使用 List 儲存每個 Chunk 的資料 (簡單實作，可優化為 Pre-allocated Tensor)
        self.k_cache = [] 
        self.v_cache = []

    def clear(self):
        """清空 Cache"""
        self.k_cache = []
        self.v_cache = []

    def store_chunk(self, k_data: dict, v_data: dict):
        """
        [修正] 通用儲存介面。
        允許 K 和 V 擁有不同的狀態 (例如 K 是 Warmup FP16, V 是 Quantized)。
        
        Args:
            k_data (dict): 包含 K 的資料，可能是 {"type": "warmup", ...} 或 {"type": "quantized", ...}
            v_data (dict): 包含 V 的資料
        """
        self.k_cache.append(k_data)
        self.v_cache.append(v_data)

    def get_all_chunks(self):
        """回傳目前所有的 Cache Chunks (用於 Attention Core 重建)"""
        return self.k_cache, self.v_cache
    
    def current_length(self):
        """計算目前 Cache 的總長度 (用於 Positional Embedding 等)"""
        length = 0
        for chunk in self.k_cache:
            if chunk["type"] == "warmup":
                length += chunk["data"].size(-2)
            else:
                length += chunk["quantized_data"].size(-2)
        return length