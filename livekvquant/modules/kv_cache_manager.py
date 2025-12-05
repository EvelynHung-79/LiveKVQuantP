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

    def store_warmup(self, k: torch.Tensor, v: torch.Tensor):
        """儲存 Warm-up 階段的 FP16 資料 """
        # 為了統一介面，包裝成 dict
        self.k_cache.append({"type": "warmup", "data": k})
        self.v_cache.append({"type": "warmup", "data": v})

    def store_quantized(self, k_compressed: dict, v_compressed: dict):
        """儲存 Quantization 階段的壓縮資料 """
        self.k_cache.append(k_compressed)
        self.v_cache.append(v_compressed)

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