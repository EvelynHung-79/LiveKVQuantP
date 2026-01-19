import torch

class KVCacheManager:
    def __init__(self, config, max_seq_len=None, max_batch_size=None):
        self.config = config
        self.k_chunks = [] 
        self.v_chunks = []
        # [還原重點] 這裡沒有 Buffer，只有 List
        self.current_len = 0 # Dummy，只為了相容介面呼叫

    def store_chunk(self, k_data, v_data):
        """
        直接儲存 Chunk 資料到 List。
        """
        self.k_chunks.append(k_data)
        self.v_chunks.append(v_data)
        
        # 簡單估算長度，避免外部呼叫報錯
        if "data" in k_data:
            self.current_len += k_data["data"].shape[-2]
        elif "quantized_data" in k_data:
            self.current_len += k_data["quantized_data"].shape[-2]

    def get_all_chunks(self):
        """
        回傳 List，供 PyTorch Fallback 使用。
        """
        return list(self.k_chunks), list(self.v_chunks)

    def get_cache_view(self):
        # [還原重點] 舊版沒有這個功能，若 Triton 被誤觸發會報錯，但因為我們改了 layer_controller，不會走到這
        raise NotImplementedError("List-based manager does not support get_cache_view (Triton Path).")

    def clear(self):
        self.k_chunks = []
        self.v_chunks = []
        self.current_len = 0