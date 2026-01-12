import torch

class KVCacheManager:
    def __init__(self, config):
        self.config = config
        self.k_chunks = [] 
        self.v_chunks = []

    def store_chunk(self, k_data, v_data):
        """
        直接儲存已處理好(量化或Warmup)的 Chunk 資料。
        """
        self.k_chunks.append(k_data)
        self.v_chunks.append(v_data)

    def get_all_chunks(self):
        """
        回傳目前所有已儲存的 Chunks。
        """
        return list(self.k_chunks), list(self.v_chunks)

    def clear(self):
        self.k_chunks = []
        self.v_chunks = []