import torch
from typing import List, Tuple

class KVCacheManager:
    def __init__(self, config, max_seq_len=None, max_batch_size=None):
        self.config = config
        self._k_chunks = [] 
        self._v_chunks = []
        self.current_len = 0 

    def __len__(self):
        """提供標準的長度存取方式"""
        return self.current_len
    
    def store_chunk(self, k_chunk, v_chunk):
        """
        儲存 KVChunk 物件。
        """
        self._k_chunks.append(k_chunk)
        self._v_chunks.append(v_chunk)
        
        # 簡單估算長度，避免外部呼叫報錯
        self.current_len += k_chunk.shape[-2]

    def get_reconstructed_cache(self, target_dtype: torch.dtype) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        [新增] 負責將內部的 Chunk 還原為可用的 Tensor List。
        這是將邏輯從 AttentionCore 移回這裡 (Move Method)。
        """
        k_list = [c.reconstruct(target_dtype) for c in self._k_chunks]
        v_list = [c.reconstruct(target_dtype) for c in self._v_chunks]
        return k_list, v_list
    
    def get_cache_view(self):
        # 目前架構僅支援 List-based (PyTorch Path)
        # 若需要支援 Triton，需改寫 storage 為預先分配的 Buffer
        raise NotImplementedError("List-based manager does not support get_cache_view (Triton Path).")

    def clear(self):
        self._k_chunks = []
        self._v_chunks = []
        self.current_len = 0