import torch
from typing import Tuple

# Decode buffer 預分配大小（超過時自動擴展）
_DECODE_BUFFER_INIT_SIZE = 256


class KVCacheManager:
    def __init__(self, config, max_seq_len=None, max_batch_size=None):
        self.config = config
        # 快取已 dequantize 的 chunk tensor，避免重複 reconstruct
        self._k_chunk_cache = None
        self._v_chunk_cache = None
        # Pre-allocated decode buffer（避免每個 token 都 torch.cat）
        self._k_decode_buf = None
        self._v_decode_buf = None
        self._decode_len = 0  # 目前 decode buffer 已使用的長度
        self._decode_capacity = 0  # decode buffer 的容量
        # 完整 KV 快取（chunk_cache + decode_buffer 合併後的結果）
        self._k_full_cache = None
        self._v_full_cache = None
        self._full_cache_valid = False  # 標記 full cache 是否需要重建
        self.current_len = 0

    def __len__(self):
        return self.current_len

    def _ensure_decode_buffer(self, k_token):
        """確保 decode buffer 有足夠空間，不夠時 2x 擴展。"""
        if self._k_decode_buf is None or self._decode_len >= self._decode_capacity:
            new_cap = max(_DECODE_BUFFER_INIT_SIZE, self._decode_capacity * 2)
            # shape: (batch, heads, capacity, head_dim)
            shape = list(k_token.shape)
            shape[-2] = new_cap
            new_k = torch.zeros(shape, dtype=k_token.dtype, device=k_token.device)
            new_v = torch.zeros(shape, dtype=k_token.dtype, device=k_token.device)
            if self._k_decode_buf is not None and self._decode_len > 0:
                new_k[..., :self._decode_len, :] = self._k_decode_buf[..., :self._decode_len, :]
                new_v[..., :self._decode_len, :] = self._v_decode_buf[..., :self._decode_len, :]
            self._k_decode_buf = new_k
            self._v_decode_buf = new_v
            self._decode_capacity = new_cap

    def store_decode_token(self, k_token, v_token):
        """將 decode token 寫入 pre-allocated buffer（O(1)，不做 cat）。"""
        n_tokens = k_token.shape[-2]
        self._ensure_decode_buffer(k_token)
        self._k_decode_buf[..., self._decode_len:self._decode_len + n_tokens, :] = k_token
        self._v_decode_buf[..., self._decode_len:self._decode_len + n_tokens, :] = v_token
        self._decode_len += n_tokens
        self.current_len += n_tokens
        self._full_cache_valid = False  # decode buffer 有新 token，full cache 需更新

    def store_chunk(self, k_chunk, v_chunk):
        """
        儲存 KVChunk：立即 reconstruct 後 append 到 chunk cache tensor。
        不再保留原始 chunk 物件（省 memory）。
        """
        target_dtype = torch.float16
        k_recon = k_chunk.reconstruct(target_dtype)
        v_recon = v_chunk.reconstruct(target_dtype)

        if self._k_chunk_cache is None:
            self._k_chunk_cache = k_recon
            self._v_chunk_cache = v_recon
        else:
            self._k_chunk_cache = torch.cat([self._k_chunk_cache, k_recon], dim=-2)
            self._v_chunk_cache = torch.cat([self._v_chunk_cache, v_recon], dim=-2)

        self.current_len += k_recon.shape[-2]
        self._full_cache_valid = False

    def get_full_kv(self, target_dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        回傳完整 KV tensor。利用 dirty flag 避免不必要的重複 cat。
        """
        if self._full_cache_valid and self._k_full_cache is not None:
            return self._k_full_cache.to(dtype=target_dtype), self._v_full_cache.to(dtype=target_dtype)

        k = self._k_chunk_cache
        v = self._v_chunk_cache

        if self._decode_len > 0:
            k_dec = self._k_decode_buf[..., :self._decode_len, :]
            v_dec = self._v_decode_buf[..., :self._decode_len, :]
            if k is not None:
                k = torch.cat([k, k_dec], dim=-2)
                v = torch.cat([v, v_dec], dim=-2)
            else:
                k = k_dec
                v = v_dec

        if k is None:
            return None, None

        self._k_full_cache = k
        self._v_full_cache = v
        self._full_cache_valid = True
        return k.to(dtype=target_dtype), v.to(dtype=target_dtype)

    def clear(self):
        self._k_chunk_cache = None
        self._v_chunk_cache = None
        self._k_decode_buf = None
        self._v_decode_buf = None
        self._decode_len = 0
        self._decode_capacity = 0
        self._k_full_cache = None
        self._v_full_cache = None
        self._full_cache_valid = False
        self.current_len = 0
