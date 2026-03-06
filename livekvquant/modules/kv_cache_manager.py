import torch
from typing import Tuple, List, Optional
from .chunk import KVChunk

# Decode buffer 預分配大小（超過時自動擴展）
_DECODE_BUFFER_INIT_SIZE = 256


class KVCacheManager:
    """
    管理 KV Cache 的儲存與重建。

    核心設計：
    - Prefill chunks 以壓縮格式（KVChunk 物件）儲存在 list 中，不立即 reconstruct。
    - 只在 get_full_kv() 被呼叫時才 lazy reconstruct + 快取結果。
    - Decode tokens 使用預分配 buffer，O(1) 寫入。
    """

    def __init__(self, config, max_seq_len=None, max_batch_size=None):
        self.config = config

        # Prefill chunks：以壓縮格式儲存
        self._k_chunks: List[KVChunk] = []
        self._v_chunks: List[KVChunk] = []
        self._chunk_seq_len = 0  # chunks 中的總 token 數

        # Lazy reconstruct 快取
        self._k_recon_cache: Optional[torch.Tensor] = None
        self._v_recon_cache: Optional[torch.Tensor] = None
        self._recon_valid = False

        # Pre-allocated decode buffer
        self._k_decode_buf = None
        self._v_decode_buf = None
        self._decode_len = 0
        self._decode_capacity = 0

        self.current_len = 0

    def __len__(self):
        return self.current_len

    # ------------------------------------------------------------------ #
    #  Prefill: Store Raw FP16 for current chunk (will be quantized later)
    # ------------------------------------------------------------------ #

    def store_raw(self, k_raw: torch.Tensor, v_raw: torch.Tensor):
        """
        暫存當前 chunk 的 raw FP16 KV（尚未量化）。
        這些 raw tensors 會在下一個 chunk 開始前被 finalize_last_chunk() 替換為壓縮格式。
        """
        k_chunk = KVChunk(chunk_type="raw", data=k_raw)
        v_chunk = KVChunk(chunk_type="raw", data=v_raw)
        self._k_chunks.append(k_chunk)
        self._v_chunks.append(v_chunk)
        self._chunk_seq_len += k_raw.shape[-2]
        self.current_len += k_raw.shape[-2]
        self._recon_valid = False

    def finalize_last_chunk(self, k_chunk: KVChunk, v_chunk: KVChunk):
        """
        將最後一個 raw chunk 替換為壓縮格式（quantized 或 warmup）。
        由 LayerController 在 attention 算完後呼叫。
        Prefill 階段同時清掉 recon cache，避免壓縮 chunks 與重建 cache 同時佔用記憶體。
        """
        if not self._k_chunks:
            return
        self._k_chunks[-1] = k_chunk
        self._v_chunks[-1] = v_chunk
        self._k_recon_cache = None
        self._v_recon_cache = None
        self._recon_valid = False

    # ------------------------------------------------------------------ #
    #  Decode: Pre-allocated buffer
    # ------------------------------------------------------------------ #

    def _ensure_decode_buffer(self, k_token):
        """確保 decode buffer 有足夠空間，不夠時 2x 擴展。"""
        if self._k_decode_buf is None or self._decode_len >= self._decode_capacity:
            new_cap = max(_DECODE_BUFFER_INIT_SIZE, self._decode_capacity * 2)
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

    # ------------------------------------------------------------------ #
    #  Read: Lazy reconstruct
    # ------------------------------------------------------------------ #

    def _reconstruct_chunks(self, target_dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """從壓縮的 chunk list 批量 reconstruct，並快取結果。"""
        if self._recon_valid and self._k_recon_cache is not None:
            return self._k_recon_cache.to(dtype=target_dtype), self._v_recon_cache.to(dtype=target_dtype)

        if not self._k_chunks:
            self._k_recon_cache = None
            self._v_recon_cache = None
            self._recon_valid = True
            return None, None

        k_parts = []
        v_parts = []
        for k_chunk, v_chunk in zip(self._k_chunks, self._v_chunks):
            k_parts.append(k_chunk.reconstruct(target_dtype))
            v_parts.append(v_chunk.reconstruct(target_dtype))

        self._k_recon_cache = torch.cat(k_parts, dim=-2) if len(k_parts) > 1 else k_parts[0]
        self._v_recon_cache = torch.cat(v_parts, dim=-2) if len(v_parts) > 1 else v_parts[0]
        self._recon_valid = True
        return self._k_recon_cache.to(dtype=target_dtype), self._v_recon_cache.to(dtype=target_dtype)

    def get_full_kv(self, target_dtype: torch.dtype) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        回傳完整 KV tensor (chunks + decode buffer)。
        不額外 cache 合併結果，避免記憶體膨脹。
        """
        k, v = self._reconstruct_chunks(target_dtype)

        if self._decode_len > 0:
            k_dec = self._k_decode_buf[..., :self._decode_len, :].to(dtype=target_dtype)
            v_dec = self._v_decode_buf[..., :self._decode_len, :].to(dtype=target_dtype)
            if k is not None:
                k = torch.cat([k, k_dec], dim=-2)
                v = torch.cat([v, v_dec], dim=-2)
            else:
                k = k_dec
                v = v_dec

        if k is None:
            return None, None

        return k, v

    # ------------------------------------------------------------------ #
    #  Introspection (for tests / debugging)
    # ------------------------------------------------------------------ #

    def get_all_chunks(self) -> Tuple[List[KVChunk], List[KVChunk]]:
        """回傳所有壓縮的 chunk 物件（供測試/除錯用）。"""
        return list(self._k_chunks), list(self._v_chunks)

    # ------------------------------------------------------------------ #
    #  Reset
    # ------------------------------------------------------------------ #

    def clear(self):
        self._k_chunks.clear()
        self._v_chunks.clear()
        self._chunk_seq_len = 0
        self._k_recon_cache = None
        self._v_recon_cache = None
        self._recon_valid = False
        self._k_decode_buf = None
        self._v_decode_buf = None
        self._decode_len = 0
        self._decode_capacity = 0
        self.current_len = 0
