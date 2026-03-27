import torch
from dataclasses import dataclass, field
from typing import Optional
from ..utils.quant_utils import (
    dequantize_symmetric, pack_int4, unpack_int4,
    dequantize_asymmetric, pack_uint4, unpack_uint4,
)
from ..utils.outliers import restore_outliers

@dataclass
class KVChunk:
    """
    封裝一個 KV Cache Chunk 的所有資料與行為。

    chunk_type:
      - "raw"       : 尚未量化的原始 FP16/BF16 tensor（暫存用，會被 finalize 替換）
      - "warmup"    : Warm-up 階段，保留原始精度
      - "quantized" : INT4 量化 + BF16 sparse outliers（quantized_data 以 packed INT4 儲存）
    """
    chunk_type: str  # "raw", "warmup", or "quantized"

    # raw / warmup 階段專用
    data: Optional[torch.Tensor] = None

    # quantized 階段專用
    quantized_data: Optional[torch.Tensor] = None  # packed int8 tensor (2 int4 per byte)
    scale: Optional[torch.Tensor] = None
    zero_point: Optional[torch.Tensor] = None       # asymmetric 量化專用（int8，值域 [0,15]）
    sparse_values: Optional[torch.Tensor] = None     # 保持原始 dtype (bf16)
    sparse_indices: Optional[torch.Tensor] = None     # int32

    # 記錄 pack 前的最後一維大小，供 shape property 和 unpack 使用
    _unpacked_last_dim: Optional[int] = field(default=None, repr=False)
    # 標記是否已做 nibble packing（False = prefill 期間的 int8，True = packed int4）
    _is_packed: bool = field(default=False, repr=False)

    @property
    def is_asymmetric(self) -> bool:
        return self.zero_point is not None

    def __post_init__(self):
        """建立時不自動 pack，由外部在 prefill 結束後統一呼叫 pack()。"""
        pass

    def pack(self):
        """
        將 quantized_data 做 INT4 nibble packing（兩個 int4 塞進一個 int8）。
        prefill 結束後由 KVCacheManager.pack_all_chunks() 統一呼叫，只能呼叫一次。
        symmetric 用 pack_int4（signed），asymmetric 用 pack_uint4（unsigned）。
        """
        if self.chunk_type != "quantized" or self._is_packed or self.quantized_data is None:
            return
        self._unpacked_last_dim = self.quantized_data.shape[-1]
        if self.is_asymmetric:
            self.quantized_data = pack_uint4(self.quantized_data)
        else:
            self.quantized_data = pack_int4(self.quantized_data)
        self._is_packed = True

    def reconstruct(self, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """將 Chunk 還原為 Dense Tensor。"""
        if self.chunk_type in ("raw", "warmup"):
            if self.data is None:
                raise ValueError(f"{self.chunk_type} chunk must have 'data'.")
            return self.data.to(dtype=target_dtype)

        # Quantized path
        if self.quantized_data is None:
            raise ValueError("Quantized chunk must have 'quantized_data'.")

        if self.is_asymmetric:
            # Asymmetric path
            if self._is_packed:
                unpacked = unpack_uint4(self.quantized_data)
            else:
                unpacked = self.quantized_data
            dequantized = dequantize_asymmetric(unpacked, self.scale, self.zero_point)
        else:
            # Symmetric path
            if self._is_packed:
                unpacked = unpack_int4(self.quantized_data)
                dequantized = dequantize_symmetric(unpacked, self.scale)
            else:
                dequantized = dequantize_symmetric(self.quantized_data, self.scale)

        reconstructed = restore_outliers(
            dequantized,
            self.sparse_values,
            self.sparse_indices
        )

        return reconstructed.to(dtype=target_dtype)

    @property
    def seq_len(self) -> int:
        """回傳此 chunk 的 sequence length（dim=-2 的大小）。"""
        return self.shape[-2]

    @property
    def device(self):
        if self.data is not None:
            return self.data.device
        return self.quantized_data.device

    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape
        if self._unpacked_last_dim is not None:
            return (*self.quantized_data.shape[:-1], self._unpacked_last_dim)
        return self.quantized_data.shape

    @property
    def memory_bytes(self) -> int:
        """估算此 chunk 在 GPU 上佔用的 bytes（供 profiling 用）。"""
        total = 0
        if self.data is not None:
            total += self.data.nelement() * self.data.element_size()
        if self.quantized_data is not None:
            total += self.quantized_data.nelement() * self.quantized_data.element_size()
        if self.scale is not None:
            total += self.scale.nelement() * self.scale.element_size()
        if self.zero_point is not None:
            total += self.zero_point.nelement() * self.zero_point.element_size()
        if self.sparse_values is not None:
            total += self.sparse_values.nelement() * self.sparse_values.element_size()
        if self.sparse_indices is not None:
            total += self.sparse_indices.nelement() * self.sparse_indices.element_size()
        return total
