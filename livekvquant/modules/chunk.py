import torch
from dataclasses import dataclass
from typing import Optional
from ..utils.quant_utils import dequantize_symmetric
from ..utils.outliers import restore_outliers

@dataclass
class KVChunk:
    """
    封裝一個 KV Cache Chunk 的所有資料與行為。

    chunk_type:
      - "raw"       : 尚未量化的原始 FP16/BF16 tensor（暫存用，會被 finalize 替換）
      - "warmup"    : Warm-up 階段，保留原始精度
      - "quantized" : INT4 量化 + BF16 sparse outliers
    """
    chunk_type: str  # "raw", "warmup", or "quantized"

    # raw / warmup 階段專用
    data: Optional[torch.Tensor] = None

    # quantized 階段專用
    quantized_data: Optional[torch.Tensor] = None  # int8 tensor
    scale: Optional[torch.Tensor] = None
    sparse_values: Optional[torch.Tensor] = None   # 保持原始 dtype (bf16)
    sparse_indices: Optional[torch.Tensor] = None   # int32

    def reconstruct(self, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """將 Chunk 還原為 Dense Tensor。"""
        if self.chunk_type in ("raw", "warmup"):
            if self.data is None:
                raise ValueError(f"{self.chunk_type} chunk must have 'data'.")
            return self.data.to(dtype=target_dtype)

        # Quantized path
        if self.quantized_data is None:
            raise ValueError("Quantized chunk must have 'quantized_data'.")

        dequantized = dequantize_symmetric(
            self.quantized_data,
            self.scale
        )

        reconstructed = restore_outliers(
            dequantized,
            self.sparse_values,
            self.sparse_indices
        )

        return reconstructed.to(dtype=target_dtype)

    @property
    def device(self):
        if self.data is not None:
            return self.data.device
        return self.quantized_data.device

    @property
    def shape(self):
        if self.data is not None:
            return self.data.shape
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
        if self.sparse_values is not None:
            total += self.sparse_values.nelement() * self.sparse_values.element_size()
        if self.sparse_indices is not None:
            total += self.sparse_indices.nelement() * self.sparse_indices.element_size()
        return total