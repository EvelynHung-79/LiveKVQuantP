import torch
from dataclasses import dataclass
from typing import Optional
from ..utils.quant_utils import dequantize_symmetric
from ..utils.outliers import restore_outliers

@dataclass
class KVChunk:
    """
    封裝一個 KV Cache Chunk 的所有資料與行為。
    取代原本鬆散的 Dictionary 結構。
    """
    chunk_type: str  # "warmup" or "quantized"
    
    # Warmup 階段專用
    data: Optional[torch.Tensor] = None
    
    # Quantized 階段專用
    quantized_data: Optional[torch.Tensor] = None
    scale: Optional[torch.Tensor] = None
    sparse_values: Optional[torch.Tensor] = None
    sparse_indices: Optional[torch.Tensor] = None

    def reconstruct(self, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        將 Chunk 還原為 Dense Tensor。
        (邏輯從 AttentionCore._reconstruct_tensor 搬移至此)
        """
        if self.chunk_type == "warmup":
            if self.data is None:
                raise ValueError("Warmup chunk must have 'data'.")
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
        """Helper to check device"""
        if self.data is not None:
            return self.data.device
        return self.quantized_data.device

    @property
    def shape(self):
        """Helper to check shape (length)"""
        if self.data is not None:
            return self.data.shape
        return self.quantized_data.shape