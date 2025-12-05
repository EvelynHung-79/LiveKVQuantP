import torch
from ..utils.quant_utils import calculate_symmetric_scale, quantize_symmetric
from ..utils.outliers import isolate_outliers

class RealTimeQuantizer:
    """
    負責執行 Outlier Isolation 與 Symmetric Quantization。
    對應論文 3.4 Real-time Quantizer Module.
    """
    def __init__(self, config):
        self.bits = config.bits
        self.outlier_ratio = config.outlier_ratio

    def compress(self, tensor: torch.Tensor, scale_basis: torch.Tensor):
        """
        執行壓縮流程：Outlier Isolation -> Scale Calculation -> Quantization
        
        Args:
            tensor: 原始 FP16 Tensor (Key or Value)
            scale_basis: StatisticsManager 算出來的基礎統計量 (mu_t or m_t)
        """
        # 1. Outlier Isolation (Sparse Part) 
        # 傳回 dense_tensor (outliers 歸零), sparse_values, sparse_indices
        dense_tensor, sparse_vals, sparse_idxs = isolate_outliers(tensor, self.outlier_ratio)
        
        # 2. 計算 Scale Factor (s) 
        # Eq 3-5: s = mu_t / (2^(b-1) - 1)
        # 注意：這裡使用 scale_basis (即 mu_t 或 m_t) 來計算 s
        s = calculate_symmetric_scale(scale_basis, self.bits)
        
        # 3. Quantization Core (Dense Part) 
        # Eq 3-6: X_quant = Clamp(Round(X / s))
        quantized_data = quantize_symmetric(dense_tensor, s, self.bits)
        
        return {
            "type": "quantized",
            "quantized_data": quantized_data, # INT8 storage (representing INT4)
            "scale": s,                       # FP16 Scale
            "sparse_values": sparse_vals,     # FP16 Outliers
            "sparse_indices": sparse_idxs     # Indices
        }