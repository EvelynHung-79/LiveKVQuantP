import torch
from ..utils.quant_utils import calculate_symmetric_scale, quantize_symmetric
from ..utils.outliers import isolate_outliers

class RealTimeQuantizer:
    def __init__(self, config):
        self.bits = config.bits
        self.outlier_ratio = config.outlier_ratio

    def isolate(self, tensor: torch.Tensor):
        """步驟 1: 分離 Outliers，回傳 Dense Tensor 供統計計算"""
        dense_tensor, sparse_vals, sparse_idxs = isolate_outliers(tensor, self.outlier_ratio)
        return dense_tensor, sparse_vals, sparse_idxs

    def quantize_dense(self, dense_tensor: torch.Tensor, scale: torch.Tensor):
        """步驟 2: 使用針對 Dense 計算出的 Scale 進行量化"""
        # Eq 3-6: X_quant = Clamp(Round(X / s))
        quantized_data = quantize_symmetric(dense_tensor, scale, self.bits)
        return quantized_data