import torch
from ..utils.quant_utils import calculate_symmetric_scale, quantize_symmetric
from ..utils.outliers import isolate_outliers

class RealTimeQuantizer:
    def __init__(self, config):
        self.bits = config.bits
        self.outlier_ratio = config.outlier_ratio

    def isolate(self, tensor: torch.Tensor, outlier_dim: int):
        """
        Args:
            outlier_dim: 指定沿著哪個軸抓 Outlier
        """
        dense_tensor, sparse_vals, sparse_idxs = isolate_outliers(tensor, self.outlier_ratio, dim=outlier_dim)
        return dense_tensor, sparse_vals, sparse_idxs

    def quantize_dense(self, dense_tensor: torch.Tensor, absmax: torch.Tensor):
        """
        [CRITICAL FIX] 修正量化邏輯
        Input: 
            dense_tensor: 要量化的資料
            absmax: 統計出的絕對最大值 (來自 StatisticsManager)
        Output:
            quantized_data: INT8 Tensor
            scale: 實際使用的 Scale Factor (存入 Cache 用)
        """
        # 1. 將 AbsMax 轉換為 Scale Factor (s = absmax / 7)
        scale = calculate_symmetric_scale(absmax, self.bits)
        
        # 2. 使用正確的 Scale 進行量化 (X_q = round(X / s))
        quantized_data = quantize_symmetric(dense_tensor, scale, self.bits)
        
        return quantized_data, scale