import torch
import logging
from ..utils.quant_utils import calculate_symmetric_scale, quantize_symmetric
from ..utils.outliers import isolate_outliers

# 取得 Logger
logger = logging.getLogger(__name__)

class RealTimeQuantizer:
    def __init__(self, config):
        self.bits = config.bits
        self.outlier_ratio = config.outlier_ratio
        
        # [DEBUG] 用來限制 Log 數量的計數器，避免洗版
        self.debug_counter = 0 

    def isolate(self, tensor: torch.Tensor, outlier_dim: int):
        """
        Args:
            outlier_dim: 指定沿著哪個軸抓 Outlier
        """
        dense_tensor, sparse_vals, sparse_idxs = isolate_outliers(tensor, self.outlier_ratio, dim=outlier_dim)
        return dense_tensor, sparse_vals, sparse_idxs

    def quantize_dense(self, dense_tensor: torch.Tensor, absmax: torch.Tensor):
        """
        Input: 
            dense_tensor: 要量化的資料 (Outlier 已經歸零)
            absmax: 統計出的絕對最大值 (來自 StatisticsManager)
        """
        # 1. 將 AbsMax 轉換為 Scale Factor (s = absmax / 7)
        scale = calculate_symmetric_scale(absmax, self.bits)
        
        # 2. 使用正確的 Scale 進行量化 (X_q = round(X / s))
        quantized_data = quantize_symmetric(dense_tensor, scale, self.bits)
        
        # ================= [DEBUG LOG START] =================
        # 我們只在前 20 個 chunk 印出 Log，或者當「數值崩壞」時強制印出
        
        # 計算 0 的比例
        # total_elements = quantized_data.numel()
        # num_zeros = (quantized_data == 0).sum().item()
        # zero_ratio = num_zeros / total_elements
        
        # # 只有在 (1) 剛開始跑 (2) 或者 0 的比例異常高 (>95%) 時才印 Log
        # if self.debug_counter < 20 or zero_ratio > 0.95:
        #     self.debug_counter += 1
            
        #     # 取得一些統計值 (轉成 float item 避免佔用 GPU 記憶體)
        #     dense_max = dense_tensor.abs().max().item()
        #     scale_val = scale.mean().item() # scale 可能是 tensor
        #     absmax_val = absmax.mean().item()
            
            # log_msg = (
            #     f"[Quant Debug] Zeros: {zero_ratio:.2%} | "
            #     f"Scale: {scale_val:.4f} (AbsMax: {absmax_val:.4f}) | "
            #     f"Dense Max: {dense_max:.4f}"
            # )
            
            # if zero_ratio > 0.95:
            #     logger.error(f"⚠️ QUANTIZATION COLLAPSE: {log_msg}")
            # elif self.debug_counter < 5:
            #     logger.info(log_msg)
        # ================= [DEBUG LOG END] =================
        
        return quantized_data, scale