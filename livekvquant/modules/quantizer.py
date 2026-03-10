import torch
import logging
from ..utils.quant_utils import (
    calculate_symmetric_scale, quantize_symmetric,
    calculate_asymmetric_params, quantize_asymmetric,
)
from ..utils.outliers import isolate_outliers

# 取得 Logger
logger = logging.getLogger(__name__)

class RealTimeQuantizer:
    def __init__(self, config):
        self.config = config
        self.bits = config.bits
        self.outlier_ratio = config.outlier_ratio

    def isolate(self, tensor: torch.Tensor, outlier_dim: int, sink_length: int = 0):
        """
        分離 Dense 與 Sparse 部分，並選擇性保護 Sink Token。

        Args:
            tensor: 輸入 Tensor
            outlier_dim: 指定沿著哪個軸抓 Outlier (-1 or -2)
            sink_length: 若 > 0，則前 N 個 Token 會被強制視為 Sparse 保留，
                         並從 Dense 部分移除（歸零）。
        """
        # Ablation: 停用 outlier isolation，直接回傳全 dense
        if not self.config.use_outlier_isolation:
            empty = torch.empty(0, device=tensor.device)
            empty_idx = torch.empty(0, device=tensor.device, dtype=torch.int32)
            return tensor, empty, empty_idx

        # 1. 準備 Working Tensor
        # 只在需要修改 sink 區域時才 clone，否則直接使用原始 tensor
        if sink_length > 0:
            working_tensor = tensor.clone()
            working_tensor[..., :sink_length, :] = 0
        else:
            working_tensor = tensor

        # 3. 執行標準 Outlier 分離
        # dense_tensor 裡面的 outlier 已經被移除 (或平滑化)
        dense_tensor, sparse_vals, sparse_idxs = isolate_outliers(
            working_tensor,
            self.outlier_ratio,
            dim=outlier_dim
        )

        # 4. Append Sink (Post-isolation)
        # 將原始的 Sink 數值加回 Sparse 列表
        if sink_length > 0:
            # 建立 Sink Mask
            sink_mask = torch.zeros_like(tensor, dtype=torch.bool)
            sink_mask[..., :sink_length, :] = True

            # 取得 Sink 的數值與 Flatten Indices
            sink_values = tensor[..., :sink_length, :].flatten()

            # 這裡假設 sparse_idxs 是使用 flatten index (與原專案邏輯一致)
            sink_indices = torch.nonzero(sink_mask.flatten(), as_tuple=False).squeeze().to(torch.int32)

            # 合併 (Sink + Outliers)
            sparse_vals = torch.cat([sparse_vals, sink_values])
            sparse_idxs = torch.cat([sparse_idxs, sink_indices])

            # 確保 Dense Tensor 的 Sink 區域為 0 (雙重保險)
            dense_tensor[..., :sink_length, :] = 0

        return dense_tensor, sparse_vals, sparse_idxs

    def quantize_dense(self, dense_tensor: torch.Tensor, stats):
        """
        量化 dense tensor。

        Args:
            dense_tensor: 要量化的資料 (Outlier 已經歸零)
            stats: 來自 StatisticsManager：
                - ema_absmax: 單一 absmax tensor（symmetric）
                - ema_minmax: (ema_max, ema_min) tuple（asymmetric）

        Returns:
            - symmetric:  (quantized_data, scale)
            - asymmetric: (quantized_data, scale, zero_point)
        """
        if isinstance(stats, tuple):
            # Asymmetric quantization (ema_minmax)
            ema_max, ema_min = stats
            scale, zero_point = calculate_asymmetric_params(ema_max, ema_min, self.bits)
            quantized_data = quantize_asymmetric(dense_tensor, scale, zero_point, self.bits)
            return quantized_data, scale, zero_point
        else:
            # Symmetric quantization (ema_absmax，原有邏輯)
            absmax = stats
            scale = calculate_symmetric_scale(absmax, self.bits)
            quantized_data = quantize_symmetric(dense_tensor, scale, self.bits)
            return quantized_data, scale
