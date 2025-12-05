import torch
from typing import Tuple

def isolate_outliers(tensor: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分離異常值 (Vector Outlier Isolation)。
    對應論文 3.4.2: Top magnitudes stored in FP16, rest quantized.

    Args:
        tensor (X): 原始 FP16 Tensor。
        ratio (R): 異常值比例 (e.g., 0.01 for 1%)。

    Returns:
        dense_tensor: 剔除異常值後的 Tensor (異常值位置被設為 0，準備進行量化)。
        sparse_values: 異常值的原始 FP16 數值。
        sparse_indices: 異常值在原始 Tensor 中的索引位置。
    """
    # 1. 計算要保留的異常值數量 k
    total_elements = tensor.numel()
    k = int(total_elements * ratio)
    
    if k == 0:
        # 如果不需要分離，回傳空
        return tensor, torch.empty(0, device=tensor.device), torch.empty(0, device=tensor.device)

    # 2. 找出絕對值最大的 Top-K 索引
    # flatten() 是為了簡化 topk 計算，實際應用可根據維度調整
    flat_tensor = tensor.flatten()
    topk_values, topk_indices = torch.topk(torch.abs(flat_tensor), k)
    
    # 3. 建立 Sparse Mask
    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    mask[topk_indices] = True
    
    # 4. 提取 Sparse Values (FP16)
    # 這裡我們只存 flat 後的 indices，重建時再還原
    sparse_values = flat_tensor[mask]
    
    # 5. 準備 Dense Tensor
    # 將異常值位置設為 0 (或保留原值讓量化器去夾斷，但設為 0 對 Scale 計算影響較小)
    # 論文中提到 "Top magnitudes stored in FP16, rest quantized"，
    # 為了讓 Dense Part 的 Absmax 變小，這裡建議將 Outlier 位置的數值歸零或抑制。
    dense_flat = flat_tensor.clone()
    dense_flat[mask] = 0 
    dense_tensor = dense_flat.view_as(tensor)
    
    return dense_tensor, sparse_values, topk_indices

def restore_outliers(dequantized_tensor: torch.Tensor, 
                     sparse_values: torch.Tensor, 
                     sparse_indices: torch.Tensor) -> torch.Tensor:
    """
    還原異常值 (Outlier Integration)。
    對應論文 Eq 3-9: X_past = Combine(X_dequant, X_sparse)
    """
    if sparse_values.numel() == 0:
        return dequantized_tensor

    # 為了操作方便，先 flatten
    flat_recon = dequantized_tensor.flatten()
    
    # 將 FP16 的異常值填回原始位置 (覆蓋掉反量化出來的值)
    flat_recon[sparse_indices] = sparse_values
    
    # Reshape 回原始形狀
    return flat_recon.view_as(dequantized_tensor)