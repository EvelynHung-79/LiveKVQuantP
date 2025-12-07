import torch
from typing import Tuple

def isolate_outliers(tensor: torch.Tensor, ratio: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分離異常值 (Vector Outlier Isolation)。
    
    [修正重點] 直接使用 topk_indices 提取數值，確保 sparse_values 與 sparse_indices 順序一致。
    """
    # 1. 計算要保留的異常值數量 k
    total_elements = tensor.numel()
    k = int(total_elements * ratio)
    
    if k == 0:
        return tensor, torch.empty(0, device=tensor.device), torch.empty(0, device=tensor.device)

    # 2. 找出絕對值最大的 Top-K 索引
    # 注意：我們對 abs 取 topk，但需要取回原始 signed values
    flat_tensor = tensor.flatten()
    topk_abs, topk_indices = torch.topk(torch.abs(flat_tensor), k)
    
    # 3. [核心修正] 直接使用索引提取原始數值
    # 這樣 sparse_values[i] 就精確對應到 sparse_indices[i]
    sparse_values = flat_tensor[topk_indices]
    
    # 4. 準備 Dense Tensor (將 Outliers 歸零)
    # 這裡我們需要一個 Mask 來把這些位置設為 0
    dense_flat = flat_tensor.clone()
    mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
    mask[topk_indices] = True
    dense_flat[mask] = 0 
    
    dense_tensor = dense_flat.view_as(tensor)
    
    return dense_tensor, sparse_values, topk_indices

def restore_outliers(dequantized_tensor: torch.Tensor, 
                     sparse_values: torch.Tensor, 
                     sparse_indices: torch.Tensor) -> torch.Tensor:
    """
    還原異常值 (Outlier Integration)。
    """
    if sparse_values.numel() == 0:
        return dequantized_tensor

    if dequantized_tensor.dtype != sparse_values.dtype:
        dequantized_tensor = dequantized_tensor.to(sparse_values.dtype)

    flat_recon = dequantized_tensor.flatten()
    
    # 將 FP16 的異常值填回原始位置
    # 因為 isolate 階段我們修正了順序，現在這裡可以直接賦值
    flat_recon[sparse_indices] = sparse_values
    
    return flat_recon.view_as(dequantized_tensor)