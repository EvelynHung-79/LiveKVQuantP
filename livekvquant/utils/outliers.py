import torch
from typing import Tuple

def isolate_outliers(tensor: torch.Tensor, ratio: float, dim: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    [CRITICAL FIX] 確保回傳的 sparse_indices 是全域的 Flat Indices。
    """
    # 1. 計算要保留的 k
    if dim is not None:
        dim_size = tensor.size(dim)
        k = max(1, int(dim_size * ratio)) # 至少抓 1 個
    else:
        k = int(tensor.numel() * ratio)
    
    if k == 0: # 防呆
        return tensor, torch.empty(0, device=tensor.device), torch.empty(0, device=tensor.device)

    # 2. 建立 Mask 來標記 Outlier 位置
    mask = torch.zeros_like(tensor, dtype=torch.bool)
    
    if dim is not None:
        # Dimension-Aware: 在指定軸向找 Top-K
        _, topk_indices = torch.topk(torch.abs(tensor), k, dim=dim)
        # 使用 scatter 將 True 填入正確的全域位置
        mask.scatter_(dim, topk_indices, True)
    else:
        # Global: 舊版邏輯
        flat_abs = torch.abs(tensor.flatten())
        _, topk_indices = torch.topk(flat_abs, k)
        flat_mask = mask.flatten()
        flat_mask[topk_indices] = True
        mask = flat_mask.view_as(tensor)

    # 3. [關鍵步驟] 提取數值與 "Flat Indices"
    # 使用 mask.flatten() 來取得全域扁平索引
    flat_mask = mask.flatten()
    sparse_indices = torch.nonzero(flat_mask, as_tuple=False).squeeze()
    
    # 確保 values 的順序跟 indices 一致
    flat_tensor = tensor.flatten()
    sparse_values = flat_tensor[sparse_indices]
    
    # 4. 製作 Dense Tensor (將 Outlier 歸零)
    dense_tensor = tensor.clone()
    dense_tensor.masked_fill_(mask, 0) # 使用 mask 歸零比 scatter 更直觀
    
    return dense_tensor, sparse_values, sparse_indices

def restore_outliers(dequantized_tensor: torch.Tensor, 
                     sparse_values: torch.Tensor, 
                     sparse_indices: torch.Tensor) -> torch.Tensor:
    """
    (保持不變，但現在輸入的 sparse_indices 已經是正確的 Flat Indices 了)
    """
    if sparse_values.numel() == 0:
        return dequantized_tensor

    if dequantized_tensor.dtype != sparse_values.dtype:
        dequantized_tensor = dequantized_tensor.to(sparse_values.dtype)

    flat_recon = dequantized_tensor.flatten()
    
    # 因為 sparse_indices 已經修正為 Flat Index，這裡的操作就安全了
    flat_recon[sparse_indices] = sparse_values
    
    return flat_recon.view_as(dequantized_tensor)