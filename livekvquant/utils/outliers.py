import torch
from typing import Tuple

def isolate_outliers(tensor: torch.Tensor, ratio: float, dim: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分離異常值，支援指定維度 (dim)。
    
    Args:
        dim: 
          - 對於 K (Per-Channel)，應設為 -2 (沿著 Sequence 軸抓 Outlier)。
          - 對於 V (Per-Token)，應設為 -1 (沿著 Head Dim 軸抓 Outlier)。
          - 若為 None，則維持舊版 Global Flatten 行為 (不建議)。
    """
    # 1. 計算要保留的 k (數量)
    if dim is None:
        # 舊版 Global Logic
        total_elements = tensor.numel()
        k = int(total_elements * ratio)
        flat_tensor = tensor.flatten()
        topk_abs, topk_indices = torch.topk(torch.abs(flat_tensor), k)
        sparse_values = flat_tensor[topk_indices]
        
        dense_flat = flat_tensor.clone()
        mask = torch.zeros_like(flat_tensor, dtype=torch.bool)
        mask[topk_indices] = True
        dense_flat[mask] = 0
        return dense_flat.view_as(tensor), sparse_values, topk_indices

    else:
        # === 新版 Dimension-Aware Logic ===
        # 計算該軸向的 k
        dim_size = tensor.size(dim)
        k = int(dim_size * ratio)
        if k < 1: k = 1 # 至少抓 1 個，避免 ratio 太小沒抓到

        # 在指定維度找 Top-K
        # values, indices 形狀會跟 tensor 一樣，但在 dim 那軸變成 k
        topk_vals, topk_indices = torch.topk(torch.abs(tensor), k, dim=dim)
        
        # 為了儲存方便 (配合原本的架構)，我們通常還是得把 index 轉成 flattened 格式，
        # 或是使用 scatter 將其取出。這裡示範用 scatter 把 dense 歸零。
        
        # 建立一個全 0 的 Mask
        mask = torch.zeros_like(tensor, dtype=torch.bool)
        # 使用 scatter 把 Top-K 的位置標記為 True
        mask.scatter_(dim, topk_indices, True)
        
        # 取出 Sparse Values (使用 masked_select 會變成 1D tensor)
        sparse_values = torch.masked_select(tensor, mask)
        
        # 取得 Flattened Indices (為了相容你原本的 restore_outliers)
        #這步很重要，因為你的 restore 是用 flat index
        sparse_indices = torch.nonzero(mask.flatten(), as_tuple=False).squeeze()
        
        # 製作 Dense Tensor
        dense_tensor = tensor.clone()
        dense_tensor.masked_fill_(mask, 0) # 把 Outlier 位置填 0
        
        return dense_tensor, sparse_values, sparse_indices

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