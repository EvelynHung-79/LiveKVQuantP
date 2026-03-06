import torch
from typing import Tuple

def isolate_outliers(tensor: torch.Tensor, ratio: float, dim: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    分離 Outlier：回傳 (dense_tensor, sparse_values, sparse_flat_indices)。
    dense_tensor 中 outlier 位置已歸零。
    """
    # 1. 計算要保留的 k
    if dim is not None:
        dim_size = tensor.size(dim)
        k = max(1, int(dim_size * ratio))
    else:
        k = int(tensor.numel() * ratio)

    if k == 0:
        return tensor, torch.empty(0, device=tensor.device), torch.empty(0, device=tensor.device, dtype=torch.int32)

    # 2. 建立 Mask 來標記 Outlier 位置
    mask = torch.zeros_like(tensor, dtype=torch.bool)

    if dim is not None:
        _, topk_indices = torch.topk(torch.abs(tensor), k, dim=dim)
        mask.scatter_(dim, topk_indices, True)
    else:
        flat_abs = torch.abs(tensor.flatten())
        _, topk_indices = torch.topk(flat_abs, k)
        flat_mask = mask.flatten()
        flat_mask[topk_indices] = True
        mask = flat_mask.view_as(tensor)

    # 3. 提取 sparse values 和 flat indices（只 flatten 一次）
    # sparse_values 保持原始 dtype（bf16），sparse_indices 用 int32 節省記憶體
    flat_mask = mask.flatten()
    sparse_indices = flat_mask.nonzero(as_tuple=True)[0].to(torch.int32)
    sparse_values = tensor.flatten()[sparse_indices]

    # 4. 用 masked_fill_ 原地歸零（不需 clone 整個 tensor）
    dense_tensor = tensor.masked_fill(mask, 0)

    return dense_tensor, sparse_values, sparse_indices

def restore_outliers(dequantized_tensor: torch.Tensor,
                     sparse_values: torch.Tensor,
                     sparse_indices: torch.Tensor) -> torch.Tensor:
    if sparse_values.numel() == 0:
        return dequantized_tensor

    # 確保 dtype 一致後，直接在 flat view 上寫入
    # sparse_indices 可能是 int32，indexing 需要 long
    flat_recon = dequantized_tensor.to(sparse_values.dtype).flatten()
    flat_recon[sparse_indices.long()] = sparse_values
    return flat_recon.view_as(dequantized_tensor)
