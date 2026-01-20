import torch

def truncate_input_ids(input_ids: torch.Tensor, max_len: int, tokenizer) -> torch.Tensor:
    """
    執行 Head+Tail Truncation。
    為了效能，盡量操作 Tensor，避免來回 Decode。
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    seq_len = input_ids.shape[1]
    if seq_len <= max_len:
        return input_ids
        
    half = int(max_len / 2)
    
    # 直接使用 Tensor Concatenation
    front_ids = input_ids[:, :half]
    back_ids = input_ids[:, -half:]
    
    # 拼接
    truncated_ids = torch.cat([front_ids, back_ids], dim=1)
    
    return truncated_ids