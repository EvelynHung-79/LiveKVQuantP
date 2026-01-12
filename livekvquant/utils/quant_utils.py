import torch

def calculate_symmetric_scale(absmax: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    計算對稱量化的 Scale。
    Format: scale = absmax / (2^(bits-1) - 1)
    """
    # INT4: range [-7, 7] -> max_val = 7
    max_val = 2 ** (bits - 1) - 1
    
    # [FIX] 加入 1e-6 epsilon 防止 absmax 為 0 時導致 Scale=0 -> 除以零錯誤
    scale = (absmax + 1e-6) / max_val
    
    return scale

def quantize_symmetric(tensor: torch.Tensor, scale: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    對稱量化： x_q = round(x / scale)
    """
    max_val = 2 ** (bits - 1) - 1
    min_val = -max_val
    
    # [FIX] Clamp 確保數值不會越界
    quantized = torch.round(tensor / scale).clamp(min_val, max_val)
    
    return quantized.to(torch.int8)

def dequantize_symmetric(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    反量化： x = x_q * scale
    """
    return quantized.float() * scale