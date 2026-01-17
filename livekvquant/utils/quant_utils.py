import torch

def calculate_symmetric_scale(absmax: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    計算對稱量化的 Scale。
    Format: scale = absmax / (2^(bits-1) - 1)
    """
    # INT4: range [-7, 7] -> max_val = 7
    max_val = 2 ** (bits - 1) - 1
    
    # [FIX] 建立與 absmax 同型態的 epsilon (BF16 or FP16)
    # 避免 absmax(BF16) + 1e-6(Float) -> Float32
    epsilon = torch.tensor(1e-6, dtype=absmax.dtype, device=absmax.device)
    
    scale = (absmax + epsilon) / max_val
    
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