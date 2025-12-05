import torch

def calculate_symmetric_scale(absmax: torch.Tensor, bits: int) -> torch.Tensor:
    """
    計算對稱量化的 Scale Factor (s)。
    對應論文 Eq 3-5: s = mu_t / (2^(b-1) - 1)
    """
    # 對稱量化的範圍是 [-2^(b-1), 2^(b-1) - 1]
    # 例如 INT4: [-8, 7], max_int = 7
    max_int = 2 ** (bits - 1) - 1
    
    # 避免除以零的保護機制 (雖然 Absmax 通常 > 0)
    scale = absmax / max_int
    return torch.clamp(scale, min=1e-8)

def quantize_symmetric(tensor: torch.Tensor, scale: torch.Tensor, bits: int) -> torch.Tensor:
    """
    執行對稱量化 (Symmetric Quantization)。
    對應論文 Eq 3-6: X_quant = Clamp(Round(X / s), min, max)
    """
    max_int = 2 ** (bits - 1) - 1
    min_int = - (2 ** (bits - 1))
    
    # 1. Scaling & Rounding
    # 使用 round() 將浮點數轉為最近的整數
    scaled = tensor / scale
    rounded = torch.round(scaled)
    
    # 2. Clamping (Eq 3-7)
    # 限制數值在 INT4 範圍內
    quantized = torch.clamp(rounded, min=min_int, max=max_int)
    
    return quantized.to(torch.int8)

def dequantize_symmetric(quantized_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    執行反量化 (Dequantization)。
    對應論文 Eq 3-8: X_dequant = X_quant * s
    """
    # 必須先轉回 float 才能跟 scale 相乘
    return quantized_tensor.float() * scale