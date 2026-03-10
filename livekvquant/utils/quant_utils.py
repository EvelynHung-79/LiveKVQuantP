import torch

def pack_int4(quantized: torch.Tensor) -> torch.Tensor:
    """
    將 int8 tensor（存 4-bit 值，範圍 [-7, 7]）沿最後一維兩兩打包成一個 int8。
    最後一維必須是偶數（head_dim=128 永遠滿足）。
    節省 50% 記憶體。
    """
    assert quantized.shape[-1] % 2 == 0, "Last dim must be even for INT4 packing"
    lo = quantized[..., 0::2]  # even indices
    hi = quantized[..., 1::2]  # odd indices
    # 在 int32 做 bit 操作避免 overflow/sign 問題
    packed = (lo.to(torch.int32) & 0xF) | ((hi.to(torch.int32) & 0xF) << 4)
    return packed.to(torch.int8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    將 pack_int4 產生的 int8 tensor 解包，還原成 int8 tensor（值域 [-7, 7]）。
    最後一維會變成兩倍。
    """
    # & 0xFF 消除 int8→int32 時的 sign extension
    p32 = packed.to(torch.int32) & 0xFF
    lo = ((p32 << 28) >> 28).to(torch.int8)          # sign-extend low nibble
    hi = ((p32 >> 4) << 28 >> 28).to(torch.int8)     # sign-extend high nibble
    result = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2,
        dtype=torch.int8, device=packed.device
    )
    result[..., 0::2] = lo
    result[..., 1::2] = hi
    return result


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
    直接用 scale 的 dtype（通常是 fp16/bf16）計算，避免多餘的 fp32 中間 tensor。
    """
    return quantized.to(scale.dtype) * scale


# ------------------------------------------------------------------ #
#  Asymmetric Quantization (用於 EMA MinMax ablation)
# ------------------------------------------------------------------ #

def calculate_asymmetric_params(ema_max: torch.Tensor, ema_min: torch.Tensor, bits: int = 4):
    """
    計算非對稱量化的 scale 和 zero_point。
    unsigned INT4: 值域 [0, 15]。
      scale = (ema_max - ema_min) / 15
      zero_point = round(-ema_min / scale)，clamp 到 [0, 15]
    """
    qmax = 2 ** bits - 1  # 15
    scale = (ema_max - ema_min + 1e-6) / qmax
    zero_point = torch.round(-ema_min / scale).clamp(0, qmax).to(torch.int8)
    return scale, zero_point


def quantize_asymmetric(tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """
    非對稱量化：x_q = round(x / scale) + zero_point，clamp 到 [0, 2^bits - 1]。
    回傳 uint8 語意的 int8 tensor（值域 [0, 15]）。
    """
    qmax = 2 ** bits - 1
    quantized = torch.round(tensor / scale) + zero_point.to(tensor.dtype)
    return quantized.clamp(0, qmax).to(torch.int8)


def dequantize_asymmetric(quantized: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """
    非對稱反量化：x = (x_q - zero_point) * scale
    """
    return (quantized.to(scale.dtype) - zero_point.to(scale.dtype)) * scale


def pack_uint4(quantized: torch.Tensor) -> torch.Tensor:
    """
    將 unsigned int4 tensor（值域 [0, 15]，存在 int8 裡）沿最後一維兩兩打包。
    與 pack_int4 的差異：不需要 sign extension，直接用 4-bit mask。
    """
    assert quantized.shape[-1] % 2 == 0
    lo = quantized[..., 0::2]
    hi = quantized[..., 1::2]
    packed = (lo.to(torch.int32) & 0xF) | ((hi.to(torch.int32) & 0xF) << 4)
    return packed.to(torch.int8)


def unpack_uint4(packed: torch.Tensor) -> torch.Tensor:
    """
    解包 pack_uint4 產生的 tensor，還原成 int8（值域 [0, 15]）。
    與 unpack_int4 的差異：不做 sign extension，直接用 mask。
    """
    p32 = packed.to(torch.int32) & 0xFF
    lo = (p32 & 0xF).to(torch.int8)
    hi = ((p32 >> 4) & 0xF).to(torch.int8)
    result = torch.empty(
        *packed.shape[:-1], packed.shape[-1] * 2,
        dtype=torch.int8, device=packed.device
    )
    result[..., 0::2] = lo
    result[..., 1::2] = hi
    return result