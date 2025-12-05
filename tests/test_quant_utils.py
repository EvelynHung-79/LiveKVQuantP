import torch
import pytest
from livekvquant.utils.quant_utils import calculate_symmetric_scale, quantize_symmetric, dequantize_symmetric
from livekvquant.utils.outliers import isolate_outliers, restore_outliers

def test_outlier_isolation_and_restore():
    """
    [核心測試] 驗證異常值分離與還原 (Eq 3-9)。
    測試 Dense 部分歸零，Sparse 部分保留 FP16。
    """
    # 建立一個 Tensor，包含明顯的 Outlier (100.0)
    tensor = torch.tensor([[1.0, 2.0, 100.0], [3.0, 4.0, 5.0]], dtype=torch.float16)
    ratio = 0.2 # 6個元素 * 0.2 = 1.2 -> 取 top 1
    
    # 1. Isolate
    dense, sparse_vals, sparse_idxs = isolate_outliers(tensor, ratio)
    
    # 驗證 Outlier 是否被提取
    assert sparse_vals.numel() == 1
    assert sparse_vals[0] == 100.0
    # 驗證 Dense Tensor 對應位置是否被歸零 (或其他處理)
    # 注意：flat index 2 對應 (0, 2)
    assert dense.flatten()[2] == 0 
    
    # 2. Restore (假設沒有量化誤差，直接還原)
    reconstructed = restore_outliers(dense, sparse_vals, sparse_idxs)
    
    # 驗證是否與原圖一致 (FP16 精度下)
    assert torch.allclose(tensor, reconstructed)

def test_quantization_flow():
    """
    測試 INT4 量化與反量化流程 (Eq 3-5 ~ Eq 3-8)。
    """
    bits = 4
    # 範圍 [-8, 7]
    # 設 Scale 讓 7.0 對應 int 7
    # Absmax = 7.0 -> Scale = 7.0 / 7 = 1.0
    tensor = torch.tensor([3.5, -7.0, 0.0, 7.0])
    absmax = torch.tensor([7.0])
    
    # 1. Calculate Scale
    scale = calculate_symmetric_scale(absmax, bits) # expect 1.0
    
    # 2. Quantize
    q_tensor = quantize_symmetric(tensor, scale, bits)
    
    # 3. Check INT values
    # 3.5 -> 3.5 -> round 4 (or 3 depending on implementation)
    # -7.0 -> -7
    assert q_tensor.dtype == torch.int8
    assert q_tensor[1] == -7
    
    # 4. Dequantize
    recon = dequantize_symmetric(q_tensor, scale)
    
    # 檢查誤差是否在可接受範圍 (Scale=1, 誤差應 < 0.5)
    assert torch.allclose(tensor, recon, atol=0.6)