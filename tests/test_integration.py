import torch
import pytest
from types import SimpleNamespace
from livekvquant.modules.layer_controller import TransformerLayerController

# 建立一個 Mock Config
@pytest.fixture
def mock_config():
    return SimpleNamespace(
        chunk_size=128,
        n_warmup=2,
        bits=4,
        ema_alpha=0.1,
        clip_factor_n=1.5,
        outlier_ratio=0.01
    )

def test_phase_transition(mock_config):
    """
    [核心測試] 驗證 LayerController 是否正確在 Warm-up 與 Quantization 模式間切換。
    """
    controller = TransformerLayerController(mock_config, layer_idx=0)
    
    # 模擬 Tensor 輸入
    batch, heads, seq, dim = 1, 4, 64, 32
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    
    # --- Chunk 0 (Warm-up Phase) ---
    controller.set_chunk_idx(0)
    controller(q, k, v)
    
    # 檢查 Cache 類型
    k_chunks, v_chunks = controller.kv_manager.get_all_chunks()
    assert len(k_chunks) == 1
    assert k_chunks[0]["type"] == "warmup" # [cite: 713]
    
    # --- Chunk 1 (Warm-up Phase) ---
    controller.set_chunk_idx(1)
    controller(q, k, v)
    k_chunks, _ = controller.kv_manager.get_all_chunks()
    assert k_chunks[1]["type"] == "warmup"
    
    # --- Chunk 2 (Quantization Phase, since N_warmup=2) ---
    controller.set_chunk_idx(2)
    controller(q, k, v)
    
    k_chunks, v_chunks = controller.kv_manager.get_all_chunks()
    assert len(k_chunks) == 3
    assert k_chunks[2]["type"] == "quantized" # [cite: 843]
    
    # 檢查 Quantized Data 是否存在
    assert "quantized_data" in k_chunks[2]
    assert "sparse_values" in k_chunks[2]

def test_attention_output_shape(mock_config):
    """
    驗證完整的 Forward Pass 是否能產生正確維度的 Attention Output。
    """
    controller = TransformerLayerController(mock_config, layer_idx=0)
    
    batch, heads, seq, dim = 1, 8, 512, 64
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)
    
    # 執行一次 Forward
    output = controller(q, k, v)
    
    # 輸出形狀應為 [batch, heads, seq, dim] (假設沒有改變 dim)
    # 注意：Attention 是 Q * K^T * V，所以 output sequence length 跟 Q 一樣
    assert output.shape == (batch, heads, seq, dim)
    assert output.dtype == torch.float32 # 或者是 float16，取決於輸入