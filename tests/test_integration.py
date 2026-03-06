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
        outlier_ratio=0.01,
        quant_start_layer=0
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

    # 檢查 Cache 類型（現在回傳 KVChunk 物件）
    k_chunks, v_chunks = controller.kv_manager.get_all_chunks()
    assert len(k_chunks) == 1
    assert k_chunks[0].chunk_type == "warmup"

    # --- Chunk 1 (Warm-up Phase) ---
    controller.set_chunk_idx(1)
    controller(q, k, v)
    k_chunks, _ = controller.kv_manager.get_all_chunks()
    assert k_chunks[1].chunk_type == "warmup"

    # --- Chunk 2 (Quantization Phase, since N_warmup=2) ---
    controller.set_chunk_idx(2)
    controller(q, k, v)

    k_chunks, v_chunks = controller.kv_manager.get_all_chunks()
    assert len(k_chunks) == 3
    assert k_chunks[2].chunk_type == "quantized"

    # 檢查 Quantized Data 是否存在
    assert k_chunks[2].quantized_data is not None
    assert k_chunks[2].sparse_values is not None
    # sparse_indices 應為 int32
    assert k_chunks[2].sparse_indices.dtype == torch.int32

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

    # 輸出形狀應為 [batch, heads, seq, dim]
    assert output.shape == (batch, heads, seq, dim)
    assert output.dtype == torch.float32

def test_memory_compression(mock_config):
    """
    [新測試] 驗證量化後的 chunks 確實比 raw FP16 佔用更少記憶體。
    """
    controller = TransformerLayerController(mock_config, layer_idx=0)

    batch, heads, seq, dim = 1, 4, 128, 64
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    # 跑 3 個 chunks（2 warmup + 1 quantized）
    for i in range(3):
        controller.set_chunk_idx(i)
        controller(q, k, v)

    k_chunks, v_chunks = controller.kv_manager.get_all_chunks()

    # warmup chunk 佔用 = raw fp32 size
    warmup_bytes = k_chunks[0].memory_bytes
    # quantized chunk: int8 + scale + sparse → 應該明顯更小
    quant_bytes = k_chunks[2].memory_bytes

    assert quant_bytes < warmup_bytes, (
        f"Quantized chunk ({quant_bytes} B) should be smaller than warmup ({warmup_bytes} B)"
    )

def test_lazy_reconstruct(mock_config):
    """
    驗證 KVCacheManager 的 lazy reconstruct 機制。
    多次 get_full_kv 不改 chunks 時，應該使用快取。
    """
    controller = TransformerLayerController(mock_config, layer_idx=0)

    batch, heads, seq, dim = 1, 4, 64, 32
    q = torch.randn(batch, heads, seq, dim)
    k = torch.randn(batch, heads, seq, dim)
    v = torch.randn(batch, heads, seq, dim)

    controller.set_chunk_idx(0)
    controller(q, k, v)

    kv_mgr = controller.kv_manager

    # 第一次呼叫：reconstruct
    k1, v1 = kv_mgr.get_full_kv(torch.float32)
    assert kv_mgr._recon_valid

    # 第二次呼叫：應該用快取（同一物件）
    k2, v2 = kv_mgr.get_full_kv(torch.float32)
    assert k1.data_ptr() == k2.data_ptr()