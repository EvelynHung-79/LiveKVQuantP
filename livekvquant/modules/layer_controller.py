import torch
import torch.nn as nn
import logging
from .statistics_manager import StatisticsManager
from .quantizer import RealTimeQuantizer
from .kv_cache_manager import KVCacheManager
from .attention_core import AttentionCore

logger = logging.getLogger(__name__)

class TransformerLayerController(nn.Module):
    """
    中央協調器：管理 Warm-up/Quantization 狀態切換與數據流。
    對應論文 3.2 Transformer Layer Controller Module.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.chunk_idx = 0
        self.is_decoding = False # 標記是否進入 Decoding 階段

        # 初始化四大核心模組
        self.stats_manager = StatisticsManager(config)
        self.quantizer = RealTimeQuantizer(config)
        self.kv_manager = KVCacheManager(config)
        self.attn_core = AttentionCore(config)

    def set_chunk_idx(self, idx: int):
        """由 Model Wrapper 呼叫，設定當前 Chunk 索引"""
        self.chunk_idx = idx
        self.is_decoding = False

    def set_decoding_mode(self):
        """進入 Decoding 階段 (Token-by-token generation)"""
        self.is_decoding = True

    def reset_cache(self):
        """重置 Cache 與 統計狀態 (用於新 Prompt)"""
        self.kv_manager.clear()
        self.stats_manager.reset()
        self.chunk_idx = 0
        self.is_decoding = False

    def forward(self, q_tensor, k_tensor, v_tensor):
        """
        處理單層的 KV Cache 壓縮與 Attention 計算。
        流程：Stats -> Quantize/Warmup -> Store -> Compute Attention
        """
        # 1. 判斷模式 (Eq 3-1)
        # 如果是 Decoding 階段，視為 Quantization Mode (沿用最後的 EMA)
        # 如果是 Prefill 階段，檢查 chunk_idx < N_warmup
        is_warmup = (not self.is_decoding) and (self.chunk_idx < self.config.n_warmup)
        
        # 2. 統計數據更新 (Statistics Manager) 
        # 即使是 Warm-up，EMA 也要更新以進行 Scale Stabilization
        # Key: Pre-RoPE (假設傳入的 k_tensor 是 Pre-RoPE，或由外部處理)
        # Value: Post-RoPE
        k_scale = self.stats_manager.update_key_stats(k_tensor)
        v_scale = self.stats_manager.get_value_stats(v_tensor)

        # 3. 壓縮與儲存 (Quantizer & KV Manager)
        if is_warmup:
            # Warm-up Phase: 存 FP16 
            self.kv_manager.store_warmup(k_tensor, v_tensor)
        else:
            # Quantization Phase: 執行 Outlier Isolation + INT4 量化 
            k_compressed = self.quantizer.compress(k_tensor, k_scale)
            v_compressed = self.quantizer.compress(v_tensor, v_scale)
            self.kv_manager.store_quantized(k_compressed, v_compressed)

        # 4. Attention 計算 (Attention Core) 
        # 重建完整 FP16 KV (包含歷史 Cache) 並計算 Output
        attn_output = self.attn_core.compute_attention(q_tensor, self.kv_manager)
        
        return attn_output