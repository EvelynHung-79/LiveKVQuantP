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

        # [新增] 用來持有 Llama 的 RoPE 模組
        self.rotary_emb_module = None

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
        # 1. Stats Update
        k_scale = self.stats_manager.update_key_stats(k_tensor)
        v_scale = self.stats_manager.get_value_stats(v_tensor)

        # 2. Store
        if (not self.is_decoding) and (self.chunk_idx < self.config.n_warmup):
            self.kv_manager.store_warmup(k_tensor, v_tensor)
        else:
            k_compressed = self.quantizer.compress(k_tensor, k_scale)
            v_compressed = self.quantizer.compress(v_tensor, v_scale)
            self.kv_manager.store_quantized(k_compressed, v_compressed)

        # 3. Compute Attention
        # [修正] 傳遞 rotary_emb_module 給 Core
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            rotary_emb_module=self.rotary_emb_module
        )
        
        return attn_output