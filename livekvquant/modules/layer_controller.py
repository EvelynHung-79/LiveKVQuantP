import torch
import torch.nn as nn
import logging
from .statistics_manager import StatisticsManager
from .quantizer import RealTimeQuantizer
from .kv_cache_manager import KVCacheManager
from .attention_core import AttentionCore
from .chunk import KVChunk

logger = logging.getLogger(__name__)

class TransformerLayerController(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.chunk_idx = 0
        self.is_decoding = False 

        self.rotary_emb_module = None
        self.stats_manager = StatisticsManager(config)
        self.quantizer = RealTimeQuantizer(config)
        self.kv_manager = KVCacheManager(config)
        self.attn_core = AttentionCore(config)

    def set_chunk_idx(self, idx: int):
        self.chunk_idx = idx
        self.is_decoding = False

    def set_decoding_mode(self):
        self.is_decoding = True

    def reset_cache(self):
        self.kv_manager.clear()
        self.stats_manager.reset()
        self.chunk_idx = 0
        self.is_decoding = False

    def forward(self, q_tensor, k_tensor, v_tensor, position_ids=None):
        
        # === 0. Bypass 機制 (Warmup Layers) ===
        if self.layer_idx < self.config.quant_start_layer:
            attn_output = self.attn_core.compute_attention(
                q_tensor, self.kv_manager, 
                current_k=k_tensor,
                current_v=v_tensor,
                rotary_emb_module=self.rotary_emb_module, 
                position_ids=position_ids
            )
            
            # 使用 KVChunk 包裝 Warmup Data
            k_chunk = KVChunk(chunk_type="warmup", data=k_tensor)
            v_chunk = KVChunk(chunk_type="warmup", data=v_tensor)
            
            self.kv_manager.store_chunk(k_chunk, v_chunk)
            return attn_output

        # === 1. 正常層：先計算 Attention (PyTorch Path) ===
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            current_k=k_tensor,
            current_v=v_tensor,
            rotary_emb_module=self.rotary_emb_module,
            position_ids=position_ids
        )

        # === 2. 量化與儲存流程 ===
        SINK_LENGTH = 4
        # 判斷是否需要保護 Sink (僅在 Prefill 的第一個 Chunk)
        is_prefill_start = (self.chunk_idx == 0 and not self.is_decoding)
        current_sink_len = SINK_LENGTH if is_prefill_start else 0

        # [重構後] 直接呼叫 Quantizer，不需要手動 Clone 或 Mask
        # Quantizer 會處理 Copy, Mask Sink, Isolate Outlier, Append Sink
        
        # 處理 Key (Outlier dim = -2)
        k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(
            k_tensor, 
            outlier_dim=-2, 
            sink_length=current_sink_len
        )
        
        # 處理 Value (Outlier dim = -1)
        v_dense, v_sp_val, v_sp_idx = self.quantizer.isolate(
            v_tensor, 
            outlier_dim=-1, 
            sink_length=current_sink_len
        )

        # D. Statistics Update 
        k_absmax = self.stats_manager.update_key_stats(k_dense)
        v_absmax = self.stats_manager.get_value_stats(v_dense)

        # E. Quantization & Packaging 
        is_k_warmup = (self.chunk_idx < self.config.n_warmup)

        if is_k_warmup:
            k_chunk = KVChunk(chunk_type="warmup", data=k_tensor)
        else:
            k_quant, k_scale = self.quantizer.quantize_dense(k_dense, k_absmax)
            k_chunk = KVChunk(
                chunk_type="quantized",
                quantized_data=k_quant,
                scale=k_scale,
                sparse_values=k_sp_val,
                sparse_indices=k_sp_idx
            )

        v_quant, v_scale = self.quantizer.quantize_dense(v_dense, v_absmax)
        v_chunk = KVChunk(
            chunk_type="quantized",
            quantized_data=v_quant,
            scale=v_scale,
            sparse_values=v_sp_val,
            sparse_indices=v_sp_idx
        )

        self.kv_manager.store_chunk(k_chunk, v_chunk)
        
        return attn_output