import torch
import torch.nn as nn
import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # [New] Import
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
        
        # === [Critical] Post-RoPE Transformation Step ===
        # 無論是 Warmup 還是正常層，我們先取得 "Rotated Key" (k_rot)
        # 這是為了確保 Attention 計算與量化存儲的對象一致 (都是 Post-RoPE)
        k_rot = k_tensor
        if self.rotary_emb_module is not None and position_ids is not None:
            # 取得 cos, sin (使用 v_tensor 的 device 與 dtype 資訊)
            cos, sin = self.rotary_emb_module(v_tensor, position_ids)
            # 這裡只取 k_rot。Q 的旋轉留給 AttentionCore 做 (或者也可以這裡做，但為了最小改動，留給 Core)
            # apply_rotary_pos_emb 回傳 (q_embed, k_embed)，我們只拿 k_embed
            _, k_rot = apply_rotary_pos_emb(q_tensor, k_tensor, cos, sin)

        # === 0. Bypass 機制 (Warmup Layers) ===
        if self.layer_idx < self.config.quant_start_layer:
            # 傳入 k_rot，並告訴 Core "K 已經轉過了"
            attn_output = self.attn_core.compute_attention(
                q_tensor, self.kv_manager, 
                current_k=k_rot,  # [Change] 使用 k_rot
                current_v=v_tensor,
                rotary_emb_module=self.rotary_emb_module, 
                position_ids=position_ids,
                k_is_rotated=True # [New] Flag
            )
            
            # [Change] 儲存 Post-RoPE 的 Key
            k_chunk = KVChunk(chunk_type="warmup", data=k_rot)
            v_chunk = KVChunk(chunk_type="warmup", data=v_tensor)
            
            self.kv_manager.store_chunk(k_chunk, v_chunk)
            return attn_output

        # === 1. 正常層：先計算 Attention ===
        # 同樣傳入 k_rot
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            current_k=k_rot,  # [Change] 使用 k_rot
            current_v=v_tensor,
            rotary_emb_module=self.rotary_emb_module,
            position_ids=position_ids,
            k_is_rotated=True # [New] Flag
        )

        # === 2. 量化與儲存流程 (針對 k_rot) ===
        SINK_LENGTH = 4
        is_prefill_start = (self.chunk_idx == 0 and not self.is_decoding)
        current_sink_len = SINK_LENGTH if is_prefill_start else 0

        # [Change] 量化對象改為 k_rot
        k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(
            k_rot, 
            outlier_dim=-2, 
            sink_length=current_sink_len
        )
        
        # Value 維持不變 (Value 不需要 RoPE)
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
            # [Change] 即使是 chunk warmup，也要存 k_rot
            k_chunk = KVChunk(chunk_type="warmup", data=k_rot)
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