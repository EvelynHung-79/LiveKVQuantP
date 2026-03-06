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
        # --- RoPE for K ---
        k_rot = k_tensor
        if self.rotary_emb_module is not None and position_ids is not None:
            cos, sin = self.rotary_emb_module(v_tensor, position_ids)
            _, k_rot = apply_rotary_pos_emb(q_tensor, k_tensor, cos, sin)

        # === 0. Bypass 層（前幾層不量化，全 FP16）===
        if self.layer_idx < self.config.quant_start_layer:
            # 先暫存 raw KV，讓 attention 能看到
            if not self.is_decoding:
                self.kv_manager.store_raw(k_rot, v_tensor)

            attn_output = self.attn_core.compute_attention(
                q_tensor, self.kv_manager,
                rotary_emb_module=self.rotary_emb_module,
                position_ids=position_ids,
                k_is_rotated=True
            )

            if self.is_decoding:
                self.kv_manager.store_decode_token(k_rot, v_tensor)
            else:
                # Finalize: bypass 層直接存 warmup（原始精度）
                k_chunk = KVChunk(chunk_type="warmup", data=k_rot)
                v_chunk = KVChunk(chunk_type="warmup", data=v_tensor)
                self.kv_manager.finalize_last_chunk(k_chunk, v_chunk)
            return attn_output

        # === 1. Prefill: 先暫存 raw KV → 用原始精度算 Attention ===
        if not self.is_decoding:
            self.kv_manager.store_raw(k_rot, v_tensor)

        attn_output = self.attn_core.compute_attention(
            q_tensor,
            self.kv_manager,
            rotary_emb_module=self.rotary_emb_module,
            position_ids=position_ids,
            k_is_rotated=True
        )

        # === 2. Decode: 直接存 FP16，不量化 ===
        if self.is_decoding:
            self.kv_manager.store_decode_token(k_rot, v_tensor)
            return attn_output

        # === 3. Attention 算完後，才做量化壓縮（延後壓縮）===
        SINK_LENGTH = 4
        is_prefill_start = (self.chunk_idx == 0)
        current_sink_len = SINK_LENGTH if is_prefill_start else 0

        # Outlier 分離
        k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(
            k_rot, outlier_dim=-2, sink_length=current_sink_len
        )
        v_dense, v_sp_val, v_sp_idx = self.quantizer.isolate(
            v_tensor, outlier_dim=-1, sink_length=current_sink_len
        )

        # EMA 統計更新
        k_absmax = self.stats_manager.update_key_stats(k_dense)
        v_absmax = self.stats_manager.get_value_stats(v_dense)

        # 量化 & 封裝
        is_k_warmup = (self.chunk_idx < self.config.n_warmup)

        if is_k_warmup:
            k_chunk = KVChunk(chunk_type="warmup", data=k_rot)
        else:
            k_quant, k_scale = self.quantizer.quantize_dense(k_dense, k_absmax)
            k_chunk = KVChunk(
                chunk_type="quantized",
                quantized_data=k_quant, scale=k_scale,
                sparse_values=k_sp_val, sparse_indices=k_sp_idx
            )

        v_quant, v_scale = self.quantizer.quantize_dense(v_dense, v_absmax)
        v_chunk = KVChunk(
            chunk_type="quantized",
            quantized_data=v_quant, scale=v_scale,
            sparse_values=v_sp_val, sparse_indices=v_sp_idx
        )

        # 用壓縮格式替換掉暫存的 raw chunk
        self.kv_manager.finalize_last_chunk(k_chunk, v_chunk)

        return attn_output