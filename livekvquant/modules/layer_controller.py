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

    def pack_kv_chunks(self):
        """Prefill 結束後統一 pack 所有 KV chunks，委派給 KVCacheManager。"""
        self.kv_manager.pack_all_chunks()

    def reset_cache(self):
        self.kv_manager.clear()
        self.stats_manager.reset()
        self.chunk_idx = 0
        self.is_decoding = False

    @staticmethod
    def _make_quantized_chunk(quant_result, sp_val, sp_idx):
        """將 quantizer.quantize_dense 的回傳值封裝成 KVChunk。"""
        if len(quant_result) == 3:
            # Asymmetric: (quantized_data, scale, zero_point)
            q_data, scale, zero_point = quant_result
            return KVChunk(
                chunk_type="quantized",
                quantized_data=q_data, scale=scale, zero_point=zero_point,
                sparse_values=sp_val, sparse_indices=sp_idx
            )
        else:
            # Symmetric: (quantized_data, scale)
            q_data, scale = quant_result
            return KVChunk(
                chunk_type="quantized",
                quantized_data=q_data, scale=scale,
                sparse_values=sp_val, sparse_indices=sp_idx
            )

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
        # ema_absmax 回傳 tensor，ema_minmax 回傳 (max, min) tuple
        k_stats = self.stats_manager.update_key_stats(k_dense)
        v_stats = self.stats_manager.get_value_stats(v_dense)

        # 量化 & 封裝
        is_k_warmup = self.config.use_warmup and (self.chunk_idx < self.config.n_warmup)

        if is_k_warmup:
            k_chunk = KVChunk(chunk_type="warmup", data=k_rot)
        else:
            k_result = self.quantizer.quantize_dense(k_dense, k_stats)
            k_chunk = self._make_quantized_chunk(k_result, k_sp_val, k_sp_idx)

        v_result = self.quantizer.quantize_dense(v_dense, v_stats)
        v_chunk = self._make_quantized_chunk(v_result, v_sp_val, v_sp_idx)

        # 用壓縮格式替換掉暫存的 raw chunk
        self.kv_manager.finalize_last_chunk(k_chunk, v_chunk)

        return attn_output