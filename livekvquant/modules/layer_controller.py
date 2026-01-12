import torch
import torch.nn as nn
import logging
from .statistics_manager import StatisticsManager
from .quantizer import RealTimeQuantizer
from .kv_cache_manager import KVCacheManager
from .attention_core import AttentionCore

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
        should_bypass = (self.layer_idx < self.config.quant_start_layer)
        
        if should_bypass:
            k_data = {"type": "warmup", "data": k_tensor}
            v_data = {"type": "warmup", "data": v_tensor}
            self.kv_manager.store_chunk(k_data, v_data)
            
            return self.attn_core.compute_attention(
                q_tensor, 
                self.kv_manager, 
                rotary_emb_module=self.rotary_emb_module, 
                position_ids=position_ids
            )

        # === 1. [優先] 計算 Attention (使用原始 BF16) ===
        # 這是為了確保精度：使用無損的當前 token 進行計算
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            current_k=k_tensor,
            current_v=v_tensor,
            rotary_emb_module=self.rotary_emb_module,
            position_ids=position_ids
        )

        # === 2. 量化流程 (為了存給未來使用) ===
        SINK_LENGTH = 4
        
        k_to_isolate = k_tensor.clone()
        v_to_isolate = v_tensor.clone()
        
        is_decoding_step = (k_tensor.size(-2) == 1)

        # A. Mask Sink (Prefill Only)
        if self.chunk_idx == 0 and not self.is_decoding:
            k_to_isolate[:, :, :SINK_LENGTH, :] = 0
            v_to_isolate[:, :, :SINK_LENGTH, :] = 0

        # B. Outlier Isolation
        if is_decoding_step:
            # Decoding: 跳過 isolate，避免把唯一 token 當 outlier
            k_dense = k_to_isolate
            v_dense = v_to_isolate
            k_sp_val, k_sp_idx = torch.empty(0, device=k_tensor.device), torch.empty(0, device=k_tensor.device)
            v_sp_val, v_sp_idx = torch.empty(0, device=v_tensor.device), torch.empty(0, device=v_tensor.device)
        else:
            # Prefill: 正常執行
            k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(k_to_isolate, outlier_dim=-2)
            v_dense, v_sp_val, v_sp_idx = self.quantizer.isolate(v_to_isolate, outlier_dim=-1)

        # C. 保護 Sink Token
        if self.chunk_idx == 0 and not self.is_decoding:
            def extract_and_append_sink(tensor, sp_val, sp_idx):
                sink_data = tensor[:, :, :SINK_LENGTH, :]
                sink_mask = torch.zeros_like(tensor, dtype=torch.bool)
                sink_mask[:, :, :SINK_LENGTH, :] = True
                sink_indices = torch.nonzero(sink_mask.flatten(), as_tuple=False).squeeze()
                sink_values = sink_data.flatten()
                return torch.cat([sp_val, sink_values]), torch.cat([sp_idx, sink_indices])

            v_sp_val, v_sp_idx = extract_and_append_sink(v_tensor, v_sp_val, v_sp_idx)
            is_k_warmup = (self.chunk_idx < self.config.n_warmup)
            if not is_k_warmup:
                k_sp_val, k_sp_idx = extract_and_append_sink(k_tensor, k_sp_val, k_sp_idx)

        # D. Statistics Update
        k_absmax = self.stats_manager.update_key_stats(k_dense)
        v_absmax = self.stats_manager.get_value_stats(v_dense)

        # E. Quantization & Packaging
        is_k_warmup = (self.chunk_idx < self.config.n_warmup)

        if is_k_warmup:
            k_data = {"type": "warmup", "data": k_tensor}
        else:
            k_quant, k_scale = self.quantizer.quantize_dense(k_dense, k_absmax)
            k_data = {
                "type": "quantized", 
                "quantized_data": k_quant, 
                "scale": k_scale, 
                "sparse_values": k_sp_val, 
                "sparse_indices": k_sp_idx
            }

        v_quant, v_scale = self.quantizer.quantize_dense(v_dense, v_absmax)
        v_data = {
            "type": "quantized", 
            "quantized_data": v_quant, 
            "scale": v_scale, 
            "sparse_values": v_sp_val, 
            "sparse_indices": v_sp_idx
        }

        # F. Store
        self.kv_manager.store_chunk(k_data, v_data)
        
        return attn_output