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
    Transformer Layer Controller (TLC)
    
    負責協調 LiveKVQuant-P 的核心流程：
    1. 攔截 KV Cache 的資料流。
    2. 決定是否進行 Bypass (Decoding 階段)。
    3. 執行異常值分離 (Outlier Isolation) 與保護 Sink Token。
    4. 更新統計數據 (Statistics Update)。
    5. 執行量化 (Quantization) 並存入 KV Cache Manager。
    6. 呼叫 Attention Core 計算最終的 Attention Output。
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.chunk_idx = 0
        self.is_decoding = False 

        # 用來持有 Llama 的 RoPE 模組 (由 Model Wrapper 注入)
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

    def forward(self, q_tensor, k_tensor, v_tensor, position_ids=None):
        """
        Forward Pass:
        Input: Q, K, V (FP16, Pre-RoPE)
        Output: Attention Output (FP16)
        """
        
        # === 0. Bypass 機制 (Decoding 階段或前幾層) ===
        # Decoding 階段：因為 seq_len=1，不適合做分塊統計，直接存 FP16。
        # Start Layer Bypass：前幾層通常含有重要語義，保留全精度有助於穩定性。
        should_bypass = (self.layer_idx < self.config.quant_start_layer)
        
        if self.is_decoding or should_bypass:
            # 建立 Warmup 格式的數據包 (標記為 warmup 即代表 FP16)
            k_data = {"type": "warmup", "data": k_tensor}
            v_data = {"type": "warmup", "data": v_tensor}
            
            # 存入 Cache
            self.kv_manager.store_chunk(k_data, v_data)
            
            # 計算 Attention
            return self.attn_core.compute_attention(
                q_tensor, 
                self.kv_manager,
                rotary_emb_module=self.rotary_emb_module
            )

        # === Prefill 階段：即時量化流程 ===

        # 設定 Sink Token 長度 (通常前 4 個 token 為 attention sink)
        SINK_LENGTH = 4
        
        # --- 1. 準備數據與 Masking (關鍵修正) ---
        # 為了避免 Sink Token 被 isolate() 重複抓取，我們先建立副本並將 Sink 區域歸零。
        # 這樣 isolate 就只會處理 "非 Sink" 的部分。
        
        k_to_isolate = k_tensor.clone()
        v_to_isolate = v_tensor.clone()
        
        if self.chunk_idx == 0:
            # 將前 SINK_LENGTH 個位置設為 0
            k_to_isolate[:, :, :SINK_LENGTH, :] = 0
            v_to_isolate[:, :, :SINK_LENGTH, :] = 0

        # --- 2. Outlier Isolation (分離異常值) ---
        # Key: 沿 Seq 軸 (dim=-2) 找 Channel Outliers
        k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(k_to_isolate, outlier_dim=-2)
        
        # Value: 沿 Head Dim 軸 (dim=-1) 找 Token Outliers
        v_dense, v_sp_val, v_sp_idx = self.quantizer.isolate(v_to_isolate, outlier_dim=-1)

        # --- 3. 手動保護 Sink Token (將其加入 Sparse 列表) ---
        # 因為我們在第 1 步把它們 Mask 掉了，現在要手動把它們 "完整地" 加回 Sparse 列表。
        # 這樣 Sink Token 就會以 FP16 形式被保存。
        
        if self.chunk_idx == 0:
            # Helper function: 提取 Sink 並轉為 Sparse 格式
            def extract_and_append_sink(tensor, sp_val, sp_idx):
                # 提取 Sink 數值
                sink_data = tensor[:, :, :SINK_LENGTH, :]
                
                # 建立對應的 Mask 以取得正確的 indices
                sink_mask = torch.zeros_like(tensor, dtype=torch.bool)
                sink_mask[:, :, :SINK_LENGTH, :] = True
                
                # 取得 flatten indices
                sink_indices = torch.nonzero(sink_mask.flatten(), as_tuple=False).squeeze()
                sink_values = sink_data.flatten()
                
                # 合併到原本的 sparse list
                new_sp_val = torch.cat([sp_val, sink_values])
                new_sp_idx = torch.cat([sp_idx, sink_indices])
                return new_sp_val, new_sp_idx

            # (A) 處理 Value (V) Sink - 絕對必要
            v_sp_val, v_sp_idx = extract_and_append_sink(v_tensor, v_sp_val, v_sp_idx)

            # (B) 處理 Key (K) Sink
            # 判斷 K 是否處於 Warmup 狀態
            is_k_warmup = (self.chunk_idx < self.config.n_warmup)
            
            # 如果 K *不是* Warmup (即將被量化)，我們必須保護它的 Sinks
            if not is_k_warmup:
                k_sp_val, k_sp_idx = extract_and_append_sink(k_tensor, k_sp_val, k_sp_idx)

        # --- 4. Statistics & Scale Update (更新統計值) ---
        # 使用 k_dense/v_dense 更新 EMA。
        # 注意：因為 Sink 已經被 Mask 成 0，所以不會影響這裡的統計值 (這很好！)
        k_absmax = self.stats_manager.update_key_stats(k_dense)
        v_absmax = self.stats_manager.get_value_stats(v_dense)

        # --- 5. Quantization Strategy (量化) ---
        
        # 5.1 處理 Key (K)
        # 重新確認 Warmup 狀態 (K 需要 Warmup 以穩定 EMA)
        is_k_warmup = (self.chunk_idx < self.config.n_warmup)

        if is_k_warmup:
            # K 在 Warmup 期間：直接存 FP16 (包含 Sinks)
            k_data = {"type": "warmup", "data": k_tensor}
        else:
            # K 量化：Dense 部分轉 INT4，Sparse (含 Sinks) 留 FP16
            k_quant, k_scale = self.quantizer.quantize_dense(k_dense, k_absmax)
            
            k_data = {
                "type": "quantized",
                "quantized_data": k_quant,
                "scale": k_scale, # 這裡存的是真正的 Scale (absmax/7)，不是 AbsMax
                "sparse_values": k_sp_val,
                "sparse_indices": k_sp_idx
            }

        # 5.2 處理 Value (V)
        # V 不需要 Warmup，總是進行量化
        v_quant, v_scale = self.quantizer.quantize_dense(v_dense, v_absmax)
        
        v_data = {
            "type": "quantized",
            "quantized_data": v_quant,
            "scale": v_scale, # 存入正確 Scale
            "sparse_values": v_sp_val,
            "sparse_indices": v_sp_idx
        }

        # 6. Store (存入 KV Cache)
        self.kv_manager.store_chunk(k_data, v_data)

        # 7. Attention (計算注意力)
        # 委託 AttentionCore 還原資料並計算
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            rotary_emb_module=self.rotary_emb_module,
            position_ids=position_ids
        )
        
        return attn_output