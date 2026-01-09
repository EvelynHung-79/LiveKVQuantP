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

    def forward(self, q_tensor, k_tensor, v_tensor):
        """
        執行 LiveKVQuant-P 的核心流程：
        1. Decoding Bypass (生成階段直接存 FP16)
        2. Outlier Isolation (Prefill 階段分離異常值)
        3. Statistics Update (更新 EMA 統計)
        4. Quantization (量化)
        5. Storage (存入 KV Cache)
        6. Attention (計算注意力機制)
        """
        
        # === 0. Decoding Bypass (生成階段保護機制) ===
        # 如果是 Decoding 階段 (seq_len=1)，且我們的方法只針對 Prefill 進行壓縮。
        # 為了避免 1 個 token 被誤判為 100% Outlier 或造成 EMA 衰減，
        # 這裡直接將其視為 Warmup (FP16) 儲存，跳過所有統計與量化步驟。
        
        # === [新增邏輯] Layer Bypass (前幾層保持全精度) ===
        # 判斷標準：如果是 Decoding 階段 OR 目前層數小於設定的起始層
        # 這些情況都直接存 FP16 ("warmup" 格式)
        should_bypass = (self.layer_idx < self.config.quant_start_layer)
        
        if self.is_decoding or should_bypass:
            k_data = {"type": "warmup", "data": k_tensor}
            v_data = {"type": "warmup", "data": v_tensor}
            
            # 直接儲存
            self.kv_manager.store_chunk(k_data, v_data)
            
            # 計算 Attention 並回傳
            return self.attn_core.compute_attention(
                q_tensor, 
                self.kv_manager,
                rotary_emb_module=self.rotary_emb_module
            )

        # === 以下為 Prefill 階段的量化流程 ===

        # --- 1. Outlier Isolation (分離異常值) ---
        # [關鍵修正] 針對不同 Tensor 的特性，沿著正確的軸向抓 Outlier
        
        # Key (K): Scale 是 Per-Channel 的，所以要抓出撐大 Channel 的兇手 (沿 Seq 軸, dim=-2)
        k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(k_tensor, outlier_dim=-2)
        
        # Value (V): Scale 是 Per-Token 的，所以要抓出撐大 Token 的兇手 (沿 Head Dim 軸, dim=-1)
        v_dense, v_sp_val, v_sp_idx = self.quantizer.isolate(v_tensor, outlier_dim=-1)

        # Attention Sink Tokens 處理
        # 在第一個 Chunk 時，將前幾個 Sink Tokens 也視為 Outlier
        SINK_LENGTH = 4
        if self.chunk_idx == 0:
            # 1. 找出 Sink Tokens 的區域 (前 SINK_LENGTH 個 tokens)
            # v_tensor shape: [Batch, Heads, Seq, Head_Dim]
            sink_data = v_tensor[:, :, :SINK_LENGTH, :]
            
            # 2. 取得它們的 Flat Indices (全域索引)
            # 建立一個與 v_tensor 形狀相同的 mask
            sink_mask = torch.zeros_like(v_tensor, dtype=torch.bool)
            sink_mask[:, :, :SINK_LENGTH, :] = True
            
            # 轉為 flat indices
            sink_indices = torch.nonzero(sink_mask.flatten(), as_tuple=False).squeeze()
            sink_values = v_tensor.flatten()[sink_indices]
            
            # 3. 從 v_dense 中移除這些值 (設為 0，因為已經搬去 Sparse 了)
            # 注意：原本的 isolate 可能已經抓過這些值了，但重複歸零沒關係
            # 為了簡單，我們直接在 v_dense 上操作
            v_dense_flat = v_dense.flatten()
            v_dense_flat[sink_indices] = 0
            v_dense = v_dense_flat.view_as(v_dense)

            # 4. 合併「數值 Outlier」與「Sink Outlier」
            # 我們將 Sink 的數據拼接到原本的 sparse list 後面
            v_sp_val = torch.cat([v_sp_val, sink_values])
            v_sp_idx = torch.cat([v_sp_idx, sink_indices])

        # --- 2. Statistics & Scale Update (使用 Dense Tensor) ---
        # 更新 EMA (Key) 或取得瞬時 Scale (Value)
        k_scale = self.stats_manager.update_key_stats(k_dense)
        v_scale = self.stats_manager.get_value_stats(v_dense)

        # --- 3. Quantization Strategy ---
        
        # 3.1 處理 Key (K)
        # 判斷是否在 Warm-up 階段 (只看 chunk_idx，因為 decoding 已經被 bypass 了)
        is_k_warmup = (self.chunk_idx < self.config.n_warmup)

        if is_k_warmup:
            # K 在 Warm-up 期間：存 FP16 (為了讓 EMA 穩定)
            k_data = {"type": "warmup", "data": k_tensor}
        else:
            # K 在 Warm-up 結束後：進行 INT4 壓縮
            # 使用穩定的 EMA Scale 量化 k_dense
            k_quant = self.quantizer.quantize_dense(k_dense, k_scale)
            k_data = {
                "type": "quantized",
                "quantized_data": k_quant,
                "scale": k_scale,
                "sparse_values": k_sp_val,
                "sparse_indices": k_sp_idx
            }

        # 3.2 處理 Value (V)
        # V 不需要 Warm-up，直接壓縮
        v_quant = self.quantizer.quantize_dense(v_dense, v_scale)
        v_data = {
            "type": "quantized",
            "quantized_data": v_quant,
            "scale": v_scale,
            "sparse_values": v_sp_val,
            "sparse_indices": v_sp_idx
        }

        # 4. Store (存入 KV Cache)
        self.kv_manager.store_chunk(k_data, v_data)

        # 5. Attention (計算注意力)
        # AttentionCore 負責還原 (Dequantize + Restore Outliers) 並計算
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            rotary_emb_module=self.rotary_emb_module
        )
        
        return attn_output