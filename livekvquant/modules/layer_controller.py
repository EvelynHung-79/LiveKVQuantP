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

        # 用來持有 Llama 的 RoPE 模組
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
        1. Outlier Isolation (分離異常值)
        2. Statistics Update (更新 EMA 統計)
        3. Quantization (使用 Dense Scale 進行量化)
        4. Storage (存入 KV Cache)
        5. Attention (計算注意力機制)
        """
        
        # 判斷是否為 Baseline 模式 (全 FP16 不壓縮)
        is_baseline = getattr(self.config, 'baseline', False)
        
        k_data = None
        v_data = None

        if is_baseline:
            # Baseline: 純 FP16，但仍需更新 stats 保持狀態一致
            self.stats_manager.update_key_stats(k_tensor) 
            # 這裡使用 "warmup" 標記來代表未壓縮的 FP16 數據
            k_data = {"type": "warmup", "data": k_tensor}
            v_data = {"type": "warmup", "data": v_tensor}
        else:
            # 1. Outlier Isolation (區分 Axis!)
            # Key (K): [Batch, Head, Seq, Dim]。Scale 是 Per-Channel (沿著 Seq 找最大)。
            # 所以 Outlier 也要沿著 Seq (dim=-2) 抓，把撐大 Channel 的兇手抓出來。
            k_dense, k_sp_val, k_sp_idx = self.quantizer.isolate(k_tensor, outlier_dim=-2)

            # Value (V): [Batch, Head, Seq, Dim]。Scale 是 Per-Token (沿著 Dim 找最大)。
            # 所以 Outlier 也要沿著 Dim (dim=-1) 抓，把撐大 Token 的兇手抓出來。
            v_dense, v_sp_val, v_sp_idx = self.quantizer.isolate(v_tensor, outlier_dim=-1)

            # 2. Statistics & Scale Update (保持不變)
            k_scale = self.stats_manager.update_key_stats(k_dense) # 內部用 dim=-2
            v_scale = self.stats_manager.get_value_stats(v_dense)  # 內部用 dim=-1

            # --- 3. Quantization Strategy ---
            # 判斷 Key 是否還在 Warm-up 階段
            is_k_warmup = (not self.is_decoding) and (self.chunk_idx < self.config.n_warmup)

            # 3.1 處理 Key (K)
            if is_k_warmup:
                # K 在 Warm-up 期間：存 FP16 (為了讓 EMA 有足夠數據穩定下來)
                k_data = {"type": "warmup", "data": k_tensor}
            else:
                # K 在 Warm-up 結束後：進行壓縮 (使用穩定的 EMA Scale 量化 k_dense)
                # 注意：這裡呼叫的是 quantize_dense，而不是舊的 compress
                k_quant = self.quantizer.quantize_dense(k_dense, k_scale)
                k_data = {
                    "type": "quantized",
                    "quantized_data": k_quant,
                    "scale": k_scale,
                    "sparse_values": k_sp_val,
                    "sparse_indices": k_sp_idx
                }

            # 3.2 處理 Value (V)
            # V 使用 Per-token 瞬時統計，不需要 Warm-up，總是直接壓縮
            v_quant = self.quantizer.quantize_dense(v_dense, v_scale)
            v_data = {
                "type": "quantized",
                "quantized_data": v_quant,
                "scale": v_scale,
                "sparse_values": v_sp_val,
                "sparse_indices": v_sp_idx
            }

        # 4. Store
        # 使用通用介面儲存 (K 和 V 狀態可能不同)
        self.kv_manager.store_chunk(k_data, v_data)

        # 5. Attention
        # 記得：AttentionCore 必須負責還原資料並計算
        attn_output = self.attn_core.compute_attention(
            q_tensor, 
            self.kv_manager,
            rotary_emb_module=self.rotary_emb_module
        )
        
        return attn_output