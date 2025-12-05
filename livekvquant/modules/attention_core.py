import torch
import torch.nn.functional as F
import math
from ..utils.quant_utils import dequantize_symmetric
from ..utils.outliers import restore_outliers

class AttentionCore:
    """
    負責 FP16 資料重建與 Attention 計算。
    對應論文 3.6 Attention Core Module.
    """
    def __init__(self, config):
        self.config = config

    def _reconstruct_tensor(self, chunk_data: dict) -> torch.Tensor:
        """
        將單個 Chunk 還原為 FP16。
        對應論文 3.6.2 Data Assembly and Dequantization.
        """
        if chunk_data["type"] == "warmup":
            return chunk_data["data"] # 直接回傳 FP16
        
        # Quantized Chunk 重建
        # 1. Dequantization 
        # X_dequant = X_quant * s
        dequantized = dequantize_symmetric(
            chunk_data["quantized_data"], 
            chunk_data["scale"]
        )
        
        # 2. Outlier Integration 
        # 將 Sparse Outliers 填回
        reconstructed = restore_outliers(
            dequantized, 
            chunk_data["sparse_values"], 
            chunk_data["sparse_indices"]
        )
        
        return reconstructed

    def compute_attention(self, q_tensor: torch.Tensor, kv_manager) -> torch.Tensor:
        """
        執行標準 Scaled Dot-Product Attention。
        對應 Eq 3-10, 3-11, 3-12.
        
        Args:
            q_tensor: 當前 Chunk 的 Query [batch, heads, seq, head_dim]
            kv_manager: 儲存了歷史 KV 的 Manager
        """
        k_chunks, v_chunks = kv_manager.get_all_chunks()
        
        # 1. 重建所有歷史 KV (包含當前 Chunk) 
        # 注意：這在生產環境可以優化為只重建新 chunk 並與 cache 拼接
        # 這裡為了學術清晰度，展示完整重建邏輯
        k_list = [self._reconstruct_tensor(c) for c in k_chunks]
        v_list = [self._reconstruct_tensor(c) for c in v_chunks]
        
        # 2. Concatenate 
        # X_full = [X_past, X_current]
        k_full = torch.cat(k_list, dim=-2)
        v_full = torch.cat(v_list, dim=-2)
        
        # 3. Final Attention Operation (全部在 FP16 執行) 
        # Attn = Softmax(Q * K^T / sqrt(d)) * V
        head_dim = q_tensor.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        
        # [batch, heads, q_len, k_len]
        attn_scores = torch.matmul(q_tensor, k_full.transpose(-2, -1)) * scale
        
        # (這裡省略了 Causal Mask 的實作細節，實際跑 Llama 需要加上 mask)
        # 假設這是一個簡單的 Causal LM 推論，mask 會由 model wrapper 處理或在這裡加
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # [batch, heads, q_len, head_dim]
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output