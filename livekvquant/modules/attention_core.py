import torch
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

from ..utils.quant_utils import dequantize_symmetric
from ..utils.outliers import restore_outliers

class AttentionCore:
    """
    負責 FP16 資料重建、RoPE 應用與 Attention 計算。
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
            return chunk_data["data"] # 直接回傳 FP16 (Warmup 存的是 Pre-RoPE)
        
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

    def compute_attention(
        self, 
        q_tensor: torch.Tensor, 
        kv_manager,
        rotary_emb_module = None
    ) -> torch.Tensor:
        """
        執行標準 Scaled Dot-Product Attention。
        """
        k_chunks, v_chunks = kv_manager.get_all_chunks()
        
        # 1. 重建所有歷史 KV 
        k_list = [self._reconstruct_tensor(c) for c in k_chunks]
        v_list = [self._reconstruct_tensor(c) for c in v_chunks]
        
        # 2. Concatenate 
        k_full = torch.cat(k_list, dim=-2)
        v_full = torch.cat(v_list, dim=-2)
        
        # 3. [核心修正] 動態生成全長的 RoPE
        if rotary_emb_module is not None:
            total_seq_len = k_full.shape[-2]
            position_ids = torch.arange(
                0, total_seq_len, device=k_full.device, dtype=torch.long
            ).unsqueeze(0)
            
            cos, sin = rotary_emb_module(k_full, position_ids)
            
            # [修正] transformers 不支援 q 為 None，我們將 k_full 同時傳入作為 placeholder
            _, k_full = apply_rotary_pos_emb(k_full, k_full, cos, sin)
        
        # --- [新增] 處理 GQA (Grouped Query Attention) ---
        # 檢查 Query Heads (32) 與 KV Heads (8) 是否不同
        num_q_heads = q_tensor.size(1)
        num_k_heads = k_full.size(1)
        
        if num_q_heads != num_k_heads:
            n_rep = num_q_heads // num_k_heads
            # 將 KV heads 重複 n_rep 次以匹配 Q heads
            # [batch, n_kv, seq, dim] -> [batch, n_kv, n_rep, seq, dim] -> [batch, n_q, seq, dim]
            k_full = k_full[:, :, None, :, :].expand(k_full.size(0), num_k_heads, n_rep, k_full.size(2), k_full.size(3)).reshape(k_full.size(0), num_k_heads * n_rep, k_full.size(2), k_full.size(3))
            v_full = v_full[:, :, None, :, :].expand(v_full.size(0), num_k_heads, n_rep, v_full.size(2), v_full.size(3)).reshape(v_full.size(0), num_k_heads * n_rep, v_full.size(2), v_full.size(3))
        # -----------------------------------------------

        # 4. Final Attention Operation (全部在 FP16 執行) 
        head_dim = q_tensor.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        
        attn_scores = torch.matmul(q_tensor, k_full.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output