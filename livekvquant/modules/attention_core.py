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
        """
        if chunk_data["type"] == "warmup":
            return chunk_data["data"] # 直接回傳 FP16 (Warmup 存的是 Pre-RoPE)
        
        # Quantized Chunk 重建
        dequantized = dequantize_symmetric(
            chunk_data["quantized_data"], 
            chunk_data["scale"]
        )
        
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
        rotary_emb_module = None,
        position_ids = None,
    ) -> torch.Tensor:
        """
        執行標準 Scaled Dot-Product Attention，並加入 Causal Mask。
        """
        k_chunks, v_chunks = kv_manager.get_all_chunks()
        
        # 1. 重建所有歷史 KV 
        k_list = [self._reconstruct_tensor(c) for c in k_chunks]
        v_list = [self._reconstruct_tensor(c) for c in v_chunks]
        
        # 2. Concatenate 
        k_full = torch.cat(k_list, dim=-2)
        v_full = torch.cat(v_list, dim=-2)
        
        # 3. 動態生成全長的 RoPE
        if rotary_emb_module is not None and position_ids is not None:
            # 1. 對 K (Full History) 生成 RoPE
            # K 的長度是 total_len，位置是 0 到 total_len
            k_len = k_full.shape[-2]
            k_pos_ids = torch.arange(0, k_len, device=k_full.device, dtype=torch.long).unsqueeze(0)
            
            cos_k, sin_k = rotary_emb_module(k_full, k_pos_ids)
            _, k_full = apply_rotary_pos_emb(k_full, k_full, cos_k, sin_k)

            # 2. 對 Q (Current Chunk) 應用 RoPE
            # 注意：這裡假設傳進來的 q_tensor 是 "未旋轉" (Pre-RoPE) 的
            # 我們使用傳入的 position_ids (對應當前 Chunk 的絕對位置)
            cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
            q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)
        
        # --- 處理 GQA ---
        num_q_heads = q_tensor.size(1)
        num_k_heads = k_full.size(1)
        
        if num_q_heads != num_k_heads:
            n_rep = num_q_heads // num_k_heads
            k_full = k_full[:, :, None, :, :].expand(k_full.size(0), num_k_heads, n_rep, k_full.size(2), k_full.size(3)).reshape(k_full.size(0), num_k_heads * n_rep, k_full.size(2), k_full.size(3))
            v_full = v_full[:, :, None, :, :].expand(v_full.size(0), num_k_heads, n_rep, v_full.size(2), v_full.size(3)).reshape(v_full.size(0), num_k_heads * n_rep, v_full.size(2), v_full.size(3))

        # 4. Final Attention Operation
        head_dim = q_tensor.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        
        # [Batch, Heads, Q_Len, K_Len]
        attn_scores = torch.matmul(q_tensor, k_full.transpose(-2, -1)) * scale
        
        # === [核心修正] 加入 Causal Mask ===
        q_len = q_tensor.size(-2)
        k_len = k_full.size(-2)
        
        # 建立全長的 Mask (包含 History)
        # 1. 只有當前 Chunk 的 Causal Mask (下三角)
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=q_tensor.device, dtype=torch.bool))
        
        # 2. 轉換數值 (0.0 可見, min_value 不可見)
        min_value = torch.finfo(q_tensor.dtype).min
        mask_tensor = torch.full((q_len, q_len), min_value, device=q_tensor.device, dtype=q_tensor.dtype)
        mask_tensor.masked_fill_(causal_mask, 0.0) 
        
        # 3. 處理 History (History 永遠可見)
        past_len = k_len - q_len
        if past_len > 0:
            # History 部分全 0 (可見)
            history_mask = torch.zeros((q_len, past_len), device=q_tensor.device, dtype=q_tensor.dtype)
            full_mask = torch.cat([history_mask, mask_tensor], dim=-1)
        else:
            full_mask = mask_tensor
            
        # 4. 強制套用 Mask (確保維度廣播正確)
        # attn_scores shape: [B, H, q_len, k_len]
        # full_mask shape: [q_len, k_len] -> [1, 1, q_len, k_len]
        attn_scores = attn_scores + full_mask.unsqueeze(0).unsqueeze(0)            
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output