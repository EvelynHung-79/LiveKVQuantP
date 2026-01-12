import torch
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from ..utils.quant_utils import dequantize_symmetric
from ..utils.outliers import restore_outliers

class AttentionCore:
    def __init__(self, config):
        self.config = config

    def _reconstruct_tensor(self, chunk_data: dict, target_dtype: torch.dtype) -> torch.Tensor:
        # [還原] 移除 isinstance(Tensor) 的檢查，因為現在只會有 dict
        if chunk_data["type"] == "warmup":
            return chunk_data["data"].to(dtype=target_dtype)
        
        dequantized = dequantize_symmetric(
            chunk_data["quantized_data"], 
            chunk_data["scale"]
        )
        
        reconstructed = restore_outliers(
            dequantized, 
            chunk_data["sparse_values"], 
            chunk_data["sparse_indices"]
        )
        
        return reconstructed.to(dtype=target_dtype)

    def compute_attention(self, q_tensor, kv_manager, 
                          current_k=None, current_v=None, 
                          rotary_emb_module=None, position_ids=None):
        k_chunks, v_chunks = kv_manager.get_all_chunks()
        
        target_dtype = q_tensor.dtype
        
        k_list = [self._reconstruct_tensor(c, target_dtype) for c in k_chunks]
        v_list = [self._reconstruct_tensor(c, target_dtype) for c in v_chunks]
        
        # 將當前無損的 Tensor 加入列表
        if current_k is not None:
            k_list.append(current_k.to(dtype=target_dtype))
            v_list.append(current_v.to(dtype=target_dtype))
        
        if len(k_list) == 0:
             return torch.zeros_like(q_tensor)

        k_full = torch.cat(k_list, dim=-2)
        v_full = torch.cat(v_list, dim=-2)
        
        # RoPE 應用
        if rotary_emb_module is not None and position_ids is not None:
            k_len = k_full.shape[-2]
            k_pos_ids = torch.arange(0, k_len, device=k_full.device, dtype=torch.long).unsqueeze(0)
            
            cos_k, sin_k = rotary_emb_module(k_full, k_pos_ids)
            _, k_full = apply_rotary_pos_emb(k_full, k_full, cos_k, sin_k)

            cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
            q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)
        
        # GQA 處理
        num_q_heads = q_tensor.size(1)
        num_k_heads = k_full.size(1)
        if num_q_heads != num_k_heads:
            n_rep = num_q_heads // num_k_heads
            k_full = k_full[:, :, None, :, :].expand(k_full.size(0), num_k_heads, n_rep, k_full.size(2), k_full.size(3)).reshape(k_full.size(0), num_k_heads * n_rep, k_full.size(2), k_full.size(3))
            v_full = v_full[:, :, None, :, :].expand(v_full.size(0), num_k_heads, n_rep, v_full.size(2), v_full.size(3)).reshape(v_full.size(0), num_k_heads * n_rep, v_full.size(2), v_full.size(3))

        # Attention 計算
        head_dim = q_tensor.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(q_tensor, k_full.transpose(-2, -1)) * scale
        
        # Masking
        q_len = q_tensor.size(-2)
        k_len = k_full.size(-2)
        
        causal_mask = torch.tril(torch.ones(q_len, q_len, device=q_tensor.device, dtype=torch.bool))
        min_value = torch.finfo(q_tensor.dtype).min
        mask_tensor = torch.full((q_len, q_len), min_value, device=q_tensor.device, dtype=q_tensor.dtype)
        mask_tensor.masked_fill_(causal_mask, 0.0)
        
        past_len = k_len - q_len
        if past_len > 0:
            history_mask = torch.zeros((q_len, past_len), device=q_tensor.device, dtype=q_tensor.dtype)
            full_mask = torch.cat([history_mask, mask_tensor], dim=-1)
        else:
            full_mask = mask_tensor
            
        attn_scores = attn_scores + full_mask.unsqueeze(0).unsqueeze(0)
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output