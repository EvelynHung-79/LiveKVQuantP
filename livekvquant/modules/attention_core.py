import torch
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class AttentionCore:
    def __init__(self, config):
        self.config = config

    def compute_attention(self, q_tensor, kv_manager, 
                          current_k=None, current_v=None, 
                          rotary_emb_module=None, position_ids=None):
        
        target_dtype = q_tensor.dtype
        
        k_list, v_list = kv_manager.get_reconstructed_cache(target_dtype)
        
        if current_k is not None:
            k_list.append(current_k.to(dtype=target_dtype))
            v_list.append(current_v.to(dtype=target_dtype))
        
        if len(k_list) == 0:
             return torch.zeros_like(q_tensor)

        k_full = torch.cat(k_list, dim=-2)
        v_full = torch.cat(v_list, dim=-2)
        
        # RoPE
        if rotary_emb_module is not None and position_ids is not None:
            k_len = k_full.shape[-2]
            k_pos_ids = torch.arange(0, k_len, device=k_full.device, dtype=torch.long).unsqueeze(0)
            cos_k, sin_k = rotary_emb_module(k_full, k_pos_ids)
            _, k_full = apply_rotary_pos_emb(k_full, k_full, cos_k, sin_k)

            cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
            q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)
        
        # GQA Expand
        num_q_heads = q_tensor.size(1)
        num_k_heads = k_full.size(1)
        if num_q_heads != num_k_heads:
            n_rep = num_q_heads // num_k_heads
            k_full = k_full.repeat_interleave(n_rep, dim=1)
            v_full = v_full.repeat_interleave(n_rep, dim=1)

        # Attention
        head_dim = q_tensor.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        attn_scores = torch.matmul(q_tensor, k_full.transpose(-2, -1)) * scale
        
        # Causal Mask (Simplified)
        if q_tensor.shape[-2] > 1:
             q_len = q_tensor.size(-2)
             k_len = k_full.size(-2)
             mask = torch.triu(torch.ones(q_len, k_len, device=q_tensor.device), diagonal=k_len-q_len+1).bool()
             attn_scores.masked_fill_(mask, torch.finfo(target_dtype).min)

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output