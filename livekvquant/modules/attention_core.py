import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class AttentionCore:
    def __init__(self, config):
        self.config = config

    def compute_attention(self, q_tensor, kv_manager,
                          rotary_emb_module=None, position_ids=None,
                          k_is_rotated=False):
        """
        計算 Attention。
        當前 chunk 的 KV 已經透過 kv_manager.store_raw() 存入，
        get_full_kv() 會包含所有歷史 chunks + 當前 raw chunk。
        """
        target_dtype = q_tensor.dtype

        k_full, v_full = kv_manager.get_full_kv(target_dtype)

        if k_full is None:
            return torch.zeros_like(q_tensor)

        # RoPE
        if rotary_emb_module is not None and position_ids is not None:
            cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
            q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)

            if not k_is_rotated:
                k_len = k_full.shape[-2]
                k_pos_ids = torch.arange(0, k_len, device=k_full.device, dtype=torch.long).unsqueeze(0)
                cos_k, sin_k = rotary_emb_module(k_full, k_pos_ids)
                _, k_full = apply_rotary_pos_emb(k_full, k_full, cos_k, sin_k)

        # flash_attn_func 要求 layout: (batch, seq_len, num_heads, head_dim)
        # 目前 tensor layout: (batch, num_heads, seq_len, head_dim)
        q = q_tensor.transpose(1, 2)    # (b, q_len, n_q_heads, hd)
        k = k_full.transpose(1, 2)      # (b, k_len, n_kv_heads, hd)
        v = v_full.transpose(1, 2)      # (b, k_len, n_kv_heads, hd)

        # flash_attn_func 原生支援 GQA（n_q_heads 可以是 n_kv_heads 的整數倍）
        # 不需要 repeat_interleave，也不需要手動 attn_mask
        is_causal = (q.shape[1] > 1)

        # flash_attn 要求 fp16 或 bf16
        input_dtype = q.dtype
        if input_dtype not in (torch.float16, torch.bfloat16):
            q, k, v = q.to(torch.float16), k.to(torch.float16), v.to(torch.float16)

        attn_output = flash_attn_func(q, k, v, causal=is_causal)

        if attn_output.dtype != input_dtype:
            attn_output = attn_output.to(input_dtype)

        # 轉回 (batch, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2)

        return attn_output