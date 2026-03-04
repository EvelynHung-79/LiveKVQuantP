import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class AttentionCore:
    def __init__(self, config):
        self.config = config

    def compute_attention(self, q_tensor, kv_manager,
                          current_k=None, current_v=None,
                          rotary_emb_module=None, position_ids=None,
                          k_is_rotated=False):

        target_dtype = q_tensor.dtype

        # 使用快取的 full KV tensor，不會重複 dequantize
        k_full, v_full = kv_manager.get_full_kv(target_dtype)

        if current_k is not None:
            current_k = current_k.to(dtype=target_dtype)
            current_v = current_v.to(dtype=target_dtype)
            if k_full is not None:
                k_full = torch.cat([k_full, current_k], dim=-2)
                v_full = torch.cat([v_full, current_v], dim=-2)
            else:
                k_full = current_k
                v_full = current_v

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

        # GQA Expand
        num_q_heads = q_tensor.size(1)
        num_k_heads = k_full.size(1)
        if num_q_heads != num_k_heads:
            n_rep = num_q_heads // num_k_heads
            k_full = k_full.repeat_interleave(n_rep, dim=1)
            v_full = v_full.repeat_interleave(n_rep, dim=1)

        # Flash Attention via SDPA（自動選擇最快的 backend）
        is_causal = (q_tensor.shape[-2] == k_full.shape[-2] and q_tensor.shape[-2] > 1)

        if is_causal:
            # Prefill 且 Q/K 長度相同：可以直接用 is_causal=True
            attn_output = F.scaled_dot_product_attention(
                q_tensor, k_full, v_full, is_causal=True
            )
        elif q_tensor.shape[-2] > 1:
            # Prefill chunk 但前面有舊 cache（Q < K）：需要手動 causal mask
            q_len = q_tensor.size(-2)
            k_len = k_full.size(-2)
            # SDPA 的 attn_mask: True = 允許 attend, False = mask 掉
            attn_mask = ~torch.triu(
                torch.ones(q_len, k_len, device=q_tensor.device, dtype=torch.bool),
                diagonal=k_len - q_len + 1
            )
            attn_output = F.scaled_dot_product_attention(
                q_tensor, k_full, v_full, attn_mask=attn_mask
            )
        else:
            # Decode：Q length = 1，不需要 causal mask
            attn_output = F.scaled_dot_product_attention(
                q_tensor, k_full, v_full
            )

        return attn_output