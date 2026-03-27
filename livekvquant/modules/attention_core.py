import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

class AttentionCore:
    def __init__(self, config):
        self.config = config
        self.use_chunked_attn = getattr(config, 'use_chunked_attn', False)

    def compute_attention_chunked(self, q_tensor, kv_manager,
                                  rotary_emb_module=None, position_ids=None):
        """
        Per-chunk dequant + online softmax。
        每次只讓一個 chunk 的 FP16 K/V 存在 memory，避免全序列 dequant 的 peak memory。
        K chunks 已在儲存前做過 RoPE，不需再次套用。
        數學上與標準 attention 等價。
        """
        B, H, Sq, D = q_tensor.shape
        scale = D ** -0.5
        dtype = q_tensor.dtype
        device = q_tensor.device

        # RoPE on Q (once)
        q_rot = q_tensor
        if rotary_emb_module is not None and position_ids is not None:
            cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
            q_rot, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)

        # Online softmax accumulators
        mi = torch.full((B, H, Sq, 1), float('-inf'), device=device, dtype=dtype)
        li = torch.zeros(B, H, Sq, 1, device=device, dtype=dtype)
        oi = torch.zeros(B, H, Sq, D, device=device, dtype=dtype)

        has_any_kv = False

        for k_chunk, v_chunk, kv_start, raw_k, raw_v in kv_manager.iter_kv_chunks():
            if k_chunk is not None:
                # Prefill chunks (raw / warmup / quantized)
                k = k_chunk.reconstruct(dtype)   # (B, kv_H, chunk_len, D)
                v = v_chunk.reconstruct(dtype)
            else:
                # Decode buffer
                k = raw_k.to(dtype=dtype)
                v = raw_v.to(dtype=dtype)

            chunk_len = k.shape[-2]
            kv_H = k.shape[1]

            # GQA expand: repeat KV heads to match Q heads
            if kv_H != H:
                groups = H // kv_H
                k = k.repeat_interleave(groups, dim=1)
                v = v.repeat_interleave(groups, dim=1)

            # scores: (B, H, Sq, chunk_len)
            scores = torch.einsum('bhqd,bhkd->bhqk', q_rot * scale, k)

            # Causal mask: only needed when Sq > 1
            # q_abs_pos: absolute position of each Q token
            if Sq > 1:
                if position_ids is not None:
                    q_abs = position_ids[0]  # (Sq,) — use first batch item (all same for prefill)
                else:
                    q_abs = torch.arange(Sq, device=device)
                q_idx = q_abs.unsqueeze(-1)                                          # (Sq, 1)
                k_idx = torch.arange(kv_start, kv_start + chunk_len, device=device) # (chunk_len,)
                mask = q_idx < k_idx  # True where Q should NOT attend to K
                scores = scores.masked_fill(mask[None, None], float('-inf'))

            # Online softmax update
            chunk_max = scores.amax(dim=-1, keepdim=True)  # (B, H, Sq, 1)
            mi_new = torch.maximum(mi, chunk_max)
            exp_s = torch.exp(scores - mi_new)             # (B, H, Sq, chunk_len)
            correction = torch.exp(mi - mi_new)
            li = correction * li + exp_s.sum(dim=-1, keepdim=True)
            oi = correction * oi + torch.einsum('bhqk,bhkd->bhqd', exp_s, v)
            mi = mi_new
            has_any_kv = True

            del k, v, scores, exp_s

        if not has_any_kv:
            return torch.zeros_like(q_tensor)

        return oi / li  # (B, H, Sq, D)

    def compute_attention(self, q_tensor, kv_manager,
                          rotary_emb_module=None, position_ids=None,
                          k_is_rotated=False):
        """
        計算 Attention。
        當前 chunk 的 KV 已經透過 kv_manager.store_raw() 存入，
        get_full_kv() 會包含所有歷史 chunks + 當前 raw chunk。
        """
        if self.use_chunked_attn:
            return self.compute_attention_chunked(
                q_tensor, kv_manager,
                rotary_emb_module=rotary_emb_module,
                position_ids=position_ids
            )

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