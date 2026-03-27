"""
Fused INT4 Dequant + Flash Attention (Triton 3.x) — Optimized v2

核心思路：
  - 從 HBM 載入 INT4 packed K/V（每 byte 含 2 個 int4 值）
  - 在 SRAM 內 unpack → dequant → FP16 → 用 tensor core 計算 attention
  - FP16 K/V 從來不寫到 HBM，peak memory = O(chunk_size) 而非 O(seq_len)

v2 optimizations (vs v1):
  - 消除 even/odd split：unpack INT4 → full HEAD_DIM FP16, 單次 tl.dot
  - FP16 tensor core for QK^T and PV (2x throughput vs FP32)
  - FP32 online softmax (numerical stability)
  - BLOCK_K=128 減少 loop iterations
  - Output 直接寫 full dim，不需要 post-hoc interleave
  - Warmup/raw chunks 用 Triton FP16 kernel，不回退 PyTorch einsum
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused INT4 dequant + attention (per-chunk stateful update)
# ---------------------------------------------------------------------------

@triton.jit
def _int4_dequant_attn_kernel(
    # Q: (B, n_q_heads, Sq, HEAD_DIM) float16
    Q_ptr, stride_qb, stride_qh, stride_qq, stride_qd,
    # K packed: (B, n_kv_heads, Sk, HEAD_DIM//2) int8
    K_ptr, stride_kb, stride_kh, stride_kk, stride_kd,
    # K scale: (B, n_kv_heads, 1, HEAD_DIM) float16  [per feature-dim]
    Ks_ptr, stride_ksb, stride_ksh, stride_ksd,
    # V packed: (B, n_kv_heads, Sk, HEAD_DIM//2) int8
    V_ptr, stride_vb, stride_vh, stride_vk, stride_vd,
    # V scale: (B, n_kv_heads, Sk, 1) float16  [per token]
    Vs_ptr, stride_vsb, stride_vsh, stride_vsk,
    # Running state (FP32)
    Mi_ptr, stride_mib, stride_mih, stride_miq,          # (B, H, Sq)
    Li_ptr, stride_lib, stride_lih, stride_liq,          # (B, H, Sq)
    Oi_ptr, stride_oib, stride_oih, stride_oiq, stride_oid,  # (B, H, Sq, D)
    # Sizes
    Sq, Sk,
    KV_START,
    SCALE,
    # Constexprs
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_HALF: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GQA_GROUPS: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
):
    # Program IDs
    q_tile = tl.program_id(0)
    bh     = tl.program_id(1)
    b      = bh // N_Q_HEADS
    h_q    = bh %  N_Q_HEADS
    h_kv   = h_q // GQA_GROUPS

    q_start = q_tile * BLOCK_Q
    q_offs  = q_start + tl.arange(0, BLOCK_Q)
    q_mask  = q_offs < Sq

    d_offs     = tl.arange(0, HEAD_DIM)        # (HEAD_DIM,)
    half_offs  = tl.arange(0, HEAD_DIM_HALF)   # (HEAD_DIM_HALF,)

    # Load Q: (BLOCK_Q, HEAD_DIM) fp16 → fp16 (stays fp16 for tensor core)
    q_base = b * stride_qb + h_q * stride_qh
    Q_block = tl.load(
        Q_ptr + q_base + q_offs[:, None] * stride_qq + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0
    ).to(tl.float16)  # (BLOCK_Q, HEAD_DIM)

    # Load K scale: (HEAD_DIM,) — constant for all K tokens in this chunk
    ks_base = b * stride_ksb + h_kv * stride_ksh
    k_scale = tl.load(Ks_ptr + ks_base + d_offs * stride_ksd).to(tl.float16)  # (HEAD_DIM,)

    # Load running state
    mi_base = b * stride_mib + h_q * stride_mih
    li_base = b * stride_lib + h_q * stride_lih
    oi_base = b * stride_oib + h_q * stride_oih

    mi = tl.load(Mi_ptr + mi_base + q_offs * stride_miq,
                 mask=q_mask, other=float('-inf'))       # (BLOCK_Q,) fp32
    li = tl.load(Li_ptr + li_base + q_offs * stride_liq,
                 mask=q_mask, other=0.0)                 # (BLOCK_Q,) fp32
    oi = tl.load(
        Oi_ptr + oi_base + q_offs[:, None] * stride_oiq + d_offs[None, :] * stride_oid,
        mask=q_mask[:, None], other=0.0
    ).to(tl.float32)  # (BLOCK_Q, HEAD_DIM) fp32

    # Inner loop: iterate over KV tiles
    for k_start in tl.range(0, Sk, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Sk

        # ---- Load & unpack K INT4 → (BLOCK_K, HEAD_DIM) fp16 ----
        k_packed = tl.load(
            K_ptr + b * stride_kb + h_kv * stride_kh +
            k_offs[:, None] * stride_kk + half_offs[None, :] * stride_kd,
            mask=k_mask[:, None], other=0
        ).to(tl.int32)  # (BLOCK_K, HEAD_DIM_HALF)

        k_byte = k_packed & 0xFF
        k_lo = (k_byte & 0xF).to(tl.int8)          # low nibble (even dims)
        k_hi = ((k_byte >> 4) & 0xF).to(tl.int8)   # high nibble (odd dims)
        # Sign-extend: 8..15 → -8..-1 via arithmetic: val = (val << 4) >> 4 in int8
        # Simpler: subtract 16 if > 7
        k_lo = tl.where(k_lo > 7, k_lo - 16, k_lo).to(tl.float16)  # (BLOCK_K, HD_HALF)
        k_hi = tl.where(k_hi > 7, k_hi - 16, k_hi).to(tl.float16)

        # Interleave to full HEAD_DIM: even dims = lo, odd dims = hi
        # Build (BLOCK_K, HEAD_DIM) by interleaving
        K_full = tl.interleave(k_lo, k_hi)  # (BLOCK_K, HEAD_DIM)

        # Dequant: per-dim scale
        K_dq = K_full * k_scale[None, :]  # (BLOCK_K, HEAD_DIM) fp16

        # ---- QK^T via tensor core: (BLOCK_Q, HEAD_DIM) @ (HEAD_DIM, BLOCK_K) ----
        scores = tl.dot(Q_block, tl.trans(K_dq)).to(tl.float32) * SCALE  # (BLOCK_Q, BLOCK_K) fp32

        # ---- Masks (fused) ----
        combined_mask = k_mask[None, :]  # start with padding mask
        if IS_CAUSAL:
            q_abs = q_start + tl.arange(0, BLOCK_Q)
            k_abs = KV_START + k_start + tl.arange(0, BLOCK_K)
            combined_mask = combined_mask & (q_abs[:, None] >= k_abs[None, :])
        scores = tl.where(combined_mask, scores, float('-inf'))

        # ---- Online softmax update (FP32) ----
        chunk_max  = tl.max(scores, axis=1)               # (BLOCK_Q,)
        mi_new     = tl.maximum(mi, chunk_max)
        correction = tl.exp(mi - mi_new)                  # (BLOCK_Q,)
        exp_s      = tl.exp(scores - mi_new[:, None])     # (BLOCK_Q, BLOCK_K) fp32

        li = correction * li + tl.sum(exp_s, axis=1)
        mi = mi_new

        # ---- Load & unpack V INT4 → (BLOCK_K, HEAD_DIM) fp16 ----
        v_packed = tl.load(
            V_ptr + b * stride_vb + h_kv * stride_vh +
            k_offs[:, None] * stride_vk + half_offs[None, :] * stride_vd,
            mask=k_mask[:, None], other=0
        ).to(tl.int32)

        v_byte = v_packed & 0xFF
        v_lo = (v_byte & 0xF).to(tl.int8)
        v_hi = ((v_byte >> 4) & 0xF).to(tl.int8)
        v_lo = tl.where(v_lo > 7, v_lo - 16, v_lo).to(tl.float16)
        v_hi = tl.where(v_hi > 7, v_hi - 16, v_hi).to(tl.float16)

        V_full = tl.interleave(v_lo, v_hi)  # (BLOCK_K, HEAD_DIM)

        # V scale: per-token
        v_scale_tile = tl.load(
            Vs_ptr + b * stride_vsb + h_kv * stride_vsh + k_offs * stride_vsk,
            mask=k_mask, other=1.0
        ).to(tl.float16)  # (BLOCK_K,)
        V_dq = V_full * v_scale_tile[:, None]  # (BLOCK_K, HEAD_DIM) fp16

        # ---- Accumulate output: P @ V ----
        # exp_s is fp32 (BLOCK_Q, BLOCK_K), V_dq is fp16 (BLOCK_K, HEAD_DIM)
        # tl.dot needs matching types → cast exp_s to fp16 for tensor core
        pv = tl.dot(exp_s.to(tl.float16), V_dq).to(tl.float32)  # (BLOCK_Q, HEAD_DIM)
        oi = correction[:, None] * oi + pv

    # Write back running state
    tl.store(Mi_ptr + mi_base + q_offs * stride_miq, mi, mask=q_mask)
    tl.store(Li_ptr + li_base + q_offs * stride_liq, li, mask=q_mask)
    tl.store(
        Oi_ptr + oi_base + q_offs[:, None] * stride_oiq + d_offs[None, :] * stride_oid,
        oi, mask=q_mask[:, None]
    )


# ---------------------------------------------------------------------------
# Triton kernel: FP16 attention (for warmup/raw/decode chunks)
# Same online-softmax stateful design, but no INT4 unpacking.
# ---------------------------------------------------------------------------

@triton.jit
def _fp16_chunk_attn_kernel(
    Q_ptr, stride_qb, stride_qh, stride_qq, stride_qd,
    K_ptr, stride_kb, stride_kh, stride_kk, stride_kd,
    V_ptr, stride_vb, stride_vh, stride_vk, stride_vd,
    Mi_ptr, stride_mib, stride_mih, stride_miq,
    Li_ptr, stride_lib, stride_lih, stride_liq,
    Oi_ptr, stride_oib, stride_oih, stride_oiq, stride_oid,
    Sq, Sk,
    KV_START,
    SCALE,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    GQA_GROUPS: tl.constexpr,
    N_Q_HEADS: tl.constexpr,
):
    q_tile = tl.program_id(0)
    bh     = tl.program_id(1)
    b      = bh // N_Q_HEADS
    h_q    = bh %  N_Q_HEADS
    h_kv   = h_q // GQA_GROUPS

    q_start = q_tile * BLOCK_Q
    q_offs  = q_start + tl.arange(0, BLOCK_Q)
    q_mask  = q_offs < Sq
    d_offs  = tl.arange(0, HEAD_DIM)

    q_base = b * stride_qb + h_q * stride_qh
    Q_block = tl.load(
        Q_ptr + q_base + q_offs[:, None] * stride_qq + d_offs[None, :] * stride_qd,
        mask=q_mask[:, None], other=0.0
    ).to(tl.float16)

    mi_base = b * stride_mib + h_q * stride_mih
    li_base = b * stride_lib + h_q * stride_lih
    oi_base = b * stride_oib + h_q * stride_oih

    mi = tl.load(Mi_ptr + mi_base + q_offs * stride_miq,
                 mask=q_mask, other=float('-inf'))
    li = tl.load(Li_ptr + li_base + q_offs * stride_liq,
                 mask=q_mask, other=0.0)
    oi = tl.load(
        Oi_ptr + oi_base + q_offs[:, None] * stride_oiq + d_offs[None, :] * stride_oid,
        mask=q_mask[:, None], other=0.0
    ).to(tl.float32)

    for k_start in tl.range(0, Sk, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < Sk

        # Load K fp16: (BLOCK_K, HEAD_DIM)
        K_block = tl.load(
            K_ptr + b * stride_kb + h_kv * stride_kh +
            k_offs[:, None] * stride_kk + d_offs[None, :] * stride_kd,
            mask=k_mask[:, None], other=0.0
        ).to(tl.float16)

        # QK^T
        scores = tl.dot(Q_block, tl.trans(K_block)).to(tl.float32) * SCALE

        combined_mask = k_mask[None, :]
        if IS_CAUSAL:
            q_abs = q_start + tl.arange(0, BLOCK_Q)
            k_abs = KV_START + k_start + tl.arange(0, BLOCK_K)
            combined_mask = combined_mask & (q_abs[:, None] >= k_abs[None, :])
        scores = tl.where(combined_mask, scores, float('-inf'))

        chunk_max  = tl.max(scores, axis=1)
        mi_new     = tl.maximum(mi, chunk_max)
        correction = tl.exp(mi - mi_new)
        exp_s      = tl.exp(scores - mi_new[:, None])

        li = correction * li + tl.sum(exp_s, axis=1)
        mi = mi_new

        V_block = tl.load(
            V_ptr + b * stride_vb + h_kv * stride_vh +
            k_offs[:, None] * stride_vk + d_offs[None, :] * stride_vd,
            mask=k_mask[:, None], other=0.0
        ).to(tl.float16)

        pv = tl.dot(exp_s.to(tl.float16), V_block).to(tl.float32)
        oi = correction[:, None] * oi + pv

    tl.store(Mi_ptr + mi_base + q_offs * stride_miq, mi, mask=q_mask)
    tl.store(Li_ptr + li_base + q_offs * stride_liq, li, mask=q_mask)
    tl.store(
        Oi_ptr + oi_base + q_offs[:, None] * stride_oiq + d_offs[None, :] * stride_oid,
        oi, mask=q_mask[:, None]
    )


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def _launch_int4_kernel(q, k_packed, k_scale, v_packed, v_scale,
                        mi, li, oi, kv_start, is_causal):
    B, n_q_heads, Sq, D = q.shape
    _, n_kv_heads, Sk, _ = k_packed.shape
    D_half = D // 2
    gqa_groups = n_q_heads // n_kv_heads
    scale = math.sqrt(D) ** -1

    BLOCK_Q = max(16, min(32, triton.next_power_of_2(Sq)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(Sk)))
    grid = (triton.cdiv(Sq, BLOCK_Q), B * n_q_heads)

    _int4_dequant_attn_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_packed, k_packed.stride(0), k_packed.stride(1), k_packed.stride(2), k_packed.stride(3),
        k_scale, k_scale.stride(0), k_scale.stride(1), k_scale.stride(3),
        v_packed, v_packed.stride(0), v_packed.stride(1), v_packed.stride(2), v_packed.stride(3),
        v_scale, v_scale.stride(0), v_scale.stride(1), v_scale.stride(2),
        mi, mi.stride(0), mi.stride(1), mi.stride(2),
        li, li.stride(0), li.stride(1), li.stride(2),
        oi, oi.stride(0), oi.stride(1), oi.stride(2), oi.stride(3),
        Sq, Sk, kv_start, scale,
        HEAD_DIM=D,
        HEAD_DIM_HALF=D_half,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        IS_CAUSAL=is_causal,
        GQA_GROUPS=gqa_groups,
        N_Q_HEADS=n_q_heads,
    )


def _launch_fp16_kernel(q, k_fp16, v_fp16, mi, li, oi, kv_start, is_causal):
    B, n_q_heads, Sq, D = q.shape
    _, n_kv_heads, Sk, _ = k_fp16.shape
    gqa_groups = n_q_heads // n_kv_heads
    scale = math.sqrt(D) ** -1

    BLOCK_Q = max(16, min(32, triton.next_power_of_2(Sq)))
    BLOCK_K = max(16, min(64, triton.next_power_of_2(Sk)))
    grid = (triton.cdiv(Sq, BLOCK_Q), B * n_q_heads)

    _fp16_chunk_attn_kernel[grid](
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_fp16, k_fp16.stride(0), k_fp16.stride(1), k_fp16.stride(2), k_fp16.stride(3),
        v_fp16, v_fp16.stride(0), v_fp16.stride(1), v_fp16.stride(2), v_fp16.stride(3),
        mi, mi.stride(0), mi.stride(1), mi.stride(2),
        li, li.stride(0), li.stride(1), li.stride(2),
        oi, oi.stride(0), oi.stride(1), oi.stride(2), oi.stride(3),
        Sq, Sk, kv_start, scale,
        HEAD_DIM=D,
        BLOCK_Q=BLOCK_Q,
        BLOCK_K=BLOCK_K,
        IS_CAUSAL=is_causal,
        GQA_GROUPS=gqa_groups,
        N_Q_HEADS=n_q_heads,
    )


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def fused_int4_flash_attn(
    q: torch.Tensor,           # (B, n_q_heads, Sq, D) float16
    kv_manager,
    rotary_emb_module=None,
    position_ids=None,
    fallback_fn=None,
) -> torch.Tensor:
    from flash_attn import flash_attn_func
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    B, H, Sq, D = q.shape
    device = q.device
    dtype  = q.dtype

    # Apply RoPE to Q once
    q_rot = q
    if rotary_emb_module is not None and position_ids is not None:
        cos, sin = rotary_emb_module(q, position_ids)
        q_rot, _ = apply_rotary_pos_emb(q, q, cos, sin)

    # -------------------------------------------------------------------
    # Decode path (Sq=1): dequant all chunks → concat → single flash_attn
    # This avoids per-chunk kernel launch overhead which dominates at Sq=1.
    # Memory cost is O(total_kv_len * D) FP16, but only briefly.
    # -------------------------------------------------------------------
    if Sq == 1:
        k_parts = []
        v_parts = []
        for k_chunk, v_chunk, kv_start, raw_k, raw_v in kv_manager.iter_kv_chunks():
            if k_chunk is not None:
                k_parts.append(k_chunk.reconstruct(dtype))
                v_parts.append(v_chunk.reconstruct(dtype))
            else:
                k_parts.append(raw_k.to(dtype))
                v_parts.append(raw_v.to(dtype))

        if not k_parts:
            return torch.zeros_like(q)

        k_full = torch.cat(k_parts, dim=2)  # (B, kv_H, total_len, D)
        v_full = torch.cat(v_parts, dim=2)

        # flash_attn layout: (B, seq, H, D)
        q_fa = q_rot.transpose(1, 2)       # (B, 1, H, D)
        k_fa = k_full.transpose(1, 2)      # (B, total_len, kv_H, D)
        v_fa = v_full.transpose(1, 2)

        out = flash_attn_func(q_fa, k_fa, v_fa, causal=False)
        return out.transpose(1, 2)  # (B, H, 1, D)

    # -------------------------------------------------------------------
    # Prefill path (Sq>1): per-chunk Triton kernels to save peak memory.
    # Only one chunk's FP16 K/V lives in SRAM at a time.
    # -------------------------------------------------------------------
    mi = torch.full((B, H, Sq), float('-inf'), device=device, dtype=torch.float32)
    li = torch.zeros((B, H, Sq),               device=device, dtype=torch.float32)
    oi = torch.zeros((B, H, Sq, D),            device=device, dtype=torch.float32)

    has_any = False

    for k_chunk, v_chunk, kv_start, raw_k, raw_v in kv_manager.iter_kv_chunks():

        if k_chunk is None:
            k_fp = raw_k.to(dtype).contiguous()
            v_fp = raw_v.to(dtype).contiguous()
            _launch_fp16_kernel(q_rot, k_fp, v_fp, mi, li, oi, kv_start, True)
            has_any = True
            continue

        chunk_type = k_chunk.chunk_type

        if chunk_type in ("raw", "warmup") or k_chunk.is_asymmetric:
            k_fp = k_chunk.reconstruct(dtype).contiguous()
            v_fp = v_chunk.reconstruct(dtype).contiguous()
            _launch_fp16_kernel(q_rot, k_fp, v_fp, mi, li, oi, kv_start, True)
            has_any = True
            continue

        # Symmetric INT4: fused Triton kernel
        k_packed = k_chunk.quantized_data
        v_packed = v_chunk.quantized_data
        k_scale  = k_chunk.scale
        v_scale  = v_chunk.scale

        if not k_chunk._is_packed:
            from ..utils.quant_utils import pack_int4
            k_packed = pack_int4(k_packed)
            v_packed = pack_int4(v_packed)

        k_packed = k_packed.contiguous()
        v_packed = v_packed.contiguous()

        _launch_int4_kernel(
            q_rot, k_packed, k_scale, v_packed, v_scale,
            mi, li, oi,
            kv_start=kv_start, is_causal=True,
        )
        has_any = True

    if not has_any:
        return torch.zeros_like(q)

    out = oi / li.unsqueeze(-1)
    return out.to(dtype)
