import torch
import triton
import triton.language as tl
import math

# =========================================================
# 1. Key Kernel (Dequant + RoPE + Q@K) [FIXED: Explicit FP32 Math]
# =========================================================

@triton.jit
def _fused_rope_quant_attention_kernel(
    # --- Pointers ---
    Q_ptr,              # Query [batch, num_heads, head_dim]
    K_ptr,              # Key Cache [batch, num_kv_heads, max_seq, head_dim] (Int8)
    K_scale_ptr,        # Key Scales [batch, num_kv_heads, 1, head_dim] (FP16/BF16)
    Cos_ptr, Sin_ptr,   # RoPE Cache [max_seq, head_dim]
    Out_ptr,            # Output [batch, num_heads, max_seq]
    
    # --- Strides ---
    stride_qb, stride_qh, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_ks_b, stride_ks_h, stride_ks_n, stride_ks_d,
    stride_cos_n, stride_cos_d,
    stride_out_b, stride_out_h, stride_out_n,
    
    # --- Constants ---
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_n = tl.program_id(2)
    pid_kv_h = pid_h // GROUP_SIZE
    
    start_n = pid_n * BLOCK_N
    offs_n = start_n + tl.arange(0, BLOCK_N)
    mask_n = offs_n < SEQ_LEN
    offs_d_lower = tl.arange(0, HALF_DIM)
    offs_d_upper = tl.arange(HALF_DIM, HEAD_DIM)
    
    # Load Q
    q_ptr_base = Q_ptr + (pid_b * stride_qb + pid_h * stride_qh)
    q_lower = tl.load(q_ptr_base + offs_d_lower * stride_qd)
    q_upper = tl.load(q_ptr_base + offs_d_upper * stride_qd)
    
    # Load K (Int8)
    k_ptr_base = K_ptr + (pid_b * stride_kb + pid_kv_h * stride_kh)
    k_ptrs_lower = k_ptr_base + (offs_n[:, None] * stride_kn + offs_d_lower[None, :] * stride_kd)
    k_int8_lower = tl.load(k_ptrs_lower, mask=mask_n[:, None], other=0.0)
    k_ptrs_upper = k_ptr_base + (offs_n[:, None] * stride_kn + offs_d_upper[None, :] * stride_kd)
    k_int8_upper = tl.load(k_ptrs_upper, mask=mask_n[:, None], other=0.0)
    
    # Load Scales & Dequantize to FP32
    # [FIX] 顯式轉型為 float32，避免 BF16 Scale 與 Int8 運算的潛在錯誤
    ks_ptr_base = K_scale_ptr + (pid_b * stride_ks_b + pid_kv_h * stride_ks_h)
    ks_ptrs_lower = ks_ptr_base + (offs_n[:, None] * stride_ks_n + offs_d_lower[None, :] * stride_ks_d)
    ks_lower = tl.load(ks_ptrs_lower, mask=mask_n[:, None], other=1.0)
    k_val_lower_f32 = k_int8_lower.to(tl.float32) * ks_lower.to(tl.float32)
    ks_ptrs_upper = ks_ptr_base + (offs_n[:, None] * stride_ks_n + offs_d_upper[None, :] * stride_ks_d)
    ks_upper = tl.load(ks_ptrs_upper, mask=mask_n[:, None], other=1.0)
    k_val_upper_f32 = k_int8_upper.to(tl.float32) * ks_upper.to(tl.float32)
    
    # Load RoPE
    cos_ptrs_l = Cos_ptr + (offs_n[:, None] * stride_cos_n + offs_d_lower[None, :] * stride_cos_d)
    sin_ptrs_l = Sin_ptr + (offs_n[:, None] * stride_cos_n + offs_d_lower[None, :] * stride_cos_d)
    cos_ptrs_u = Cos_ptr + (offs_n[:, None] * stride_cos_n + offs_d_upper[None, :] * stride_cos_d)
    sin_ptrs_u = Sin_ptr + (offs_n[:, None] * stride_cos_n + offs_d_upper[None, :] * stride_cos_d)
    
    cos_l = tl.load(cos_ptrs_l, mask=mask_n[:, None], other=1.0)
    sin_l = tl.load(sin_ptrs_l, mask=mask_n[:, None], other=0.0)
    cos_u = tl.load(cos_ptrs_u, mask=mask_n[:, None], other=1.0)
    sin_u = tl.load(sin_ptrs_u, mask=mask_n[:, None], other=0.0)
    
    # Apply RoPE (in FP32)
    cos_l_f32 = cos_l.to(tl.float32)
    sin_l_f32 = sin_l.to(tl.float32)
    cos_u_f32 = cos_u.to(tl.float32)
    sin_u_f32 = sin_u.to(tl.float32)
    
    k_rot_lower_f32 = k_val_lower_f32 * cos_l_f32 - k_val_upper_f32 * sin_l_f32
    k_rot_upper_f32 = k_val_upper_f32 * cos_u_f32 + k_val_lower_f32 * sin_u_f32
    
    # Compute Score (Q * K)
    q_lower_f32 = q_lower[None, :].to(tl.float32)
    q_upper_f32 = q_upper[None, :].to(tl.float32)
    
    prod_lower = q_lower_f32 * k_rot_lower_f32
    prod_upper = q_upper_f32 * k_rot_upper_f32
    
    score_lower = tl.sum(prod_lower, axis=1)
    score_upper = tl.sum(prod_upper, axis=1)
    score = (score_lower + score_upper) * (1.0 / math.sqrt(HEAD_DIM))
    
    # Store Output
    out_ptr_base = Out_ptr + (pid_b * stride_out_b + pid_h * stride_out_h)
    out_ptrs = out_ptr_base + offs_n * stride_out_n
    tl.store(out_ptrs, score, mask=mask_n)


# =========================================================
# 2. Value Kernel (Prob @ Dequant(V)) [FIXED: Explicit FP32 Math]
# =========================================================

@triton.jit
def _fused_value_attention_kernel(
    # --- Pointers ---
    Prob_ptr, V_ptr, V_scale_ptr, Out_ptr,
    # --- Strides ---
    stride_pb, stride_ph, stride_pn,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_vs_b, stride_vs_h, stride_vs_n, stride_vs_1,
    stride_ob, stride_oh, stride_od,
    # --- Constants ---
    SEQ_LEN: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr, GROUP_SIZE: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kv_h = pid_h // GROUP_SIZE
    
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offs_d = tl.arange(0, HEAD_DIM)
    
    prob_ptr_base = Prob_ptr + (pid_b * stride_pb + pid_h * stride_ph)
    v_ptr_base = V_ptr + (pid_b * stride_vb + pid_kv_h * stride_vh)
    vs_ptr_base = V_scale_ptr + (pid_b * stride_vs_b + pid_kv_h * stride_vs_h)
    
    for start_n in range(0, SEQ_LEN, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < SEQ_LEN
        
        prob_vals = tl.load(prob_ptr_base + offs_n * stride_pn, mask=mask_n, other=0.0)
        
        v_ptrs = v_ptr_base + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)
        v_int8 = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        
        vs_ptrs = vs_ptr_base + (offs_n[:, None] * stride_vs_n)
        v_scale = tl.load(vs_ptrs, mask=mask_n[:, None], other=0.0)
        
        # [FIX] Explicit FP32 conversion for safe multiplication
        v_val_f32 = v_int8.to(tl.float32) * v_scale.to(tl.float32)
        weighted_v = prob_vals[:, None].to(tl.float32) * v_val_f32
        
        acc += tl.sum(weighted_v, axis=0)
        
    out_ptr_base = Out_ptr + (pid_b * stride_ob + pid_h * stride_oh)
    tl.store(out_ptr_base + offs_d * stride_od, acc)


# =========================================================
# Wrappers
# =========================================================

def fused_kv_attention(q, k_cache, k_scales, cos_cache, sin_cache, output_buffer):
    """ Key Part Wrapper """
    batch_size, num_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    seq_len = k_cache.shape[2]
    assert head_dim % 2 == 0
    BLOCK_N = 32
    grid = (batch_size, num_heads, triton.cdiv(seq_len, BLOCK_N))

    s_ks_b, s_ks_h, s_ks_n, s_ks_d = k_scales.stride()
    if k_scales.shape[-1] == 1:
        s_ks_d = 0  # 強制最後一維 Stride 為 0，實現 Broadcast
    
    _fused_rope_quant_attention_kernel[grid](
        q, k_cache, k_scales, cos_cache, sin_cache, output_buffer,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        s_ks_b, s_ks_h, s_ks_n, s_ks_d,  # 使用修正後的 Stride
        cos_cache.stride(0), cos_cache.stride(1),
        output_buffer.stride(0), output_buffer.stride(1), output_buffer.stride(2),
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim, HALF_DIM=head_dim // 2, BLOCK_N=BLOCK_N, GROUP_SIZE=num_heads // num_kv_heads
    )
    return output_buffer

def fused_value_attention(probs, v_cache, v_scales, output_buffer=None):
    """ Value Part Wrapper """
    batch_size, num_heads, seq_len = probs.shape
    num_kv_heads = v_cache.shape[1]
    head_dim = v_cache.shape[3]
    
    if output_buffer is None:
        # [FIX] 使用 probs.dtype 作為預設輸出型態，而不是 float16
        output_buffer = torch.empty((batch_size, num_heads, head_dim), dtype=probs.dtype, device=probs.device)
        
    BLOCK_N = 128
    grid = (batch_size, num_heads)
    
    _fused_value_attention_kernel[grid](
        probs, v_cache, v_scales, output_buffer,
        probs.stride(0), probs.stride(1), probs.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        v_scales.stride(0), v_scales.stride(1), v_scales.stride(2), v_scales.stride(3),
        output_buffer.stride(0), output_buffer.stride(1), output_buffer.stride(2),
        SEQ_LEN=seq_len,
        HEAD_DIM=head_dim,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE=num_heads // num_kv_heads
    )
    return output_buffer