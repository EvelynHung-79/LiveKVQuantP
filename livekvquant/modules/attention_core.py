import torch
import torch.nn.functional as F
import math
import logging
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from .kernels import fused_kv_attention, fused_value_attention

# 設定 Logger
logger = logging.getLogger(__name__)

def debug_tensor(name, tensor):
    """輔助函式：印出 Tensor 的型態與形狀"""
    if tensor is None:
        print(f"[DEBUG] {name}: None")
    elif isinstance(tensor, torch.Tensor):
        # [Log Restored] 恢復 Log 輸出
        print(f"[DEBUG] {name}: dtype={tensor.dtype}, shape={tensor.shape}, device={tensor.device}")
    else:
        print(f"[DEBUG] {name}: Type={type(tensor)}")

class AttentionCore:
    def __init__(self, config):
        self.config = config

    def compute_attention(self, q_tensor, kv_manager, 
                          current_k=None, current_v=None, 
                          rotary_emb_module=None, position_ids=None):
        
        # 1. 安全解析 Q
        if q_tensor.dim() == 4:
            batch_size, num_heads, q_len, head_dim = q_tensor.shape
        else:
            batch_size, num_heads, head_dim = q_tensor.shape
            q_len = 1
            q_tensor = q_tensor.unsqueeze(2)

        k_cache, k_scales, v_cache, v_scales = kv_manager.get_views()
        
        has_int8_cache = (isinstance(k_cache, torch.Tensor) and k_cache.dtype == torch.int8)
        is_decoding = (q_len == 1)
        
        # GQA Info
        if current_k is not None:
            num_kv_heads = current_k.shape[1]
        elif isinstance(k_cache, torch.Tensor):
            num_kv_heads = k_cache.shape[1]
        elif isinstance(k_cache, list) and len(k_cache) > 0:
            num_kv_heads = k_cache[0].shape[1]
        else:
            num_kv_heads = num_heads
        n_rep = num_heads // num_kv_heads
        
        # =================================================================
        # Path A: Hybrid Decoding
        # =================================================================
        if has_int8_cache and is_decoding:
            seq_len_hist = k_cache.shape[2]
            
            # A-1. Query RoPE (優先)
            if rotary_emb_module is not None:
                cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
                cos_q = cos_q.to(dtype=q_tensor.dtype)
                sin_q = sin_q.to(dtype=q_tensor.dtype)
                q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)
            
            q_squeezed = q_tensor.squeeze(2) 
            
            # [FIX] Output dtype matches Query dtype
            hist_scores = torch.empty(
                (batch_size, num_heads, seq_len_hist), 
                dtype=q_tensor.dtype, device=q_tensor.device
            )
            
            if rotary_emb_module is not None:
                pos_ids_hist = torch.arange(seq_len_hist, device=q_tensor.device).unsqueeze(0)
                cos_hist, sin_hist = rotary_emb_module(q_tensor, pos_ids_hist)
                
                cos_hist = cos_hist.squeeze(0).to(dtype=q_tensor.dtype).contiguous()
                sin_hist = sin_hist.squeeze(0).to(dtype=q_tensor.dtype).contiguous()
                
                fused_kv_attention(q_squeezed, k_cache, k_scales, cos_hist, sin_hist, hist_scores)
            else:
                raise NotImplementedError("Fused kernel requires RoPE")

            # A-2. Current Token Score
            if rotary_emb_module is not None:
                cos_k, sin_k = rotary_emb_module(current_k, position_ids)
                cos_k = cos_k.to(dtype=current_k.dtype)
                sin_k = sin_k.to(dtype=current_k.dtype)
                curr_k_rope, _ = apply_rotary_pos_emb(current_k, current_k, cos_k, sin_k)
            else:
                curr_k_rope = current_k
            
            # GQA
            if n_rep > 1:
                curr_k_rope = curr_k_rope.repeat_interleave(n_rep, dim=1)
                
            curr_score = torch.matmul(q_tensor, curr_k_rope.transpose(-1, -2)) 
            curr_score = curr_score.squeeze(2) * (1.0 / math.sqrt(head_dim))

            # A-3. Merge
            all_scores = torch.cat([hist_scores, curr_score], dim=-1)
            all_probs = F.softmax(all_scores, dim=-1)
            
            probs_hist = all_probs[:, :, :seq_len_hist].contiguous() 
            probs_curr = all_probs[:, :, seq_len_hist:]
            
            # Value Kernel
            out_hist = fused_value_attention(probs_hist, v_cache, v_scales)
            
            curr_v_expanded = current_v
            if n_rep > 1:
                curr_v_expanded = curr_v_expanded.repeat_interleave(n_rep, dim=1)
            out_curr = torch.matmul(probs_curr.unsqueeze(2), curr_v_expanded).squeeze(2)
            
            attn_output = out_hist + out_curr
            return attn_output

        # =================================================================
        # Path B: Standard Attention
        # =================================================================
        else:
            target_dtype = q_tensor.dtype
            
            # 1. 重建完整 K/V
            if has_int8_cache:
                k_hist = k_cache.to(target_dtype) * k_scales.to(target_dtype)
                v_hist = v_cache.to(target_dtype) * v_scales.to(target_dtype)
            elif isinstance(k_cache, list) and len(k_cache) > 0:
                k_hist = torch.cat([c.to(target_dtype) for c in k_cache], dim=-2)
                v_hist = torch.cat([c.to(target_dtype) for c in v_cache], dim=-2)
            else:
                k_hist = torch.empty(batch_size, num_kv_heads, 0, head_dim, device=q_tensor.device, dtype=target_dtype)
                v_hist = torch.empty(batch_size, num_kv_heads, 0, head_dim, device=q_tensor.device, dtype=target_dtype)
            
            if current_k is not None:
                curr_k_target = current_k.to(target_dtype)
                curr_v_target = current_v.to(target_dtype)

                # [FIX] 檢查維度並自動修正 GQA Mismatch (安全網，以防 model_wrapper 修正失敗)
                if k_hist.shape[1] != curr_k_target.shape[1]:
                    # 只有真的發生錯誤時才 Log，避免洗版
                    logger.warning(
                        f"⚠️ Shape Mismatch in Path B! k_hist={k_hist.shape}, current_k={curr_k_target.shape}. "
                        f"Attempting GQA fix..."
                    )
                    if k_hist.shape[1] == num_heads and curr_k_target.shape[1] == num_kv_heads:
                        curr_k_target = curr_k_target.repeat_interleave(n_rep, dim=1)
                        curr_v_target = curr_v_target.repeat_interleave(n_rep, dim=1)
                        n_rep = 1 
                        logger.warning("✅ Fixed: Expanded current_k/v to match cache heads.")

                k_full = torch.cat([k_hist, curr_k_target], dim=-2)
                v_full = torch.cat([v_hist, curr_v_target], dim=-2)
            else:
                k_full, v_full = k_hist, v_hist
                
            # debug_tensor("k_full", k_full)

            # 2. RoPE
            seq_len = k_full.shape[2]
            if rotary_emb_module is not None:
                cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
                cos_q = cos_q.to(dtype=q_tensor.dtype)
                sin_q = sin_q.to(dtype=q_tensor.dtype)
                q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)
                
                k_pos_ids = torch.arange(seq_len, device=q_tensor.device).unsqueeze(0)
                cos_k, sin_k = rotary_emb_module(k_full, k_pos_ids)
                cos_k = cos_k.to(dtype=k_full.dtype)
                sin_k = sin_k.to(dtype=k_full.dtype)
                k_full, _ = apply_rotary_pos_emb(k_full, k_full, cos_k, sin_k)

            # 3. GQA Repeat
            if n_rep > 1:
                k_full = k_full.repeat_interleave(n_rep, dim=1)
                v_full = v_full.repeat_interleave(n_rep, dim=1)

            # 4. Attention
            scores = torch.matmul(q_tensor, k_full.transpose(-1, -2)) / math.sqrt(head_dim)
            
            if q_len > 1:
                # [FIX] Mask Dtype Fix (BF16)
                min_val = torch.finfo(q_tensor.dtype).min
                mask = torch.triu(
                    torch.full((q_len, seq_len), min_val, device=q_tensor.device, dtype=q_tensor.dtype),
                    diagonal=seq_len - q_len + 1
                )
                scores = scores + mask.unsqueeze(0).unsqueeze(0)
            
            probs = F.softmax(scores, dim=-1)
            output = torch.matmul(probs, v_full)
            
            if q_len == 1:
                output = output.squeeze(2)
                
            return output
        