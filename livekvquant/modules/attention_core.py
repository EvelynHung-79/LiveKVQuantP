import torch
import torch.nn.functional as F
import math
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from ..utils.quant_utils import dequantize_symmetric
from ..utils.outliers import restore_outliers

try:
    from .kernels import fused_kv_attention, fused_value_attention
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

class AttentionCore:
    def __init__(self, config):
        self.config = config

    def _reconstruct_tensor(self, chunk_data: dict, target_dtype: torch.dtype) -> torch.Tensor:
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
        
        # 1. Triton Kernel 加速路徑
        # 條件：有 Triton、在 GPU、Decoding 階段 (q_len=1)、且沒有遺留的 current_k
        use_triton = TRITON_AVAILABLE and q_tensor.is_cuda and (q_tensor.shape[-2] == 1)
        
        if use_triton:
            # 若有 current_k (通常發生在 prefill 剛結束)，目前簡單策略是 fallback
            # 除非我們在這裡也即時存入 buffer (但通常 current_k 應由 model_wrapper 處理)
            if current_k is None and kv_manager.current_len > 0:
                # [FAST PATH] 直接取得 Buffer View (Zero-Copy)
                k_cache, k_scale, v_cache, v_scale = kv_manager.get_cache_view()

                # A. 對 Q 進行 RoPE
                if rotary_emb_module is not None and position_ids is not None:
                    cos_q, sin_q = rotary_emb_module(q_tensor, position_ids)
                    q_tensor, _ = apply_rotary_pos_emb(q_tensor, q_tensor, cos_q, sin_q)

                    # B. 準備 Kernel 用的 RoPE (對應 K 的長度)
                    total_len = k_cache.shape[2]
                    k_pos_ids = torch.arange(0, total_len, device=q_tensor.device).unsqueeze(0)
                    cos_k, sin_k = rotary_emb_module(k_cache, k_pos_ids)
                    
                    # 壓縮維度至 [Seq, Dim]
                    while cos_k.dim() > 2:
                        cos_k = cos_k.squeeze(0)
                        sin_k = sin_k.squeeze(0)
                else:
                    cos_k = torch.ones((k_cache.shape[2], q_tensor.shape[-1]), device=q_tensor.device)
                    sin_k = torch.zeros_like(cos_k)

                # C. 執行 Kernel
                q_in = q_tensor.squeeze(2) if q_tensor.dim() == 4 else q_tensor
                batch, heads, dim = q_in.shape
                seq_len = k_cache.shape[2]
                
                scores_out = torch.empty((batch, heads, seq_len), device=q_tensor.device, dtype=torch.float32)
                
                # Kernel 1: Q * K
                fused_kv_attention(q_in, k_cache, k_scale, cos_k, sin_k, scores_out)
                
                # Softmax
                attn_probs = F.softmax(scores_out, dim=-1).to(q_tensor.dtype)
                
                # Kernel 2: Score * V
                output = fused_value_attention(attn_probs, v_cache, v_scale)
                
                return output.unsqueeze(2)

        # 2. PyTorch Fallback 路徑 (保留原樣以支援 Prefill 或 CPU)
        k_chunks, v_chunks = kv_manager.get_all_chunks()
        target_dtype = q_tensor.dtype
        
        k_list = [self._reconstruct_tensor(c, target_dtype) for c in k_chunks]
        v_list = [self._reconstruct_tensor(c, target_dtype) for c in v_chunks]
        
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