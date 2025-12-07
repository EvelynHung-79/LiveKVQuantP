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
        rotary_emb_module = None
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
        if rotary_emb_module is not None:
            total_seq_len = k_full.shape[-2]
            position_ids = torch.arange(
                0, total_seq_len, device=k_full.device, dtype=torch.long
            ).unsqueeze(0)
            
            cos, sin = rotary_emb_module(k_full, position_ids)
            _, k_full = apply_rotary_pos_emb(k_full, k_full, cos, sin)
        
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
        # Llama 是 Causal Model，Query 只能看到它自己以前的 Key
        # 當 Q_len > 1 時 (Prefill 階段)，我們必須遮住右上角 (Future tokens)
        
        q_len = q_tensor.size(-2)
        k_len = k_full.size(-2)
        
        # 只有在 Prefill (q_len > 1) 時才需要複雜的 Mask
        # Decoding (q_len == 1) 時，Q 是最後一個 token，它可以看所有 K，無需 Mask
        if q_len > 1:
            # 建立一個全 -inf 的矩陣
            # causal_mask: 下三角為 0 (可見)，上三角為 -inf (不可見)
            # triu(diagonal=1) 會把對角線以上設為 True
            mask = torch.full((q_len, q_len), float("-inf"), device=q_tensor.device)
            mask = torch.triu(mask, diagonal=1)
            
            # 我們的 K_full 包含 [History (Past Chunks), Current Chunk]
            # History 部分是全開的 (全 0)，Current 部分需要 Causal Mask
            past_len = k_len - q_len
            
            if past_len > 0:
                # 歷史部分全 0
                history_mask = torch.zeros((q_len, past_len), device=q_tensor.device)
                # 拼接：[History (0), Current (Causal)]
                full_mask = torch.cat([history_mask, mask], dim=-1)
            else:
                full_mask = mask
            
            # 加入 Mask (廣播到 Batch 和 Heads)
            attn_scores += full_mask.unsqueeze(0).unsqueeze(0)
        # =================================
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output