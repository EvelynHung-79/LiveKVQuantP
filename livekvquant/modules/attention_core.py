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
        對應論文 3.6.2 Data Assembly and Dequantization.
        """
        if chunk_data["type"] == "warmup":
            return chunk_data["data"] # 直接回傳 FP16 (Warmup 存的是 Pre-RoPE)
        
        # Quantized Chunk 重建
        # 1. Dequantization 
        # X_dequant = X_quant * s
        dequantized = dequantize_symmetric(
            chunk_data["quantized_data"], 
            chunk_data["scale"]
        )
        
        # 2. Outlier Integration 
        # 將 Sparse Outliers 填回
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
        執行標準 Scaled Dot-Product Attention。
        對應 Eq 3-10, 3-11, 3-12.
        
        Args:
            q_tensor: 當前 Chunk 的 Query [batch, heads, seq, head_dim]。
                      注意：這個 Q 已經在 model_wrapper 中做過 RoPE 了。
            kv_manager: 儲存了歷史 KV 的 Manager。
            rotary_emb: (cos, sin) tuple，用於對重建後的 K 補做 RoPE。
        """
        k_chunks, v_chunks = kv_manager.get_all_chunks()
        
        # 1. 重建所有歷史 KV (包含當前 Chunk) 
        # 重建出來的 K 是 Pre-RoPE 的 (Raw Key)
        k_list = [self._reconstruct_tensor(c) for c in k_chunks]
        v_list = [self._reconstruct_tensor(c) for c in v_chunks]
        
        # 2. Concatenate 
        # k_full: [batch, num_key_heads, total_seq_len, head_dim]
        k_full = torch.cat(k_list, dim=-2)
        v_full = torch.cat(v_list, dim=-2)
        
        # 3. [核心修正] 動態生成全長的 RoPE
        if rotary_emb_module is not None:
            # 取得目前的總序列長度 (包含歷史 + 當前 chunk)
            total_seq_len = k_full.shape[-2]
            
            # 使用 LlamaRotaryEmbedding 生成對應長度的 cos/sin
            # 這裡我們傳入 k_full 作為 device/dtype 的參考
            cos, sin = rotary_emb_module(k_full, seq_len=total_seq_len)
            
            # 執行旋轉
            # q 傳入 None 即可，因為我們只旋轉 k_full
            # apply_rotary_pos_emb 回傳 (q_rot, k_rot)
            _, k_full = apply_rotary_pos_emb(None, k_full, cos, sin)
        
        # 4. Final Attention Operation (全部在 FP16 執行) 
        # Attn = Softmax(Q * K^T / sqrt(d)) * V
        head_dim = q_tensor.size(-1)
        scale = 1.0 / math.sqrt(head_dim)
        
        # [batch, heads, q_len, k_len]
        # 這裡的 Q (Rotated) 與 K (Rotated) 終於可以在同一個空間計算了
        attn_scores = torch.matmul(q_tensor, k_full.transpose(-2, -1)) * scale
        
        # 這裡假設是 Causal LM，需要加上 Causal Mask
        # 實作上通常會傳入 attention_mask，這裡為了簡化省略
        # 若 q_len > 1 (Prefill)，你需要在此處 apply causal mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # [batch, heads, q_len, head_dim]
        attn_output = torch.matmul(attn_probs, v_full)
        
        return attn_output