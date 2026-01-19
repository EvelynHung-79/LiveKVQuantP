import torch
import unittest
import math
from livekvquant.modules.kv_cache_manager import KVCacheManager
from livekvquant.modules.attention_core import AttentionCore
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# === Mock Config ===
class MockConfig:
    def __init__(self):
        self.num_attention_heads = 32
        self.num_key_value_heads = 32
        self.hidden_size = 4096
        self.head_dim = 128
        self.outlier_ratio = 0.01
        self.bits = 8

# === Mock Rotary Embedding ===
class MockRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim, max_pos=4096, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
        t = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos().to(dtype)
        self.sin_cached = emb.sin().to(dtype)

    def forward(self, x, position_ids=None, seq_len=None):
        if seq_len is not None:
            return self.cos_cached, self.sin_cached
        return self.cos_cached[position_ids].squeeze(0), self.sin_cached[position_ids].squeeze(0)

class TestFusedKernel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.test_dtype = torch.bfloat16

        if self.device == "cpu":
            print("Warning: Triton kernels require GPU. Tests will run on CPU path only.")

        self.config = MockConfig()
        # 初始化 Manager
        self.kv_manager = KVCacheManager(self.config, max_seq_len=2048)
        self.attn_core = AttentionCore(self.config)
        self.rope = MockRotaryEmbedding(self.config.head_dim, device=self.device, dtype=self.test_dtype)

    def test_decoding_attention_correctness(self):
        """
        測試 Decoding 階段 (Q_len=1, KV_len=Long) 的正確性。
        這會觸發 Triton Fused Kernel (如果有的話)。
        """
        if self.device == "cpu":
            return

        print("\n=== Start Testing Fused Kernel (Decoding Mode) ===")

        # 1. 設定參數
        batch_size = 1
        num_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim
        dtype = self.test_dtype
        
        past_seq_len = 128
        q_seq_len = 1  
        
        # 2. 準備資料
        q = torch.randn(batch_size, num_heads, q_seq_len, head_dim, device=self.device, dtype=dtype)
        
        k_hist = torch.randn(batch_size, num_heads, past_seq_len, head_dim, device=self.device, dtype=dtype)
        v_hist = torch.randn(batch_size, num_heads, past_seq_len, head_dim, device=self.device, dtype=dtype)
        
        # [FIX] 配合新的 Per-Channel Key Quantization
        # K Scale: [B, H, 1, D] (Per-Channel AbsMax)
        # 注意 dim=-2 reduction
        k_absmax = k_hist.abs().max(dim=-2, keepdim=True).values
        k_scale = k_absmax / 127.0
        k_scale = torch.clamp(k_scale, min=1e-5)
        # Broadcast scale to [B, H, Seq, D] for quantization
        k_int8 = (k_hist / k_scale).round().clamp(-127, 127).to(torch.int8)
        
        # V Scale: [B, H, Seq, 1] (Per-Token, 維持不變)
        v_absmax = v_hist.abs().max(dim=-1, keepdim=True).values 
        v_scale = v_absmax / 127.0
        v_scale = torch.clamp(v_scale, min=1e-5)
        v_int8 = (v_hist / v_scale).round().clamp(-127, 127).to(torch.int8)
        
        self.kv_manager.clear()
        
        # 準備要存入的 Chunk
        k_chunk = {
            "quantized_data": k_int8,
            "scale": k_scale,
            "sparse_values": torch.empty(0, device=self.device),
            "sparse_indices": torch.empty(0, device=self.device),
            "type": "quantized"
        }
        v_chunk = {
            "quantized_data": v_int8,
            "scale": v_scale,
            "sparse_values": torch.empty(0, device=self.device),
            "sparse_indices": torch.empty(0, device=self.device),
            "type": "quantized"
        }
        
        # 這裡會觸發新的 buffer 機制
        self.kv_manager.store_chunk(k_chunk, v_chunk)

        position_ids = torch.tensor([[past_seq_len]], device=self.device)

        # ==================================================
        # 3. Ground Truth (PyTorch 全精度模擬)
        # ==================================================
        # 還原時要用剛剛算好的正確 scale
        k_hist_dequant = k_int8.to(dtype) * k_scale
        v_hist_dequant = v_int8.to(dtype) * v_scale
        
        # RoPE Q (Current)
        cos_q, sin_q = self.rope(q, position_ids)
        cos_q = cos_q.unsqueeze(0)
        sin_q = sin_q.unsqueeze(0)
        q_rope, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        
        # RoPE K (History)
        k_pos_ids = torch.arange(0, past_seq_len, device=self.device).unsqueeze(0)
        cos_k, sin_k = self.rope(k_hist_dequant, k_pos_ids)
        cos_k = cos_k.unsqueeze(0)
        sin_k = sin_k.unsqueeze(0)
        
        k_hist_rope, _ = apply_rotary_pos_emb(k_hist_dequant, k_hist_dequant, cos_k, sin_k)
        
        # Attention (Q * K^T)
        attn_scores = torch.matmul(q_rope, k_hist_rope.transpose(-1, -2)) / math.sqrt(head_dim)
        attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1).to(dtype)
        
        # Output (Prob * V)
        output_ref = torch.matmul(attn_probs, v_hist_dequant)
        
        # ==================================================
        # 4. Fused Kernel Execution
        # ==================================================
        # 這會去呼叫新的 get_cache_view() 並跑 Triton
        output_fused = self.attn_core.compute_attention(
            q, self.kv_manager, 
            current_k=None, current_v=None,
            rotary_emb_module=self.rope, 
            position_ids=position_ids
        )
        
        # ==================================================
        # 5. 驗證結果
        # ==================================================
        print(f"\n[DEBUG INFO]")
        print(f"Ref Output Shape: {output_ref.shape}")
        print(f"Fused Output Shape: {output_fused.shape}")
        
        # Cosine Similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            output_ref.flatten().to(torch.float32), 
            output_fused.flatten().to(torch.float32), 
            dim=0
        )
        print(f"Cosine Similarity: {cos_sim.item():.6f}")
        
        self.assertTrue(cos_sim.item() > 0.99, f"Kernel accuracy issue! Sim: {cos_sim.item()}")
        print(">>> Test Passed: Triton Kernel matches PyTorch logic.")

if __name__ == "__main__":
    unittest.main()