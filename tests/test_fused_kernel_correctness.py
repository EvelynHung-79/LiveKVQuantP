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
    # [FIX] 加入 dtype 參數，預設使用 bfloat16 以匹配測試環境
    def __init__(self, head_dim, max_pos=4096, device="cuda", dtype=torch.bfloat16):
        super().__init__()
        self.dim = head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float().to(device) / head_dim))
        t = torch.arange(max_pos, device=device, dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # [FIX] 使用傳入的 dtype (bfloat16)
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
        if self.device == "cpu":
            print("Warning: Triton kernels require GPU. Skipping tests.")
            return
        
        # [FIX] 統一測試用的 dtype
        self.test_dtype = torch.bfloat16

        self.config = MockConfig()
        self.kv_manager = KVCacheManager(self.config, max_seq_len=2048, max_batch_size=1)
        self.attn_core = AttentionCore(self.config)
        
        # [FIX] 初始化 RoPE 時指定 dtype=bfloat16
        self.rope = MockRotaryEmbedding(self.config.head_dim, device=self.device, dtype=self.test_dtype)

    def test_key_fused_correctness(self):
        if self.device == "cpu": return

        print("\n=== Start Testing Fused Kernel (Hybrid Mode) ===")

        # 1. 準備資料
        batch_size = 1
        total_seq_len = 128
        hist_len = 127
        head_dim = self.config.head_dim
        num_heads = self.config.num_attention_heads
        dtype = self.test_dtype
        
        print(f"[DEBUG] Test Dtype: {dtype}")
        
        # Q: [1, 32, 128] (Current Token)
        q = torch.randn(batch_size, num_heads, head_dim, device=self.device, dtype=dtype)
        
        # Full Data (History + Current)
        k_full = torch.randn(batch_size, num_heads, total_seq_len, head_dim, device=self.device, dtype=dtype)
        v_full = torch.randn(batch_size, num_heads, total_seq_len, head_dim, device=self.device, dtype=dtype)
        
        # Splitting Data for Hybrid Testing
        k_hist = k_full[:, :, :hist_len, :]
        v_hist = v_full[:, :, :hist_len, :]
        k_curr = k_full[:, :, hist_len:, :]
        v_curr = v_full[:, :, hist_len:, :]
        position_ids = torch.tensor([[hist_len]], device=self.device)
        
        # ==================================================
        # 2. Ground Truth (Full BF16 PyTorch)
        # ==================================================
        print("\n[DEBUG] --- Calculating Ground Truth ---")
        
        # RoPE Q
        cos_q, sin_q = self.rope(q, position_ids)
        print(f"[DEBUG] cos_q dtype: {cos_q.dtype}") # Checkpoint 1
        
        q_ref, _ = apply_rotary_pos_emb(q, q, cos_q, sin_q)
        print(f"[DEBUG] q_ref dtype: {q_ref.dtype}") # Checkpoint 2
        
        # RoPE K (Full)
        k_pos_ids = torch.arange(0, total_seq_len, device=self.device).unsqueeze(0)
        cos_k, sin_k = self.rope(k_full, k_pos_ids)
        cos_k = cos_k.unsqueeze(0)
        sin_k = sin_k.unsqueeze(0)
        k_ref_rope, _ = apply_rotary_pos_emb(k_full, k_full, cos_k, sin_k)
        
        # Attention
        scores_ref = torch.matmul(q_ref.unsqueeze(2), k_ref_rope.transpose(-1, -2)) / math.sqrt(head_dim)
        scores_ref = scores_ref.squeeze(2)
        print(f"[DEBUG] scores_ref dtype: {scores_ref.dtype}") # Checkpoint 3
        
        # Output
        probs_ref = torch.nn.functional.softmax(scores_ref, dim=-1)
        print(f"[DEBUG] probs_ref dtype: {probs_ref.dtype}") # Checkpoint 4
        print(f"[DEBUG] v_full dtype: {v_full.dtype}")       # Checkpoint 5

        # [FIX] 如果這裡還是 Float32 (雖然改了 Mock 應該不會)，強制轉型保護
        if probs_ref.dtype != dtype:
            print(f"[WARN] probs_ref mismatch! Casting {probs_ref.dtype} -> {dtype}")
            probs_ref = probs_ref.to(dtype)

        output_ref = torch.matmul(probs_ref.unsqueeze(2), v_full).squeeze(2)

        # ==================================================
        # 3. Fused Kernel Setup (Hybrid)
        # ==================================================
        print("\n[DEBUG] --- Running Fused Kernel ---")
        
        k_scale = torch.rand_like(k_hist, dtype=dtype) + 0.01
        k_int8 = (k_hist / k_scale).round().clamp(-127, 127).to(torch.int8)
        
        v_absmax = v_hist.abs().max(dim=3, keepdim=True).values 
        v_scale = v_absmax / 127.0
        v_int8 = (v_hist / v_scale).round().clamp(-127, 127).to(torch.int8)
        
        self.kv_manager.clear()
        self.kv_manager.store_tokens(k_int8, k_scale, v_int8, v_scale, valid_len=hist_len)
        
        # ==================================================
        # 4. Execute (Hybrid Call)
        # ==================================================
        output_fused = self.attn_core.compute_attention(
            q, self.kv_manager, 
            current_k=k_curr, current_v=v_curr,
            rotary_emb_module=self.rope, 
            position_ids=position_ids
        )
        
        # ==================================================
        # 5. Compare
        # ==================================================
        if output_fused.ndim == 4: output_fused = output_fused.squeeze(2)

        print(f"\n[DEBUG INFO]")
        print(f"Ref Output Shape: {output_ref.shape}")
        print(f"Fused Output Shape: {output_fused.shape}")
        print(f"Ref Output Mean: {output_ref.abs().mean().item():.4f}")
        print(f"Fused Output Mean: {output_fused.abs().mean().item():.4f}")
        
        # [FIX] Cosine Similarity 在 BF16 有時精度不夠計算，轉 float32 算相似度比較準
        cos_sim = torch.nn.functional.cosine_similarity(
            output_ref.flatten().to(torch.float32), 
            output_fused.flatten().to(torch.float32), 
            dim=0
        )
        print(f"Cosine Similarity: {cos_sim.item():.4f}")
        
        self.assertTrue(cos_sim.item() > 0.95, f"Similarity too low: {cos_sim.item()}")

if __name__ == "__main__":
    unittest.main()