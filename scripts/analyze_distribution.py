import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# 加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_kv_distribution(model_id):
    print(f"Loading model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 使用 float16 或 bfloat16
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return

    # 準備測試文本
    text = "This is a long text to analyze the distribution of Key and Value caches. " * 50
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print("Running full model forward pass to capture hidden states...")
    
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    
    all_hidden_states = outputs.hidden_states
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    head_dim = hidden_dim // num_q_heads
    
    print(f"Model Config: Layers={num_layers}, KV_Heads={num_kv_heads}, Head_Dim={head_dim}")
    
    # 儲存統計數據
    stats_sum = torch.zeros((num_layers, 3, head_dim), device="cuda")
    
    print("Analyzing layers...")
    
    bsz, seq_len = inputs.input_ids.shape
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    
    with torch.no_grad():
        for i in range(num_layers):
            layer = model.model.layers[i]
            
            input_state = all_hidden_states[i].to(layer.self_attn.q_proj.weight.device)
            norm_out = layer.input_layernorm(input_state)
            
            q_proj = layer.self_attn.q_proj(norm_out)
            k_proj = layer.self_attn.k_proj(norm_out)
            v_proj = layer.self_attn.v_proj(norm_out)
            
            q_states = q_proj.view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
            k_states = k_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            k_pre_rope = k_states
            
            if hasattr(layer.self_attn, "rotary_emb"):
                rotary_emb = layer.self_attn.rotary_emb
            elif hasattr(model.model, "rotary_emb"):
                rotary_emb = model.model.rotary_emb
            else:
                rotary_emb = lambda x, p: (torch.zeros_like(x), torch.zeros_like(x)) 
            
            cos, sin = rotary_emb(v_states, position_ids)
            _, k_post_rope = apply_rotary_pos_emb(q_states, k_pre_rope, cos, sin)
            
            # 統計 Head 0
            stats_sum[i, 0] += torch.sum(torch.abs(k_pre_rope[:, 0, :, :]), dim=(0, 1)).squeeze() # K Pre
            stats_sum[i, 1] += torch.sum(torch.abs(k_post_rope[:, 0, :, :]), dim=(0, 1)).squeeze() # K Post
            stats_sum[i, 2] += torch.sum(torch.abs(v_states[:, 0, :, :]), dim=(0, 1)).squeeze()    # V

    print("Plotting results...")
    avg_stats = stats_sum / (bsz * seq_len)
    avg_stats = avg_stats.cpu().float().numpy()
    
    # === [修改重點] ===
    # 1. 選擇 0, 16, 31 層
    target_layers = [0, 16, 31]
    
    # 2. 調整畫布大小 (高度增加到 15，因為有 3 行)
    plt.figure(figsize=(18, 15))
    
    model_short_name = model_id.split("/")[-1]
    
    for idx, layer_idx in enumerate(target_layers):
        # 3. subplot 改成 (3, 3, ...) 代表 3 行 3 列
        
        # 1. Key Pre-RoPE
        plt.subplot(3, 3, idx*3 + 1)
        data = avg_stats[layer_idx, 0, :] 
        plt.bar(range(head_dim), data, color='green', alpha=0.7)
        plt.title(f"[{model_short_name}] L{layer_idx} Key (Pre-RoPE) Head 0")
        plt.xlabel("Channel")
        plt.ylabel("Mean Abs")
        plt.grid(True, alpha=0.3)

        # 2. Key Post-RoPE
        plt.subplot(3, 3, idx*3 + 2)
        data = avg_stats[layer_idx, 1, :] 
        plt.bar(range(head_dim), data, color='blue', alpha=0.7)
        plt.title(f"[{model_short_name}] L{layer_idx} Key (Post-RoPE) Head 0")
        plt.xlabel("Channel")
        plt.grid(True, alpha=0.3)
        
        # 3. Value
        plt.subplot(3, 3, idx*3 + 3)
        data = avg_stats[layer_idx, 2, :] 
        plt.bar(range(head_dim), data, color='orange', alpha=0.7)
        plt.title(f"[{model_short_name}] L{layer_idx} Value Head 0")
        plt.xlabel("Channel")
        plt.grid(True, alpha=0.3)
        
    plt.tight_layout()
    output_png = f"kv_dist_{model_short_name}.png"
    plt.savefig(output_png)
    print(f"Saved analysis to {output_png}")

import gc
import torch  # 確保有 import torch

if __name__ == "__main__":
    models_to_compare = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf",  # 13B 很大，這一步特別容易爆
        "meta-llama/Meta-Llama-3.1-8B-Instruct"
    ]
    
    print("=== Starting Comparative Analysis ===")
    for model_id in models_to_compare:
        print(f"\nProcessing {model_id}...")
        analyze_kv_distribution(model_id)

        # 清理記憶體
        print(f"Cleaning up memory after {model_id}...")
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\nAll done! Check the generated .png files.")