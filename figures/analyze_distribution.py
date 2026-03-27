import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import gc
from mpl_toolkits.mplot3d import Axes3D  # 用於 3D 繪圖
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# 加入路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_kv_distribution_3d(model_id):
    print(f"Loading model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # 使用 float16 節省記憶體
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return

    # 準備測試文本 (長度適中，約 500-800 tokens 以便觀察趨勢)
    text = "The quick brown fox jumps over the lazy dog. " * 50 
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print("Running forward pass...")
    
    # 這裡我們不需要拿全部的 hidden_states，為了省記憶體，我們只跑一次 forward
    # 但為了拿到每一層的 K/V，我們需要 hook 或者直接遍歷層 (這裡維持你的邏輯，先跑 forward 拿 hidden states)
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    
    all_hidden_states = outputs.hidden_states
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    head_dim = hidden_dim // num_q_heads
    
    # === 設定目標層 ===
    # Llama-2-13B 有 40 層，Llama-3-8B 有 32 層
    # 確保 31 不會超過層數 (如果模型較小，自動調整)
    target_layers = [16, 31]
    target_layers = [l for l in target_layers if l < num_layers]
    
    print(f"Analyzing layers: {target_layers}")
    
    # 儲存要畫圖的 Raw Data (不再是平均值)
    # 結構: {layer_idx: {'pre_k': numpy, 'post_k': numpy, 'v': numpy}}
    captured_data = {}
    
    bsz, seq_len = inputs.input_ids.shape
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    
    # 限制畫圖的 Token 數量，避免圖太擠看不清楚
    plot_seq_len = min(seq_len, 600) 
    
    with torch.no_grad():
        for i in target_layers:
            print(f"Extracting data from Layer {i}...")
            layer = model.model.layers[i]
            
            # 重建該層的輸入
            input_state = all_hidden_states[i].to(layer.self_attn.q_proj.weight.device)
            norm_out = layer.input_layernorm(input_state)
            
            # 投影
            q_proj = layer.self_attn.q_proj(norm_out)
            k_proj = layer.self_attn.k_proj(norm_out)
            v_proj = layer.self_attn.v_proj(norm_out)
            
            q_states = q_proj.view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
            k_states = k_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            # Pre-RoPE Key
            k_pre_rope = k_states
            
            # RoPE 計算
            if hasattr(layer.self_attn, "rotary_emb"):
                rotary_emb = layer.self_attn.rotary_emb
            elif hasattr(model.model, "rotary_emb"):
                rotary_emb = model.model.rotary_emb
            else:
                rotary_emb = lambda x, p: (torch.zeros_like(x), torch.zeros_like(x))
            
            cos, sin = rotary_emb(v_states, position_ids)
            _, k_post_rope = apply_rotary_pos_emb(q_states, k_pre_rope, cos, sin)
            
            # 抓取 Head 0 的資料 (Batch 0, Head 0, :plot_seq_len, :)
            # 取絕對值 (Magnitude)
            data_pre_k = k_pre_rope[0, 0, :plot_seq_len, :].abs().cpu().float().numpy()
            data_post_k = k_post_rope[0, 0, :plot_seq_len, :].abs().cpu().float().numpy()
            data_v = v_states[0, 0, :plot_seq_len, :].abs().cpu().float().numpy()
            
            captured_data[i] = {
                'pre_k': data_pre_k,
                'post_k': data_post_k,
                'v': data_v
            }

    # 釋放模型記憶體，避免繪圖時 OOM
    del model, all_hidden_states, outputs
    torch.cuda.empty_cache()

    print("Plotting 3D results...")
    
    # === 3D 繪圖設定 ===
    fig = plt.figure(figsize=(20, 12))
    # 調整間距
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.92, wspace=0.1, hspace=0.2)
    
    model_short_name = model_id.split("/")[-1]
    fig.suptitle(f"KV Cache Magnitude Distribution (3D Surface) - {model_short_name}", fontsize=16)

    # 準備 Grid: 2 rows (Layer 16, 31), 3 cols (Pre-K, Post-K, V)
    rows = len(target_layers)
    cols = 3
    
    # 定義 X, Y 軸網格
    # X: Channel (0 ~ head_dim), Y: Token (0 ~ plot_seq_len)
    X, Y = np.meshgrid(np.arange(head_dim), np.arange(plot_seq_len))
    
    # 統一 Z 軸高度上限，讓比較更有意義 (可選)
    # z_max = np.max([d['pre_k'].max() for d in captured_data.values()])
    
    for row_idx, layer_idx in enumerate(target_layers):
        layer_data = captured_data[layer_idx]
        
        # 定義這一行的 3 個數據來源
        plots_info = [
            ("Pre-RoPE Key", layer_data['pre_k']),
            ("Post-RoPE Key", layer_data['post_k']),
            ("Value", layer_data['v'])
        ]
        
        for col_idx, (title, data) in enumerate(plots_info):
            # 計算 subplot 位置 (1-based index)
            plot_idx = row_idx * cols + col_idx + 1
            
            ax = fig.add_subplot(rows, cols, plot_idx, projection='3d')
            
            # 繪製 Surface
            # cmap='coolwarm': 藍色低 -> 紅色高 (符合你要的色系)
            # rstride, cstride: 降採樣步長，設大一點畫得快，設 1 最精細
            surf = ax.plot_surface(X, Y, data, cmap='coolwarm', edgecolor='none', alpha=0.9)
            
            # 設定視角 (KIVI 風格視角)
            ax.view_init(elev=35, azim=-60)
            
            # 標題與標籤
            ax.set_title(f"L{layer_idx} {title}", fontsize=12, fontweight='bold')
            if row_idx == 1: # 只在最下面一行標示 X/Y 軸名稱，保持整潔
                ax.set_xlabel('Channel', fontsize=9)
                ax.set_ylabel('Token', fontsize=9)
            
            ax.set_zlabel('Mag', fontsize=9)
            
            # 為了讓 Outlier 更明顯，可以手動設定 Z 軸範圍 (視情況)
            # ax.set_zlim(0, 10) 
            
    # 加入 Colorbar (共用一個，或是每個 Row 一個)
    # 這裡簡單加在圖邊
    # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    output_png = f"kv_dist_3d_{model_short_name}.png"
    plt.savefig(output_png, dpi=150)
    print(f"Saved 3D analysis to {output_png}")

def analyze_early_layers_distribution(model_id):
    print(f"Loading model: {model_id}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="cuda")
    except Exception as e:
        print(f"Error loading model {model_id}: {e}")
        return

    # 使用較長的文本來觀察是否有 Early Token Sink 的現象
    text = "The quick brown fox jumps over the lazy dog. " * 50 
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
    
    all_hidden_states = outputs.hidden_states
    
    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    head_dim = hidden_dim // num_q_heads
    
    # === [重點修改] 觀察前幾層的變化 ===
    # 我們密集觀察 0, 1, 2，然後跳到 4, 8 看看差異
    target_layers = [0, 1, 2, 4, 8]
    target_layers = [l for l in target_layers if l < num_layers]
    
    print(f"Analyzing Early Layers: {target_layers}")
    
    captured_data = {}
    bsz, seq_len = inputs.input_ids.shape
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0)
    plot_seq_len = min(seq_len, 400) # 只看前 400 個 token
    
    with torch.no_grad():
        for i in target_layers:
            print(f"Extracting data from Layer {i}...")
            layer = model.model.layers[i]
            
            input_state = all_hidden_states[i].to(layer.self_attn.q_proj.weight.device)
            norm_out = layer.input_layernorm(input_state)
            
            # 投影 Key 和 Value
            k_proj = layer.self_attn.k_proj(norm_out)
            v_proj = layer.self_attn.v_proj(norm_out)
            q_proj = layer.self_attn.q_proj(norm_out) # 需要 Q 來算 RoPE (雖然這裡只存 K/V)

            q_states = q_proj.view(bsz, seq_len, num_q_heads, head_dim).transpose(1, 2)
            k_states = k_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            v_states = v_proj.view(bsz, seq_len, num_kv_heads, head_dim).transpose(1, 2)
            
            k_pre_rope = k_states
            
            # RoPE
            if hasattr(layer.self_attn, "rotary_emb"):
                rotary_emb = layer.self_attn.rotary_emb
            elif hasattr(model.model, "rotary_emb"):
                rotary_emb = model.model.rotary_emb
            else:
                rotary_emb = lambda x, p: (torch.zeros_like(x), torch.zeros_like(x))
            
            cos, sin = rotary_emb(v_states, position_ids)
            _, k_post_rope = apply_rotary_pos_emb(q_states, k_pre_rope, cos, sin)
            
            # 抓取 Head 0 的 Magnitude
            # 我們特別關注 Key (Pre-RoPE)，因為那是 Outlier 的源頭
            data_pre_k = k_pre_rope[0, 0, :plot_seq_len, :].abs().cpu().float().numpy()
            data_v = v_states[0, 0, :plot_seq_len, :].abs().cpu().float().numpy()
            
            captured_data[i] = {
                'pre_k': data_pre_k,
                'v': data_v
            }

    del model, all_hidden_states, outputs
    torch.cuda.empty_cache()

    print("Plotting Early Layers Analysis...")
    
    # === 繪圖設定: 每一層一列 (Row)，左邊 Key，右邊 Value ===
    rows = len(target_layers)
    cols = 2 
    fig = plt.figure(figsize=(16, 4 * rows)) # 高度隨層數增加
    
    X, Y = np.meshgrid(np.arange(head_dim), np.arange(plot_seq_len))
    
    for row_idx, layer_idx in enumerate(target_layers):
        layer_data = captured_data[layer_idx]
        
        # Plot 1: Key (Pre-RoPE)
        ax1 = fig.add_subplot(rows, cols, row_idx * 2 + 1, projection='3d')
        surf1 = ax1.plot_surface(X, Y, layer_data['pre_k'], cmap='coolwarm', edgecolor='none', alpha=0.9)
        ax1.view_init(elev=35, azim=-60)
        ax1.set_title(f"L{layer_idx} Key (Pre-RoPE)", fontsize=12, fontweight='bold')
        ax1.set_zlabel('Mag')
        if row_idx == rows - 1:
            ax1.set_xlabel('Channel')
            ax1.set_ylabel('Token')

        # Plot 2: Value
        ax2 = fig.add_subplot(rows, cols, row_idx * 2 + 2, projection='3d')
        surf2 = ax2.plot_surface(X, Y, layer_data['v'], cmap='viridis', edgecolor='none', alpha=0.9)
        ax2.view_init(elev=35, azim=-60)
        ax2.set_title(f"L{layer_idx} Value", fontsize=12, fontweight='bold')
        ax2.set_zlabel('Mag')
        if row_idx == rows - 1:
            ax2.set_xlabel('Channel')
            ax2.set_ylabel('Token')

    model_short = model_id.split("/")[-1]
    plt.tight_layout()
    plt.savefig(f"early_layers_dist_{model_short}.png", dpi=120)
    print(f"Saved analysis to early_layers_dist_{model_short}.png")

if __name__ == "__main__":
    # 你可以只留 Llama-3.1-8B
    models_to_compare = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Llama-2-7b-hf", 
    ]
    
    print("=== Starting 3D Distribution Analysis ===")
    for model_id in models_to_compare:
        analyze_early_layers_distribution(model_id)
        # analyze_kv_distribution_3d(model_id)
        gc.collect()
        torch.cuda.empty_cache()
        
    print("\nAll done!")