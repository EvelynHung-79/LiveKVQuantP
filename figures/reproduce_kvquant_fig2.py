"""
Reproduce KVQuant Figure 2: 3D surface plots of Key (Pre-RoPE), Key (Post-RoPE), Value
distributions for different LLaMA models.

Usage:
    python figures/reproduce_kvquant_fig2.py [--models MODEL1 MODEL2 ...] [--layer LAYER]
"""

import sys
import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import gc
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from datasets import load_dataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def get_wikitext2_sample(tokenizer, target_seq_len=2048):
    """Load a sample from Wikitext-2 and tokenize to ~2K tokens, matching KVQuant paper."""
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Concatenate text until we have enough tokens
    full_text = ""
    for item in dataset:
        full_text += item["text"] + " "
        tokens = tokenizer(full_text, return_tensors="pt")
        if tokens.input_ids.shape[1] >= target_seq_len:
            break
    tokens = tokenizer(full_text, return_tensors="pt", max_length=target_seq_len, truncation=True)
    return tokens.to("cuda")


def extract_kv_data(model, tokenizer, layer_idx, seq_len=2048, head_idx=0):
    """Extract Pre-RoPE Key, Post-RoPE Key, and Value data for a specific layer."""
    inputs = get_wikitext2_sample(tokenizer, target_seq_len=seq_len)
    actual_seq_len = inputs.input_ids.shape[1]
    print(f"  Input sequence length: {actual_seq_len}")

    num_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    num_q_heads = model.config.num_attention_heads
    num_kv_heads = getattr(model.config, "num_key_value_heads", num_q_heads)
    head_dim = hidden_dim // num_q_heads

    if layer_idx >= num_layers:
        layer_idx = num_layers // 2
        print(f"  Adjusted layer index to {layer_idx} (model has {num_layers} layers)")

    bsz, seq_len_actual = inputs.input_ids.shape
    position_ids = torch.arange(seq_len_actual, device="cuda").unsqueeze(0)

    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True)
        all_hidden_states = outputs.hidden_states

        layer = model.model.layers[layer_idx]
        input_state = all_hidden_states[layer_idx].to(layer.self_attn.q_proj.weight.device)
        norm_out = layer.input_layernorm(input_state)

        q_proj = layer.self_attn.q_proj(norm_out)
        k_proj = layer.self_attn.k_proj(norm_out)
        v_proj = layer.self_attn.v_proj(norm_out)

        q_states = q_proj.view(bsz, seq_len_actual, num_q_heads, head_dim).transpose(1, 2)
        k_states = k_proj.view(bsz, seq_len_actual, num_kv_heads, head_dim).transpose(1, 2)
        v_states = v_proj.view(bsz, seq_len_actual, num_kv_heads, head_dim).transpose(1, 2)

        k_pre_rope = k_states.clone()

        # RoPE
        if hasattr(layer.self_attn, "rotary_emb"):
            rotary_emb = layer.self_attn.rotary_emb
        elif hasattr(model.model, "rotary_emb"):
            rotary_emb = model.model.rotary_emb
        else:
            rotary_emb = lambda x, p: (torch.zeros_like(x), torch.zeros_like(x))

        cos, sin = rotary_emb(v_states, position_ids)
        _, k_post_rope = apply_rotary_pos_emb(q_states, k_pre_rope, cos, sin)

        # Extract Head 0, take absolute values (magnitude)
        data_pre_k = k_pre_rope[0, head_idx, :, :].abs().cpu().float().numpy()
        data_post_k = k_post_rope[0, head_idx, :, :].abs().cpu().float().numpy()
        data_v = v_states[0, head_idx, :, :].abs().cpu().float().numpy()

    del outputs, all_hidden_states
    torch.cuda.empty_cache()

    return {
        "pre_k": data_pre_k,
        "post_k": data_post_k,
        "v": data_v,
        "layer_idx": layer_idx,
        "head_dim": head_dim,
        "seq_len": seq_len_actual,
    }


def plot_single_model_fig2(data, model_name, output_path):
    """Plot KVQuant Figure 2 style: 1 row, 3 columns (Pre-K, Post-K, V)."""
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle(
        f"KV Cache Activation Distribution — {model_name} (Layer {data['layer_idx']})",
        fontsize=14, fontweight="bold",
    )

    head_dim = data["head_dim"]
    seq_len = data["seq_len"]
    X, Y = np.meshgrid(np.arange(head_dim), np.arange(seq_len))

    titles = ["Keys (Pre-RoPE)", "Keys (Post-RoPE)", "Values"]
    keys = ["pre_k", "post_k", "v"]
    cmaps = ["coolwarm", "coolwarm", "coolwarm"]

    for i, (title, key, cmap) in enumerate(zip(titles, keys, cmaps)):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        surf = ax.plot_surface(X, Y, data[key], cmap=cmap, edgecolor="none", alpha=0.9,
                               rstride=max(1, seq_len // 200), cstride=max(1, head_dim // 64))
        ax.view_init(elev=35, azim=-60)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Channel", fontsize=9)
        ax.set_ylabel("Token", fontsize=9)
        ax.set_zlabel("Magnitude", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison(all_data, output_path):
    """Plot all models side by side for comparison: N rows (models) x 3 cols (Pre-K, Post-K, V)."""
    n_models = len(all_data)
    fig = plt.figure(figsize=(18, 6 * n_models))
    fig.suptitle(
        "KV Cache Activation Distribution Comparison (KVQuant Fig.2 Style)",
        fontsize=16, fontweight="bold", y=0.98,
    )

    titles = ["Keys (Pre-RoPE)", "Keys (Post-RoPE)", "Values"]
    keys = ["pre_k", "post_k", "v"]

    for row_idx, (model_name, data) in enumerate(all_data.items()):
        head_dim = data["head_dim"]
        seq_len = data["seq_len"]
        X, Y = np.meshgrid(np.arange(head_dim), np.arange(seq_len))

        for col_idx, (title, key) in enumerate(zip(titles, keys)):
            ax = fig.add_subplot(n_models, 3, row_idx * 3 + col_idx + 1, projection="3d")
            ax.plot_surface(X, Y, data[key], cmap="coolwarm", edgecolor="none", alpha=0.9,
                            rstride=max(1, seq_len // 200), cstride=max(1, head_dim // 64))
            ax.view_init(elev=35, azim=-60)
            subtitle = f"{model_name}\nL{data['layer_idx']} {title}"
            ax.set_title(subtitle, fontsize=11, fontweight="bold")
            ax.set_xlabel("Channel", fontsize=8)
            ax.set_ylabel("Token", fontsize=8)
            ax.set_zlabel("Mag", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Reproduce KVQuant Figure 2")
    parser.add_argument("--models", nargs="+", default=[
        "huggyllama/llama-7b",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
    ], help="Model IDs to analyze")
    parser.add_argument("--layer", type=int, default=16,
                        help="Layer index to analyze (default: 16, middle layer)")
    parser.add_argument("--seq_len", type=int, default=2048,
                        help="Sequence length (default: 2048, matching KVQuant paper)")
    parser.add_argument("--head", type=int, default=0,
                        help="Attention head index to visualize (default: 0)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = {}

    for model_id in args.models:
        model_short = model_id.split("/")[-1]
        print(f"\n{'='*60}")
        print(f"Processing: {model_id}")
        print(f"{'='*60}")

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map="auto"
            )
        except Exception as e:
            print(f"  ERROR loading {model_id}: {e}")
            print(f"  Skipping this model.")
            continue

        data = extract_kv_data(model, tokenizer, args.layer, args.seq_len, args.head)

        # Save individual figure
        output_path = os.path.join(OUTPUT_DIR, f"kvquant_fig2_{model_short}_layer{data['layer_idx']}.png")
        plot_single_model_fig2(data, model_short, output_path)

        all_data[model_short] = data

        # Free GPU memory
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Save comparison figure
    if len(all_data) > 1:
        comparison_path = os.path.join(OUTPUT_DIR, "kvquant_fig2_comparison.png")
        plot_comparison(all_data, comparison_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
