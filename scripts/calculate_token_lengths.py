#!/usr/bin/env python3
"""
計算 LongBench v1 和 v2 每個任務的平均 token 長度 (使用 llama3.1-8b-instruct tokenizer)
"""

import json
import sys
import os
from pathlib import Path
from collections import defaultdict
from transformers import AutoTokenizer
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()


def load_tokenizer(hf_token=None):
    """載入 llama3.1-8b-instruct tokenizer"""
    print("Loading llama3.1-8b-instruct tokenizer...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            trust_remote_code=True,
            token=hf_token
        )
        print("✓ Successfully loaded llama3.1-8b-instruct tokenizer")
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load llama3.1-8b-instruct: {e}")
        print("Trying Llama-2-7b as fallback...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-2-7b-hf",
                trust_remote_code=True
            )
            print("✓ Loaded Llama-2-7b tokenizer")
            return tokenizer
        except Exception as e2:
            print(f"Error: Could not load any llama tokenizer: {e2}")
            sys.exit(1)


def process_v1(tokenizer, output_dir="./"):
    """處理 LongBench v1 數據"""
    print("\n" + "=" * 70)
    print("Processing LongBench V1")
    print("=" * 70)

    v1_results = {}
    v1_dir = Path("data/longbench_v1")

    if not v1_dir.exists():
        print(f"Error: {v1_dir} not found")
        return v1_results

    for task_file in sorted(v1_dir.glob("*.jsonl")):
        task_name = task_file.stem
        print(f"Processing {task_name:30s}...", end=" ", flush=True)

        task_tokens = []
        try:
            with open(task_file) as f:
                for line in f:
                    sample = json.loads(line)
                    context = sample.get('context', '')
                    input_text = sample.get('input', '')
                    combined = context + " " + input_text

                    # 計算 token 數（不包含 special tokens）
                    token_count = len(tokenizer.encode(combined, add_special_tokens=False))
                    task_tokens.append(token_count)

            if task_tokens:
                avg_tokens = sum(task_tokens) / len(task_tokens)
                min_tokens = min(task_tokens)
                max_tokens = max(task_tokens)

                v1_results[task_name] = {
                    'count': len(task_tokens),
                    'avg': round(avg_tokens, 2),
                    'min': min_tokens,
                    'max': max_tokens,
                    'total': sum(task_tokens)
                }

                print(f"✓ {len(task_tokens):3d} samples, avg={avg_tokens:7.0f} tokens")
            else:
                print("✗ No samples found")

        except Exception as e:
            print(f"✗ Error: {e}")

    return v1_results


def process_v2(tokenizer, output_dir="./"):
    """處理 LongBench v2 數據"""
    print("\n" + "=" * 70)
    print("Processing LongBench V2")
    print("=" * 70)

    v2_results = {}
    v2_dir = Path("data/longbench_v2")

    if not v2_dir.exists():
        print(f"Error: {v2_dir} not found")
        return v2_results

    for task_file in sorted(v2_dir.glob("*.json")):
        task_name = task_file.stem
        print(f"Processing {task_name:40s}...", end=" ", flush=True)

        task_tokens = []
        try:
            with open(task_file) as f:
                data = json.load(f)

            for sample in data:
                context = sample.get('context', '')
                question = sample.get('question', '')
                combined = context + " " + question

                # 計算 token 數（不包含 special tokens）
                token_count = len(tokenizer.encode(combined, add_special_tokens=False))
                task_tokens.append(token_count)

            if task_tokens:
                avg_tokens = sum(task_tokens) / len(task_tokens)
                min_tokens = min(task_tokens)
                max_tokens = max(task_tokens)

                v2_results[task_name] = {
                    'count': len(task_tokens),
                    'avg': round(avg_tokens, 2),
                    'min': min_tokens,
                    'max': max_tokens,
                    'total': sum(task_tokens)
                }

                print(f"✓ {len(task_tokens):3d} samples, avg={avg_tokens:7.0f} tokens")
            else:
                print("✗ No samples found")

        except Exception as e:
            print(f"✗ Error: {e}")

    return v2_results


def save_results(v1_results, v2_results, output_file="longbench_token_lengths.json"):
    """保存結果為 JSON"""
    results = {
        'v1': v1_results,
        'v2': v2_results
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Results saved to {output_file}")
    return results


def print_summary(v1_results, v2_results):
    """打印摘要"""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if v1_results:
        v1_avgs = [r['avg'] for r in v1_results.values()]
        print(f"\nV1: {len(v1_results)} tasks")
        print(f"  Average token length: {sum(v1_avgs) / len(v1_avgs):.0f}")
        print(f"  Min: {min(v1_avgs):.0f}, Max: {max(v1_avgs):.0f}")

    if v2_results:
        v2_avgs = [r['avg'] for r in v2_results.values()]
        print(f"\nV2: {len(v2_results)} task categories")
        print(f"  Average token length: {sum(v2_avgs) / len(v2_avgs):.0f}")
        print(f"  Min: {min(v2_avgs):.0f}, Max: {max(v2_avgs):.0f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    tokenizer = load_tokenizer(hf_token=hf_token)

    v1_results = process_v1(tokenizer)
    v2_results = process_v2(tokenizer)

    results = save_results(v1_results, v2_results)
    print_summary(v1_results, v2_results)

    print("\n✓ Done! Results can be used to generate markdown documentation.")
