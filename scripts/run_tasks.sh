#!/bin/bash

set -e
source venv/bin/activate
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# === LongBench v1: LiveKVQuantP ===
# v2 full results

# python scripts/run_liveKVQuantP.py --task_type few-shot --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1


# === Ablations (baseline: alpha=0.1, clip=4.0, outlier=0.01, chunk=512, qsl=3) ===

# --- Architectural ablations (Layer 1) ---
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --use_warmup false
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --use_outlier_isolation false
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --stats_method ema_minmax
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --quant_start_layer 0

# --- Chunk size sensitivity (Layer 2) ---
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --chunk_size 256
# chunk=512 is baseline (already done)
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --chunk_size 1024
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --chunk_size 2048
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1 --chunk_size 4096

# --- Outlier ratio sensitivity (Layer 2) ---
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.005 --num_samples -1
# outlier=0.01 is baseline (already done)
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.05  --num_samples -1

# --- EMA alpha sensitivity (Layer 2) ---
# alpha=0.1 = baseline (already done)
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.2 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.3 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.5 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1

# --- Clip factor sensitivity (Layer 2) ---
# python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 3.0 --outlier_ratio 0.01 --num_samples -1
# clip=4.0 = baseline (already done)
python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 5.0 --outlier_ratio 0.01 --num_samples -1
python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 7.0 --outlier_ratio 0.01 --num_samples -1
python scripts/run_liveKVQuantP.py --task_type narrativeqa --ema_alpha 0.1 --clip_factor_n 1.0 --outlier_ratio 0.01 --num_samples -1

# === LongBench v2: fullKV ===
# python scripts/run_fullKV.py --bench_version v2 --task_type single-doc --num_samples -1
# python scripts/run_fullKV.py --bench_version v2 --task_type multi-doc --num_samples -1
# python scripts/run_fullKV.py --bench_version v2 --task_type long-context --num_samples -1
# python scripts/run_fullKV.py --bench_version v2 --task_type dialogue --num_samples -1
# python scripts/run_fullKV.py --bench_version v2 --task_type code --num_samples -1
# python scripts/run_fullKV.py --bench_version v2 --task_type structured --num_samples -1

# # === LongBench v2: liveKVQuantP ===
# python scripts/run_liveKVQuantP.py --bench_version v2 --task_type single-doc --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --bench_version v2 --task_type multi-doc --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --bench_version v2 --task_type long-context --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --bench_version v2 --task_type dialogue --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --bench_version v2 --task_type code --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --bench_version v2 --task_type structured --ema_alpha 0.1 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1

echo "All Done!"

# To run this script in the background and log output to a file, use:
# nohup bash scripts/run_tasks.sh > run.log 2>&1 &
