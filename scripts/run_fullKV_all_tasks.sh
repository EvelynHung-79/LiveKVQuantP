#!/bin/bash

# 當任何一行指令回傳錯誤 (error exit code) 時，立刻停止執行整個腳本
set -e

# echo "Running experiment 1..."
# python scripts/run_fullKV.py   --bench_version v1   --task_type single-doc   --num_samples -1

# echo "Running experiment 2..."
# python scripts/run_fullKV.py   --bench_version v1   --task_type multi-doc   --num_samples -1

# echo "Running experiment 3..."
# python scripts/run_fullKV.py   --bench_version v1   --task_type summarization   --num_samples -1

# echo "Running experiment 4..."
# python scripts/run_fullKV.py   --bench_version v1   --task_type few-shot   --num_samples -1

# echo "Running experiment 5..."
# python scripts/run_fullKV.py   --bench_version v1   --task_type synthetic   --num_samples -1

# echo "Running experiment 6..."
# python scripts/run_fullKV.py   --bench_version v1   --task_type code   --num_samples -1

python scripts/run_liveKVQuantP.py --task_type single-doc -ema_alpha 0.2 --clip_factor_n 4.0 --outlier_ratio 0.01   --num_samples -1

python scripts/run_liveKVQuantP.py --task_type single-doc -ema_alpha 0.1 --clip_factor_n 2.0 --outlier_ratio 0.01   --num_samples -1

echo "All Done!"