#!/bin/bash

set -e
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/run_fullKV.py   --bench_version v1   --task_type qmsum  --num_samples -1
python scripts/run_fullKV.py   --bench_version v1   --task_type multi_news  --num_samples -1

# python scripts/run_liveKVQuantP.py --task_type samsum --ema_alpha 0.2 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples 10
# python scripts/run_liveKVQuantP.py --task_type multi-doc --ema_alpha 0.2 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1
# python scripts/run_liveKVQuantP.py --task_type code --ema_alpha 0.2 --clip_factor_n 4.0 --outlier_ratio 0.01 --num_samples -1

echo "All Done!"

# To run this script in the background and log output to a file, use:
# nohup bash scripts/run_fullKV_all_tasks.sh > run_log.txt 2>&1 &