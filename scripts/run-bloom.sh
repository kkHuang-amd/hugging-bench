#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=2; break;;
    $MI100) batch_size=1; break;;
    $MI50) batch_size=1; break;;
    $A100) batch_size=1; break;;
    $V100) batch_size=1; break;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

python3 /workspace/transformers/examples/pytorch/language-modeling/run_clm.py\
    --output_dir output \
    --model_name_or_path bigscience/bloom-560m  \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --block_size 128 \
    --do_eval \
    --do_train \
    --label_smoothing 0.1 \
    --logging_steps 1 \
    --logging_dir log\
    --fp16 \
    --dataloader_num_workers 1 \
    --skip_memory_metrics \
    --per_device_train_batch_size=$batch_size \
    --per_device_eval_batch_size=$batch_size \
    --overwrite_output_dir \
    --max_steps 150 \
     "$@" \
    2>&1 | tee log.txt