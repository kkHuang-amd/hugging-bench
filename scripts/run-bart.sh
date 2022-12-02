#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=24; break;;
    $MI100) batch_size=16; break;;
    $MI50) batch_size=2; break;;
    $A100) batch_size=16; break;;
    $V100) batch_size=2; break;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

python3 /workspace/transformers/examples/pytorch/translation/run_translation.py \
        --dataset_name wmt16 \
       --dataset_config ro-en \
       --model_name_or_path facebook/bart-large \
       --output_dir /tmp/tst-translation \
       --do_train \
       --label_smoothing 0.1 \
       --logging_steps 1 \
       --overwrite_output_dir \
       --per_device_train_batch_size $batch_size \
       --predict_with_generate \
       --source_lang en \
       --target_lang ro \
       --warmup_steps 5 \
       --fp16 \
       --max_steps 150 \
       --skip_memory_metrics=True \
       "$@" \
    # 2>&1 | tee log.txt