#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=24;;
    $MI100) batch_size=16;;
    $MI50) batch_size=1;;
    $A100) batch_size=8;;
    $V100) batch_size=1;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

python /workspace/transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path bert-large-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --max_steps 150 \
    --logging_steps 1 \
    --output_dir /tmp/test-mlm-bbu \
    --overwrite_output_dir \
    --per_device_train_batch_size ${batch_size} \
    --fp16 \
    --skip_memory_metrics=True
