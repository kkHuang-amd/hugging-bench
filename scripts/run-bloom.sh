#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

case ${gpu_architecture} in
    $MI200) batch_size=${BATCH_SIZE:-2};;
    $MI100) batch_size=${BATCH_SIZE:-1};;
    $MI50) batch_size=${BATCH_SIZE:-1};;
    $A100) batch_size=${BATCH_SIZE:-1};;
    $V100) batch_size=${BATCH_SIZE:-1};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

# Print parameters
echo "Number of GCDs: ${NGCD}"
echo "Batch size: ${batch_size}"

python -m torch.distributed.launch --nproc_per_node=$NGCD /workspace/transformers/examples/pytorch/language-modeling/run_clm.py\
    --cache_dir /data \
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
    #  "$@" \
    # 2>&1 | tee log.txt