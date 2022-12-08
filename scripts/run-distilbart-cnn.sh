#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

case ${gpu_architecture} in
    $MI200) batch_size=${BATCH_SIZE:-40};;
    $MI100) batch_size=${BATCH_SIZE:-4};;
    $MI50) batch_size=${BATCH_SIZE:-4};;
    $A100) batch_size=${BATCH_SIZE:-20};;
    $V100) batch_size=${BATCH_SIZE:-4};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

# Print parameters
echo "Number of GCDs: ${n_gcd}"
echo "Batch size: ${batch_size}"

python -m torch.distributed.launch --nproc_per_node=$n_gcd /workspace/transformers/examples/pytorch/summarization/run_summarization.py \
        --cache_dir /data \
        --model_name_or_path sshleifer/distilbart-cnn-6-6 \
        --dataset_name xsum \
        --max_steps 150 \
        --do_train \
        --output_dir /tmp/tst-summarization \
        --per_device_train_batch_size=$batch_size \
        --per_device_eval_batch_size=1 \
        --overwrite_output_dir \
        --predict_with_generate \
        # "$@" \
        2>&1 | tee log.txt 

# output performance metric
performance=$(cat log.txt | grep -Eo "stable_train_samples_per_second':[^,]+" | sed "s/stable_train_samples_per_second': //g")

echo "performance: $performance samples_per_second"