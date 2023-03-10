#!/bin/bash

# Load user-specified params
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh "$@"
# Load GPU information
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh


# Container-specific Python path
export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}


# Use default params if not specified by user
max_steps=${max_steps:-150}
n_gcd=${n_gcd:-1}

# GPU-specific default params
case ${gpu_architecture} in
    $MI200) batch_size=${batch_size:-20};;
    $MI100) batch_size=${batch_size:-8};;
    $MI50) batch_size=${batch_size:-4};;
    $H100) batch_size=${batch_size:-25};;
    $A100) batch_size=${batch_size:-25};;
    $V100) batch_size=${batch_size:-4};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; batch_size=${batch_size:-1};;
esac

# Print parameters
echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"
echo "Number of GCDs: ${n_gcd}"
echo "Batch size: ${batch_size}"
echo "Max steps: ${max_steps}"

python -m torch.distributed.launch --nproc_per_node=$n_gcd /workspace/transformers/examples/pytorch/language-modeling/run_clm.py\
    --cache_dir /data \
    --output_dir output \
    --model_name_or_path gpt2 \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --label_smoothing 0.1 \
    --logging_steps 1 \
    --logging_dir log\
    --fp16 \
    --dataloader_num_workers 1 \
    --skip_memory_metrics \
    --per_device_train_batch_size=$batch_size \
    --overwrite_output_dir \
    --max_steps $max_steps