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
    $MI200) batch_size=${batch_size:-30};;
    $MI100) batch_size=${batch_size:-1};;
    $MI50) batch_size=${batch_size:-1};;
    $H100) batch_size=${batch_size:-40};;
    $A100) batch_size=${batch_size:-40};;
    $V100) batch_size=${batch_size:-1};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; batch_size=${batch_size:-1};;
esac

# Print parameters
echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"
echo "Number of GCDs: ${n_gcd}"
echo "Batch size: ${batch_size}"
echo "Max steps: ${max_steps}"

python -m torch.distributed.launch --nproc_per_node=$n_gcd /workspace/transformers/examples/pytorch/summarization/run_summarization.py \
    --cache_dir /data \
    --model_name_or_path google/pegasus-xsum \
    --dataset_name xsum \
    --max_steps $max_steps \
	--do_train \
	--output_dir /tmp/tst-summarization \
	--per_device_train_batch_size=$batch_size \
	--per_device_eval_batch_size=1 \
    --overwrite_output_dir \
    --predict_with_generate \
	--max_source_length 512