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
    $MI200) batch_size=${batch_size:-64};;
    $MI100) batch_size=${batch_size:-24};;
    $MI50) batch_size=${batch_size:-1};;
	$H100) batch_size=${batch_size:-32};;
    $A100) batch_size=${batch_size:-32};;
    $V100) batch_size=${batch_size:-1};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; batch_size=${batch_size:-1};;
esac

# Print parameters
echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"
echo "Number of GCDs: ${n_gcd}"
echo "Batch size: ${batch_size}"
echo "Max steps: ${max_steps}"

python -m torch.distributed.launch --nproc_per_node=$n_gcd /workspace/transformers/examples/pytorch/text-classification/run_glue.py \
	--cache_dir /data \
	--model_name_or_path microsoft/deberta-v2-xlarge \
	--task_name MRPC \
	--do_train \
	--max_seq_length 128 \
	--per_device_train_batch_size $batch_size \
	--learning_rate 3e-6 \
	--max_steps $max_steps \
	--output_dir /tmp/deberta_res \
	--overwrite_output_dir \
	--logging_steps 1 \
	--fp16 \
	--skip_memory_metrics=True