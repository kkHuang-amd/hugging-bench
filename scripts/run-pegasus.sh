#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

max_steps={max_steps:-150}

case ${gpu_architecture} in
    $MI200) batch_size=${batch_size:-20};;
    $MI100) batch_size=${batch_size:-1};;
    $MI50) batch_size=${batch_size:-1};;
    $A100) batch_size=${batch_size:-12};;
    $V100) batch_size=${batch_size:-1};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

# Print parameters
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