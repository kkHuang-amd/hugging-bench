#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

max_steps={max_steps:-150}

case ${gpu_architecture} in
    $MI200) batch_size=${batch_size:-24};;
    $MI100) batch_size=${batch_size:-32};;
    $MI50) batch_size=${batch_size:-4};;
    $A100) batch_size=${batch_size:-32};;
    $V100) batch_size=${batch_size:-4};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

# Print parameters
echo "Number of GCDs: ${n_gcd}"
echo "Batch size: ${batch_size}"
echo "Max steps: ${max_steps}"

python -m torch.distributed.launch --nproc_per_node=$n_gcd /workspace/transformers/examples/pytorch/language-modeling/run_mlm.py \
	--cache_dir /data \
	--model_name_or_path distilbert-base-uncased \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--do_train \
	--max_steps $max_steps \
	--logging_steps 1 \
	--output_dir /tmp/test-mlm-bbu \
	--overwrite_output_dir \
	--per_device_train_batch_size $batch_size \
	--fp16 \
	--skip_memory_metrics=True