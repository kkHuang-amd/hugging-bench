#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

case ${gpu_architecture} in
    $MI200) batch_size=${batch_size:-16};;
    $MI100) batch_size=${batch_size:-8};;
    $MI50) batch_size=${batch_size:-2};;
    $A100) batch_size=${batch_size:-16};;
    $V100) batch_size=${batch_size:-2};;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

# Print parameters
echo "Number of GCDs: ${n_gcd}"
echo "Batch size: ${batch_size}"

python -m torch.distributed.launch --nproc_per_node=$n_gcd /workspace/transformers/examples/pytorch/translation/run_translation.py \
	--cache_dir /data \
	--source_prefix "translate English to Romanian:"  \
	--dataset_name wmt16 \
	--dataset_config ro-en \
	--model_name_or_path t5-large \
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
	--skip_memory_metrics=True