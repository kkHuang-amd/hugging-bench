#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=64; break;;
    $MI100) batch_size=24; break;;
    $MI50) batch_size=1; break;;
    $A100) batch_size=32; break;;
    $V100) batch_size=1; break;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

python3 /workspace/transformers/examples/pytorch/text-classification/run_glue.py \
	--model_name_or_path microsoft/deberta-v2-xlarge \
	--task_name MRPC \
	--do_train \
	--max_seq_length 128 \
	--per_device_train_batch_size $batch_size \
	--learning_rate 3e-6 \
	--max_steps 150 \
	--output_dir /tmp/deberta_res \
	--overwrite_output_dir \
	--logging_steps 1 \
	--fp16 \
	--skip_memory_metrics=True \
        "$@" \
    2>&1 | tee log-deberta-v2-xxlarge.txt

# output performance metric
performance=$(cat log-deberta-v2-xxlarge.txt | grep -Eo "stable_train_samples_per_second':[^,]+" | sed "s/stable_train_samples_per_second': //g")

echo "performance: $performance samples_per_second"