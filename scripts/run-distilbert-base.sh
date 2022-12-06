#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=24;;
    $MI100) batch_size=32;;
    $MI50) batch_size=4;;
    $A100) batch_size=32;;
    $V100) batch_size=4;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

# Print parameters
echo "Number of GCDs: ${NGCD}"

python -m torch.distributed.launch --nproc_per_node=$NGCD /workspace/transformers/examples/pytorch/language-modeling/run_mlm.py \
	--model_name_or_path distilbert-base-uncased \
	--dataset_name wikitext \
	--dataset_config_name wikitext-2-raw-v1 \
	--do_train \
	--max_steps 150 \
	--logging_steps 1 \
	--output_dir /tmp/test-mlm-bbu \
	--overwrite_output_dir \
	--per_device_train_batch_size $batch_size \
	--fp16 \
	--skip_memory_metrics=True \
        # "$@" \
    2>&1 | tee log.txt

# output performance metric
performance=$(cat log.txt | grep -Eo "stable_train_samples_per_second':[^,]+" | sed "s/stable_train_samples_per_second': //g")

echo "performance: $performance samples_per_second"