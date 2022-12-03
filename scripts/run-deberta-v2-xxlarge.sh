#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=64;;
    $MI100) batch_size=24;;
    $MI50) batch_size=1;;
    $A100) batch_size=32;;
    $V100) batch_size=1;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

# Workaround for https://ontrack-internal.amd.com/browse/SWDEV-317794
# for var in "$@"; do 
#     if [ "$var" == "--ort" ]; then
#         patch src/transformers/models/deberta_v2/modeling_deberta_v2.py deberta_softmax_backward.patch
#         patch $HF_PATH/src/transformers/models/deberta_v2/modeling_deberta_v2.py deberta_softmax_backward.patch
#     fi
# done

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

python /workspace/transformers/examples/pytorch/text-classification/run_glue.py \
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