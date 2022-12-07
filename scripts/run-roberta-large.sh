#!/bin/bash
source $(dirname "${BASH_SOURCE[0]}")/detect-gpu.sh

echo "GPU Vendor: ${gpu_vendor}"
echo "GPU architecture: ${gpu_architecture}"

case ${gpu_architecture} in
    $MI200) batch_size=24;;
    $MI100) batch_size=24;;
    $MI50) batch_size=2;;
    $A100) batch_size=24;;
    $V100) batch_size=2;;
    *) echo "Unrecognized GPU architecture: ${gpu_architecture}"; exit 1;;
esac

export PYTHONPATH=/workspace/transformers/src:${PATHONPATH}

# Load user-specified parameters
source $(dirname "${BASH_SOURCE[0]}")/load-params.sh $@

# Print parameters
echo "Number of GCDs: ${NGCD}"

python -m torch.distributed.launch --nproc_per_node=$NGCD /workspace/transformers/examples/pytorch/question-answering/run_qa.py \
       --cache_dir /data \
       --model_name_or_path roberta-large \
       --dataset_name squad \
       --do_train \
       --per_device_train_batch_size $batch_size \
       --learning_rate 3e-5 \
       --max_steps 150 \
       --max_seq_length 384 \
       --doc_stride 128 \
       --output_dir /tmp/roberta_res \
       --overwrite_output_dir \
       --logging_steps 1 \
       --fp16 \
       --skip_memory_metrics=True \
    #    "$@" \
    2>&1 | tee log.txt

# output performance metric
performance=$(cat log.txt | grep -Eo "stable_train_samples_per_second':[^,]+" | sed "s/stable_train_samples_per_second': //g")

echo "performance: $performance samples_per_second"