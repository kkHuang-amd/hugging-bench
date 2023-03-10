#!/bin/bash

outdir=results/rocm_hf/"$(date +"%Y%m%dT%H%M")"
NUM_ITERATIONS=5

mkdir -p ${outdir}

if [ -f /bench/bin/sutinfo-gpuperf-main.pyz ]; then
	sudo python3 /bench/bin/sutinfo-gpuperf-main.pyz -o ${outdir}/sutinfo.json
fi

docker build -f Dockerfile_rocm_hft -t hugging-bench-hft:latest .

for i in $(seq 1 $NUM_ITERATIONS); do
	echo
	echo
	echo "Hugging Face iteration ${i}"
	date
	echo
	for model in bart bert bloom deberta-v2-xlarge distilbart-cnn distilbert-base gpt-neo gpt2 pegasus roberta-large t5-large; do
		docker run --name ${model} --rm -it --ipc=host --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined hugging-bench-hft:latest scripts/run-${model}.sh | tee ${outdir}/${model}_${i}.log
		python3 utils/logextract.py -f ${outdir}/${model}_${i}.log > ${outdir}/${model}_${i}.json
	done
done | tee ${outdir}/run.log
