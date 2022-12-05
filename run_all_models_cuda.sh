#!/bin/bash

outdir=results/20221201a
NUM_ITERATIONS=5

mkdir -p ${outdir}

if [ -f /bench/bin/sutinfo-gpuperf-main.pyz ]; then
	sudo python3 /bench/bin/sutinfo-gpuperf-main.pyz -o ${outdir}/sutinfo.json
fi

docker build -f Dockerfile_cuda -t hugging-bench:cuda-latest .

for i in $(seq 1 $NUM_ITERATIONS); do
	echo
	echo
	echo "Hugging Face iteration ${i}"
	date
	echo
	for model in bart bert bloom deberta-v2-xxlarge distilbart-cnn distilbert-base gpt-neo gpt2 pegasus roberta-large t5-large; do
		docker run --rm -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 hugging-bench:cuda-latest scripts/run-${model}.sh | tee ${outdir}/${model}_${i}.log
		python3 utils/logextract.py -f ${outdir}/${model}_${i}.log > ${outdir}/${model}_${i}.json
	done
done | tee ${outdir}/run.log