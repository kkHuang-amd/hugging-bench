#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/load_execute_params.sh $@

echo "Docker base image tag: ${BASE_DOCKER_TAG}"
echo "Hugging-Bench Docker image tag: ${HB_DOCKER_TAG}"
echo "Number of GCDs: ${NGCD}"
echo "Output directory: ${OUTDIR}"

NUM_ITERATIONS=5

mkdir -p ${OUTDIR}
mkdir -p cache_dir

if [ -f /bench/bin/sutinfo-gpuperf-main.pyz ]; then
	sudo python3 /bench/bin/sutinfo-gpuperf-main.pyz -o ${OUTDIR}/sutinfo.json
fi

docker build --build-arg BASE_DOCKER_TAG=${BASE_DOCKER_TAG} -f Dockerfile_cuda -t hugging-bench:cuda-${HB_DOCKER_TAG} .

for i in $(seq 1 $NUM_ITERATIONS); do
	echo
	echo
	echo "Hugging Face iteration ${i}"
	date
	echo
	for model in bart bert bloom deberta-v2-xxlarge distilbart-cnn distilbert-base gpt-neo gpt2 pegasus roberta-large t5-large; do
		docker run --rm -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v ${PWD}/cache_dir:/data hugging-bench:cuda-${HB_DOCKER_TAG} scripts/run-${model}.sh --n_gcd ${NGCD} | tee ${OUTDIR}/${model}_${i}.log
		python3 utils/logextract.py -f ${OUTDIR}/${model}_${i}.log > ${OUTDIR}/${model}_${i}.json
	done
done | tee ${OUTDIR}/run.log
