#!/bin/bash

source $(dirname "${BASH_SOURCE[0]}")/load_execute_params.sh "$@"
source $(dirname "${BASH_SOURCE[0]}")/execute_common.sh

BASE_DOCKER_TAG=${BASE_DOCKER_TAG:-rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1}
HB_DOCKER_TAG=${HB_DOCKER_TAG:-latest}


# Build Docker image
docker build --build-arg BASE_DOCKER_TAG=${BASE_DOCKER_TAG} -f Dockerfile_rocm -t hugging-bench:${HB_DOCKER_TAG} .


# Log execute parameters
EXECUTE_LOG=${OUTDIR}/execute.log
touch ${EXECUTE_LOG}

echo "OUTDIR: ${OUTDIR}" | tee -a ${EXECUTE_LOG}
echo "CACHEDIR: ${CACHEDIR}" | tee -a ${EXECUTE_LOG}
echo "NUM_ITERATIONS: ${NUM_ITERATIONS}" | tee -a ${EXECUTE_LOG}
echo "MODELS: ${MODELS}" | tee -a ${EXECUTE_LOG}
echo "NGCD: ${NGCD}" | tee -a ${EXECUTE_LOG}
echo "BATCH_SIZE: ${BATCH_SIZE}" | tee -a ${EXECUTE_LOG}
echo "BASE_DOCKER_TAG: ${BASE_DOCKER_TAG}" | tee -a ${EXECUTE_LOG}
echo "HB_DOCKER_TAG: ${HB_DOCKER_TAG}" | tee -a ${EXECUTE_LOG}


# Log SUT Info
if [ -f /bench/bin/sutinfo-gpuperf-main.pyz ]; then
	sudo python3 /bench/bin/sutinfo-gpuperf-main.pyz -o ${OUTDIR}/sutinfo.json
fi


# Execute iterations
for i in $(seq 1 $NUM_ITERATIONS); do
	echo
	echo
	echo "Hugging Face iteration ${i}"
	date
	echo
	for model in ${MODELS}; do
		docker run --name ${model} --rm -it --ipc=host --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined -v ${CACHEDIR}:/data hugging-bench:${HB_DOCKER_TAG} scripts/run-${model}.sh --n_gcd ${NGCD}  --batch_size ${BATCH_SIZE} | tee ${OUTDIR}/${model}_${i}.log
		python3 utils/logextract.py -f ${OUTDIR}/${model}_${i}.log > ${OUTDIR}/${model}_${i}.json
	done
done | tee ${OUTDIR}/run.log