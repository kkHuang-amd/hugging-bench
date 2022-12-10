#!/bin/bash

BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE:-nvcr.io/nvidia/pytorch}
BASE_DOCKER_TAG=${BASE_DOCKER_TAG:-22.11-py3}
TRANSFORMERS_REPO=${TRANSFORMERS_REPO:-https://github.com/ROCmSoftwarePlatform/transformers}
TRANSFORMERS_BRANCH_OR_TAG=${TRANSFORMERS_BRANCH_OR_TAG:-master}
HB_DOCKER_TAG=${HB_DOCKER_TAG:-latest}

echo "Building pre-image"
image=$(docker build \
    --build-arg BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE}:${BASE_DOCKER_TAG} \
    --build-arg TRANSFORMERS_REPO=${TRANSFORMERS_REPO} \
    -f Dockerfile_cuda \
    . | grep "Successfully built" | cut -d ' ' -f 3)
echo "Built pre-image $image"

full_pytorch_version=$(docker run --rm ${image} python3 -c "import torch; print(torch.__version__)" | tail -n 1)
pytorch_version=$(echo $full_pytorch_version | cut -d '+' -f 1)
# Sample output line from nvcc --version:
# Build cuda_11.8.r11.8/compiler.31833905_0
cuda_version=$(docker run --rm ${image} nvcc --version | grep "^Build" | cut -d '_' -f 2 | cut -d '.' -f 1-2)
transformers_rev=$(docker run --rm ${image} git -C /workspace/transformers rev-parse HEAD | tail -n 1)
huggingbench_rev=$(git rev-parse HEAD)

echo "FROM ${image}" | docker build \
    --label "org.opencontainers.image.title=Hugging Face benchmarks for ROCm" \
    --label "com.amd.image.build.hugging-bench.rev=${huggingbench_rev}" \
    --label "com.amd.image.build.base_image=${BASE_DOCKER_IMAGE}:${BASE_DOCKER_TAG}" \
    --label "com.amd.image.build.transformers_repo=${TRANSFORMERS_REPO}" \
    --label "com.amd.image.build.transformers_rev=${transformers_rev}" \
    --label "com.amd.image.cuda.version=${cuda_version}" \
    --label "com.amd.image.pytorch.version=${pytorch_version}" \
    --label "com.amd.image.pytorch.version.full=${full_pytorch_version}" \
    --label "com.amd.image.build.info=hugging-bench rev ${huggingbench_rev}, pytorch ${full_pytorch_version}, transformers ${TRANSFORMERS_REPO} rev ${transformers_rev}" \
    -t hugging-bench-cuda:${HB_DOCKER_TAG} \
    -
