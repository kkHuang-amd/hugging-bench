#!/bin/bash

BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE:-rocm/pytorch}
BASE_DOCKER_TAG=${BASE_DOCKER_TAG:-latest-release}
TRANSFORMERS_REPO=${TRANSFORMERS_REPO:-https://github.com/ROCmSoftwarePlatform/transformers}
TRANSFORMERS_BRANCH_OR_TAG=${TRANSFORMERS_BRANCH_OR_TAG:-master}
HB_DOCKER_TAG=${HB_DOCKER_TAG:-latest}

image=$(docker build \
    --build-arg BASE_DOCKER_IMAGE=${BASE_DOCKER_IMAGE}:${BASE_DOCKER_TAG} \
    --build-arg TRANSFORMERS_REPO=${TRANSFORMERS_REPO} \
    -f Dockerfile_rocm \
    . | grep "Successfully built" | cut -d ' ' -f 3)
echo "Built pre-image $image"

full_pytorch_version=$(docker run --rm ${image} python3 -c "import torch; print(torch.__version__)")
pytorch_version=$(echo $full_pytorch_version | cut -d '+' -f 1)
full_rocm_version=$(docker run --rm ${image} dpkg -s rocm-dev | grep Version | cut -d ' ' -f 2)
rocm_version=$(echo $full_rocm_version | cut -d '.' -f 1-3)
transformers_rev=$(docker run --rm ${image} git -C /workspace/transformers rev-parse HEAD)

echo "FROM ${image}" | docker build \
    --label "org.opencontainers.image.title=Hugging Face benchmarks for ROCm" \
    --label "com.amd.image.build.hugging-bench.rev=$(git rev-parse HEAD)" \
    --label "com.amd.image.build.base_image=${BASE_DOCKER_IMAGE}:${BASE_DOCKER_TAG}" \
    --label "com.amd.image.build.transformers_repo=${TRANSFORMERS_REPO}" \
    --label "com.amd.image.build.transformers_rev=${transformers_rev}" \
    --label "com.amd.image.rocm.version=${rocm_version}" \
    --label "com.amd.image.rocm.version.full=${full_rocm_version}" \
    --label "com.amd.image.pytorch.version=${pytorch_version}" \
    --label "com.amd.image.pytorch.version.full=${full_pytorch_version}" \
    -t hugging-bench:${HB_DOCKER_TAG} \
    -
