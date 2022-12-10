# hugging-bench
Dockerfiles and scripts for benchmarking Hugging Face models.

## Building Docker container

There are build_container_* scripts for building containers for both ROCm and CUDA.  By default, both containers will use the master branch of the https://github.com/ROCmSoftwarePlatform/transformers repository and be based on a recent pytorch base container.  These details can be changed through environment variables.  See the scripts for a list of available environment variables.  For example:

ROCm:
```
BASE_DOCKER_TAG=rocm5.3_ubuntu20.04_py3.7_pytorch_1.12.1 \
HB_DOCKER_TAG=rocm-5.3 \
./build_container_rocm.sh
```

CUDA:
```
BASE_DOCKER_TAG=22.11-py3 \
./build_container_cuda.sh
```

## Running Docker container

ROCm (BERT):
```
docker run --rm -it --ipc=host --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined hugging-bench:latest scripts/run-bert.sh
```

CUDA (BERT):
```
docker run --rm -it --gpus=all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 hugging-bench:cuda-latest scripts/run-bert.sh
```

run-bert.sh can be replaced with run-bart.sh, run-bloom.sh run-deberta-v2-xxlarge.sh, run-distilbart-cnn.sh, run-distilbert-base.sh, run-gpt-neo.sh, run-gpt2.sh, run-pegasus.sh, run-roberta-large.sh, or run-t5-large.sh to run other tests.


## Running all models
ROCm example: all models; `2` iterations; `16` GCDs; `24` batch size; rocm/pytorch base image tag `rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1`
```
./execute_rocm.sh -m "all" -i 2 -g 16 -bs 24 -bt rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
```

CUDA example: all models; `5` iterations; `8` GPUs; `16` batch size; nvidia/pytorch base image tag `22.11-py3`
```
./execute_cuda.sh -m "all" -i 5 -g 8 -bs 16 -bt 22.11-py3
```


## Running specific models
ROCm example: models BLOOM, PEGASUS, & T5-large; `2` iterations; `16` GCDs; `24` batch size; rocm/pytorch base image tag `rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1`
```
./execute_rocm.sh -m "bloom pegasus t5-large" -i 2 -g 16 -bs 24 -bt rocm5.4_ubuntu20.04_py3.8_pytorch_1.12.1
```

CUDA example: models BART & GPT-2; `5` iterations; `8` GPUs; `16` batch size; nvidia/pytorch base image tag `22.11-py3`
```
./execute_cuda.sh -m "bart gpt2" -i 5 -g 8 -bs 16 -bt 22.11-py3
```