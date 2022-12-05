# hugging-bench
Dockerfiles and scripts for benchmarking Hugging Face models.

## Building Docker container

ROCm:
```
docker build -f Dockerfile_rocm -t hugging-bench:latest .
```

CUDA:
```
docker build -f Dockerfile_cuda -t hugging-bench:cuda-latest .
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