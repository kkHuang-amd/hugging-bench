# hugging-bench
Dockerfiles and scripts for benchmarking Hugging Face models.

## Building Docker container

```
docker build -f Dockerfile_rocm -t hugging-bench:latest .
```

## Running Docker container

BERT:
```
docker run --rm -it --ipc=host --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined hugging-bench:latest scripts/run-bert.sh
```