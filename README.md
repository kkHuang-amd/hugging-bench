# hugging-bench
Dockerfiles and scripts for benchmarking Hugging Face models.

## Building Docker container

```
docker build -f Dockerfile_rocm -t hugging-bench:latest .
```

## Running Docker container

BERT:
```
docker run --rm -it --ipc=host --device /dev/dri --device /dev/kfd --security-opt seccomp=unconfined hugging-bench:latest python3 transformers/examples/pytorch/language-modeling/run_mlm.py --model_name_or_path bert-large-uncased --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --max_steps 150 --logging_steps 1 --output_dir /tmp/test-mlm-bbu --overwrite_output_dir --per_device_train_batch_size 24 --fp16 --skip_memory_metrics=True
```