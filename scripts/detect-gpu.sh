#!/bin/bash
if [ -f "/usr/bin/nvidia-smi" ]; then
    echo "NVIDIA GPU detected"
    gpu_vendor="NVIDIA"
    gpu_architecture=$(nvidia-smi --query-gpu=name --format=csv,noheader | grep -m 1 -E -o ".{0,1}100"| xargs )
elif [ -f "/opt/rocm/bin/rocm-smi" ]; then
    echo "AMD GPU detected."
    gpu_vendor="AMD"
    gpu_architecture=$(rocminfo | grep -o -m 1 'gfx.*'| xargs )
else
    echo "Unable to detect GPU vendor"
    exit 1
fi

MI200="gfx90a"
MI100="gfx908"
MI50="gfx906"
H100="H100"
A100="A100"
V100="V100"
