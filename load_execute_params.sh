#!/bin/bash

# Default parameters
NGCD=8
HB_DOCKER_TAG=latest
OUTDIR=results/latest

# User-defined parameters
while [ ! -z "$1" ]; do
    case $1 in
        # Docker base image tag
        --base_docker_tag|-bt)
            shift
            BASE_DOCKER_TAG=$1
            ;;
        # Hugging-Bench Docker image tag
        --hb_docker_tag|-t)
            shift
            HB_DOCKER_TAG=$1
            ;;
        # Output directory
        --outdir|-o)
            shift
            OUTDIR=$1
            ;;
        # Data cache directory
        --cache_dir|-c)
            shift
            CACHEDIR=$1
            ;;
        # Number of iterations
        --iterations|-i)
            shift
            ITERATIONS=$1
            ;;
        # Models
        --models|-m)
            shift
            MODELS=$1
            ;;
        # Number of GCDs
        --n_gcd|-g)
            shift
            NGCD=$1
            ;;
        # Batch size
        --batch_size|-bs)
            shift
            BATCH_SIZE=$1
            ;;
        *)
        show_usage
        exit 1
        ;;
    esac
    shift
done