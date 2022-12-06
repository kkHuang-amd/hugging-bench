#!/bin/bash

# Default parameters
NGCD=8
HB_DOCKER_TAG=latest
outdir=results/latest

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
        # Number of GCDs
        --n_gcd|-g)
            shift
            NGCD=$1
            ;;
        *)
        show_usage
        exit 1
        ;;
    esac
    shift
done