#!/bin/bash

# Default parameters
NGCD=8

# User-defined parameters
while [ ! -z "$1" ]; do
    case $1 in
        # Number of GCDs
        --n_gcd|-g)
            shift
            NGCD=$1
            ;;
        # Number of GCDs
        --batch_size|-bs)
            shift
            BATCH_SIZE=$1
            ;;
        #
        *)
        show_usage
        exit 1
        ;;
    esac
    shift
done