#!/bin/bash

# User-defined parameters
while [ ! -z "$1" ]; do
    case $1 in
        # Number of GCDs
        --n_gcd|-g)
            shift
            n_gcd=$1
            ;;
        # Batch size
        --batch_size|-bs)
            shift
            batch_size=$1
            ;;
        # Maximum number of training steps
        --max_steps|-s)
            shift
            max_steps=$1
            ;;
        #
        *)
        show_usage
        exit 1
        ;;
    esac
    shift
done