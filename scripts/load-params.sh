#!/bin/bash

# Default parameters
n_gcd=8

# User-defined parameters
while [ ! -z "$1" ]; do
    case $1 in
        # Number of GCDs
        --n_gcd|-g)
            shift
            n_gcd=$1
            ;;
        # Number of GCDs
        --batch_size|-bs)
            shift
            batch_size=$1
            ;;
        #
        *)
        show_usage
        exit 1
        ;;
    esac
    shift
done