# Default param values
OUTDIR=${OUTDIR:-results}
CACHEDIR=${CACHEDIR:-~/data/hugging-bench}
NUM_ITERATIONS=${NUM_ITERATIONS:-5}
MODELS=${MODELS:-all}
NGCD=${NGCD:-1}
BATCH_SIZE=${BATCH_SIZE:-}

if [[ ${MODELS} == "all" ]]; then
    MODELS='bart bert bloom deberta-v2-xlarge distilbart-cnn distilbert-base gpt-neo gpt2 pegasus roberta-large t5-large'
fi

OUTDIR=${OUTDIR}/"$(date +"%Y%m%dT%H%M")"

# Create directories
mkdir -p ${OUTDIR}
mkdir -p ${CACHEDIR}
