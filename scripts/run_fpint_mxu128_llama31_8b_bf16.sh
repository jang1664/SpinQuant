#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

export MODEL_NAME=llama31-8b
export COMPUTE_DTYPE=bf16
export ROTATION_DTYPE=bf16
export MODEL=${MODEL:-./models/llama3.1-8b}
export ROTATION=${ROTATION:-rotation_llama-3.1-8b/a16w4kv4-vasym-bf16/R.bin}
export CHECKPOINT=${CHECKPOINT:-saved_models/w4-gptq-wclip-wgs128-spinquant-optrot/llama3.1-8b/w4-gptq-bf16-fpint-v2.pt}
export OUTPUT_ROOT=${OUTPUT_ROOT:-results/fpint-mxu128-llama31-8b-bf16}
export LOG_ROOT=${LOG_ROOT:-logs/fpint-mxu128-llama31-8b-bf16}
export REPORT=${REPORT:-docs/FP-INT-hw-acc/mxu128_llama31_bf16_model_results.md}
export RANDOM_RESULT=${RANDOM_RESULT:-results/fpint-mxu128-gemm/bf16-int4-reference.json}

if [[ ! -s "${ROTATION}" ]]; then
    rotation_dir=$(dirname -- "${ROTATION}")
    PYTHON_BIN=${PYTHON_BIN:-python} CUDA_DEVICE=${ROTATION_CUDA_DEVICE:-0} \
        MASTER_PORT=${ROTATION_MASTER_PORT:-29500} \
        "${SCRIPT_DIR}/optimize_rotation_one.sh" \
        "${MODEL}" "${rotation_dir}" bf16
fi

exec "${SCRIPT_DIR}/run_fpint_mxu128_llama31_8b.sh"
