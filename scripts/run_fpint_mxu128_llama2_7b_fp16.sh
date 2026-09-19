#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

export MODEL_NAME=llama2-7b
export COMPUTE_DTYPE=fp16
export ROTATION_DTYPE=fp16
export MODEL=${MODEL:-./models/llama2-7b}
export ROTATION=${ROTATION:-rotation_llama-2-7b/a16w4kv4-vasym/R.bin}
export CHECKPOINT=${CHECKPOINT:-saved_models/w4-gptq-wclip-wgs128-spinquant-optrot/llama2-7b/w4-gptq-fp16-fpint-v2.pt}
export OUTPUT_ROOT=${OUTPUT_ROOT:-results/fpint-mxu128-llama2-7b-fp16}
export LOG_ROOT=${LOG_ROOT:-logs/fpint-mxu128-llama2-7b-fp16}
export REPORT=${REPORT:-docs/FP-INT-hw-acc/mxu128_llama2_fp16_model_results.md}
export RANDOM_RESULT=${RANDOM_RESULT:-results/fpint-mxu128-gemm/fp16-int4-reference.json}

if [[ ! -s "${ROTATION}" ]]; then
    rotation_dir=$(dirname -- "${ROTATION}")
    PYTHON_BIN=${PYTHON_BIN:-python} CUDA_DEVICE=${ROTATION_CUDA_DEVICE:-0} \
        MASTER_PORT=${ROTATION_MASTER_PORT:-29500} \
        "${SCRIPT_DIR}/optimize_rotation_one.sh" \
        "${MODEL}" "${rotation_dir}" fp16
fi

exec "${SCRIPT_DIR}/run_fpint_mxu128_llama31_8b.sh"
