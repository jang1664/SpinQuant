#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

export ZP_INT8=0 SIGNED_KV=0 ZP_CLAMP=1 SCALE_NO_UPCAST=1
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICE=${CUDA_DEVICE:-}
MATRIX_MIN_FREE_MIB=${MATRIX_MIN_FREE_MIB:-38000}
SEQUENCE_LENGTH=${MATRIX_SEQUENCE_LENGTH:-2048}
RESULT_ROOT=${MATRIX_RESULT_ROOT:-results/aqp-matrix-output-accuracy}
MAX_DOCUMENTS=${MATRIX_MAX_DOCUMENTS:-}
MAX_TOKENS=${MATRIX_MAX_TOKENS_PER_DOCUMENT:-}

if (( $# != 0 && $# != 3 )); then
    echo "Usage: $0 [LLAMA2_7B_MODEL LLAMA32_3B_MODEL LLAMA31_8B_MODEL]" >&2
    exit 2
fi

LLAMA2_7B_MODEL=${1:-${LLAMA2_7B_MODEL:-./models/llama2-7b}}
LLAMA32_3B_MODEL=${2:-${LLAMA32_3B_MODEL:-./models/llama3.2-3b}}
LLAMA31_8B_MODEL=${3:-${LLAMA31_8B_MODEL:-./models/llama3.1-8b}}
FP_RESULT_DIR=${FP_RESULT_DIR:-results/w16a16kv16-fp16-baseline}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-saved_models/w4-gptq-wclip-wgs-1-spinquant-optrot}

if [[ -z ${CUDA_DEVICE} ]]; then
    command -v nvidia-smi >/dev/null 2>&1 || {
        echo "Set CUDA_DEVICE explicitly because nvidia-smi is unavailable." >&2
        exit 1
    }
    while IFS=',' read -r gpu free_mib; do
        gpu=${gpu//[[:space:]]/}
        free_mib=${free_mib//[[:space:]]/}
        if (( free_mib >= MATRIX_MIN_FREE_MIB )); then
            CUDA_DEVICE=${gpu}
            break
        fi
    done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
    [[ -n ${CUDA_DEVICE} ]] || {
        echo "No GPU with at least ${MATRIX_MIN_FREE_MIB} MiB free; set CUDA_DEVICE to override." >&2
        exit 1
    }
fi
echo "Using GPU ${CUDA_DEVICE} (MATRIX_SEQUENCE_LENGTH=${SEQUENCE_LENGTH})"

run_case() {
    local name=$1 model=$2 rotation=$3
    local checkpoint="${CHECKPOINT_ROOT}/${name}/w4-gptq.pt"
    local fp_result="${FP_RESULT_DIR}/${name}.json"
    local output="${RESULT_ROOT}/${name}/matrix-output-accuracy.json"
    local -a limits=()
    [[ -n ${MAX_DOCUMENTS} ]] && limits+=(--max-documents "${MAX_DOCUMENTS}")
    [[ -n ${MAX_TOKENS} ]] && limits+=(--max-tokens-per-document "${MAX_TOKENS}")
    for required in "${model}" "${rotation}" "${checkpoint}" "${fp_result}"; do
        [[ -e ${required} ]] || { echo "Required input not found: ${required}" >&2; return 1; }
    done
    mkdir -p "$(dirname -- "${output}")"
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} "${PYTHON_BIN}" measure_matrix_output_accuracy.py \
        --input-model "${model}" --load-qmodel-path "${checkpoint}" \
        --fp-results-path "${fp_result}" --rotation-path "${rotation}" \
        --sequence-length "${SEQUENCE_LENGTH}" --output "${output}" "${limits[@]}"
}

run_case llama2-7b "${LLAMA2_7B_MODEL}" "${LLAMA2_7B_ROTATION:-rotation_llama-2-7b/a16w4kv4-vasym/R.bin}"
run_case llama3.1-8b "${LLAMA31_8B_MODEL}" "${LLAMA31_8B_ROTATION:-rotation_llama-3.1-8b/a16w4kv4-vasym/R.bin}"
run_case llama3.2-3b "${LLAMA32_3B_MODEL}" "${LLAMA32_3B_ROTATION:-rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin}"

"${PYTHON_BIN}" summarize_matrix_output_accuracy.py \
    "${RESULT_ROOT}"/*/matrix-output-accuracy.json \
    --output "${RESULT_ROOT}/SUMMARY.md"
