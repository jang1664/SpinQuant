#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

CUDA_DEVICE=${CUDA_DEVICE:-0}

# Concurrent torchrun jobs otherwise contend for the default rendezvous port
# 29500. Standalone mode asks torchrun to allocate a free local port. Set
# TORCHRUN_MASTER_PORT only when a fixed port is explicitly required.
if [[ -n ${TORCHRUN_MASTER_PORT:-} ]]; then
    TORCHRUN_RDZV_ARGS=(--master-port "${TORCHRUN_MASTER_PORT}")
else
    TORCHRUN_RDZV_ARGS=(--standalone)
fi

W_BITS=16
A_BITS=16
K_BITS=16
V_BITS=16
W_GROUPSIZE=-1
A_GROUPSIZE=-1
K_GROUPSIZE=-1
V_GROUPSIZE=-1

BASELINE_CONFIG_ID="w${W_BITS}a${A_BITS}kv${K_BITS}-fp16-baseline"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    echo "Usage: $0 [LLAMA2_7B_MODEL LLAMA32_3B_MODEL LLAMA31_8B_MODEL]"
    echo "Model arguments can be local paths or Hugging Face model IDs."
    echo "CUDA device can be selected with CUDA_DEVICE (default: 0)."
    echo "torchrun uses an automatically selected free port; set TORCHRUN_MASTER_PORT to override it."
    echo "Evaluates unquantized FP16 Llama-2 7B, Llama-3.2 3B, and Llama-3.1 8B."
    echo "No rotation, weight quantization, activation quantization, or KV-cache quantization is applied."
    echo "Models with a completed result JSON are skipped individually."
    exit 0
fi
if (( $# != 0 && $# != 3 )); then
    echo "Expected zero or three model arguments. Use --help for usage." >&2
    exit 2
fi

LLAMA2_7B_MODEL=${1:-${LLAMA2_7B_MODEL:-./models/llama2-7b}}
LLAMA32_3B_MODEL=${2:-${LLAMA32_3B_MODEL:-./models/llama3.2-3b}}
LLAMA31_8B_MODEL=${3:-${LLAMA31_8B_MODEL:-./models/llama3.1-8b}}

RESULT_DIR="results/${BASELINE_CONFIG_ID}"
LOG_DIR="logs/${BASELINE_CONFIG_ID}"
mkdir -p "${RESULT_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

echo "Baseline config: ${BASELINE_CONFIG_ID}"
echo "torchrun rendezvous: ${TORCHRUN_RDZV_ARGS[*]}"

python - <<'PY'
from importlib.metadata import version

from packaging.version import Version

installed = Version(version("datasets"))
if installed < Version("4.0.0"):
    raise SystemExit(
        f"datasets>=4.0 is required for current Hugging Face dataset schemas; "
        f"found {installed}. Run: python -m pip install 'datasets>=4.0,<5'"
    )
PY

run_baseline() {
    local name=$1
    local model=$2
    local result_path="${RESULT_DIR}/${name}.json"
    local log_path="${LOG_DIR}/${name}.log"

    if [[ -s "${result_path}" ]]; then
        echo "[${name}] result already exists; skipping: ${result_path}"
        return 0
    fi

    case "${model}" in
        /*|./*|../*)
            if [[ ! -e "${model}" ]]; then
                echo "Local model not found: ${model}" >&2
                return 1
            fi
            ;;
    esac

    echo "[${name}] model=${model}"
    echo "[${name}] precision=W16A16KV16 (unquantized FP16 baseline)"
    echo "[${name}] rotation=disabled quantization=disabled"

    if [[ -s "${log_path}" ]]; then
        echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Resuming ${name}" >> "${log_path}"
    fi

    torchrun "${TORCHRUN_RDZV_ARGS[@]}" --nnodes=1 --nproc_per_node=1 ptq.py \
        --input_model "${model}" \
        --do_train False \
        --do_eval True \
        --per_device_eval_batch_size 4 \
        --model_max_length 2048 \
        --fp16 True \
        --bf16 False \
        --save_safetensors False \
        --w_bits "${W_BITS}" \
        --w_groupsize "${W_GROUPSIZE}" \
        --a_bits "${A_BITS}" \
        --a_groupsize "${A_GROUPSIZE}" \
        --k_bits "${K_BITS}" \
        --k_groupsize "${K_GROUPSIZE}" \
        --v_bits "${V_BITS}" \
        --v_groupsize "${V_GROUPSIZE}" \
        --no-rotate \
        --results_path "${result_path}" \
        2>&1 | tee -a "${log_path}"

    if [[ ! -s "${result_path}" ]]; then
        echo "[${name}] baseline evaluation did not produce ${result_path}" >&2
        return 1
    fi

    echo "[${name}] baseline accuracy/PPL JSON: ${result_path}"
    echo "[${name}] full log: ${log_path}"
}

run_baseline "llama2-7b" "${LLAMA2_7B_MODEL}"
run_baseline "llama3.2-3b" "${LLAMA32_3B_MODEL}"
run_baseline "llama3.1-8b" "${LLAMA31_8B_MODEL}"

echo "All FP16 baseline evaluations completed for ${BASELINE_CONFIG_ID}."
echo "Results: ${RESULT_DIR}"
echo "Logs: ${LOG_DIR}"
