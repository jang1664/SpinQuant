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

W_BITS=4
A_BITS=16
K_BITS=4
V_BITS=4
W_GROUPSIZE=-1
A_GROUPSIZE=-1
K_GROUPSIZE=128
V_GROUPSIZE=128

# Match run_ptq_w4a8kv4.sh so activation precision is the only experimental
# difference. A_ASYM has no numerical effect when A_BITS=16, but retaining the
# flag keeps the invocation directly comparable with the A8 run.
export ZP_INT8=0
export SIGNED_KV=0
export ZP_CLAMP=1
export SCALE_NO_UPCAST=1

# The rotated GPTQ weights do not depend on runtime activation/KV precision.
# Reuse the checkpoint produced by run_ptq_w4a8kv4.sh when it is available.
WEIGHT_CONFIG_ID="w${W_BITS}-gptq-wclip-wgs${W_GROUPSIZE}-spinquant-optrot"
KV_CONFIG="kv${K_BITS}"
QUANT_CONFIG_ID="w${W_BITS}a${A_BITS}${KV_CONFIG}-gptq-wclip-aasym-kasym-vasym-wgs${W_GROUPSIZE}-ags${A_GROUPSIZE}-kgs${K_GROUPSIZE}-vgs${V_GROUPSIZE}-zpint8${ZP_INT8}-signedkv${SIGNED_KV}-zpclamp${ZP_CLAMP}-scalenoup${SCALE_NO_UPCAST}"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    echo "Usage: $0 [LLAMA2_7B_MODEL LLAMA32_3B_MODEL LLAMA31_8B_MODEL]"
    echo "       $0 [LLAMA31_8B_MODEL]"
    echo "Model arguments can be local paths or Hugging Face model IDs."
    echo "With one argument, only the Llama-3.1 path is overridden; all three models are still evaluated."
    echo "CUDA device can be selected with CUDA_DEVICE (default: 0)."
    echo "torchrun uses an automatically selected free port; set TORCHRUN_MASTER_PORT to override it."
    echo "Evaluates Llama-2 7B, Llama-3.2 3B, and Llama-3.1 8B with W4A16KV4."
    echo "Existing W4 checkpoints are reused; set FORCE_REQUANTIZE=1 to regenerate them."
    echo "Models with a completed result JSON are skipped individually."
    exit 0
fi
if (( $# != 0 && $# != 1 && $# != 3 )); then
    echo "Expected zero, one, or three model arguments. Use --help for usage." >&2
    exit 2
fi

if (( $# == 1 )); then
    LLAMA2_7B_MODEL=${LLAMA2_7B_MODEL:-./models/llama2-7b}
    LLAMA32_3B_MODEL=${LLAMA32_3B_MODEL:-./models/llama3.2-3b}
    LLAMA31_8B_MODEL=$1
else
    LLAMA2_7B_MODEL=${1:-${LLAMA2_7B_MODEL:-./models/llama2-7b}}
    LLAMA32_3B_MODEL=${2:-${LLAMA32_3B_MODEL:-./models/llama3.2-3b}}
    LLAMA31_8B_MODEL=${3:-${LLAMA31_8B_MODEL:-./models/llama3.1-8b}}
fi

LLAMA2_7B_ROTATION=${LLAMA2_7B_ROTATION:-rotation_llama-2-7b/a16w4kv4-vasym/R.bin}
LLAMA32_3B_ROTATION=${LLAMA32_3B_ROTATION:-rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin}
LLAMA31_8B_ROTATION=${LLAMA31_8B_ROTATION:-rotation_llama-3.1-8b/a16w4kv4-vasym/R.bin}

MODEL_OUTPUT_ROOT="saved_models/${WEIGHT_CONFIG_ID}"
RESULT_DIR="results/${QUANT_CONFIG_ID}"
LOG_DIR="logs/${QUANT_CONFIG_ID}"

mkdir -p "${MODEL_OUTPUT_ROOT}" "${RESULT_DIR}" "${LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"

echo "Quantization config: ${QUANT_CONFIG_ID}"
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

run_ptq() {
    local name=$1
    local model=$2
    local rotation=$3
    local result_path="${RESULT_DIR}/${name}.json"
    local log_path="${LOG_DIR}/${name}.log"
    local model_output_dir="${MODEL_OUTPUT_ROOT}/${name}"
    local model_output_path="${model_output_dir}/w4-gptq.pt"
    local -a checkpoint_args

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

    mkdir -p "${model_output_dir}"

    if [[ -s "${model_output_path}" && ${FORCE_REQUANTIZE:-0} != 1 ]]; then
        echo "[${name}] existing quantized model found; skipping rotation/GPTQ"
        echo "[${name}] checkpoint=${model_output_path}"
        checkpoint_args=(--load_qmodel_path "${model_output_path}")
    else
        if [[ ! -f "${rotation}" ]]; then
            echo "Rotation checkpoint not found: ${rotation}" >&2
            return 1
        fi
        echo "[${name}] rotation=${rotation}"
        checkpoint_args=(
            --optimized_rotation_path "${rotation}"
            --save_qmodel_path "${model_output_path}"
        )
    fi

    echo "[${name}] model=${model}"
    echo "[${name}] quantization=W4A16KV4 (GPTQ weight)"
    echo "[${name}] activation=FP16 KV=head-wise (k/v_groupsize=${K_GROUPSIZE})"
    echo "[${name}] ZP_INT8=0 SIGNED_KV=0 ZP_CLAMP=1 SCALE_NO_UPCAST=1"

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
        --v_bits "${V_BITS}" \
        --w_clip \
        --a_asym \
        --k_asym \
        --v_asym \
        --k_groupsize "${K_GROUPSIZE}" \
        --v_groupsize "${V_GROUPSIZE}" \
        --rotate \
        "${checkpoint_args[@]}" \
        --results_path "${result_path}" \
        2>&1 | tee -a "${log_path}"

    if [[ ! -s "${result_path}" ]]; then
        echo "[${name}] PTQ evaluation did not produce ${result_path}" >&2
        return 1
    fi

    echo "[${name}] quantized model: ${model_output_path}"
    echo "[${name}] accuracy/PPL JSON: ${result_path}"
    echo "[${name}] full log: ${log_path}"
}

run_ptq "llama2-7b" "${LLAMA2_7B_MODEL}" "${LLAMA2_7B_ROTATION}"
run_ptq "llama3.2-3b" "${LLAMA32_3B_MODEL}" "${LLAMA32_3B_ROTATION}"
run_ptq "llama3.1-8b" "${LLAMA31_8B_MODEL}" "${LLAMA31_8B_ROTATION}"

echo "All PTQ/evaluation runs completed for ${QUANT_CONFIG_ID}."
echo "Results: ${RESULT_DIR}"
echo "Logs: ${LOG_DIR}"
echo "Shared quantized weight models: ${MODEL_OUTPUT_ROOT}"
