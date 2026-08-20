#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

if (( $# != 3 )); then
    echo "Usage: $0 MODEL ROTATION_CHECKPOINT OUTPUT_DIR" >&2
    exit 2
fi

MODEL=$1
ROTATION_CHECKPOINT=$2
OUTPUT_DIR=$3
CUDA_DEVICE=${CUDA_DEVICE:-0}
WEIGHT_CHECKPOINT=${WEIGHT_CHECKPOINT:-"${OUTPUT_DIR}/w4-gptq.pt"}
TASKS=${EVAL_TASKS:-mmlu,gsm8k_cot,bbh_cot_zeroshot,gpqa_diamond_zeroshot}

if [[ ! -f "${ROTATION_CHECKPOINT}" ]]; then
    echo "Rotation checkpoint not found: ${ROTATION_CHECKPOINT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export ZP_INT8=0 SIGNED_KV=0 ZP_CLAMP=1 SCALE_NO_UPCAST=1

if [[ -n ${TORCHRUN_MASTER_PORT:-} ]]; then
    RDZV_ARGS=(--master-port "${TORCHRUN_MASTER_PORT}")
else
    RDZV_ARGS=(--standalone)
fi

COMMON_ARGS=(
    --input_model "${MODEL}" --do_train False --do_eval True
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-1}"
    --model_max_length "${MODEL_MAX_LENGTH:-2048}"
    --fp16 True --bf16 False --save_safetensors False --seed "${SEED:-0}"
    --eval_tasks "${TASKS}"
    --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE:-1}"
)
if [[ -n ${EVAL_LIMIT:-} ]]; then
    COMMON_ARGS+=(--eval_limit "${EVAL_LIMIT}")
fi

run_eval() {
    local name=$1; shift
    local result_path="${OUTPUT_DIR}/results/${name}.json"
    local log_path="${OUTPUT_DIR}/logs/${name}.log"
    if [[ -s "${result_path}" ]]; then
        echo "[${name}] result exists; skipping: ${result_path}"
        return
    fi
    torchrun "${RDZV_ARGS[@]}" --nnodes=1 --nproc_per_node=1 ptq.py \
        "${COMMON_ARGS[@]}" "$@" --results_path "${result_path}" \
        2>&1 | tee "${log_path}"
    [[ -s "${result_path}" ]] || { echo "Missing ${result_path}" >&2; return 1; }
}

# FP reference: no rotation, no weight/KV/activation quantization.
run_eval fp_base \
    --attention_backend "${FP_ATTENTION_BACKEND:-eager}" \
    --w_bits 16 --w_groupsize -1 --a_bits 16 --a_groupsize -1 \
    --k_bits 16 --k_groupsize -1 --v_bits 16 --v_groupsize -1 \
    --q_bits 16 --q_groupsize -1 --p_bits 16 --p_groupsize -1

# Generate/reuse the common rotated W4/K4/V4 checkpoint.
CHECKPOINT_ARGS=()
if [[ -s "${WEIGHT_CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--load_qmodel_path "${WEIGHT_CHECKPOINT}")
else
    CHECKPOINT_ARGS=(--save_qmodel_path "${WEIGHT_CHECKPOINT}")
fi

run_eval aqp16 \
    --attention_backend eager --rotate \
    --optimized_rotation_path "${ROTATION_CHECKPOINT}" \
    --w_bits 4 --w_groupsize -1 --w_clip \
    --k_bits 4 --k_groupsize 128 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_asym \
    --a_bits 16 --a_groupsize -1 --a_asym \
    --q_bits 16 --q_groupsize 128 --p_bits 16 \
    "${CHECKPOINT_ARGS[@]}"

# AQP8 must load the exact checkpoint produced/reused by AQP16.
if [[ -s "${WEIGHT_CHECKPOINT}" ]]; then
    CHECKPOINT_ARGS=(--load_qmodel_path "${WEIGHT_CHECKPOINT}")
fi

run_eval aqp8 \
    --attention_backend eager --rotate \
    --optimized_rotation_path "${ROTATION_CHECKPOINT}" \
    --w_bits 4 --w_groupsize -1 --w_clip \
    --k_bits 4 --k_groupsize 128 --k_asym \
    --v_bits 4 --v_groupsize 128 --v_asym \
    --a_bits 8 --a_groupsize -1 --a_asym \
    --q_bits 8 --q_groupsize 128 --p_bits 8 \
    "${CHECKPOINT_ARGS[@]}"

echo "Hard workload comparison complete: ${OUTPUT_DIR}/results"
