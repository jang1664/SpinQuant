#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

if (( $# != 3 )); then
    echo "Usage: $0 MODEL ROTATION_CHECKPOINT OUTPUT_DIR" >&2
    echo "Runs fixed W4/KV4 with A/Q/P=(16,16,16) and (4,4,4)." >&2
    exit 2
fi

MODEL=$1
ROTATION_CHECKPOINT=$2
OUTPUT_DIR=$3
CUDA_DEVICE=${CUDA_DEVICE:-0}
WEIGHT_CHECKPOINT=${WEIGHT_CHECKPOINT:-"${OUTPUT_DIR}/w4-gptq.pt"}
read -r -a AQP_BIT_VALUES <<< "${AQP_BITS:-16 4}"

if [[ ! -f "${ROTATION_CHECKPOINT}" ]]; then
    echo "Rotation checkpoint not found: ${ROTATION_CHECKPOINT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}/results" "${OUTPUT_DIR}/logs"
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export ZP_INT8=0
export SIGNED_KV=0
export ZP_CLAMP=1
export SCALE_NO_UPCAST=1

if [[ -n ${TORCHRUN_MASTER_PORT:-} ]]; then
    RDZV_ARGS=(--master-port "${TORCHRUN_MASTER_PORT}")
else
    RDZV_ARGS=(--standalone)
fi

COMMON_ARGS=(
    --input_model "${MODEL}"
    --do_train False
    --do_eval True
    --per_device_eval_batch_size "${EVAL_BATCH_SIZE:-4}"
    --model_max_length "${MODEL_MAX_LENGTH:-2048}"
    --fp16 True
    --bf16 False
    --save_safetensors False
    --seed "${SEED:-0}"
    --attention_backend eager
    --w_bits 4
    --w_groupsize -1
    --k_bits 4
    --v_bits 4
    --k_groupsize 128
    --v_groupsize 128
    --w_clip
    --a_asym
    --k_asym
    --v_asym
    --rotate
)

run_condition() {
    local name=$1
    local aqp_bits=$2
    local result_path="${OUTPUT_DIR}/results/${name}.json"
    local log_path="${OUTPUT_DIR}/logs/${name}.log"
    local -a checkpoint_args

    if [[ -s "${result_path}" ]]; then
        echo "[${name}] result exists; skipping: ${result_path}"
        return
    fi
    if [[ -s "${WEIGHT_CHECKPOINT}" ]]; then
        checkpoint_args=(
            --load_qmodel_path "${WEIGHT_CHECKPOINT}"
            --optimized_rotation_path "${ROTATION_CHECKPOINT}"
        )
    else
        checkpoint_args=(
            --optimized_rotation_path "${ROTATION_CHECKPOINT}"
            --save_qmodel_path "${WEIGHT_CHECKPOINT}"
        )
    fi

    echo "[${name}] W4/KV4, A/Q/P=${aqp_bits}/${aqp_bits}/${aqp_bits}"
    torchrun "${RDZV_ARGS[@]}" --nnodes=1 --nproc_per_node=1 ptq.py \
        "${COMMON_ARGS[@]}" \
        --a_bits "${aqp_bits}" \
        --a_groupsize -1 \
        --q_bits "${aqp_bits}" \
        --q_groupsize 128 \
        --p_bits "${aqp_bits}" \
        --p_groupsize -1 \
        "${checkpoint_args[@]}" \
        --results_path "${result_path}" \
        2>&1 | tee -a "${log_path}"

    if [[ ! -s "${result_path}" ]]; then
        echo "[${name}] evaluation did not produce ${result_path}" >&2
        return 1
    fi
}

# Generate/reuse one rotated W4 checkpoint so W/KV/R1/R2 remain fixed.
for aqp_bits in "${AQP_BIT_VALUES[@]}"; do
    if (( aqp_bits < 1 || aqp_bits > 16 )); then
        echo "Invalid AQP bit-width: ${aqp_bits}" >&2
        exit 2
    fi
    run_condition "aqp${aqp_bits}" "${aqp_bits}"
done

echo "A/Q/P comparison complete: ${OUTPUT_DIR}/results"
