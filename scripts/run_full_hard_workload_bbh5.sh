#!/usr/bin/env bash

# Long-running full evaluation.
# Conditions: FP base, A/Q/P bypassed at 16-bit, and A/Q/P at 8-bit.
# Workloads: full MMLU, full GSM8K, full GPQA Diamond, and five BBH subtasks.

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

if (( $# != 3 )); then
    echo "Usage: $0 MODEL ROTATION_CHECKPOINT OUTPUT_DIR" >&2
    echo "Runs a multi-day full evaluation; no EVAL_LIMIT is applied." >&2
    exit 2
fi

MODEL=$1
ROTATION_CHECKPOINT=$2
OUTPUT_DIR=$3

# Keep all settings overridable from the environment.
export CUDA_DEVICE=${CUDA_DEVICE:-0}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
export LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-1}
export MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-2048}
export SEED=${SEED:-0}
export TORCHRUN_MASTER_PORT=${TORCHRUN_MASTER_PORT:-}

# Do not inherit a smoke-test limit from the shell environment.
unset EVAL_LIMIT

export EVAL_TASKS="mmlu,gsm8k_cot,bbh_cot_zeroshot_boolean_expressions,bbh_cot_zeroshot_logical_deduction_three_objects,bbh_cot_zeroshot_navigate,bbh_cot_zeroshot_multistep_arithmetic_two,bbh_cot_zeroshot_tracking_shuffled_objects_three_objects,gpqa_diamond_zeroshot"

if [[ -z ${WEIGHT_CHECKPOINT:-} ]]; then
    export WEIGHT_CHECKPOINT="${OUTPUT_DIR}/w4-gptq.pt"
fi

if [[ ! -f "${ROTATION_CHECKPOINT}" ]]; then
    echo "Rotation checkpoint not found: ${ROTATION_CHECKPOINT}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Starting long-running hard-workload evaluation"
echo "  model=${MODEL}"
echo "  output=${OUTPUT_DIR}"
echo "  cuda_device=${CUDA_DEVICE}"
echo "  model_max_length=${MODEL_MAX_LENGTH}"
echo "  lm_eval_batch_size=${LM_EVAL_BATCH_SIZE}"
echo "  tasks=${EVAL_TASKS}"
echo "  conditions=fp_base,aqp16,aqp8"
echo "  eval_limit=<none>"

exec "${SCRIPT_DIR}/run_hard_workload_comparison.sh" \
    "${MODEL}" "${ROTATION_CHECKPOINT}" "${OUTPUT_DIR}"
