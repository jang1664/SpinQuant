#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

LOGIT_CONFIG=${LOGIT_CONFIG:-both}
LOGIT_SEQUENCE_LENGTH=${LOGIT_SEQUENCE_LENGTH:-2048}
LOGIT_TOKEN_CHUNK_SIZE=${LOGIT_TOKEN_CHUNK_SIZE:-16}
LOGIT_TEMPERATURE=${LOGIT_TEMPERATURE:-1.0}
LOGIT_EAR_TOP_K=${LOGIT_EAR_TOP_K:-10}
LOGIT_MIN_FREE_MIB=${LOGIT_MIN_FREE_MIB:-38000}
LOGIT_MAX_GPU_UTIL=${LOGIT_MAX_GPU_UTIL:-20}
LOGIT_GPU_POLL_SECONDS=${LOGIT_GPU_POLL_SECONDS:-30}
LOGIT_WAIT_FOR_GPU=${LOGIT_WAIT_FOR_GPU:-0}
PYTHON_BIN=${PYTHON_BIN:-python}

export ZP_INT8=0
export SIGNED_KV=0
export ZP_CLAMP=1
export SCALE_NO_UPCAST=1

WEIGHT_CONFIG_ID="w4-gptq-wclip-wgs-1-spinquant-optrot"
MODEL_OUTPUT_ROOT="saved_models/${WEIGHT_CONFIG_ID}"
FP_RESULT_DIR="results/w16a16kv16-fp16-baseline"
A16_RESULT_DIR="results/w4a16kv4-gptq-wclip-aasym-kasym-vasym-wgs-1-ags-1-kgs128-vgs128-zpint80-signedkv0-zpclamp1-scalenoup1"
A8_RESULT_DIR="results/w4a8kv4-gptq-wclip-aasym-kasym-vasym-wgs-1-ags-1-kgs128-vgs128-zpint80-signedkv0-zpclamp1-scalenoup1"

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
    echo "Usage: $0 [LLAMA2_7B_MODEL LLAMA32_3B_MODEL LLAMA31_8B_MODEL]"
    echo "Compares FP16 and SpinQuant over the complete accuracy and WikiText PPL workload."
    echo "Accuracy KL uses every stored sample/choice from the existing result JSONs."
    echo "PPL KL uses every token and full-vocabulary logits in all 62 WikiText documents."
    echo ""
    echo "GPU scheduling:"
    echo "  GPUs are selected automatically using nvidia-smi and run one job each."
    echo "  LOGIT_GPUS=2,3       restricts scheduling to the listed physical GPU IDs."
    echo "  CUDA_DEVICE=2        backward-compatible single-GPU restriction."
    echo "  LOGIT_MIN_FREE_MIB   required free memory (default: 38000)."
    echo "  LOGIT_MAX_GPU_UTIL   maximum current utilization (default: 20)."
    echo "  LOGIT_WAIT_FOR_GPU=1 waits until at least one eligible GPU is available."
    echo ""
    echo "Evaluation:"
    echo "  LOGIT_CONFIG selects a16, a8, or both (default: both)."
    echo "  LOGIT_SEQUENCE_LENGTH controls rolling continuation length (default: 2048)."
    echo "  LOGIT_TOKEN_CHUNK_SIZE controls metric-memory usage (default: 16)."
    echo "  LOGIT_TEMPERATURE controls KL softmax temperature (default: 1.0)."
    echo "  LOGIT_EAR_TOP_K controls paper-style EAR support size (default: 10)."
    echo "  FORCE_LOGIT_METRICS=1 recomputes existing output JSON files."
    echo "  LOGIT_MAX_DOCUMENTS and LOGIT_MAX_TOKENS_PER_DOCUMENT are testing-only limits."
    echo "  Existing complete-workload metric JSON files are skipped."
    exit 0
fi
if (( $# != 0 && $# != 3 )); then
    echo "Expected zero or three model arguments. Use --help for usage." >&2
    exit 2
fi
case "${LOGIT_CONFIG}" in
    a16|a8|both) ;;
    *)
        echo "LOGIT_CONFIG must be a16, a8, or both; got ${LOGIT_CONFIG}" >&2
        exit 2
        ;;
esac
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required for automatic GPU scheduling." >&2
    exit 1
fi

LLAMA2_7B_MODEL=${1:-${LLAMA2_7B_MODEL:-./models/llama2-7b}}
LLAMA32_3B_MODEL=${2:-${LLAMA32_3B_MODEL:-./models/llama3.2-3b}}
LLAMA31_8B_MODEL=${3:-${LLAMA31_8B_MODEL:-./models/llama3.1-8b}}

declare -a JOB_NAMES=()
declare -a JOB_MODELS=()
declare -a JOB_A_BITS=()
declare -a JOB_RESULT_DIRS=()

add_job() {
    JOB_NAMES+=("$1")
    JOB_MODELS+=("$2")
    JOB_A_BITS+=("$3")
    JOB_RESULT_DIRS+=("$4")
}

# Submit both activation variants of larger models first. This prevents one
# worker from receiving all large jobs when several idle GPUs are available.
for model_spec in \
    "llama3.1-8b|${LLAMA31_8B_MODEL}" \
    "llama2-7b|${LLAMA2_7B_MODEL}" \
    "llama3.2-3b|${LLAMA32_3B_MODEL}"; do
    name=${model_spec%%|*}
    model=${model_spec#*|}
    if [[ ${LOGIT_CONFIG} == "a16" || ${LOGIT_CONFIG} == "both" ]]; then
        add_job "${name}" "${model}" 16 "${A16_RESULT_DIR}"
    fi
    if [[ ${LOGIT_CONFIG} == "a8" || ${LOGIT_CONFIG} == "both" ]]; then
        add_job "${name}" "${model}" 8 "${A8_RESULT_DIR}"
    fi
done

requested_gpu() {
    local gpu=$1
    local requested=${LOGIT_GPUS:-${CUDA_DEVICE:-}}
    if [[ -z ${requested} ]]; then
        return 0
    fi
    requested=",${requested// /},"
    [[ ${requested} == *",${gpu},"* ]]
}

discover_gpus() {
    ELIGIBLE_GPUS=()
    GPU_STATUS_LINES=()
    while IFS=',' read -r gpu free_mib utilization; do
        gpu=${gpu//[[:space:]]/}
        free_mib=${free_mib//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        if ! requested_gpu "${gpu}"; then
            continue
        fi
        local state="busy"
        if (( free_mib >= LOGIT_MIN_FREE_MIB && utilization <= LOGIT_MAX_GPU_UTIL )); then
            state="eligible"
            ELIGIBLE_GPUS+=("${gpu}")
        fi
        GPU_STATUS_LINES+=("GPU ${gpu}: free=${free_mib} MiB util=${utilization}% ${state}")
    done < <(nvidia-smi \
        --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits)
}

while true; do
    discover_gpus
    if (( ${#ELIGIBLE_GPUS[@]} > 0 )); then
        break
    fi
    printf '%s\n' "${GPU_STATUS_LINES[@]}"
    if [[ ${LOGIT_WAIT_FOR_GPU} != 1 ]]; then
        echo "No eligible GPU. Adjust LOGIT_MIN_FREE_MIB/LOGIT_MAX_GPU_UTIL or set LOGIT_WAIT_FOR_GPU=1." >&2
        exit 1
    fi
    echo "No eligible GPU; checking again in ${LOGIT_GPU_POLL_SECONDS}s."
    sleep "${LOGIT_GPU_POLL_SECONDS}"
done

printf '%s\n' "${GPU_STATUS_LINES[@]}"
echo "Selected GPUs: ${ELIGIBLE_GPUS[*]}"
echo "Queued jobs: ${#JOB_NAMES[@]} (LOGIT_CONFIG=${LOGIT_CONFIG})"

run_case() {
    local gpu=$1
    local name=$2
    local model=$3
    local a_bits=$4
    local result_dir=$5
    local checkpoint="${MODEL_OUTPUT_ROOT}/${name}/w4-gptq.pt"
    local fp_result="${FP_RESULT_DIR}/${name}.json"
    local quant_result="${result_dir}/${name}.json"
    local limit_suffix=""
    if [[ -n ${LOGIT_MAX_DOCUMENTS:-} || -n ${LOGIT_MAX_TOKENS_PER_DOCUMENT:-} ]]; then
        limit_suffix="-limited-d${LOGIT_MAX_DOCUMENTS:-all}-t${LOGIT_MAX_TOKENS_PER_DOCUMENT:-all}"
    fi
    local output="${result_dir}/${name}-full-workload-divergence${limit_suffix}.json"
    local log="${result_dir}/${name}-full-workload-divergence${limit_suffix}.log"
    local -a limit_args=()

    if [[ -s ${output} && ${FORCE_LOGIT_METRICS:-0} != 1 ]]; then
        echo "[GPU ${gpu}] [${name} A${a_bits}] already exists; skipping: ${output}"
        return 0
    fi
    case "${model}" in
        /*|./*|../*)
            if [[ ! -e ${model} ]]; then
                echo "Local model not found: ${model}" >&2
                return 1
            fi
            ;;
    esac
    for required in "${checkpoint}" "${fp_result}" "${quant_result}"; do
        if [[ ! -s ${required} ]]; then
            echo "Required input not found: ${required}" >&2
            return 1
        fi
    done
    if [[ -n ${LOGIT_MAX_DOCUMENTS:-} ]]; then
        limit_args+=(--max-documents "${LOGIT_MAX_DOCUMENTS}")
    fi
    if [[ -n ${LOGIT_MAX_TOKENS_PER_DOCUMENT:-} ]]; then
        limit_args+=(--max-tokens-per-document "${LOGIT_MAX_TOKENS_PER_DOCUMENT}")
    fi

    mkdir -p "${result_dir}"
    echo "[GPU ${gpu}] [${name} A${a_bits}] started; log=${log}"
    if CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" measure_logit_divergence.py \
        --input-model "${model}" \
        --load-qmodel-path "${checkpoint}" \
        --fp-results-path "${fp_result}" \
        --quant-results-path "${quant_result}" \
        --a-bits "${a_bits}" \
        --sequence-length "${LOGIT_SEQUENCE_LENGTH}" \
        --token-chunk-size "${LOGIT_TOKEN_CHUNK_SIZE}" \
        --temperature "${LOGIT_TEMPERATURE}" \
        --ear-top-k "${LOGIT_EAR_TOP_K}" \
        --output "${output}" \
        "${limit_args[@]}" \
        >"${log}" 2>&1; then
        if [[ ! -s ${output} ]]; then
            echo "[GPU ${gpu}] [${name} A${a_bits}] output was not created: ${output}" >&2
            return 1
        fi
        echo "[GPU ${gpu}] [${name} A${a_bits}] completed: ${output}"
    else
        local status=$?
        echo "[GPU ${gpu}] [${name} A${a_bits}] failed (exit ${status}); log=${log}" >&2
        return "${status}"
    fi
}

declare -A PID_GPU=()
declare -A PID_LABEL=()
next_job=0
failures=0

launch_next() {
    local gpu=$1
    if (( next_job >= ${#JOB_NAMES[@]} )); then
        return 1
    fi
    local index=${next_job}
    next_job=$((next_job + 1))
    run_case \
        "${gpu}" \
        "${JOB_NAMES[index]}" \
        "${JOB_MODELS[index]}" \
        "${JOB_A_BITS[index]}" \
        "${JOB_RESULT_DIRS[index]}" &
    local pid=$!
    PID_GPU[${pid}]="${gpu}"
    PID_LABEL[${pid}]="${JOB_NAMES[index]} A${JOB_A_BITS[index]}"
    return 0
}

for gpu in "${ELIGIBLE_GPUS[@]}"; do
    launch_next "${gpu}" || break
done

while (( ${#PID_GPU[@]} > 0 )); do
    completed_pid=""
    if wait -n -p completed_pid; then
        status=0
    else
        status=$?
    fi
    gpu=${PID_GPU[${completed_pid}]}
    label=${PID_LABEL[${completed_pid}]}
    unset 'PID_GPU['"${completed_pid}"']'
    unset 'PID_LABEL['"${completed_pid}"']'
    if (( status != 0 )); then
        echo "Job failed: ${label} on GPU ${gpu} (exit ${status})" >&2
        failures=$((failures + 1))
    fi
    launch_next "${gpu}" || true
done

if (( failures > 0 )); then
    echo "Complete-workload divergence finished with ${failures} failed job(s)." >&2
    exit 1
fi
echo "Complete-workload divergence completed for all jobs."
