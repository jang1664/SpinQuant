#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_DIR}"

MODEL_NAME=${MODEL_NAME:-llama31-8b}
COMPUTE_DTYPE=${COMPUTE_DTYPE:-fp16}
ROTATION_DTYPE=${ROTATION_DTYPE:-${COMPUTE_DTYPE}}
MODEL=${MODEL:-./models/llama3.1-8b}
ROTATION=${ROTATION:-rotation_llama-3.1-8b/a16w4kv4-vasym/R.bin}
CHECKPOINT=${CHECKPOINT:-saved_models/w4-gptq-wclip-wgs128-spinquant-optrot/llama3.1-8b/w4-gptq-${COMPUTE_DTYPE}-fpint-v2.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/fpint-mxu128-${MODEL_NAME}-${COMPUTE_DTYPE}}
LOG_ROOT=${LOG_ROOT:-logs/fpint-mxu128-${MODEL_NAME}-${COMPUTE_DTYPE}}
REPORT=${REPORT:-docs/FP-INT-hw-acc/mxu128_${MODEL_NAME}_${COMPUTE_DTYPE}_results.md}
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-29610}
RANDOM_TRIALS=${RANDOM_TRIALS:-30}
SEED=${SEED:-0}

if [[ "${COMPUTE_DTYPE}" != fp16 && "${COMPUTE_DTYPE}" != bf16 ]]; then
    echo "COMPUTE_DTYPE must be fp16 or bf16; got: ${COMPUTE_DTYPE}" >&2
    exit 1
fi
if [[ "${COMPUTE_DTYPE}" == bf16 ]]; then
    PRECISION_ARGS=(--fp16 False --bf16 True)
    EXPECTED_SAMPLER=k_scaled_finite_bf16_fields_v1
else
    PRECISION_ARGS=(--fp16 True --bf16 False)
    EXPECTED_SAMPLER=k_scaled_finite_fp16_fields_v3
fi

RANDOM_RESULT=${RANDOM_RESULT:-${OUTPUT_ROOT}/random-qcol-fp64-errors.json}
SMOKE_RESULT="${OUTPUT_ROOT}/standard-smoke.json"
PAIRED_SMOKE_RESULT="${OUTPUT_ROOT}/paired-smoke-sanity.json"
FULL_RESULT_DIR="${OUTPUT_ROOT}/full-shards"
FULL_METRICS_RESULT="${OUTPUT_ROOT}/full-metrics.json"

mkdir -p "$(dirname -- "${CHECKPOINT}")" "$(dirname -- "${RANDOM_RESULT}")" \
    "${OUTPUT_ROOT}" "${FULL_RESULT_DIR}" "${LOG_ROOT}" "$(dirname -- "${REPORT}")"
IFS=',' read -r -a GPU_IDS <<< "${CUDA_DEVICES}"
if (( ${#GPU_IDS[@]} < 4 )); then
    echo "CUDA_DEVICES must list at least four GPUs; got: ${CUDA_DEVICES}" >&2
    exit 1
fi

if [[ ! -r "${MODEL}/config.json" ]]; then
    echo "Model is not readable: ${MODEL}" >&2
    exit 1
fi
if [[ ! -r "${ROTATION}" ]]; then
    echo "Rotation checkpoint is not readable: ${ROTATION}" >&2
    exit 1
fi
if ! command -v ninja >/dev/null 2>&1; then
    echo "ninja is not on PATH; activate the spinquant conda environment first" >&2
    exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is required" >&2
    exit 1
fi

echo "Model: ${MODEL}"
echo "Model name: ${MODEL_NAME}"
echo "Compute dtype: ${COMPUTE_DTYPE}"
echo "Rotation optimization dtype: ${ROTATION_DTYPE}"
echo "Rotation: ${ROTATION}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_ROOT}"
echo "Task GPUs: ${GPU_IDS[*]:0:4}"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader

FPINT_KERNEL_SHA256=$(sha256sum fpint_emul/csrc/fpint_cuda_kernel.cu | awk '{print $1}')
ROTATION_SHA256=$(sha256sum "${ROTATION}" | awk '{print $1}')
ROTATION_METADATA="$(dirname -- "${ROTATION}")/rotation-metadata.json"
if [[ -s "${ROTATION_METADATA}" ]]; then
    "${PYTHON_BIN}" - "${ROTATION_METADATA}" "${ROTATION_DTYPE}" "${ROTATION_SHA256}" <<'PY'
import json
import sys

path, expected_dtype, expected_sha256 = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    metadata = json.load(handle)
if metadata.get("optimization_dtype") != expected_dtype:
    raise SystemExit(f"rotation dtype mismatch in {path}")
if metadata.get("rotation_sha256") != expected_sha256:
    raise SystemExit(f"rotation SHA256 mismatch in {path}")
PY
else
    echo "Warning: rotation provenance metadata is missing: ${ROTATION_METADATA}" >&2
fi
CHECKPOINT_SHA256=${SPINQUANT_CHECKPOINT_SHA256:-}
if [[ -s "${CHECKPOINT}" && -z "${CHECKPOINT_SHA256}" ]]; then
    CHECKPOINT_SHA256=$(sha256sum "${CHECKPOINT}" | awk '{print $1}')
fi
if [[ -n "${CHECKPOINT_SHA256}" ]]; then
    export SPINQUANT_CHECKPOINT_SHA256="${CHECKPOINT_SHA256}"
fi
export SPINQUANT_ROTATION_DTYPE="${ROTATION_DTYPE}"
if ! "${PYTHON_BIN}" - "${RANDOM_RESULT}" "${FPINT_KERNEL_SHA256}" "${RANDOM_TRIALS}" "${COMPUTE_DTYPE}" "${EXPECTED_SAMPLER}" <<'PY'
import json
import sys

path, kernel_sha, trials, compute_dtype, expected_sampler = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as handle:
        result = json.load(handle)
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
config = result.get("config", {})
valid = (
    result.get("status") == "pass"
    and result.get("evaluation_policy", {}).get("fp64_gpu_ground_truth") is True
    and result.get("evaluation_policy", {}).get(
        "common_finite_mask_for_paired_errors"
    ) is True
    and result.get("evaluation_policy", {}).get(
        "qcol_reference_must_be_allclose"
    ) is True
    and config.get("sampler_version") == expected_sampler
    and config.get("operation") == f"raw_{compute_dtype}_times_signed_int"
    and config.get("activation_format") == compute_dtype
    and config.get("m") == 32
    and config.get("n") == 32
    and config.get("trials") == int(trials)
    and config.get("k_values")
    == [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    and config.get(f"{compute_dtype}_fields", {}).get("sign") == [0, 1]
    and config.get(f"{compute_dtype}_fields", {}).get("exponent_min") == 0
    and config.get(f"{compute_dtype}_fields", {}).get("mantissa")
    == ([0, 1023] if compute_dtype == "fp16" else [0, 127])
    and config.get("integer_range") == [-8, 7]
    and config.get("finite_target") == 0.999
    and "conventional_err" in result.get("summary", {}).get("overall", {})
    and "fp_int_err" in result.get("summary", {}).get("overall", {})
    and result.get("environment", {}).get("fpint_cuda_kernel_sha256")
    == kernel_sha
)
raise SystemExit(0 if valid else 1)
PY
then
    CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${PYTHON_BIN}" measure_fpint_qcol_accuracy.py \
        --bits 4 --group-size 128 --mxu-rows 128 \
        --activation-format "${COMPUTE_DTYPE}" \
        --trials "${RANDOM_TRIALS}" --device cuda:0 \
        --output "${RANDOM_RESULT}" \
        2>&1 | tee "${LOG_ROOT}/random-qcol-fp64-errors.log"
fi

COMMON_ARGS=(
    --input_model "${MODEL}"
    --do_train False --do_eval True
    --per_device_eval_batch_size 1 --model_max_length 2048
    "${PRECISION_ARGS[@]}" --save_safetensors False --seed "${SEED}"
    --attention_backend eager
    --rotate --optimized_rotation_path "${ROTATION}"
    --w_bits 4 --w_groupsize 128 --w_clip
    --a_bits 16 --a_groupsize -1
    --q_bits 16 --q_groupsize -1
    --p_bits 16 --p_groupsize -1
    --k_bits 16 --k_groupsize -1
    --v_bits 16 --v_groupsize -1
    --fpint_mxu_rows 128 --fpint_extra_bits 19 --fpint_reduce_extra_bits 10
)

result_matches() {
    local result=$1
    local backend=$2
    local tasks=$3
    local batch_size=$4
    [[ -s "${result}" ]] || return 1
    "${PYTHON_BIN}" - "${result}" "${backend}" "${tasks}" "${CHECKPOINT}" "${batch_size}" "${COMPUTE_DTYPE}" "${ROTATION_DTYPE}" "${CHECKPOINT_SHA256}" "${ROTATION_SHA256}" "${FPINT_KERNEL_SHA256}" <<'PY'
import json
import sys

(
    path,
    backend,
    tasks,
    checkpoint,
    batch_size,
    compute_dtype,
    rotation_dtype,
    checkpoint_sha256,
    rotation_sha256,
    fpint_kernel_sha256,
) = sys.argv[1:]
with open(path, encoding="utf-8") as handle:
    result = json.load(handle)
metadata = result.get("spinquant_quantization", {})
expected_tasks = set(tasks.split(","))
valid = (
    set(result.get("results", {})) == expected_tasks
    and metadata.get("linear_backend") == backend
    and metadata.get("weight_bits") == 4
    and metadata.get("weight_groupsize") == 128
    and metadata.get("weight_symmetric") is True
    and metadata.get("compute_dtype") == compute_dtype
    and metadata.get("rotation_optimization_dtype") == rotation_dtype
    and metadata.get("rotation_checkpoint_sha256") == rotation_sha256
    and (
        backend != "fpint_cuda"
        or metadata.get("fpint_cuda_kernel_sha256") == fpint_kernel_sha256
    )
    and metadata.get("fpint_mxu_rows") == 128
    and metadata.get("quantized_checkpoint") == checkpoint
    and (
        not checkpoint_sha256
        or metadata.get("quantized_checkpoint_sha256") == checkpoint_sha256
    )
    and str(metadata.get("lm_eval_batch_size")) == batch_size
)
if "/full-shards/" in path:
    valid = valid and isinstance(metadata.get("evaluation_seconds"), (int, float))
raise SystemExit(0 if valid else 1)
PY
}

run_ptq() {
    local name=$1
    local backend=$2
    local tasks=$3
    local result=$4
    local limit=${5:-}
    local checkpoint_mode=$6
    local gpu=$7
    local port=$8
    local batch_size=$9
    local -a checkpoint_args
    if [[ "${checkpoint_mode}" != save ]] && result_matches "${result}" "${backend}" "${tasks}" "${batch_size}"; then
        echo "[${name}] validated result exists; skipping: ${result}"
        return
    fi
    if [[ "${checkpoint_mode}" == save ]]; then
        checkpoint_args=(--save_qmodel_path "${CHECKPOINT}")
    else
        checkpoint_args=(--load_qmodel_path "${CHECKPOINT}")
    fi
    local -a limit_args=()
    if [[ -n "${limit}" ]]; then
        limit_args=(--eval_limit "${limit}")
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m torch.distributed.run \
        --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 --master_port="${port}" ptq.py \
        "${COMMON_ARGS[@]}" \
        --linear_backend "${backend}" \
        --eval_tasks "${tasks}" \
        --lm_eval_batch_size "${batch_size}" \
        "${limit_args[@]}" \
        "${checkpoint_args[@]}" \
        --results_path "${result}" \
        2>&1 | tee "${LOG_ROOT}/${name}.log"
    [[ -s "${result}" ]] || { echo "Missing result: ${result}" >&2; return 1; }
}

if [[ -s "${CHECKPOINT}" ]]; then
    run_ptq standard-smoke standard wikitext "${SMOKE_RESULT}" 4 load "${GPU_IDS[0]}" "${MASTER_PORT_BASE}" 1
else
    run_ptq standard-smoke standard wikitext "${SMOKE_RESULT}" 4 save "${GPU_IDS[0]}" "${MASTER_PORT_BASE}" 1
fi

if [[ -z "${CHECKPOINT_SHA256}" ]]; then
    CHECKPOINT_SHA256=$("${PYTHON_BIN}" - "${SMOKE_RESULT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["spinquant_quantization"]["quantized_checkpoint_sha256"])
PY
)
    export SPINQUANT_CHECKPOINT_SHA256="${CHECKPOINT_SHA256}"
fi

if ! "${PYTHON_BIN}" - "${PAIRED_SMOKE_RESULT}" "${FPINT_KERNEL_SHA256}" "${CHECKPOINT_SHA256}" "${COMPUTE_DTYPE}" "${ROTATION_DTYPE}" <<'PY'
import json
import sys

try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        result = json.load(handle)
except (FileNotFoundError, json.JSONDecodeError):
    raise SystemExit(1)
conditions = result.get("conditions", {})
valid = (
    result.get("status") == "pass"
    and result.get("evaluation_policy", {}).get("name") == "sanity_v1"
    and result.get("environment", {}).get("fpint_cuda_kernel_sha256")
    == sys.argv[2]
    and conditions.get("quantized_checkpoint_sha256") == sys.argv[3]
    and conditions.get("compute_dtype") == sys.argv[4]
    and conditions.get("rotation_optimization_dtype") == sys.argv[5]
)
raise SystemExit(0 if valid else 1)
PY
then
    if ! CUDA_VISIBLE_DEVICES="${GPU_IDS[0]}" "${PYTHON_BIN}" measure_fpint_backend_accuracy.py \
        --input-model "${MODEL}" \
        --load-qmodel-path "${CHECKPOINT}" \
        --source-results-path "${SMOKE_RESULT}" \
        --output "${PAIRED_SMOKE_RESULT}" \
        --group-size 128 --mxu-rows 128 \
        --sequence-length 256 --max-documents 4 --max-tokens-per-document 256 \
        --capture-linears --device cuda:0 \
        --compute-dtype "${COMPUTE_DTYPE}" \
        --rotation-optimization-dtype "${ROTATION_DTYPE}" \
        2>&1 | tee "${LOG_ROOT}/paired-smoke.log"; then
        echo "Paired smoke command reported a failed sanity gate; validating its JSON result." >&2
    fi
fi

if ! "${PYTHON_BIN}" - "${PAIRED_SMOKE_RESULT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    result = json.load(handle)
if (
    result.get("status") != "pass"
    or result.get("evaluation_policy", {}).get("name") != "sanity_v1"
):
    raise SystemExit(1)
PY
then
    "${PYTHON_BIN}" summarize_fpint_mxu128.py \
        --random "${RANDOM_RESULT}" \
        --output "${REPORT}"
    echo "Stopping before full workload: paired sanity gate failed." >&2
    exit 1
fi

if [[ ${STOP_AFTER_SMOKE:-0} == 1 ]]; then
    "${PYTHON_BIN}" summarize_fpint_mxu128.py \
        --random "${RANDOM_RESULT}" \
        --output "${REPORT}"
    echo "Smoke stage complete: ${PAIRED_SMOKE_RESULT}"
    exit 0
fi

PAIRED_CHECKPOINT_SHA256=$("${PYTHON_BIN}" - "${PAIRED_SMOKE_RESULT}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle)["conditions"]["quantized_checkpoint_sha256"])
PY
)
if [[ "${PAIRED_CHECKPOINT_SHA256}" != "${CHECKPOINT_SHA256}" ]]; then
    echo "Paired smoke checkpoint hash does not match the current checkpoint" >&2
    exit 1
fi

SHARD_NAMES=(wikitext hellaswag arc-openbook arc-winogrande)
SHARD_TASKS=(wikitext hellaswag arc_easy,openbookqa arc_challenge,winogrande)
SHARD_BATCHES=(1 32 32 32)
STANDARD_RESULTS=()
FPINT_RESULTS=()
PIDS=()
for shard_index in "${!SHARD_NAMES[@]}"; do
    shard=${SHARD_NAMES[${shard_index}]}
    tasks=${SHARD_TASKS[${shard_index}]}
    gpu=${GPU_IDS[${shard_index}]}
    batch_size=${SHARD_BATCHES[${shard_index}]}
    port=$((MASTER_PORT_BASE + shard_index + 1))
    standard_result="${FULL_RESULT_DIR}/standard-${shard}.json"
    fpint_result="${FULL_RESULT_DIR}/fpint-cuda-${shard}.json"
    STANDARD_RESULTS+=("${standard_result}")
    FPINT_RESULTS+=("${fpint_result}")
    (
        run_ptq "standard-${shard}" standard "${tasks}" "${standard_result}" "" load "${gpu}" "${port}" "${batch_size}"
        run_ptq "fpint-cuda-${shard}" fpint_cuda "${tasks}" "${fpint_result}" "" load "${gpu}" "${port}" "${batch_size}"
    ) &
    PIDS+=("$!")
done

shard_failure=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then
        shard_failure=1
    fi
done
if (( shard_failure != 0 )); then
    echo "At least one full-evaluation shard failed; see ${LOG_ROOT}." >&2
    exit 1
fi

"${PYTHON_BIN}" summarize_fpint_mxu128.py \
    --random "${RANDOM_RESULT}" \
    --standard-results "${STANDARD_RESULTS[@]}" \
    --fpint-results "${FPINT_RESULTS[@]}" \
    --metrics-output "${FULL_METRICS_RESULT}" \
    --output "${REPORT}"

echo "MXU ROW 128 experiment complete: ${FULL_METRICS_RESULT}"
