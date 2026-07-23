#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

CUDA_DEVICE=${CUDA_DEVICE:-0}
read -r -a HAD_MODES <<< "${ONLINE_HAD_MODES:-factorized zero_padding}"

MODELS=(
    "llama2-7b|${LLAMA2_7B_MODEL:-./models/llama2-7b}|${LLAMA2_7B_ROTATION:-rotation_llama-2-7b/a16w4kv4-vasym/R.bin}"
    "llama3.1-8b|${LLAMA31_8B_MODEL:-./models/llama3.1-8b}|${LLAMA31_8B_ROTATION:-rotation_llama-3.1-8b/a16w4kv4-vasym/R.bin}"
    "llama3.2-3b|${LLAMA32_3B_MODEL:-./models/llama3.2-3b}|${LLAMA32_3B_ROTATION:-rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin}"
)

export ZP_INT8=0
export SIGNED_KV=0
export ZP_CLAMP=1
export SCALE_NO_UPCAST=1

if [[ -n ${TORCHRUN_MASTER_PORT:-} ]]; then
    TORCHRUN_RDZV_ARGS=(--master-port "${TORCHRUN_MASTER_PORT}")
else
    TORCHRUN_RDZV_ARGS=(--standalone)
fi

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

if [[ ${DRY_RUN:-0} != 1 ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
    /home/jaeyongjang/.conda/envs/spinquant/bin/python - <<'PY'
from importlib.metadata import version

from packaging.version import Version

installed = Version(version("datasets"))
if installed < Version("4.0.0"):
    raise SystemExit(
        "datasets>=4.0 is required for current Hugging Face dataset schemas; "
        f"found {installed}"
    )
PY
fi

for mode in "${HAD_MODES[@]}"; do
    case "${mode}" in
        factorized|zero_padding) ;;
        *)
            echo "Unsupported ONLINE_HAD_MODES entry: ${mode}" >&2
            exit 2
            ;;
    esac

    for model_spec in "${MODELS[@]}"; do
        IFS='|' read -r name model rotation <<< "${model_spec}"

        checkpoint="saved_models/online-had-comparison/${mode}/${name}/w4-gptq.pt"
        result="results/online-had-comparison/${mode}/${name}.json"
        log="logs/online-had-comparison/${mode}/${name}.log"

        if [[ ${DRY_RUN:-0} != 1 && -s ${result} ]]; then
            echo "[${mode}/${name}] result already exists; skipping: ${result}"
            continue
        fi

        if [[ ${FORCE_REQUANTIZE:-0} != 1 && -s ${checkpoint} ]]; then
            checkpoint_args=(--load_qmodel_path "${checkpoint}")
        else
            checkpoint_args=(
                --optimized_rotation_path "${rotation}"
                --save_qmodel_path "${checkpoint}"
            )
        fi

        command=(
            torchrun
            "${TORCHRUN_RDZV_ARGS[@]}"
            --nnodes=1
            --nproc_per_node=1
            ptq.py
            --input_model "${model}"
            --do_train False
            --do_eval True
            --per_device_eval_batch_size 4
            --model_max_length 2048
            --fp16 True
            --bf16 False
            --save_safetensors False
            --w_bits 4
            --w_groupsize -1
            --a_bits 8
            --a_groupsize -1
            --k_bits 4
            --v_bits 4
            --w_clip
            --a_asym
            --k_asym
            --v_asym
            --k_groupsize 128
            --v_groupsize 128
            --rotate
            --online_had_mode "${mode}"
            "${checkpoint_args[@]}"
            --results_path "${result}"
        )

        if [[ ${DRY_RUN:-0} == 1 ]]; then
            print_command "${command[@]}"
            continue
        fi

        if [[ ! -e ${model} ]]; then
            echo "Model not found: ${model}" >&2
            exit 1
        fi
        if [[ ${checkpoint_args[0]} == --optimized_rotation_path && ! -f ${rotation} ]]; then
            echo "Rotation checkpoint not found: ${rotation}" >&2
            exit 1
        fi

        mkdir -p \
            "$(dirname -- "${checkpoint}")" \
            "$(dirname -- "${result}")" \
            "$(dirname -- "${log}")"
        echo "[${mode}/${name}] checkpoint=${checkpoint}"
        echo "[${mode}/${name}] result=${result}"
        "${command[@]}" 2>&1 | tee -a "${log}"

        if [[ ! -s ${result} ]]; then
            echo "PTQ evaluation did not produce ${result}" >&2
            exit 1
        fi
    done
done
