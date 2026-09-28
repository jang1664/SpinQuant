#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/unquantized-llama-baselines}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
EVAL_LIMIT=${EVAL_LIMIT:-}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
IFS=',' read -r -a GPUS <<< "${CUDA_DEVICES}"
if (( ${#GPUS[@]} != 4 )); then
    echo "CUDA_DEVICES must list four GPUs" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

resolve_model() {
    "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
root = Path(HF_HUB_CACHE) / ('models--' + sys.argv[1].replace('/', '--')) / 'snapshots'
for snapshot in sorted(root.glob('*')):
    index = snapshot / 'model.safetensors.index.json'
    if not (snapshot / 'config.json').is_file() or not index.is_file():
        continue
    files = set(json.loads(index.read_text())['weight_map'].values())
    if all((snapshot / name).is_file() for name in files):
        print(snapshot)
        break
else:
    raise SystemExit(f'No complete local snapshot for {sys.argv[1]} in {root}')
PY
}
LLAMA2_MODEL=${LLAMA2_MODEL:-$(resolve_model meta-llama/Llama-2-7b-hf)}
LLAMA3_MODEL=${LLAMA3_MODEL:-$(resolve_model meta-llama/Meta-Llama-3.1-8B)}

run_shard() {
    local name=$1 model=$2 dtype=$3 gpu=$4 shard=$5 tasks=$6 batch=$7
    local fp16=True bf16=False
    if [[ "${dtype}" == bf16 ]]; then fp16=False; bf16=True; fi
    local limit_args=()
    if [[ -n "${EVAL_LIMIT}" ]]; then limit_args=(--eval_limit "${EVAL_LIMIT}"); fi
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=1 ptq.py \
        --input_model "${model}" --do_train False --do_eval True \
        --per_device_eval_batch_size 1 --model_max_length 2048 \
        --fp16 "${fp16}" --bf16 "${bf16}" --save_safetensors False \
        --seed 0 --attention_backend eager --linear_backend standard --no-rotate \
        --w_bits 16 --a_bits 16 --q_bits 16 --p_bits 16 --k_bits 16 --v_bits 16 \
        --w_groupsize -1 --a_groupsize -1 --q_groupsize -1 --p_groupsize -1 --k_groupsize -1 --v_groupsize -1 \
        --eval_tasks "${tasks}" --lm_eval_batch_size "${batch}" "${limit_args[@]}" \
        --results_path "${OUTPUT_ROOT}/${name}-${shard}.json" \
        > "${OUTPUT_ROOT}/${name}-${shard}.log" 2>&1
}

TASKS=hellaswag,arc_easy,arc_challenge,winogrande,openbookqa
PIDS=()
run_shard llama2-7b "${LLAMA2_MODEL}" fp16 "${GPUS[0]}" accuracy "${TASKS}" 32 &
PIDS+=("$!")
run_shard llama2-7b "${LLAMA2_MODEL}" fp16 "${GPUS[1]}" wikitext wikitext 1 &
PIDS+=("$!")
run_shard llama31-8b "${LLAMA3_MODEL}" bf16 "${GPUS[2]}" accuracy "${TASKS}" 32 &
PIDS+=("$!")
run_shard llama31-8b "${LLAMA3_MODEL}" bf16 "${GPUS[3]}" wikitext wikitext 1 &
PIDS+=("$!")
failed=0
for pid in "${PIDS[@]}"; do
    if ! wait "${pid}"; then failed=1; fi
done
if (( failed )); then
    echo "At least one evaluation failed; see ${OUTPUT_ROOT}/*.log" >&2
    exit 1
fi
echo "Completed unquantized evaluations: ${OUTPUT_ROOT}"
