#!/usr/bin/env bash
# Paired FP16 baseline and W4/K4/V4 AQP16/AQP8 evaluation for Llama-3 (not 3.1).
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."
PYTHON_BIN=${PYTHON_BIN:-python}
MODEL=${MODEL:-models/llama3-8b}
OUTPUT_DIR=${OUTPUT_DIR:-results/aqp-comparison/llama3-8b}
LOG_DIR=${LOG_DIR:-/tmp/spinquant-llama3-8b}
BASELINE_GPU=${BASELINE_GPU:-0}
QUANT_GPU=${QUANT_GPU:-1}
ROTATION_DIR="${OUTPUT_DIR}/rotation"
export LM_EVAL_BATCH_SIZE=${LM_EVAL_BATCH_SIZE:-4}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
mkdir -p "${OUTPUT_DIR}/results" "${LOG_DIR}"

# Fail before starting any GPU work if a model shard is missing.
"${PYTHON_BIN}" - "${MODEL}" <<'PY'
import json
import sys
from pathlib import Path
model = Path(sys.argv[1])
config = json.loads((model / 'config.json').read_text())
assert config['hidden_size'] == 4096 and config['num_hidden_layers'] == 32
assert config['max_position_embeddings'] == 8192, 'Expected original Llama-3-8B'
index = json.loads((model / 'model.safetensors.index.json').read_text())
for name in set(index['weight_map'].values()):
    assert (model / name).is_file(), f'Missing model shard: {name}'
assert (model / 'tokenizer.json').is_file()
PY

baseline_pid=
if [[ ! -s "${OUTPUT_DIR}/results/fp_base.json" ]]; then
    CUDA_VISIBLE_DEVICES="${BASELINE_GPU}" "${PYTHON_BIN}" -m torch.distributed.run \
        --standalone --nnodes=1 --nproc_per_node=1 ptq.py \
        --input_model "${MODEL}" --do_train False --do_eval True \
        --model_max_length 2048 --fp16 True --bf16 False --save_safetensors False \
        --seed 0 --attention_backend eager --linear_backend standard --no-rotate \
        --w_bits 16 --a_bits 16 --q_bits 16 --p_bits 16 --k_bits 16 --v_bits 16 \
        --lm_eval_batch_size "${LM_EVAL_BATCH_SIZE}" \
        --results_path "${OUTPUT_DIR}/results/fp_base.json" \
        > "${LOG_DIR}/fp_base.log" 2>&1 &
    baseline_pid=$!
    echo "FP16 baseline PID=${baseline_pid}, GPU=${BASELINE_GPU}"
fi

rotation_status=0
if [[ ! -s "${ROTATION_DIR}/rotation-metadata.json" ]]; then
    echo "Optimizing W4/A16/K4/V4 rotation on GPU=${QUANT_GPU}"
    CUDA_DEVICE="${QUANT_GPU}" PYTHON_BIN="${PYTHON_BIN}" \
        bash scripts/optimize_rotation_one.sh "${MODEL}" "${ROTATION_DIR}" fp16 \
        > "${LOG_DIR}/rotation.log" 2>&1 || rotation_status=$?
fi
if [[ -n "${baseline_pid}" ]]; then wait "${baseline_pid}"; fi
if (( rotation_status != 0 )); then exit "${rotation_status}"; fi

echo "Evaluating shared W4 checkpoint with AQP16 and AQP8"
export PATH="$(dirname -- "$(command -v "${PYTHON_BIN}")"):${PATH}"
CUDA_DEVICE="${QUANT_GPU}" AQP_BITS="16 8" SPINQUANT_ROTATION_DTYPE=fp16 \
    bash scripts/run_ptq_w4kv4_aqp_comparison.sh \
    "${MODEL}" "${ROTATION_DIR}/R.bin" "${OUTPUT_DIR}" \
    > "${LOG_DIR}/aqp-comparison.log" 2>&1
"${PYTHON_BIN}" scripts/summarize_llama3_8b_accuracy.py "${OUTPUT_DIR}" \
    > "${LOG_DIR}/summary.log" 2>&1
echo "Finished: ${OUTPUT_DIR}/results; stdout logs: ${LOG_DIR}"
