#!/usr/bin/env bash

set -euo pipefail

if (( $# != 3 )); then
    echo "Usage: $0 MODEL OUTPUT_ROTATION_DIR fp16|bf16" >&2
    exit 2
fi

MODEL=$1
OUTPUT_ROTATION_DIR=$2
COMPUTE_DTYPE=$3
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICE=${CUDA_DEVICE:-0}
MASTER_PORT=${MASTER_PORT:-29500}

case "${COMPUTE_DTYPE}" in
    fp16) PRECISION_ARGS=(--fp16 True --bf16 False) ;;
    bf16) PRECISION_ARGS=(--fp16 False --bf16 True) ;;
    *) echo "COMPUTE_DTYPE must be fp16 or bf16" >&2; exit 2 ;;
esac

if [[ ! -r "${MODEL}/config.json" ]]; then
    echo "Model is not readable: ${MODEL}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_ROTATION_DIR}" outputs logs
export ZP_INT8=0
export SIGNED_KV=0
export ZP_CLAMP=1
export SCALE_NO_UPCAST=0

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=1 --nproc_per_node=1 --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" optimize_rotation.py \
    --input_model "${MODEL}" \
    --output_rotation_path "${OUTPUT_ROTATION_DIR}" \
    --output_dir outputs/ --logging_dir logs/ \
    --model_max_length 2048 \
    "${PRECISION_ARGS[@]}" \
    --log_on_each_node False --per_device_train_batch_size 1 \
    --logging_steps 1 --learning_rate 1.5 --weight_decay 0. \
    --lr_scheduler_type cosine --gradient_checkpointing True \
    --save_safetensors False --max_steps 100 \
    --w_bits 4 --a_bits 16 --k_bits 4 --v_bits 4 \
    --w_clip --a_asym --k_asym --v_asym \
    --k_groupsize 128 --v_groupsize 128

test -s "${OUTPUT_ROTATION_DIR}/R.bin"
ROTATION_SHA256=$(sha256sum "${OUTPUT_ROTATION_DIR}/R.bin" | awk '{print $1}')
"${PYTHON_BIN}" - \
    "${OUTPUT_ROTATION_DIR}/rotation-metadata.json" \
    "${MODEL}" "${COMPUTE_DTYPE}" "${ROTATION_SHA256}" <<'PY'
import datetime
import json
import sys
from pathlib import Path

output, model, compute_dtype, sha256 = sys.argv[1:]
metadata = {
    "format_version": 1,
    "model": model,
    "optimization_dtype": compute_dtype,
    "w_bits": 4,
    "a_bits": 16,
    "k_bits": 4,
    "v_bits": 4,
    "k_groupsize": 128,
    "v_groupsize": 128,
    "max_steps": 100,
    "rotation_sha256": sha256,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
Path(output).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
PY
echo "${ROTATION_SHA256}  ${OUTPUT_ROTATION_DIR}/R.bin"
