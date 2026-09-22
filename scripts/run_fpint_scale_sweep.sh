#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."
PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda:0}
TRIALS=${TRIALS:-30}
REFERENCE_TRIALS=${REFERENCE_TRIALS:-1}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/fpint-scale-exponent-sweep}
REPORT=${REPORT:-docs/FP-INT-hw-acc/mxu128_scale_exponent_sweep.md}
mkdir -p "${OUTPUT_ROOT}"
for dtype in fp16 bf16; do
    "${PYTHON_BIN}" measure_fpint_scale_sweep.py \
        --activation-format "${dtype}" --device "${DEVICE}" \
        --trials "${TRIALS}" --reference-trials "${REFERENCE_TRIALS}" \
        --output "${OUTPUT_ROOT}/${dtype}.json" \
        2>&1 | tee "${OUTPUT_ROOT}/${dtype}.log"
done
"${PYTHON_BIN}" summarize_fpint_scale_sweep.py \
    --inputs "${OUTPUT_ROOT}/fp16.json" "${OUTPUT_ROOT}/bf16.json" --output "${REPORT}"
