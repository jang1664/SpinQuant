#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}/.."

PYTHON_BIN=${PYTHON_BIN:-python}
DEVICE=${DEVICE:-cuda:0}
TRIALS=${TRIALS:-30}
BASE_SEED=${BASE_SEED:-20260915}
SCALE_FORMAT=${SCALE_FORMAT:-activation}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/fpint-mxu128-scaled-gemm}
REPORT_ROOT=${REPORT_ROOT:-docs/FP-INT-hw-acc}
mkdir -p "${OUTPUT_ROOT}" "${REPORT_ROOT}"

for dtype in fp16 bf16; do
    result="${OUTPUT_ROOT}/${dtype}-int4-scaled.json"
    "${PYTHON_BIN}" measure_fpint_qcol_accuracy.py \
        --activation-format "${dtype}" --scale-format "${SCALE_FORMAT}" \
        --scale-mode log-uniform --scale-log2-min -4 --scale-log2-max 0 \
        --bits 4 --group-size 128 --mxu-rows 128 \
        --extra-bits 19 --reduce-extra-bits 10 \
        --m 32 --n 32 --trials "${TRIALS}" --base-seed "${BASE_SEED}" \
        --device "${DEVICE}" --output "${result}" \
        2>&1 | tee "${OUTPUT_ROOT}/${dtype}-int4-scaled.log"
    "${PYTHON_BIN}" summarize_fpint_mxu128.py \
        --random "${result}" \
        --output "${REPORT_ROOT}/mxu128_${dtype}_int4_gemm_results.md"
done
