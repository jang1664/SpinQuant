#!/usr/bin/env bash
set -euo pipefail

# Run the pinned representative subset from workloads/matrix_accuracy_v1.yaml.
# Set PYTHON_BIN and CUDA_DEVICE explicitly for reproducible cluster launches.
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_DEVICE=${CUDA_DEVICE:-2}
WORKLOADS=${WORKLOADS:-c4_validation,arc_challenge,hellaswag,winogrande,ruler_niah_single_1_2k}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/aqp-matrix-workloads}
MANIFEST=${MANIFEST:-workloads/matrix_accuracy_v1.yaml}

for model in llama2-7b llama3.1-8b llama3.2-3b; do
  case "$model" in
    llama2-7b) rotation_model=llama-2-7b ;;
    llama3.1-8b) rotation_model=llama-3.1-8b ;;
    llama3.2-3b) rotation_model=llama-3.2-3b ;;
  esac
  mkdir -p "$OUTPUT_ROOT/$model"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" "$PYTHON_BIN" measure_matrix_workloads.py \
    --input-model "./models/$model" \
    --load-qmodel-path "saved_models/w4-gptq-wclip-wgs-1-spinquant-optrot/$model/w4-gptq.pt" \
    --rotation-path "rotation_${rotation_model}/a16w4kv4-vasym/R.bin" \
    --workload-manifest "$MANIFEST" \
    --workloads "$WORKLOADS" \
    --output-dir "$OUTPUT_ROOT/$model" \
    --sequence-length 2048 \
    --max-tokens-per-example 2048
done

"$PYTHON_BIN" summarize_matrix_workloads.py \
  --legacy-dir results/aqp-matrix-output-accuracy \
  --workload-dir "$OUTPUT_ROOT" \
  --models llama2-7b llama3.1-8b llama3.2-3b \
  --output results/aqp-matrix-output-accuracy/SUMMARY.md
