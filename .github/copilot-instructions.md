# SpinQuant AI Coding Instructions

## Project Overview
SpinQuant is a framework for LLM quantization using learned rotations to reduce outliers and improve performance. It involves optimizing rotation matrices (R1, R2, etc.) and then applying Post-Training Quantization (PTQ).

## Key Components & Architecture

### 1. Entry Points
- **`optimize_rotation.py`**: Main script for learning rotation matrices. Uses FSDP for training.
- **`ptq.py`**: Main script for evaluating quantized models. Applies rotations and quantization (GPTQ/RTN).

### 2. Core Modules
- **`eval_utils/`**: Utilities for evaluation and applying quantization.
    - `main.py`: Contains `ptq_model` which orchestrates rotation and quantization.
    - `modeling_llama.py`: Modified Llama model for evaluation.
    - `rotation_utils.py`: Functions to apply rotations to model weights (in-place).
    - `gptq_utils.py`: GPTQ and RTN quantization logic.
- **`train_utils/`**: Utilities for optimizing rotations.
    - `modeling_llama_quant.py`: Modified Llama model for training (supports `RotateModule`).
    - `fsdp_trainer.py`: Custom trainer for FSDP.
- **`utils/`**: General helpers.
    - `quant_utils.py`: `ActQuantWrapper`, `QuantLinear`, and quantization search logic.
    - `hadamard_utils.py`: Hadamard matrix generation and application.

### 3. Data Flow
1.  **Rotation Optimization**:
    -   Load model (`train_utils.modeling_llama_quant`).
    -   Initialize `RotateModule` (R1, R2).
    -   Train using `FSDPTrainer` on calibration data (WikiText-2).
    -   Save optimized rotation matrices.
2.  **PTQ Evaluation**:
    -   Load model (`eval_utils.modeling_llama`).
    -   Load optimized rotation matrices.
    -   Apply rotation (`rotation_utils.rotate_model`).
    -   Apply quantization (`quant_utils.add_actquant`, `gptq_utils`).
    -   Evaluate using `lm_eval`.

## Critical Workflows

### Running Experiments
Do not run python scripts directly. Use the provided shell scripts in `scripts/` which handle arguments and environment setup.

- **Optimize Rotation**:
  ```bash
  bash scripts/10_optimize_rotation.sh <model_name> <w_bits> <a_bits> <kv_bits>
  ```
- **Evaluate PTQ**:
  ```bash
  bash scripts/2_eval_ptq.sh <model_name> <w_bits> <a_bits> <kv_bits>
  ```

### Debugging & Analysis
- **`result_analysis/`**: Contains notebooks and scripts for analyzing distributions and results.
- **`catch_data.py`**: Use `catch_tensors` to capture intermediate tensors (input, output, weights) for analysis.

## Conventions & Patterns

- **Argument Parsing**: Arguments are processed in `utils/process_args.py`. Check this file for available flags (`--rotate`, `--w_bits`, etc.).
- **Model Wrappers**: The codebase uses different model definitions for training (`train_utils`) vs evaluation (`eval_utils`). Be careful which one is imported.
- **In-Place Modification**: Rotation and quantization often modify the model weights in-place.
- **Hadamard Transform**: Relies on `fast_hadamard_transform` (C extension). Ensure it is installed.
- **ExecuTorch Export**: Specific logic exists for exporting to ExecuTorch (`utils/convert_to_executorch.py`).

## Common Tasks

- **Adding a new model**: You may need to update `modeling_llama.py` or create a new modeling file if the architecture differs significantly.
- **Changing Quantization Logic**: Modify `utils/quant_utils.py` or `eval_utils/gptq_utils.py`.
- **Analyzing Activation Distribution**: Use `result_analysis/catch_data.py` to hook into the model and capture data.
