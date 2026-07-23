# Zero-Padding Online Hadamard Design

## Goal

Extend SpinQuant with a selectable zero-padding implementation for the online
R4 Hadamard transform while preserving the current factorized implementation as
the default. Run W4A8KV4 PTQ with both implementations and compare task
accuracy and WikiText results for three Llama models.

The first experiment reuses the same optimized R1/R2 rotation checkpoints for
both modes. A separate rotation-matrix search for the zero-padding mode is out
of scope unless the first experiment produces insufficient accuracy.

## Background

SpinQuant applies an R4 Hadamard transform to the input of each MLP
`down_proj`. The transform dimension is the model's `intermediate_size`, not
its `hidden_size`.

The current factorized path handles a non-power-of-two dimension `n` by writing
it as `K * 2^m`. It applies the fast GPU kernel to the power-of-two part and
multiplies by a supported `K x K` Hadamard matrix.

The installed `fast-hadamard-transform` kernel accepts a non-power-of-two last
dimension directly. It implicitly zero-pads the input, computes the transform,
and crops the output back to the original dimension. The cropped transform is
not orthogonal. For this experiment, SpinQuant will intentionally apply this
same cropped transform to both the offline `down_proj` weight and the online
activation. The experiment therefore measures any model-function change caused
by the cropped transform in addition to PTQ error.

## User Interface

Add a command-line argument:

```text
--online_had_mode {factorized,zero_padding}
```

Its default is `factorized`, preserving existing behavior and command lines.
Invalid values are rejected by argument parsing.

The argument is supported by both:

- the PTQ/evaluation path; and
- the rotation-training preparation path, so a later zero-padding-specific
  rotation search can reuse the same interface.

## Transform Implementations

### Factorized

The `factorized` mode retains the current implementation:

1. Determine `had_K` and `K` with `get_hadK(n)`.
2. Reshape the input into `K` rows of power-of-two length.
3. Apply the fast GPU Hadamard kernel to each row.
4. Multiply by the `K x K` base Hadamard matrix.
5. Normalize by `sqrt(n)`.

No numerical or behavioral change is intended for this mode.

### Zero padding

The `zero_padding` mode:

1. Keeps the original non-power-of-two last dimension intact.
2. Passes the contiguous tensor directly to the fast GPU Hadamard kernel.
3. Relies on the kernel's implicit padding and cropping.
4. Normalizes by `sqrt(n)`, where `n` is the original last dimension.

For a power-of-two dimension, this path reduces to the same GPU kernel call and
normalization as the factorized path with `K == 1`.

### Dispatch

A single Hadamard mode dispatcher in `utils/hadamard_utils.py` selects the
implementation. Both offline weight transformation and online activation
transformation call this dispatcher. This prevents mode logic from being
duplicated across PTQ and training code.

The activation quantization wrapper stores the selected mode. Factorization
metadata (`had_K` and `K`) is initialized only when required by the factorized
implementation.

## Data Flow

For PTQ:

1. Parse `online_had_mode`.
2. Load the model and the existing optimized R1/R2 checkpoint.
3. During model rotation, transform every `down_proj` weight using the selected
   R4 mode.
4. Configure every `down_proj` activation wrapper with the same selected mode.
5. Run GPTQ and activation/KV quantization using the existing W4A8KV4
   configuration.
6. Evaluate the quantized model and save JSON metrics under a mode-specific
   path.

The zero-padding and factorized modes must use separate quantized weight
checkpoints because the R4-transformed `down_proj` weights differ for
non-power-of-two dimensions.

## Model Matrix

The comparison covers:

| Model | Hidden size | Intermediate size | Expected role |
| --- | ---: | ---: | --- |
| Llama-2 7B | 4096 | 11008 | Non-power-of-two comparison |
| Llama-3.1 8B | 4096 | 14336 | Non-power-of-two comparison |
| Llama-3.2 3B | 3072 | 8192 | Power-of-two control |

Llama-3.2 3B has a non-power-of-two hidden size, but its online R4 transform is
applied at the power-of-two intermediate size. The two modes should therefore
match for this model apart from ordinary run-to-run nondeterminism.

## Experiment Runner and Summary

Provide a comparison runner that executes each of the three models with both
modes under the existing W4A8KV4 settings. Mode names are included in
checkpoint, result, and log paths to prevent accidental reuse across modes.
Existing valid outputs may be skipped independently, allowing interrupted
experiments to resume.

Provide a summary utility that reads the six JSON results and emits a table
containing:

- factorized value;
- zero-padding value; and
- zero-padding minus factorized delta

for HellaSwag, ARC Easy, ARC Challenge, Winogrande, OpenBookQA, and WikiText.
The utility fails clearly when required result files or expected metric fields
are missing.

## Error Handling

- Argument parsing rejects unsupported mode strings.
- The zero-padding path reports a clear error when the fast CUDA kernel cannot
  be used.
- Kernel dimension and dtype restrictions are allowed to fail with contextual
  information rather than silently falling back to a different transform.
- Existing factorized dimension validation remains unchanged.
- A PTQ run does not report success unless it produced a non-empty result JSON.

## Testing

Add focused tests for:

1. Default argument parsing selects `factorized`.
2. Explicit argument parsing selects `zero_padding`.
3. Factorized mode preserves the existing transform result.
4. Zero-padding mode matches the fast kernel reference for representative
   non-power-of-two dimensions.
5. Both modes match for representative power-of-two dimensions.
6. Offline weight and online activation configuration receive the same mode.
7. Comparison path generation isolates checkpoints, logs, and JSON results by
   mode.
8. The summary utility computes task deltas correctly from fixture JSON.

GPU-dependent kernel tests are marked and skipped with an explicit reason when
CUDA or `fast-hadamard-transform` is unavailable. CPU-only tests cover argument
parsing, dispatch selection, wiring, path generation, and result summarization.

## Success Criteria

- Existing invocations without the new argument use factorized Hadamard without
  numerical or checkpoint-path changes.
- Both modes complete W4A8KV4 PTQ/evaluation for all three models using the same
  optimized R1/R2 inputs.
- Each run has an isolated checkpoint, log, and JSON result.
- A generated summary shows per-task metrics and deltas for all models.
- The Llama-3.2 3B control confirms that the two modes agree within the chosen
  numerical tolerance.
- The implementation and focused tests pass before accuracy results are
  reported.

## Deferred Work

If zero-padding accuracy is inadequate, add a follow-up experiment that
optimizes R1/R2 while the zero-padding R4 mode is active. This follow-up will
reuse `--online_had_mode zero_padding` but will receive a separate design and
experiment plan.
