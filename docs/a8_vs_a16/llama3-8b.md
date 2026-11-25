# Llama-3-8B AQP16/AQP8 accuracy experiment

This run adds the original `meta-llama/Meta-Llama-3-8B` base model, distinct
from Llama-3.1-8B and the Instruct variant. It completed on 2026-10-14 at
approximately 20:43 UTC. All three full evaluations and the paired-result
validation completed successfully.

## Completed results

| Task | Metric | Samples | FP base | AQP16 | AQP8 | AQP8 − AQP16 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | acc_norm | 10042 | 79.1077% | 77.7036% | 77.7734% | +0.0697 pp |
| ARC-Easy | acc_norm | 2376 | 77.6936% | 79.0404% | 79.0825% | +0.0421 pp |
| ARC-Challenge | acc_norm | 1172 | 53.3276% | 52.2184% | 52.2184% | +0.0000 pp |
| WinoGrande | acc | 1267 | 72.5335% | 72.0600% | 73.1650% | +1.1050 pp |
| OpenBookQA | acc_norm | 500 | 45.0000% | 44.0000% | 43.4000% | -0.6000 pp |
| WikiText | word_perplexity | 62 | 7.794194 | 8.523998 | 8.575431 | +0.051433 (+0.603%) |
| Micro average | sample-weighted accuracy | 15357 | 75.2686% | 74.4026% | 74.5263% | +0.1237 pp |

AQP8 increases word perplexity by 0.603% relative to AQP16. Downstream
accuracy changes are mixed: three increases, one decrease, and one unchanged
score. The sample-weighted aggregate rises by 0.1237 percentage points.
This single-seed comparison does not establish a consistent task-quality
degradation from A/Q/P 8-bit quantization or statistical significance of
the accuracy differences. It does not measure intermediate QK/PV errors.

Validated provenance:

- Shared quantized-weight SHA-256:
  `9f41b7f911dd8330be10250c14f34a9ee4c00b683b1c34841c26e114b83b16d9`.
- Shared rotation SHA-256:
  `95f044477096fade4f04cbb83174e509cf09bc4074c00891d3a8bd90b97d4328`.
- Full task counts and ordered document/prompt hashes match across all conditions.
- Rotation completed 100 steps in 42849.0806 seconds. `grad_norm=NaN` was
  logged at steps 18 and 94; subsequent logged gradients were finite.
  All 33 tensors in the saved rotation checkpoint were independently checked
  and contain only finite values. See `/tmp/spinquant-llama3-8b/rotation-validation.log`.
- The pipeline exited with status 0; `summary.log` records successful validation.

## Fixed configuration

- Model revision: `8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920`.
- Local model: `models/llama3-8b`, linked to the downloaded Hugging Face snapshot.
- All evaluations: FP16, eager attention, standard Linear, length 2048,
  zero-shot, seed 0, lm-eval batch size 4, no evaluation limit.
- FP base: W/A/Q/P/K/V all 16-bit bypass, no rotation.
- AQP16: W4/K4/V4, A/Q/P 16-bit bypass.
- AQP8: the same W4/K4/V4 and rotation checkpoint, A/Q/P all 8-bit.
- GPTQ weights: clipping enabled, group size -1; asymmetric A/K/V;
  K/V/Q group size 128, A/P group size -1; default symmetric Q/P.
- Rotation: existing `optimize_rotation_one.sh` recipe, W4/A16/K4/V4,
  100 steps, learning rate 1.5, batch size 1, gradient checkpointing.
  FP16 model computation retains the implementation's FP64 weight rotations.
- PTQ environment: `ZP_INT8=0`, `SIGNED_KV=0`, `ZP_CLAMP=1`, `SCALE_NO_UPCAST=1`.

## Measurements and validation

Full HellaSwag, ARC-Easy, ARC-Challenge, WinoGrande, OpenBookQA, and WikiText
are evaluated. Accuracy uses `acc_norm` except WinoGrande (`acc`). WikiText
uses lm-eval's `word_perplexity`; it is not the concatenated-token PPL used
by some papers. The five accuracy tasks contain 15,357 examples in total.

The summarizer checks all task counts, matching document/prompt hashes,
compute settings, and shared weight/rotation SHA-256 hashes before writing
`results/aqp-comparison/llama3-8b/SUMMARY.md` and `summary.json`.
Raw results are `results/{fp_base,aqp16,aqp8}.json` within that directory.
Intermediate QK/PV numerical errors are outside this run's scope.

## Reproduction and logs

```bash
conda activate spinquant
bash scripts/run_llama3_8b_accuracy.sh
```

The model snapshot must already be complete. The runner uses GPU 0 for the
baseline and GPU 1 for rotation/PTQ by default (`BASELINE_GPU`, `QUANT_GPU`).
It reuses completed artifacts in the same output directory. Use a fresh
`OUTPUT_DIR` and `LOG_DIR` when changing the configuration.

Program stdout/stderr is stored under `/tmp/spinquant-llama3-8b/`:
`driver.log`, `fp_base.log`, `rotation.log`, `aqp-comparison.log`, and
`summary.log`. The first launch's download log is `/tmp/llama3-8b-download.log`.
Generated checkpoints and raw results remain in the ignored `results/` tree.
