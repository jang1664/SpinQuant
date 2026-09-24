# AQP8 vs AQP16 experiment results

These reports compare 8-bit quantization of A/Q/P with 16-bit bypass.
Weights, keys, and values remain W4/K4/V4 in both conditions; AQP16 is
not a fully floating-point model. Where present, `FP base` is a separate
unquantized reference.

## Reports

| Report | Experiment date | Scope |
| --- | --- | --- |
| [Benchmark comparison](benchmark-comparison.md) | 2026-08-20 | Llama-3.2 3B: WikiText-2 perplexity and five zero-shot accuracy tasks, with AQP16/AQP8/AQP4. |
| [Matrix-output accuracy](matrix-output-accuracy.md) | 2026-09-10 | Llama-2 7B, Llama-3.1 8B, and Llama-3.2 3B: Linear, QK, and PV errors against FP16 across WikiText, ARC-Challenge, HellaSwag, WinoGrande, C4, and RULER inputs. |
| [Representative hard workloads](hard-workloads-representative.md) | 2026-08-20 | Small evaluation run comparing FP base, AQP16, and AQP8; not the full hard-workload evaluation. |
| [Matrix-output smoke test](matrix-output-accuracy-smoke.md) | 2026-09-10 | Llama-3.2 3B smoke-test output; use the main matrix-output report for the completed experiment. |

## Main observations

- For Llama-3.2 3B, WikiText-2 word perplexity changes from 10.9110
  (AQP16) to 10.9572 (AQP8), an increase of 0.42%. Changes in the five
  reported accuracy metrics range from -0.2560 to +1.0101 percentage points.
- The matrix-output report measures numerical error, not task accuracy.
  Its `error_A` compares FP16 with AQP8 and `error_B` compares FP16 with
  AQP16. An error ratio A/B above 1 indicates larger AQP8 error.
- The extended zero-shot workloads use representative subsets. RULER is
  an input stress workload only; the report does not measure retrieval accuracy.

## Original artifacts

The four report files were copied without content changes from the following
repository-relative paths on 2026-09-24:

| Archived report | Original path |
| --- | --- |
| `benchmark-comparison.md` | `results/aqp-comparison/llama3.2-3b/SUMMARY.md` |
| `matrix-output-accuracy.md` | `results/aqp-matrix-output-accuracy/SUMMARY.md` |
| `hard-workloads-representative.md` | `results/hard-workloads-representative/llama3.2-3b/SUMMARY.md` |
| `matrix-output-accuracy-smoke.md` | `results/aqp-matrix-output-accuracy-smoke/SUMMARY.md` |

The original artifacts were found on **atlas2**, under:

```text
/home/jaeyongjang/project.local/model_playground/quantizations/SpinQuant/
```

Raw JSON results and logs remain in the original `results/` directories.
Additional matrix-workload JSON files are in `results/aqp-matrix-workloads/`.
The later hard-workload run has `fp_base.json`, `aqp16.json`, and
`aqp8.json` under `results/hard-workloads-bbh5/llama3.2-3b/results/`
(dated 2026-09-05/06), but no Markdown summary was found there.

Because `results/` is ignored by Git, pulling the repository retrieves these
archived Markdown reports, not the raw JSON, logs, or model checkpoints.
