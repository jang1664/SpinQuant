# Llama-3.2 3B A/Q/P quantization comparison

## Fixed configuration

- Model: `models/llama3.2-3b`
- Weight / KV: W4, K4, V4
- Attention backend: eager
- Sequence length: 2048
- Seed: 0
- Rotation: `rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin`
- Rotation SHA-256: `6bbe3c394ffc0e35d59145351eb6090a6bbc2c185fd1681b047c2372935d0f2d`
- Shared W4 checkpoint: `saved_models/llama3.2-3b/a16w4kv4-vasym.pt`

## Results

| Metric | A/Q/P 16/16/16 | A/Q/P 8/8/8 | 8-bit delta vs 16 | A/Q/P 4/4/4 | 4-bit delta vs 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| WikiText-2 word perplexity | 10.9110 | 10.9572 | +0.0462 (+0.42%) | 15.5494 | +4.6384 (+42.51%) |
| HellaSwag acc_norm | 71.8482% | 71.6092% | -0.2390 pp | 65.5348% | -6.3135 pp |
| ARC-Challenge acc_norm | 41.7235% | 41.4676% | -0.2560 pp | 37.4573% | -4.2662 pp |
| ARC-Easy acc_norm | 64.7306% | 65.7407% | +1.0101 pp | 61.1111% | -3.6195 pp |
| OpenBookQA acc_norm | 41.2000% | 41.0000% | -0.2000 pp | 34.4000% | -6.8000 pp |
| Winogrande accuracy | 68.4294% | 69.1397% | +0.7103 pp | 58.8003% | -9.6290 pp |

All three result files report the same rotation checkpoint SHA-256. Only A, Q,
and P bit-widths differ between the runs; W/K/V and the eager attention backend
are fixed.
