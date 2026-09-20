# gemm_unit_wrap_opt_v3 RTL vs CUDA FP-INT

- Status: **MISMATCH**
- RTL: `/home/jaeyongjang/project.local/fpint/hw/reconfigurable/rtl/gemm_unit_wrap_opt_v3.sv`
- RTL SHA256: `4f44f0910352b59916479f973ff57976427e0e79ca471d971adc497c015b2d57`
- Verilator: `Verilator 5.020 2024-01-01 rev (Debian 5.020-1)`
- GPU: `NVIDIA RTX A6000`
- Shape per case: `M × K × N = 128 × 128 × 2`; weights are signed INT4.
- RTL path: `gemm_unit_wrap_opt_v3`, K-first (`load_i=1`), FP32 dequant scale, packed FP16/BF16 output.

## Aggregate

| Activation | CUDA geometry | Exact | Mismatch | Exact rate | Mean ULP | P95 ULP | Max ULP |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | experiment_extra19 | 518 / 1024 | 506 | 50.585938% | 127.026 | 661 | 799 |
| BF16 | rtl_matched | 518 / 1024 | 506 | 50.585938% | 127.026 | 661 | 799 |
| FP16 | experiment_extra19 | 1024 / 1024 | 0 | 100.000000% | 0 | 0 | 0 |
| FP16 | rtl_matched | 1024 / 1024 | 0 | 100.000000% | 0 | 0 | 0 |

## Isolation checks

| Activation | RTL FP32 core vs software FP32 | Correctly repacked RTL FP32 vs CUDA |
|---|---:|---:|
| BF16 | 768 / 1024 exact | 768 / 1024 exact |
| FP16 | 1024 / 1024 exact | 1024 / 1024 exact |

## Per-case result (current experiment: extra_bits=19)

| Activation | Case | Exact | Mismatch | Exact rate | Max ULP | Max abs |
|---|---|---:|---:|---:|---:|---:|
| BF16 | normal | 133 / 256 | 123 | 51.953125% | 1 | 64 |
| BF16 | wide_exponent | 129 / 256 | 127 | 50.390625% | 1 | 32768 |
| BF16 | subnormal | 0 / 256 | 256 | 0.000000% | 799 | 4.67259e-37 |
| BF16 | cancellation | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | normal | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | wide_exponent | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | subnormal | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | cancellation | 256 / 256 | 0 | 100.000000% | 0 | 0 |

## Interpretation

FP16 is bit-exact in the tested matrix. For finite normal BF16 values, the RTL FP32 core is bit-exact, but the RTL BF16 packer can be 1 ULP high: its rounding expression concatenates `15'h7fff` with the tie bit, producing a bias near `0xfffe` instead of `0x7fff + lsb`. BF16 subnormals expose two additional differences: the RTL exponent path treats raw exponent zero as zero (rather than effective exponent one), while the CUDA kernel forms `ldexpf(1.0f, binary_exponent)` first and underflows that factor to zero. The Torch/NumPy reference preserves those subnormal values.

## Candidate fixes (not applied)

1. RTL BF16 packing: replace the concatenated rounding bias with `32'h0000_7fff + fp32[16]`.
2. RTL BF16 subnormal alignment: use effective exponent 1 when exponent is zero and mantissa is nonzero.
3. CUDA BF16 subnormal restoration: compute `ldexpf(static_cast<float>(post), binary_exponent)` directly instead of multiplying by a separately formed power-of-two factor that can underflow first.

The Verilator environment replaces unavailable Synopsys DesignWare cells. FIFO and leading-zero detection use SystemVerilog compatibility models; FP32 multiply uses a DPI-C IEEE binary32 operation. The reconfigurable prealigner, MXU, column merger, int-to-FP logic, wrapper, and packer are the supplied RTL. This run does not cover multi-tile K accumulation (`load_i=0`) or asymmetric zero-point correction.

Raw JSON: `/home/jaeyongjang/project.local/SpinQuant/results/fpint-rtl-compare/summary.json`
