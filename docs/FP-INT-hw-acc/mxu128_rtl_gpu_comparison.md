# gemm_unit_wrap_opt_v3 RTL vs CUDA FP-INT

- Status: **PASS — bit-exact**
- RTL: `/home/jaeyongjang/project.local/fpint/hw/reconfigurable/rtl/gemm_unit_wrap_opt_v3.sv`
- RTL SHA256: `4f44f0910352b59916479f973ff57976427e0e79ca471d971adc497c015b2d57`
- RTL Git: `feat/bugfix` @ `bd123befd07e2dc216753d4579f1ed6b5fbbd8fc`
- BF16 packer source SHA256: `782ce1d5c5b560c48cd448eca34e92384f2bfb421a3a8251af832d99a1327bb1`
- BF16 exponent source SHA256: `6bd1baeea939aa83bfcd6805e0f4aa3ea409390169ad6a274a3db3d44054ed39`
- CUDA kernel SHA256: `267945ac381abc66e7075f50e79fe5a28f00188b6673ed921755f05cb424acc4`
- Verilator: `Verilator 5.020 2024-01-01 rev (Debian 5.020-1)`
- GPU: `NVIDIA RTX A6000`
- Shape per case: `M × K × N = 128 × 128 × 2`; weights are signed INT4.
- RTL path: `gemm_unit_wrap_opt_v3`, K-first (`load_i=1`), FP32 dequant scale, packed FP16/BF16 output.

## Aggregate

| Activation | CUDA geometry | Exact | Mismatch | Exact rate | Mean ULP | P95 ULP | Max ULP |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | experiment_extra19 | 1024 / 1024 | 0 | 100.000000% | 0 | 0 | 0 |
| BF16 | rtl_matched | 1024 / 1024 | 0 | 100.000000% | 0 | 0 | 0 |
| FP16 | experiment_extra19 | 1024 / 1024 | 0 | 100.000000% | 0 | 0 | 0 |
| FP16 | rtl_matched | 1024 / 1024 | 0 | 100.000000% | 0 | 0 | 0 |

## Isolation checks

| Activation | RTL FP32 core vs software FP32 | Correctly repacked RTL FP32 vs CUDA |
|---|---:|---:|
| BF16 | 1024 / 1024 exact | 1024 / 1024 exact |
| FP16 | 1024 / 1024 exact | 1024 / 1024 exact |

## Per-case result (current experiment: extra_bits=19)

| Activation | Case | Exact | Mismatch | Exact rate | Max ULP | Max abs |
|---|---|---:|---:|---:|---:|---:|
| BF16 | normal | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| BF16 | wide_exponent | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| BF16 | subnormal | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| BF16 | cancellation | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | normal | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | wide_exponent | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | subnormal | 256 / 256 | 0 | 100.000000% | 0 | 0 |
| FP16 | cancellation | 256 / 256 | 0 | 100.000000% | 0 | 0 |

## Interpretation

The current CUDA experiment configuration and the RTL are bit-exact for all tested outputs.

## Applied fixes

1. RTL BF16 packing: replace the concatenated rounding bias with `32'h0000_7fff + fp32[16]`.
2. RTL BF16 subnormal alignment: use effective exponent 1 when exponent is zero and mantissa is nonzero.
3. CUDA integer restoration: for normal results, round the integer once with `__ll2float_rn` and apply the power-of-two scale with `ldexpf`; for subnormal results, round the integer quotient/remainder directly to FP32 subnormal bits. This avoids both an underflowed intermediate factor and FP32 double rounding without using FP64.

The Verilator environment replaces unavailable Synopsys DesignWare cells. FIFO and leading-zero detection use SystemVerilog compatibility models; FP32 multiply uses a DPI-C IEEE binary32 operation. The reconfigurable prealigner, MXU, column merger, int-to-FP logic, wrapper, and packer are the supplied RTL. This run does not cover multi-tile K accumulation (`load_i=0`) or asymmetric zero-point correction.

Raw JSON: `/home/jaeyongjang/project.local/SpinQuant/results/fpint-rtl-compare/summary.json`
