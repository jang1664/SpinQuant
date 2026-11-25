# QROW real two's-complement backend

`fpint_linear` supports QROW with `backend="fpint_cuda"` (fused CUDA) and
`backend="fpint_torch"` (Torch). `quantization_direction="qrow"` selects QROW;
the default remains QCOL. This is a tensor-level API. Existing model PTQ
scripts still produce QCOL metadata; this addition does not change their
quantization direction or apply QROW to attention automatically.

```python
from fpint_emul import FpIntConfig, fpint_linear

config = FpIntConfig(
    weight_bits=4,
    group_size=32,          # N-axis parameter sharing
    mxu_rows=128,           # K-axis reduction tile
    activation_format="fp16",
    quantization_direction="qrow",
)
y = fpint_linear(x, weight, scale, zero, config, backend="fpint_cuda")
```

Tensor layouts:

| Tensor | Shape | Dtype |
| --- | --- | --- |
| activation | `[..., K]` | FP16 or BF16 matching config |
| weight | `[N, K]` | signed INT4/INT8 values in `torch.int8` |
| scale | `[K, ceil(N/group_size)]` | FP16 or BF16 |
| zero | same as scale | int16/int32/int64; values must fit int32 and arithmetic bounds |
| optional bias | `[N]` | same dtype as activation |
| output | `[..., N]` | same dtype as activation |

`group_size=-1` shares one scale/zero per K across all N. Unlike QCOL, QROW
group size need not be a multiple of `mxu_rows`. CUDA supports MXU sizes 1–256,
including partial K tiles and partial N groups. All tensors share a device.

For each output and K tile:

1. Compute activation × scale in FP32 and round to the activation format.
2. Find the maximum exponent of these scaled values, and prealign using
   `extra_bits` and `reduce_extra_bits` for the two integer paths.
3. Sum `aligned_main * weight` and `aligned_reduce * zero[k, group]` separately.
4. Subtract the shifted correction, restore to FP32, then accumulate tiles
   sequentially in FP32. There is no post-MAC scale multiplication.
5. Round to the activation format, then add optional bias in that format.

The CUDA implementation uses one warp per output and does not materialize an
`M × N × K` scaled-activation tensor. It emulates numerical behavior and is
not a native FP×INT Tensor Core implementation or a throughput claim.

Checked calls reject nonfinite inputs and prescaling overflow. With
`validate_values=False`, the caller must guarantee valid ranges and finite
prescaled activations; CUDA emits NaN for a tile containing nonfinite scaled
values. A supplied `has_zero=False` must accurately describe all-zero points.

The independent `qrow_real_2scomp_reference` returns NumPy FP16 (or FP32 values
representing BF16). It matches the new sequential FP32 accumulation contract.
The older `fpint_emul/py/fpint_emul.py` QROW routine accumulates tiles in Python
FP64, so multi-tile outputs are not promised to be bit-identical to it.

Tests: `tests/test_fpint_qrow.py` covers the independent oracle, legacy
single-tile comparison, both floating formats and weight widths, K-varying
zero points, signed scales, subnormals, grouping/tile tails, noncontiguous
inputs, bias, current CUDA stream, validation and FP32 accumulation order.

```bash
conda activate spinquant
pytest -q tests/test_fpint_qrow.py tests/test_fpint_emul.py
```

Validated on 2026-10-14 on an RTX A6000 (PyTorch 2.7.0+cu126): the combined
QROW and existing QCOL/model integration suite passed **121 tests** under
CUDA Compute Sanitizer memcheck, with **0 memory errors**. One existing
two-GPU device-selection test was skipped because the test process exposed
only GPU 0. Logs: `/tmp/spinquant-qrow-memcheck.log`.
