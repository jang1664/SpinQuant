from dataclasses import replace

import numpy as np
import pytest
import torch

from fpint_emul import (FpIntConfig, dequantize_weight, fpint_linear,
                        qrow_real_2scomp_reference)
from fpint_emul.py.fpint_emul import fpint_gemm_qrow_real_2scomp


def case(fmt="fp16", bits=4, group=3, tile=32, shape=(2, 3, 67), n=7, asymmetric=True):
    g = torch.Generator().manual_seed(348)
    dtype = torch.float16 if fmt == "fp16" else torch.bfloat16
    cfg = FpIntConfig(bits, group_size=group, mxu_rows=tile,
                      activation_format=fmt, quantization_direction="qrow")
    x = torch.randn(shape, generator=g).to(dtype)
    w = torch.randint(-(1 << (bits - 1)), 1 << (bits - 1), (n, shape[-1]), generator=g, dtype=torch.int8)
    ps = cfg.parameter_shape(n, shape[-1])
    s = (torch.randn(ps, generator=g) * 0.1).to(dtype)
    z = torch.randint(-7, 8, ps, generator=g, dtype=torch.int32) if asymmetric else torch.zeros(ps, dtype=torch.int32)
    return x, w, s, z, cfg


def check(data, backend, bias=None, validate_values=True):
    x, w, s, z, cfg = data
    expected = torch.from_numpy(qrow_real_2scomp_reference(x, w, s, z, cfg, bias))
    device = "cuda" if backend == "fpint_cuda" else "cpu"
    actual = fpint_linear(*(v.to(device) for v in (x, w, s, z)), cfg,
                          bias=None if bias is None else bias.to(device), backend=backend,
                          validate_values=validate_values)
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=0, rtol=0, equal_nan=True)
    return actual


@pytest.fixture(params=["fpint_torch", "fpint_cuda"])
def backend(request):
    if request.param == "fpint_cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_formats_weights_and_k_varying_zero(backend, fmt, bits, asymmetric):
    check(case(fmt, bits, asymmetric=asymmetric), backend)


@pytest.mark.parametrize("shape,n,group,tile", [
    ((35,), 5, -1, 16), ((2, 0, 33), 5, 3, 32), ((2, 7), 0, 3, 8),
    ((1, 259), 9, 5, 128), ((2, 257), 5, 2, 256),
    ((1, 69), 4, 2, 7), ((1, 5), 2, 1, 1), ((1, 4097), 3, -1, 128),
])
def test_shapes_groups_and_tile_tails(backend, shape, n, group, tile):
    check(case(shape=shape, n=n, group=group, tile=tile), backend)


def test_noncontiguous_and_bias(backend):
    x, w, s, z, cfg = case(shape=(2, 3, 67))
    x = x.transpose(0, 1)
    w, s, z = (v.T.contiguous().T for v in (w, s, z))
    check((x, w, s, z, cfg), backend, torch.linspace(-0.2, 0.2, w.shape[0]).half())


@pytest.mark.parametrize("dtype", [torch.int16, torch.int32, torch.int64])
def test_zero_storage_dtypes(backend, dtype):
    x, w, s, z, cfg = case()
    check((x, w, s, z.to(dtype), cfg), backend)


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_subnormal_wide_exponents_and_mixed_scale_dtype(backend, fmt):
    x, w, s, z, cfg = case(fmt=fmt, shape=(1, 67))
    raw = [0, 0x8000, 1, 0x007f, 0x0080, 0x3f80, 0xbf80, 0x7f7f] if fmt == "bf16" else [0, 0x8000, 1, 0x03ff, 0x0400, 0x3c00, 0xbc00, 0x7bff]
    x = torch.tensor(raw, dtype=torch.uint16).view(x.dtype).repeat(9)[:67][None]
    s = s.to(torch.float16 if fmt == "bf16" else torch.bfloat16)
    check((x, w, s, z, cfg), backend)


def test_scale_rounding_precedes_alignment(backend):
    # (1 + 2^-10)^2 rounds to 1 + 2^-9 in FP16, before cancellation.
    cfg = FpIntConfig(4, group_size=1, mxu_rows=2, quantization_direction="qrow")
    x = torch.tensor([[1.0009765625, 1.001953125]], dtype=torch.float16)
    s = torch.tensor([[1.0009765625], [1]], dtype=torch.float16)
    w = torch.tensor([[1, -1]], dtype=torch.int8)
    z = torch.zeros((2, 1), dtype=torch.int32)
    actual = check((x, w, s, z, cfg), backend)
    assert actual.item() == 0


@pytest.mark.parametrize("fmt", ["fp16", "bf16"])
def test_subnormal_output_and_custom_alignment_bits(backend, fmt):
    dtype = torch.float16 if fmt == "fp16" else torch.bfloat16
    cfg = FpIntConfig(4, group_size=1, mxu_rows=3, extra_bits=5,
                      reduce_extra_bits=2, activation_format=fmt,
                      quantization_direction="qrow")
    x = torch.tensor([1, 2, 0x8001, 0x8002], dtype=torch.uint16).view(dtype)[:, None]
    w = torch.tensor([[1], [-1]], dtype=torch.int8)
    s = torch.ones((1, 2), dtype=dtype)
    z = torch.zeros((1, 2), dtype=torch.int32)
    check((x, w, s, z, cfg), backend)


def test_legacy_single_tile_bit_exact(backend):
    # A single tile avoids the legacy FP64 vs backend FP32 accumulation difference.
    data = case(shape=(2, 32), n=5, group=32, tile=32)
    x, w, s, z, cfg = data
    legacy = fpint_gemm_qrow_real_2scomp(
        x.numpy().view(np.uint16), w.numpy().T, s.numpy().view(np.uint16),
        z.numpy().astype(np.int16), 2, 5, 32)
    actual = check(data, backend)
    np.testing.assert_array_equal(actual.cpu().numpy().view(np.uint16), legacy)


def test_standard_dequantization():
    x, w, s, z, cfg = case()
    expected = torch.stack([(w[j].int() - z[:, j // 3]) * s[:, j // 3].float() for j in range(w.shape[0])]).half()
    torch.testing.assert_close(dequantize_weight(w, s, z, cfg), expected, atol=0, rtol=0)
    torch.testing.assert_close(fpint_linear(x, w, s, z, cfg, backend="standard"),
                               torch.nn.functional.linear(x, expected), atol=0, rtol=0)


def test_sequential_fp32_accumulation(backend):
    cfg = FpIntConfig(4, group_size=1, mxu_rows=1, activation_format="bf16",
                      quantization_direction="qrow")
    x = torch.tensor([[2**24, 1, 2**24]], dtype=torch.bfloat16)
    w = torch.tensor([[1, 1, -1]], dtype=torch.int8)
    s = torch.ones((3, 1), dtype=torch.bfloat16)
    z = torch.zeros((3, 1), dtype=torch.int32)
    # FP64 accumulation would give 1. Sequential FP32 rounds away the middle 1.
    assert check((x, w, s, z, cfg), backend).item() == 0


@pytest.mark.parametrize("problem", ["shape", "weight", "scale_inf", "prescale_overflow", "zero_range", "direction"])
def test_invalid_inputs(backend, problem):
    x, w, s, z, cfg = case()
    if problem == "shape": s = s.T.contiguous()
    if problem == "weight": w[0, 0] = 8
    if problem == "scale_inf": s[0, 0] = float("inf")
    if problem == "prescale_overflow": x.fill_(65504); s.fill_(2)
    if problem == "zero_range": z = z.to(torch.int64); z[0, 0] = 1 << 31
    if problem == "direction": cfg = replace(cfg, quantization_direction="qcol", group_size=32)
    from fpint_emul.qrow import qrow_real_2scomp_cuda, qrow_real_2scomp_torch
    fn = qrow_real_2scomp_cuda if backend == "fpint_cuda" else qrow_real_2scomp_torch
    device = "cuda" if backend == "fpint_cuda" else "cpu"
    with pytest.raises((ValueError, TypeError)):
        fn(*(v.to(device) for v in (x, w, s, z)), cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_current_stream_and_unchecked_path():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        check(case(), "fpint_cuda", validate_values=False)
    stream.synchronize()
