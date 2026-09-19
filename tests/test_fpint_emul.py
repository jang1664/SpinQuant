import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace

from fpint_emul import (
    FpIntConfig,
    dequantize_weight,
    fpint_linear,
    qcol_real_2scomp_reference,
    qcol_real_2scomp_torch,
)
from utils.quant_utils import (
    ActQuantWrapper,
    WeightQuantizer,
    configure_fpint_linears,
    stash_fpint_metadata,
)
from eval_utils.gptq_utils import GPTQ, rtn_fwrd
from eval_utils.main import ptq_model
from utils.process_args import parser_gen
from utils import hadamard_utils


def _case(
    *,
    shape=(2, 3, 67),
    n=19,
    bits=4,
    group_size=64,
    mxu_rows=32,
    asymmetric=False,
    seed=0,
):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=shape).astype(np.float16)
    qmin, qmax = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    weight = rng.integers(qmin, qmax + 1, size=(n, shape[-1]), dtype=np.int16).astype(
        np.int8
    )
    groups = 1 if group_size == -1 else (shape[-1] + group_size - 1) // group_size
    scale = rng.uniform(0.002, 0.08, size=(n, groups)).astype(np.float16)
    if asymmetric:
        zero = rng.integers(qmin, qmax + 1, size=(n, groups), dtype=np.int16)
    else:
        zero = np.zeros((n, groups), dtype=np.int16)
    return x, weight, scale, zero, FpIntConfig(bits, group_size, mxu_rows)


def _assert_reference_matches_torch(case, device="cpu", bias=None):
    x, weight, scale, zero, config = case
    expected = qcol_real_2scomp_reference(
        x, weight, scale, zero, config, bias=bias
    )
    actual = qcol_real_2scomp_torch(
        torch.from_numpy(x).to(device),
        torch.from_numpy(weight).to(device),
        torch.from_numpy(scale).to(device),
        torch.from_numpy(zero).to(device),
        config,
        bias=None if bias is None else torch.from_numpy(bias).to(device),
    )
    torch.testing.assert_close(
        actual.cpu(), torch.from_numpy(expected), atol=1e-3, rtol=1e-3
    )


def _bf16_case(*, k=257, n=37, asymmetric=False, seed=0):
    generator = torch.Generator().manual_seed(seed)
    activation = torch.randn((2, 3, k), generator=generator).to(torch.bfloat16)
    weight = torch.randint(-8, 8, (n, k), generator=generator, dtype=torch.int8)
    groups = (k + 127) // 128
    scale = (
        torch.rand((n, groups), generator=generator) * 0.078 + 0.002
    ).to(torch.float16)
    zero = (
        torch.randint(-8, 8, (n, groups), generator=generator, dtype=torch.int32)
        if asymmetric
        else torch.zeros((n, groups), dtype=torch.int32)
    )
    config = FpIntConfig(
        4, group_size=128, mxu_rows=128, activation_format="bf16"
    )
    return activation, weight, scale, zero, config


def _assert_bf16_reference_matches(case, backend="fpint_torch"):
    activation, weight, scale, zero, config = case
    expected = torch.from_numpy(
        qcol_real_2scomp_reference(activation, weight, scale, zero, config)
    )
    device = "cuda" if backend == "fpint_cuda" else "cpu"
    actual = fpint_linear(
        activation.to(device),
        weight.to(device),
        scale.to(device),
        zero.to(device),
        config,
        backend=backend,
    )
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.cpu().float(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("asymmetric", [False, True])
def test_bf16_reference_matches_torch_bit_exact(asymmetric):
    _assert_bf16_reference_matches(
        _bf16_case(asymmetric=asymmetric, seed=300 + asymmetric)
    )


def test_bf16_wide_exponents_subnormal_and_signed_zero():
    bits = torch.tensor(
        [0x0000, 0x8000, 0x0001, 0x007F, 0x0080, 0x3F80, 0xBF80, 0x7F7F],
        dtype=torch.uint16,
    )
    activation = bits.view(torch.bfloat16).repeat(17)[:129].reshape(1, 129)
    weight = torch.tensor([[-8, 7]], dtype=torch.int8).repeat(3, 65)[:, :129]
    scale = torch.full((3, 2), 0.001, dtype=torch.float16)
    zero = torch.tensor([[7, -8], [-8, 7], [3, -2]], dtype=torch.int32)
    config = FpIntConfig(
        4, group_size=128, mxu_rows=128, activation_format="bf16"
    )
    _assert_bf16_reference_matches((activation, weight, scale, zero, config))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("asymmetric", [False, True])
def test_bf16_cuda_matches_independent_reference_bit_exact(asymmetric):
    _assert_bf16_reference_matches(
        _bf16_case(asymmetric=asymmetric, seed=400 + asymmetric),
        backend="fpint_cuda",
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("asymmetric", [False, True])
def test_reference_and_torch_cover_bits_and_zero_points(bits, asymmetric):
    _assert_reference_matches_torch(
        _case(bits=bits, asymmetric=asymmetric, seed=bits + asymmetric)
    )


@pytest.mark.parametrize(
    "shape,n,group_size,mxu_rows",
    [
        ((5, 35), 7, 32, 16),
        ((2, 0, 33), 5, 32, 16),
        ((2, 2, 65), 3, 64, 32),
        ((3, 47), 11, -1, 16),
        ((2, 3, 257), 37, 128, 128),
    ],
)
def test_general_shapes_tails_and_independent_grouping(
    shape, n, group_size, mxu_rows
):
    _assert_reference_matches_torch(
        _case(
            shape=shape,
            n=n,
            group_size=group_size,
            mxu_rows=mxu_rows,
            asymmetric=True,
        )
    )


def test_bias_is_added_after_fp16_fpint_output():
    case = _case(shape=(4, 32), n=9, group_size=32, asymmetric=True)
    bias = np.linspace(-0.5, 0.5, 9).astype(np.float16)
    _assert_reference_matches_torch(case, bias=bias)


def test_noncontiguous_activation_is_supported():
    case = _case(shape=(3, 34), n=5, group_size=32, asymmetric=True)
    x, weight, scale, zero, config = case
    backing = torch.empty((3, 68), dtype=torch.float16)
    backing[:, ::2] = torch.from_numpy(x)
    noncontiguous = backing[:, ::2]
    assert not noncontiguous.is_contiguous()
    actual = qcol_real_2scomp_torch(
        noncontiguous,
        torch.from_numpy(weight),
        torch.from_numpy(scale),
        torch.from_numpy(zero),
        config,
    )
    expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-3, rtol=1e-3)


def test_subnormal_wide_exponents_and_cancellation():
    x = np.array(
        [[np.nextafter(np.float16(0), np.float16(1)), -65504.0] * 17],
        dtype=np.float16,
    )[:, :33]
    weight = np.tile(np.array([[-8, 7]], dtype=np.int8), (3, 17))[:, :33]
    scale = np.full((3, 2), np.float16(0.001), dtype=np.float16)
    zero = np.array([[7, -8], [-8, 7], [3, -2]], dtype=np.int16)
    config = FpIntConfig(4, group_size=32, mxu_rows=16)
    _assert_reference_matches_torch((x, weight, scale, zero, config))
    if torch.cuda.is_available():
        expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
        actual = fpint_linear(
            torch.from_numpy(x).cuda(),
            torch.from_numpy(weight).cuda(),
            torch.from_numpy(scale).cuda(),
            torch.from_numpy(zero).cuda(),
            config,
            backend="fpint_cuda",
        )
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected))


def test_large_k_sequential_fp32_accumulation_matches_reference():
    _assert_reference_matches_torch(
        _case(
            shape=(2, 4097),
            n=5,
            bits=8,
            group_size=64,
            mxu_rows=32,
            asymmetric=True,
            seed=27,
        )
    )


def test_default_general_reference_matches_original_hardware_reference():
    path = Path(__file__).parents[1] / "fpint_emul" / "py" / "fpint_emul.py"
    spec = importlib.util.spec_from_file_location("legacy_fpint_emul", path)
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)

    x, weight, scale, zero, config = _case(
        shape=(3, 64), n=16, group_size=32, mxu_rows=32, asymmetric=True, seed=11
    )
    expected_bits = legacy.fpint_gemm_qcol_real_2scomp(
        x.view(np.uint16),
        weight.T.copy().view(np.uint8),
        scale.T.copy().view(np.uint16),
        zero.T.copy(),
        3,
        16,
        64,
    )
    expected = expected_bits.view(np.float16)
    actual = qcol_real_2scomp_reference(x, weight, scale, zero, config)
    np.testing.assert_array_equal(actual, expected)


def test_standard_backend_dequantizes_signed_asymmetric_parameters():
    case = _case(shape=(2, 35), n=7, group_size=32, asymmetric=True)
    x, weight, scale, zero, config = case
    tw = torch.from_numpy(weight)
    ts = torch.from_numpy(scale)
    tz = torch.from_numpy(zero)
    expected_weight = dequantize_weight(tw, ts, tz, config)
    actual = fpint_linear(
        torch.from_numpy(x), tw, ts, tz, config, backend="standard"
    )
    torch.testing.assert_close(
        actual, torch.nn.functional.linear(torch.from_numpy(x), expected_weight)
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"weight_bits": 3}, "4 or 8"),
        ({"weight_bits": 4, "group_size": 24, "mxu_rows": 16}, "multiple"),
        ({"weight_bits": 8, "extra_bits": 8, "reduce_extra_bits": 10}, ">="),
    ],
)
def test_invalid_configurations_fail_early(kwargs, match):
    with pytest.raises(ValueError, match=match):
        FpIntConfig(**kwargs)


def test_nonfinite_activation_and_out_of_range_int4_are_rejected():
    x, weight, scale, zero, config = _case(shape=(1, 32), n=2, group_size=32)
    x[0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN and Inf"):
        qcol_real_2scomp_reference(x, weight, scale, zero, config)
    x[0, 0] = 0
    weight[0, 0] = 8
    with pytest.raises(ValueError, match="INT4"):
        qcol_real_2scomp_reference(x, weight, scale, zero, config)


def test_large_signed_zero_points_are_supported_until_int64_would_overflow():
    x, weight, scale, zero, config = _case(
        shape=(2, 32), n=3, group_size=32, asymmetric=False
    )
    zero[:] = np.array([[-1000], [1000], [511]], dtype=np.int16)
    _assert_reference_matches_torch((x, weight, scale, zero, config))
    if torch.cuda.is_available():
        expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
        actual = fpint_linear(
            torch.from_numpy(x).cuda(),
            torch.from_numpy(weight).cuda(),
            torch.from_numpy(scale).cuda(),
            torch.from_numpy(zero).cuda(),
            config,
            backend="fpint_cuda",
        )
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected))
    overflowing = zero.astype(np.int64)
    overflowing[:] = 1 << 40
    with pytest.raises(ValueError, match="overflow int64"):
        qcol_real_2scomp_reference(x, weight, scale, overflowing, config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("bits,asymmetric", [(4, False), (4, True), (8, False), (8, True)])
def test_cuda_matches_independent_reference(bits, asymmetric):
    case = _case(bits=bits, asymmetric=asymmetric, seed=100 + bits)
    _assert_reference_matches_torch(case, device="cuda")
    x, weight, scale, zero, config = case
    expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
    actual = fpint_linear(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(weight).cuda(),
        torch.from_numpy(scale).cuda(),
        torch.from_numpy(zero).cuda(),
        config,
        backend="fpint_cuda",
    )
    torch.testing.assert_close(
        actual.cpu(), torch.from_numpy(expected), atol=1e-3, rtol=1e-3
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_backend_honors_nondefault_current_stream():
    x, weight, scale, zero, config = _case(
        shape=(2, 35), n=19, group_size=32, mxu_rows=16, asymmetric=True
    )
    expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = fpint_linear(
            torch.from_numpy(x).cuda(),
            torch.from_numpy(weight).cuda(),
            torch.from_numpy(scale).cuda(),
            torch.from_numpy(zero).cuda(),
            config,
            backend="fpint_cuda",
        )
        copied = actual.to("cpu", non_blocking=True)
    stream.synchronize()
    torch.testing.assert_close(copied, torch.from_numpy(expected), atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("asymmetric", [False, True])
def test_cuda_mxu_row_128_matches_independent_reference(asymmetric):
    case = _case(
        shape=(2, 3, 257),
        n=37,
        bits=4,
        group_size=128,
        mxu_rows=128,
        asymmetric=asymmetric,
        seed=128 + asymmetric,
    )
    x, weight, scale, zero, config = case
    expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
    actual = fpint_linear(
        torch.from_numpy(x).cuda(),
        torch.from_numpy(weight).cuda(),
        torch.from_numpy(scale).cuda(),
        torch.from_numpy(zero).cuda(),
        config,
        backend="fpint_cuda",
    )
    torch.testing.assert_close(
        actual.cpu(), torch.from_numpy(expected), atol=1e-3, rtol=1e-3
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="two CUDA GPUs are required")
def test_cuda_backend_uses_the_tensor_device_not_current_device():
    x, weight, scale, zero, config = _case(
        shape=(1, 33), n=5, group_size=-1, mxu_rows=16, asymmetric=True
    )
    expected = qcol_real_2scomp_reference(x, weight, scale, zero, config)
    torch.cuda.set_device(0)
    device = torch.device("cuda:1")
    actual = fpint_linear(
        torch.from_numpy(x).to(device),
        torch.from_numpy(weight).to(device),
        torch.from_numpy(scale).to(device),
        torch.from_numpy(zero).to(device),
        config,
        backend="fpint_cuda",
    )
    assert actual.device == device
    torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected), atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("bits,symmetric", [(4, True), (4, False), (8, True), (8, False)])
def test_weight_quantizer_preserves_signed_integer_metadata_with_k_tail(
    bits, symmetric
):
    torch.manual_seed(bits + symmetric)
    weight = torch.randn(5, 67)
    quantizer = WeightQuantizer()
    quantizer.configure(
        bits,
        perchannel=True,
        sym=symmetric,
        weight_groupsize=32,
    )
    quantizer.find_params(weight)
    dequantized, integer, scale, zero = quantizer.fake_quantize_with_metadata(weight)
    assert integer.dtype == torch.int8
    assert zero.dtype == torch.int32
    assert integer.min() >= -(1 << (bits - 1))
    assert integer.max() <= (1 << (bits - 1)) - 1
    torch.testing.assert_close(
        dequantized,
        scale * (integer.to(scale.dtype) - zero.to(scale.dtype)),
    )
    if symmetric:
        assert torch.count_nonzero(zero) == 0
    else:
        # Signed conversion applies the same offset to q and z, preserving q-z.
        unsigned = integer.to(torch.int32) + (1 << (bits - 1))
        unsigned_zero = zero + (1 << (bits - 1))
        torch.testing.assert_close(
            integer.to(torch.int32) - zero,
            unsigned - unsigned_zero,
        )


@pytest.mark.parametrize("symmetric", [True, False])
def test_weight_quantizer_training_path_retains_ste_gradient(symmetric):
    torch.manual_seed(3)
    weight = torch.randn(4, 32, requires_grad=True)
    quantizer = WeightQuantizer()
    quantizer.configure(
        4, perchannel=True, sym=symmetric, weight_groupsize=32
    )
    quantizer.find_params(weight.detach())
    quantizer.quantize(weight).sum().backward()
    torch.testing.assert_close(weight.grad, torch.ones_like(weight))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_groupwise_mse_weight_quantizer_supports_16bit_inputs(dtype):
    torch.manual_seed(31)
    weight = torch.randn(4, 35).to(dtype)
    quantizer = WeightQuantizer()
    quantizer.configure(
        4,
        perchannel=True,
        sym=True,
        mse=True,
        weight_groupsize=32,
    )
    quantizer.find_params(weight)
    assert quantizer.scale.dtype == dtype
    assert quantizer.scale.shape == weight.shape
    assert bool(torch.isfinite(quantizer.scale).all())


def _wrapper_with_pending_metadata(asymmetric=True):
    torch.manual_seed(4)
    linear = torch.nn.Linear(35, 7, bias=True).half()
    quantizer = WeightQuantizer()
    quantizer.configure(
        4,
        perchannel=True,
        sym=not asymmetric,
        weight_groupsize=32,
    )
    quantizer.find_params(linear.weight.data)
    dequantized, integer, scale, zero = quantizer.fake_quantize_with_metadata(
        linear.weight.data
    )
    linear.weight.data.copy_(dequantized)
    stash_fpint_metadata(
        linear,
        integer_weight=integer,
        expanded_scale=scale,
        expanded_zero=zero,
        bits=4,
        group_size=32,
    )
    return ActQuantWrapper(linear)


def _backend_args(backend="fpint_torch"):
    return SimpleNamespace(
        linear_backend=backend,
        fpint_mxu_rows=16,
        fpint_extra_bits=19,
        fpint_reduce_extra_bits=10,
    )


def test_wrapper_dispatches_fpint_and_reports_layer_coverage():
    wrapper = _wrapper_with_pending_metadata()
    model = torch.nn.ModuleDict({"proj": wrapper})
    coverage = configure_fpint_linears(model, _backend_args())
    assert coverage == [
        {"name": "proj", "backend": "fpint_torch", "reason": "", "bits": 4}
    ]
    x = torch.randn(2, 3, 35).half()
    expected = qcol_real_2scomp_reference(
        x.numpy(),
        wrapper.fpint_weight.numpy(),
        wrapper.fpint_scale.numpy(),
        wrapper.fpint_zero.numpy(),
        FpIntConfig(4, group_size=32, mxu_rows=16),
        bias=wrapper.bias.detach().numpy(),
    )
    actual = model["proj"](x)
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=1e-3, rtol=1e-3)


def test_wrapper_preserves_activation_qdq_before_fpint_linear():
    wrapper = _wrapper_with_pending_metadata(asymmetric=False)
    configure_fpint_linears(torch.nn.ModuleDict({"proj": wrapper}), _backend_args())
    wrapper.quantizer.configure(bits=4, groupsize=-1, sym=True)
    x = torch.randn(2, 35).half()
    wrapper.quantizer.find_params(x)
    expected_input = wrapper.quantizer(x).half()
    expected = qcol_real_2scomp_torch(
        expected_input,
        wrapper.fpint_weight,
        wrapper.fpint_scale,
        wrapper.fpint_zero,
        FpIntConfig(4, group_size=32, mxu_rows=16),
        bias=wrapper.bias,
    )
    actual = wrapper(x)
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_online_hadamard_runs_before_activation_qdq_and_fpint_cuda():
    torch.manual_seed(14)
    linear = torch.nn.Linear(32, 7, bias=False).half()
    quantizer = WeightQuantizer()
    quantizer.configure(4, perchannel=True, sym=True, weight_groupsize=32)
    quantizer.find_params(linear.weight)
    dequantized, integer, scale, zero = quantizer.fake_quantize_with_metadata(
        linear.weight
    )
    linear.weight.data.copy_(dequantized)
    stash_fpint_metadata(
        linear,
        integer_weight=integer,
        expanded_scale=scale,
        expanded_zero=zero,
        bits=4,
        group_size=32,
    )
    wrapper = ActQuantWrapper(linear).cuda()
    wrapper.online_full_had = True
    wrapper.online_had_mode = "factorized"
    wrapper.had_K, wrapper.K = hadamard_utils.get_hadK(32)
    wrapper.quantizer.configure(bits=4, groupsize=-1, sym=True)
    configure_fpint_linears(
        torch.nn.ModuleDict({"proj": wrapper}),
        SimpleNamespace(
            linear_backend="fpint_cuda",
            fpint_mxu_rows=32,
            fpint_extra_bits=19,
            fpint_reduce_extra_bits=10,
        ),
    )
    x = torch.randn(2, 32, device="cuda", dtype=torch.float16)
    rotated = hadamard_utils.matmul_hadU_cuda(
        x, wrapper.had_K, wrapper.K, mode=wrapper.online_had_mode
    )
    wrapper.quantizer.find_params(rotated)
    quantized = wrapper.quantizer(rotated).half()
    expected = fpint_linear(
        quantized,
        wrapper.fpint_weight,
        wrapper.fpint_scale,
        wrapper.fpint_zero,
        FpIntConfig(4, 32, 32),
        backend="fpint_cuda",
    )
    actual = wrapper(x)
    torch.testing.assert_close(actual, expected)


def test_fpint_metadata_state_dict_round_trip_and_legacy_load():
    source = _wrapper_with_pending_metadata()
    configure_fpint_linears(torch.nn.ModuleDict({"proj": source}), _backend_args())
    state = source.state_dict()

    restored = ActQuantWrapper(torch.nn.Linear(35, 7, bias=True).half())
    restored.load_state_dict(state, strict=True)
    restored.configure_linear_backend(
        "fpint_torch", mxu_rows=16, extra_bits=19, reduce_extra_bits=10
    )
    x = torch.randn(2, 35).half()
    torch.testing.assert_close(restored(x), source(x))

    legacy_state = {
        key: value
        for key, value in state.items()
        if not key.startswith("fpint_")
    }
    legacy = ActQuantWrapper(torch.nn.Linear(35, 7, bias=True).half())
    legacy.load_state_dict(legacy_state, strict=True)
    assert not legacy.has_fpint_metadata
    with pytest.raises(ValueError, match="without integer weight metadata"):
        legacy.configure_linear_backend("fpint_torch")


def test_grouped_act_order_metadata_fails_with_actionable_error():
    wrapper = _wrapper_with_pending_metadata()
    configure_fpint_linears(
        torch.nn.ModuleDict({"proj": wrapper}), _backend_args("standard")
    )
    wrapper.fpint_group_index = wrapper.fpint_group_index.roll(1)
    with pytest.raises(ValueError, match="act-order"):
        wrapper.configure_linear_backend("fpint_torch", mxu_rows=16)


def test_floating_point_lm_head_is_the_only_explicit_fallback():
    lm_head = ActQuantWrapper(torch.nn.Linear(8, 4).half())
    model = torch.nn.ModuleDict({"lm_head": lm_head})
    coverage = configure_fpint_linears(model, _backend_args())
    assert coverage[0]["backend"] == "standard"
    assert "floating point" in coverage[0]["reason"]

    missing = torch.nn.ModuleDict(
        {"model_layer_proj": ActQuantWrapper(torch.nn.Linear(8, 4).half())}
    )
    with pytest.raises(ValueError, match="Legacy dequantized checkpoints"):
        configure_fpint_linears(missing, _backend_args())


def test_rtn_path_stashes_asymmetric_int4_metadata_with_tail():
    layer = torch.nn.Sequential(torch.nn.Linear(35, 7, bias=False).half())
    args = SimpleNamespace(
        w_bits=4,
        w_groupsize=32,
        w_asym=True,
        w_clip=False,
        int8_down_proj=False,
        export_to_et=False,
    )
    rtn_fwrd(None, "cpu", args, custom_layers=[layer])
    pending = layer[0]._fpint_pending_metadata
    assert pending["weight"].dtype == torch.int8
    assert pending["scale"].shape == (7, 2)
    assert pending["zero"].shape == (7, 2)
    assert pending["group_index"].shape == (35,)
    dequantized = dequantize_weight(
        pending["weight"],
        pending["scale"],
        pending["zero"],
        FpIntConfig(4, group_size=32, mxu_rows=16),
    )
    torch.testing.assert_close(layer[0].weight, dequantized)


def test_rtn_per_channel_metadata_uses_one_compact_group():
    layer = torch.nn.Sequential(torch.nn.Linear(35, 7, bias=False).half())
    args = SimpleNamespace(
        w_bits=8,
        w_groupsize=-1,
        w_asym=False,
        w_clip=False,
        int8_down_proj=False,
        export_to_et=False,
    )
    rtn_fwrd(None, "cpu", args, custom_layers=[layer])
    pending = layer[0]._fpint_pending_metadata
    assert pending["scale"].shape == (7, 1)
    assert pending["zero"].shape == (7, 1)
    assert torch.count_nonzero(pending["group_index"]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("actorder", [False, True])
def test_gptq_path_preserves_integer_scale_zero_and_group_mapping(actorder):
    torch.manual_seed(9)
    layer = torch.nn.Linear(32, 6, bias=False, device="cuda").float()
    gptq = GPTQ(layer)
    # A diagonal Hessian avoids calibration cost while exercising the complete
    # quantization/error-feedback and act-order metadata path.
    gptq.H = torch.diag(torch.linspace(1.0, 3.0, 32, device="cuda"))
    gptq.quantizer = WeightQuantizer()
    gptq.quantizer.configure(4, perchannel=True, sym=False)
    gptq.fasterquant(groupsize=16, blocksize=16, actorder=actorder)
    pending = layer._fpint_pending_metadata
    assert pending["weight"].shape == layer.weight.shape
    assert pending["scale"].shape == (6, 2)
    assert pending["zero"].shape == (6, 2)
    if actorder:
        canonical = torch.arange(32, dtype=torch.int32) // 16
        assert not torch.equal(pending["group_index"], canonical)
    else:
        torch.testing.assert_close(
            pending["group_index"], torch.arange(32, dtype=torch.int32) // 16
        )


def test_fpint_cli_defaults_and_validation(monkeypatch):
    monkeypatch.setattr("sys.argv", ["ptq.py"])
    args, unknown = parser_gen()
    assert unknown == []
    assert args.linear_backend == "standard"
    assert (args.fpint_mxu_rows, args.fpint_extra_bits, args.fpint_reduce_extra_bits) == (
        32,
        19,
        10,
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "ptq.py",
            "--linear_backend",
            "fpint_torch",
            "--w_bits",
            "4",
            "--w_groupsize",
            "48",
            "--fpint_mxu_rows",
            "32",
        ],
    )
    with pytest.raises(SystemExit):
        parser_gen()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_ptq_model_fpint_checkpoint_round_trip_and_forward(tmp_path):
    from transformers import LlamaConfig

    from eval_utils.modeling_llama import LlamaForCausalLM
    from utils.quant_utils import add_actquant

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    checkpoint = tmp_path / "new-checkpoint-dir" / "tiny-fpint.pt"
    args = SimpleNamespace(
        seed=0,
        rotate=False,
        load_qmodel_path=None,
        save_qmodel_path=str(checkpoint),
        export_to_et=False,
        w_bits=4,
        w_groupsize=32,
        w_asym=True,
        w_clip=False,
        w_rtn=True,
        int8_down_proj=False,
        linear_backend="fpint_cuda",
        fpint_mxu_rows=32,
        fpint_extra_bits=19,
        fpint_reduce_extra_bits=10,
        a_bits=16,
        a_groupsize=-1,
        a_asym=False,
        a_clip_ratio=1.0,
        v_bits=16,
        v_asym=False,
        v_clip_ratio=1.0,
        q_bits=16,
        q_groupsize=-1,
        q_asym=False,
        q_clip_ratio=1.0,
        k_bits=16,
        k_groupsize=-1,
        k_asym=False,
        k_clip_ratio=1.0,
        k_pre_rope=False,
        p_bits=16,
        p_groupsize=-1,
        p_asym=True,
        p_clip_ratio=1.0,
    )
    model = LlamaForCausalLM(config).half().cuda().eval()
    model = ptq_model(args, model)
    payload = torch.load(checkpoint, weights_only=False)
    assert payload["fpint_format_version"] == 1
    assert payload["fpint_quantization"] == {
        "weight_bits": 4,
        "weight_group_size": 32,
        "weight_symmetric": False,
        "weight_clip": False,
        "gptq": False,
        "act_order": False,
    }
    assert any(key.endswith("fpint_weight") for key in payload["model"])

    restored = LlamaForCausalLM(config).half().eval()
    add_actquant(restored)
    restored.load_state_dict(payload["model"], strict=True)
    configure_fpint_linears(restored, args)
    model.cuda()
    restored.cuda()
    input_ids = torch.arange(12, device="cuda").remainder(config.vocab_size)[None]
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
        actual = restored(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected)
