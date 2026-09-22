from argparse import Namespace

import numpy as np
import pytest
import torch

from fpint_emul import FpIntConfig, fpint_linear, qcol_real_2scomp_reference
from measure_fpint_qcol_accuracy import make_case
from measure_fpint_scale_sweep import (
    CANDIDATES, aggregate, output_coverage, run, sample_scale_fields, write_outputs,
)


@pytest.mark.parametrize("dtype,bits,bias", [("fp16", 10, 15), ("bf16", 7, 127)])
def test_scale_raw_fields_uniform_and_paired(dtype, bits, bias):
    a = sample_scale_fields(17, (100000,), dtype, bias).view(torch.int16).numpy().view(np.uint16)
    b = sample_scale_fields(17, (100000,), dtype, bias + 1).view(torch.int16).numpy().view(np.uint16)
    mantissa_mask = (1 << bits) - 1
    assert np.array_equal(a >> 15, b >> 15)
    assert np.array_equal(a & mantissa_mask, b & mantissa_mask)
    assert set(a >> 15) == {0, 1}
    assert len(set(a & mantissa_mask)) == 1 << bits
    exponents = (a & 0x7FFF) >> bits
    counts = np.bincount(exponents, minlength=bias + 1)
    expected = len(a) / (bias + 1)
    assert np.max(np.abs(counts - expected)) < 6 * np.sqrt(expected)
    assert torch.isfinite(sample_scale_fields(17, (100000,), dtype, bias)).all()


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
def test_signed_zero_scale_reference_torch_cuda(dtype):
    config = FpIntConfig(4, 128, 128, activation_format=dtype)
    a, w, _, z = make_case(m=4, k=257, n=5, seed=7, config=config, exponent_max=config.exponent_bias)
    scale = sample_scale_fields(7, (5, 3), dtype, config.exponent_bias)
    scale[0] = torch.tensor([-.5, 0., .75], dtype=scale.dtype)
    expected = torch.from_numpy(qcol_real_2scomp_reference(a, w, scale, z, config)).float()
    tensors = (torch.as_tensor(a), torch.from_numpy(w), scale, torch.from_numpy(z))
    paths = [("fpint_torch", "cpu")]
    if torch.cuda.is_available():
        paths += [("fpint_torch", "cuda"), ("fpint_cuda", "cuda")]
    for backend, device in paths:
        actual = fpint_linear(*(t.to(device) for t in tensors), config, backend=backend, has_zero=False)
        torch.testing.assert_close(actual.cpu().float(), expected, atol=0, rtol=0)
    scale[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        qcol_real_2scomp_reference(a, w, scale, z, config)
    with pytest.raises(ValueError, match="finite"):
        fpint_linear(*tensors, config, has_zero=False)


def test_coverage_includes_all_failures_without_changing_denominator():
    outputs = {name: torch.ones(4) for name in CANDIDATES}
    outputs["gpu_true"][1] = float("nan")
    outputs["gpu_false"][2] = float("inf")
    outputs["fpint"][3] = -float("inf")
    coverage = output_coverage(outputs)
    assert coverage["total"] == 4
    assert coverage["common_finite"] == 1
    assert coverage["outputs"]["gpu_true"]["nan"] == 1
    assert coverage["outputs"]["gpu_false"]["positive_inf"] == 1
    assert coverage["outputs"]["fpint"]["negative_inf"] == 1
    row = {**coverage, "dequant_weight_nonfinite": 2, "reference_check": None}
    summary = aggregate([row, row], .999)
    assert summary["total"] == 8
    assert summary["common_finite_fraction"] == .25
    assert not summary["finite_target_met"]
    assert summary["dequant_weight_nonfinite"] == 4


def test_sweep_continues_when_finite_target_not_met(tmp_path):
    args = Namespace(
        device="cpu", activation_format="fp16", trials=1, k_values=(128,),
        m=4, n=4, base_seed=72, scale_exp_min=0, scale_exp_max_values=(15, 30),
        reference_trials=1, finite_target=1.,
    )
    result = run(args)
    assert result["status"] == "completed"
    assert len(result["records"]) == 2
    assert result["records"][0]["input_sha256"] == result["records"][1]["input_sha256"]
    assert not result["summary"]["30"]["all_k_meet_target"]
    path = tmp_path / "sweep.json"
    write_outputs(result, path)
    assert path.exists() and path.with_suffix(".csv").exists()
    assert path.with_name("sweep-summary.csv").exists()


@pytest.mark.parametrize("dtype,maximum", [("fp16", 15), ("bf16", 127)])
def test_accuracy_raw_fields_reproduce_sweep_scale(dtype, maximum):
    config = FpIntConfig(4, 128, 128, activation_format=dtype)
    a, w, scale, zero = make_case(
        m=4, k=256, n=5, config=config, seed=20260915,
        exponent_max=config.exponent_bias, scale_mode="raw-fields",
    )
    expected = sample_scale_fields(20260915, (5, 2), dtype, maximum)
    assert torch.equal(scale.view(torch.int16), expected.view(torch.int16))
    assert not np.any(zero)
