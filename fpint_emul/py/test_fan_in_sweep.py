"""Tests for the FP16-INT4 fan-in numerical-accuracy sweep."""

import sys
import struct
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from fan_in_sweep import (
    bits_to_fp16,
    fp16_to_bits,
    generate_trial_data,
    gpu_fp64_reference_outputs,
    gpu_model_prefix_outputs,
    plot_csv,
    qcol_real_2scomp_prefix_outputs,
    qrow_real_2scomp_prefix_outputs,
    reference_prefix_outputs,
    tensor_core_outputs,
)
from fpint_emul import (
    QBLOCK,
    QCOL,
    QROW,
    fpint_gemm_gpu,
    fpint_gemm_qcol_real_2scomp,
    fpint_gemm_qrow_real_2scomp,
    fpint_gemm_ref,
)
from visualize import ulp_diff_fp16_bits


def test_fp16_bit_round_trip():
    bits = np.arange(1 << 16, dtype=np.uint16)
    assert np.array_equal(fp16_to_bits(bits_to_fp16(bits)), bits)


def test_exact_fp16_ulp_boundaries():
    zero = np.array([0x0000], dtype=np.uint16)
    negative_zero = np.array([0x8000], dtype=np.uint16)
    min_subnormal = np.array([0x0001], dtype=np.uint16)
    negative_min_subnormal = np.array([0x8001], dtype=np.uint16)
    one = np.array([0x3C00], dtype=np.uint16)
    next_one = np.array([0x3C01], dtype=np.uint16)

    assert ulp_diff_fp16_bits(zero, negative_zero).item() == 0
    assert ulp_diff_fp16_bits(zero, min_subnormal).item() == 1
    assert ulp_diff_fp16_bits(negative_zero, negative_min_subnormal).item() == 1
    assert ulp_diff_fp16_bits(negative_min_subnormal, min_subnormal).item() == 2
    assert ulp_diff_fp16_bits(one, next_one).item() == 1
    with pytest.raises(ValueError, match="NaN"):
        ulp_diff_fp16_bits(np.array([0x7E00], dtype=np.uint16), zero)


def test_signed_asymmetric_trial_ranges_and_reproducibility():
    first = generate_trial_data(seed=17, m=2, n=16, max_k=64)
    second = generate_trial_data(seed=17, m=2, n=16, max_k=64)
    assert np.array_equal(first.input_bits, second.input_bits)
    assert np.array_equal(first.weights, second.weights)
    input_values = bits_to_fp16(first.input_bits)
    assert np.all(np.isfinite(input_values))
    assert np.all(input_values != 0)
    assert np.any(np.signbit(input_values))
    assert np.any(~np.signbit(input_values))
    assert first.weights.dtype == np.int8
    assert int(first.weights.min()) >= -8
    assert int(first.weights.max()) <= 7
    assert int(first.qcol_zero.min()) >= -4
    assert int(first.qcol_zero.max()) <= 3
    assert int(first.qrow_zero.min()) >= -4
    assert int(first.qrow_zero.max()) <= 3


@pytest.mark.parametrize("qdir", [QCOL, QROW])
def test_prefix_reference_and_gpu_model_match_scalar(qdir):
    k_values = (32, 64)
    data = generate_trial_data(seed=23, m=2, n=16, max_k=max(k_values))
    if qdir == QCOL:
        scales, zeros = data.qcol_scale_bits, data.qcol_zero
    else:
        scales, zeros = data.qrow_scale_bits, data.qrow_zero

    references = reference_prefix_outputs(
        data.input_bits, data.weights, scales, zeros, qdir, k_values
    )
    gpu_models = gpu_model_prefix_outputs(
        data.input_bits, data.weights, scales, zeros, qdir, k_values
    )
    for k in k_values:
        scale_prefix = scales[: k // QBLOCK] if qdir == QCOL else scales[:k]
        zero_prefix = zeros[: k // QBLOCK] if qdir == QCOL else zeros[:k]
        scalar_ref = fpint_gemm_ref(
            data.input_bits[:, :k],
            data.weights[:k],
            scale_prefix,
            zero_prefix,
            2,
            16,
            k,
            qdir=qdir,
        )
        scalar_gpu = fpint_gemm_gpu(
            data.input_bits[:, :k],
            data.weights[:k],
            scale_prefix,
            zero_prefix,
            2,
            16,
            k,
            qdir=qdir,
        )
        assert np.array_equal(references[k], scalar_ref)
        assert np.array_equal(gpu_models[k], scalar_gpu)


@pytest.mark.parametrize("qdir", [QCOL, QROW])
def test_vectorized_real_2scomp_matches_scalar(qdir):
    k_values = (32, 64, 128)
    data = generate_trial_data(seed=31, m=2, n=16, max_k=max(k_values))
    if qdir == QCOL:
        scales, zeros = data.qcol_scale_bits, data.qcol_zero
        vectorized = qcol_real_2scomp_prefix_outputs(
            data.input_bits, data.weights, scales, zeros, k_values
        )
        scalar_func = fpint_gemm_qcol_real_2scomp
    else:
        scales, zeros = data.qrow_scale_bits, data.qrow_zero
        vectorized = qrow_real_2scomp_prefix_outputs(
            data.input_bits, data.weights, scales, zeros, k_values
        )
        scalar_func = fpint_gemm_qrow_real_2scomp

    for k in k_values:
        scale_prefix = scales[: k // QBLOCK] if qdir == QCOL else scales[:k]
        zero_prefix = zeros[: k // QBLOCK] if qdir == QCOL else zeros[:k]
        scalar = scalar_func(
            data.input_bits[:, :k],
            data.weights[:k],
            scale_prefix,
            zero_prefix,
            2,
            16,
            k,
        )
        assert np.array_equal(vectorized[k], scalar)


def test_tensor_core_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = generate_trial_data(seed=47, m=16, n=16, max_k=32)
    outputs = tensor_core_outputs(
        data.input_bits,
        data.weights,
        data.qcol_scale_bits,
        data.qcol_zero,
        QCOL,
        (32,),
        "cuda:0",
    )
    repeated = tensor_core_outputs(
        data.input_bits,
        data.weights,
        data.qcol_scale_bits,
        data.qcol_zero,
        QCOL,
        (32,),
        "cuda:0",
    )
    assert outputs[32].shape == (16, 16)
    assert np.array_equal(outputs[32], repeated[32])
    assert np.all(np.isfinite(bits_to_fp16(outputs[32])))


def test_gpu_fp64_reference_smoke():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    data = generate_trial_data(seed=53, m=16, n=16, max_k=64)
    cpu = reference_prefix_outputs(
        data.input_bits,
        data.weights,
        data.qcol_scale_bits,
        data.qcol_zero,
        QCOL,
        (32, 64),
    )
    gpu = gpu_fp64_reference_outputs(
        data.input_bits,
        data.weights,
        data.qcol_scale_bits,
        data.qcol_zero,
        QCOL,
        (32, 64),
        "cuda:0",
    )
    for k in (32, 64):
        assert gpu[k].shape == (16, 16)
        assert np.array_equal(gpu[k], cpu[k])


def test_plot_csv_creates_fixed_column_width_figure(tmp_path):
    pd = pytest.importorskip("pandas")
    rows = []
    for seed in (1, 2):
        for qdir in ("qcol", "qrow"):
            for k in (32, 64):
                for reference, reference_scale in (
                    ("cpu_fp64", 1.0),
                    ("gpu_fp64", 1.1),
                ):
                    for method, mean_ulp in (
                        ("gpu_tensor_core", 1.0),
                        ("ours", 0.5),
                    ):
                        rows.append(
                            {
                                "seed": seed,
                                "qdir": qdir,
                                "K": k,
                                "method": method,
                                "reference": reference,
                                "mean_ulp": mean_ulp * reference_scale,
                            }
                        )
    csv_path = tmp_path / "raw_seed_metrics.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    paths = plot_csv(csv_path)

    assert len(paths) == 6
    for suffix in ("CPU", "GPU"):
        png_path = tmp_path / f"fp16_int4_real2scomp_fanin_vs_ref_{suffix}.png"
        svg_path = tmp_path / f"fp16_int4_real2scomp_fanin_vs_ref_{suffix}.svg"
        pdf_path = tmp_path / f"fp16_int4_real2scomp_fanin_vs_ref_{suffix}.pdf"
        assert png_path.exists()
        assert svg_path.exists()
        assert pdf_path.exists()
        svg = svg_path.read_text()
        assert svg.lstrip().startswith("<?xml")
        assert "Mean ULP Error" in svg
        with png_path.open("rb") as image:
            image.seek(16)
            width, height = struct.unpack(">II", image.read(8))
        assert (width, height) == (700, 224)
    assert (tmp_path / "summary.csv").exists()
