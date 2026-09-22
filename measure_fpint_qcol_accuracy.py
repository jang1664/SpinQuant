#!/usr/bin/env python3
"""Compare conventional and FPINT 16-bit outputs against a GPU FP64 reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from fpint_emul import FpIntConfig, fpint_linear, qcol_real_2scomp_reference


SAMPLER_VERSION = "k_scaled_finite_fp16_fields_v3"
BF16_SAMPLER_VERSION = "k_scaled_finite_bf16_fields_v1"
DEFAULT_K_VALUES = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
DEFAULT_EXPONENT_MAX_BY_K = {
    128: 24,
    256: 24,
    512: 24,
    1024: 23,
    2048: 23,
    4096: 22,
    8192: 21,
    16384: 21,
    32768: 20,
}
DEFAULT_BF16_EXPONENT_MAX_BY_K = {
    k: exponent - 15 + 127 for k, exponent in DEFAULT_EXPONENT_MAX_BY_K.items()
}
FP16_SIGN_MIN = 0
FP16_SIGN_MAX = 1
FP16_EXPONENT_MIN = 0
FP16_MANTISSA_MIN = 0
FP16_MANTISSA_MAX = 1023


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def parse_exponent_max_map(value: str) -> dict[int, int]:
    result: dict[int, int] = {}
    try:
        for item in value.split(","):
            k_text, exponent_text = item.split(":", maxsplit=1)
            k = int(k_text)
            exponent = int(exponent_text)
            if k <= 0 or not FP16_EXPONENT_MIN <= exponent <= 254:
                raise ValueError
            if k in result:
                raise ValueError
            result[k] = exponent
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected unique K:EXP pairs with K > 0 and 0 <= EXP <= 254"
        ) from error
    if not result:
        raise argparse.ArgumentTypeError("at least one K:EXP pair is required")
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_finite_fp16_fields(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    exponent_max: int,
) -> np.ndarray:
    """Uniformly sample independent finite IEEE FP16 fields."""

    if not FP16_EXPONENT_MIN <= exponent_max <= 30:
        raise ValueError("exponent_max must be in [0, 30]")
    sign = rng.integers(
        FP16_SIGN_MIN, FP16_SIGN_MAX + 1, size=shape, dtype=np.uint16
    )
    exponent = rng.integers(
        FP16_EXPONENT_MIN, exponent_max + 1, size=shape, dtype=np.uint16
    )
    mantissa = rng.integers(
        FP16_MANTISSA_MIN,
        FP16_MANTISSA_MAX + 1,
        size=shape,
        dtype=np.uint16,
    )
    bits = (sign << np.uint16(15)) | (exponent << np.uint16(10)) | mantissa
    values = np.ascontiguousarray(bits).view(np.float16)
    if not np.isfinite(values).all():
        raise AssertionError("finite FP16 field sampler produced NaN or Inf")
    return values


def sample_finite_bf16_fields(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    exponent_max: int,
) -> torch.Tensor:
    """Uniformly sample independent finite IEEE BF16 fields on CPU."""

    if not 0 <= exponent_max <= 254:
        raise ValueError("exponent_max must be in [0, 254]")
    sign = rng.integers(0, 2, size=shape, dtype=np.uint16)
    exponent = rng.integers(0, exponent_max + 1, size=shape, dtype=np.uint16)
    mantissa = rng.integers(0, 128, size=shape, dtype=np.uint16)
    bits = (sign << np.uint16(15)) | (exponent << np.uint16(7)) | mantissa
    values = torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16)
    if not bool(torch.isfinite(values).all()):
        raise AssertionError("finite BF16 field sampler produced NaN or Inf")
    return values


def activation_field_stats(activation) -> dict[str, int]:
    if isinstance(activation, torch.Tensor):
        bits = activation.contiguous().view(torch.uint16).numpy()
        numeric_zeros = int((activation == 0).sum().item())
    else:
        bits = np.ascontiguousarray(activation).view(np.uint16)
        numeric_zeros = int((activation == 0).sum())
    negative = int(((bits >> np.uint16(15)) != 0).sum())
    return {
        "activation_elements": int(bits.size),
        "positive_sign_bits": int(bits.size - negative),
        "negative_sign_bits": negative,
        "numeric_zeros": numeric_zeros,
    }


def sample_scale_fields(seed, shape, dtype, exponent_max, exponent_min=0):
    """Common random numbers: sign/mantissa stay fixed as exponent bounds vary."""
    mantissa_bits, largest_exponent = (10, 30) if dtype == "fp16" else (7, 254)
    if dtype not in ("fp16", "bf16"):
        raise ValueError("scale dtype must be fp16 or bf16")
    if not 0 <= exponent_min <= exponent_max <= largest_exponent:
        raise ValueError("scale exponent bounds must contain only finite fields")
    scale_seed = np.random.SeedSequence(seed).spawn(3)[2]
    rng = np.random.default_rng(scale_seed)
    sign = rng.integers(0, 2, size=shape, dtype=np.uint16)
    # Fixed draw count avoids randint rejection changing subsequent mantissas
    # when the exponent range changes. floor(U * count) samples integer bins.
    exponent = (
        exponent_min + np.floor(rng.random(shape) * (exponent_max - exponent_min + 1))
    ).astype(np.uint16)
    mantissa = rng.integers(0, 1 << mantissa_bits, size=shape, dtype=np.uint16)
    bits = (sign << 15) | (exponent << mantissa_bits) | mantissa
    result = torch.from_numpy(np.ascontiguousarray(bits)).view(
        torch.float16 if dtype == "fp16" else torch.bfloat16
    )
    return result


def make_case(
    *,
    m: int,
    k: int,
    n: int,
    config: FpIntConfig,
    seed: int,
    exponent_max: int,
    scale_mode: str = "identity",
    scale_format: str = "activation",
    scale_log2_min: float = -4.0,
    scale_log2_max: float = 0.0,
    scale_exp_min: int = 0,
    scale_exp_max: int | None = None,
):
    seed_sequence = np.random.SeedSequence(seed)
    activation_rng, weight_rng, scale_rng = [
        np.random.default_rng(child) for child in seed_sequence.spawn(3)
    ]
    sampler = (
        sample_finite_fp16_fields
        if config.activation_format == "fp16"
        else sample_finite_bf16_fields
    )
    activation = sampler(activation_rng, (m, k), exponent_max)
    qmin = -(1 << (config.weight_bits - 1))
    qmax = (1 << (config.weight_bits - 1)) - 1
    weight = weight_rng.integers(
        qmin, qmax + 1, size=(n, k), dtype=np.int16
    ).astype(np.int8)
    groups = config.group_count(k)
    scale_format = config.activation_format if scale_format == "activation" else scale_format
    if scale_mode == "raw-fields":
        if scale_exp_max is None:
            scale_exp_max = 15 if scale_format == "fp16" else 127
        scale = sample_scale_fields(seed, (n, groups), scale_format, scale_exp_max, scale_exp_min)
        return activation, weight, scale, np.zeros((n, groups), dtype=np.int32)
    if scale_mode == "identity":
        scale_values = np.ones((n, groups), dtype=np.float64)
    elif scale_mode == "log-uniform":
        scale_values = np.exp2(scale_rng.uniform(
            scale_log2_min, scale_log2_max, size=(n, groups)
        ))
    else:
        raise ValueError(f"unknown scale mode: {scale_mode}")
    scale_format = config.activation_format if scale_format == "activation" else scale_format
    if scale_format == "fp16":
        scale = scale_values.astype(np.float16)
    elif scale_format == "bf16":
        scale = torch.from_numpy(scale_values).to(torch.bfloat16)
    else:
        raise ValueError(f"unknown scale format: {scale_format}")
    zero = np.zeros((n, groups), dtype=np.int32)
    return activation, weight, scale, zero


def fp64_reference_linear(
    activation: torch.Tensor, weight: torch.Tensor,
    scale: torch.Tensor | None = None, group_size: int = 128,
) -> torch.Tensor:
    """GPU/CPU FP64 ground truth; independent of FPINT configuration."""

    weight_fp64 = weight.to(torch.float64)
    if scale is not None:
        group_size = weight.shape[1] if group_size == -1 else group_size
        groups = torch.arange(weight.shape[1], device=weight.device) // group_size
        weight_fp64 = weight_fp64 * scale[:, groups].to(torch.float64)
    return torch.nn.functional.linear(activation.to(torch.float64), weight_fp64)


def conventional_linear(
    activation: torch.Tensor, weight: torch.Tensor,
    scale: torch.Tensor | None = None, group_size: int = 128,
    reduced_precision: bool | None = None,
) -> torch.Tensor:
    """GPU Linear with FP32 dequantization then an activation-dtype weight cast.

    The reduction flag is scoped to this call and restored even on failure.
    """

    dequantized = weight.to(torch.float32)
    if scale is not None:
        group_size = weight.shape[1] if group_size == -1 else group_size
        groups = torch.arange(weight.shape[1], device=weight.device) // group_size
        dequantized = dequantized * scale[:, groups].float()
    dequantized = dequantized.to(activation.dtype)
    flag = (
        "allow_fp16_reduced_precision_reduction" if activation.dtype == torch.float16
        else "allow_bf16_reduced_precision_reduction"
    )
    previous = getattr(torch.backends.cuda.matmul, flag)
    try:
        if reduced_precision is not None:
            setattr(torch.backends.cuda.matmul, flag, reduced_precision)
        output = torch.nn.functional.linear(activation, dequantized)
        if activation.is_cuda:
            torch.cuda.synchronize(activation.device)
        return output
    finally:
        setattr(torch.backends.cuda.matmul, flag, previous)


def qcol_allclose_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    matching = torch.isclose(
        actual, expected, atol=atol, rtol=rtol, equal_nan=True
    )
    both_finite = torch.isfinite(actual) & torch.isfinite(expected)
    finite_error = (actual[both_finite] - expected[both_finite]).abs()
    return {
        "allclose": bool(matching.all()),
        "atol": atol,
        "rtol": rtol,
        "elements": matching.numel(),
        "outside_tolerance": int((~matching).sum().item()),
        "matching_fraction": float(matching.float().mean().item()),
        "max_abs": float(finite_error.max().item()) if finite_error.numel() else None,
        "reference_nonfinite": int((~torch.isfinite(expected)).sum().item()),
        "candidate_nonfinite": int((~torch.isfinite(actual)).sum().item()),
        "nonfinite_mismatch": int(((~matching) & (~both_finite)).sum().item()),
    }


def _ordered_float16(values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Map finite FP16/BF16 values to monotonic integer coordinates."""

    values = values.to(dtype=dtype, device="cpu").contiguous()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("ULP distance requires finite values")
    bits = values.view(torch.int16).to(torch.int32) & 0xFFFF
    magnitude = bits & 0x7FFF
    negative = (bits & 0x8000) != 0
    return torch.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def float16_ulp_distance(
    actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype
) -> torch.Tensor:
    if actual.shape != expected.shape:
        raise ValueError(
            f"shape mismatch: {tuple(actual.shape)} != {tuple(expected.shape)}"
        )
    return (
        _ordered_float16(actual, dtype) - _ordered_float16(expected, dtype)
    ).abs().to(torch.int32)


def fp16_ulp_distance(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
    """Backward-compatible FP16 ULP helper."""

    return float16_ulp_distance(actual, expected, torch.float16)


def _candidate_error_metrics(
    candidate: torch.Tensor,
    reference_fp64: torch.Tensor,
    rounded_reference: torch.Tensor,
    common_finite: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> dict[str, Any]:
    elements = int(common_finite.sum().item())
    if not elements:
        return {
            "elements": 0,
            "rmse": None,
            "relative_l2_error": None,
            "max_abs_error": None,
            "signed_error_mean": None,
            "signed_error_std": None,
            "mean_ulp": None,
            "std_ulp": None,
            "p50_ulp": None,
            "p95_ulp": None,
            "p99_ulp": None,
            "max_ulp": None,
            "error_sum": 0.0,
            "squared_error_sum": 0.0,
            "reference_squared_sum": 0.0,
            "ulp_sum": 0.0,
            "squared_ulp_sum": 0.0,
        }

    candidate_values = candidate[common_finite].double()
    reference_values = reference_fp64[common_finite]
    signed_error = candidate_values - reference_values
    error_sum = float(signed_error.sum().item())
    squared_error_sum = float(signed_error.square().sum().item())
    reference_squared_sum = float(reference_values.square().sum().item())
    error_mean = error_sum / elements
    error_variance = max(squared_error_sum / elements - error_mean**2, 0.0)

    ulp = float16_ulp_distance(
        candidate[common_finite], rounded_reference[common_finite], output_dtype
    ).double()
    ulp_sum = float(ulp.sum().item())
    squared_ulp_sum = float(ulp.square().sum().item())
    ulp_mean = ulp_sum / elements
    ulp_variance = max(squared_ulp_sum / elements - ulp_mean**2, 0.0)
    ulp_numpy = ulp.numpy()
    return {
        "elements": elements,
        "rmse": math.sqrt(squared_error_sum / elements),
        "relative_l2_error": math.sqrt(squared_error_sum)
        / max(math.sqrt(reference_squared_sum), 1e-300),
        "max_abs_error": float(signed_error.abs().max().item()),
        "signed_error_mean": error_mean,
        "signed_error_std": math.sqrt(error_variance),
        "mean_ulp": ulp_mean,
        "std_ulp": math.sqrt(ulp_variance),
        "p50_ulp": float(np.percentile(ulp_numpy, 50)),
        "p95_ulp": float(np.percentile(ulp_numpy, 95)),
        "p99_ulp": float(np.percentile(ulp_numpy, 99)),
        "max_ulp": int(ulp.max().item()),
        "error_sum": error_sum,
        "squared_error_sum": squared_error_sum,
        "reference_squared_sum": reference_squared_sum,
        "ulp_sum": ulp_sum,
        "squared_ulp_sum": squared_ulp_sum,
    }


def paired_error_metrics(
    conventional: torch.Tensor,
    fp_int: torch.Tensor,
    reference_fp64: torch.Tensor,
    activation_format: str = "fp16",
    conventional_full_precision: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Compute all candidate errors on one shared finite-output mask."""

    output_dtype = torch.float16 if activation_format == "fp16" else torch.bfloat16
    conventional = conventional.detach().to(dtype=output_dtype, device="cpu")
    fp_int = fp_int.detach().to(dtype=output_dtype, device="cpu")
    reference_fp64 = reference_fp64.detach().to(dtype=torch.float64, device="cpu")
    if conventional.shape != fp_int.shape or conventional.shape != reference_fp64.shape:
        raise ValueError("conventional, FPINT and FP64 outputs must have the same shape")
    rounded_reference = reference_fp64.to(output_dtype)
    reference_fp64_finite = torch.isfinite(reference_fp64)
    rounded_reference_finite = torch.isfinite(rounded_reference)
    conventional_finite = torch.isfinite(conventional)
    fp_int_finite = torch.isfinite(fp_int)
    common_finite = (
        reference_fp64_finite
        & rounded_reference_finite
        & conventional_finite
        & fp_int_finite
    )
    if conventional_full_precision is not None:
        conventional_full_precision = conventional_full_precision.detach().to(
            dtype=output_dtype, device="cpu"
        )
        if conventional_full_precision.shape != reference_fp64.shape:
            raise ValueError("baseline shapes must match the reference")
        common_finite &= torch.isfinite(conventional_full_precision)
    total = reference_fp64.numel()
    common = int(common_finite.sum().item())
    coverage = {
        "output_elements": total,
        "fp64_reference_nonfinite": int((~reference_fp64_finite).sum().item()),
        "rounded_reference_nonfinite": int((~rounded_reference_finite).sum().item()),
        f"{activation_format}_rounded_reference_nonfinite": int(
            (~rounded_reference_finite).sum().item()
        ),
        "conventional_nonfinite": int((~conventional_finite).sum().item()),
        "fp_int_nonfinite": int((~fp_int_finite).sum().item()),
        "common_finite_elements": common,
        "common_finite_fraction": common / total if total else None,
    }
    result = {
        "coverage": coverage,
        "conventional_err": _candidate_error_metrics(
            conventional,
            reference_fp64,
            rounded_reference,
            common_finite,
            output_dtype,
        ),
        "fp_int_err": _candidate_error_metrics(
            fp_int, reference_fp64, rounded_reference, common_finite, output_dtype
        ),
    }
    if conventional_full_precision is not None:
        coverage["conventional_full_precision_nonfinite"] = int(
            (~torch.isfinite(conventional_full_precision)).sum().item()
        )
        result["conventional_full_precision_err"] = _candidate_error_metrics(
            conventional_full_precision, reference_fp64, rounded_reference,
            common_finite, output_dtype,
        )
    return result


def _sample_std(values: list[float]) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _aggregate_error(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["elements"]]
    elements = sum(row["elements"] for row in valid)
    error_sum = sum(row["error_sum"] for row in valid)
    squared_error_sum = sum(row["squared_error_sum"] for row in valid)
    reference_squared_sum = sum(row["reference_squared_sum"] for row in valid)
    ulp_sum = sum(row["ulp_sum"] for row in valid)
    squared_ulp_sum = sum(row["squared_ulp_sum"] for row in valid)
    if elements:
        error_mean = error_sum / elements
        ulp_mean = ulp_sum / elements
        global_metrics = {
            "elements": elements,
            "global_rmse": math.sqrt(squared_error_sum / elements),
            "global_relative_l2_error": math.sqrt(squared_error_sum)
            / max(math.sqrt(reference_squared_sum), 1e-300),
            "global_max_abs_error": max(row["max_abs_error"] for row in valid),
            "global_signed_error_mean": error_mean,
            "global_signed_error_std": math.sqrt(
                max(squared_error_sum / elements - error_mean**2, 0.0)
            ),
            "global_mean_ulp": ulp_mean,
            "global_std_ulp": math.sqrt(
                max(squared_ulp_sum / elements - ulp_mean**2, 0.0)
            ),
            "global_max_ulp": max(row["max_ulp"] for row in valid),
        }
    else:
        global_metrics = {
            "elements": 0,
            "global_rmse": None,
            "global_relative_l2_error": None,
            "global_max_abs_error": None,
            "global_signed_error_mean": None,
            "global_signed_error_std": None,
            "global_mean_ulp": None,
            "global_std_ulp": None,
            "global_max_ulp": None,
        }

    rmse = [row["rmse"] for row in valid]
    mean_ulp = [row["mean_ulp"] for row in valid]
    p50_ulp = [row["p50_ulp"] for row in valid]
    p95_ulp = [row["p95_ulp"] for row in valid]
    return {
        **global_metrics,
        "trials_with_metrics": len(valid),
        "trial_rmse_mean": _mean(rmse),
        "trial_rmse_sample_std": _sample_std(rmse),
        "trial_mean_ulp_mean": _mean(mean_ulp),
        "trial_mean_ulp_sample_std": _sample_std(mean_ulp),
        "trial_p50_ulp_mean": _mean(p50_ulp),
        "trial_p95_ulp_mean": _mean(p95_ulp),
    }


def _delta_ratio(fp_int: float | None, conventional: float | None) -> dict[str, Any]:
    if fp_int is None or conventional is None:
        return {"delta_fp_int_minus_conventional": None, "ratio_fp_int_over_conventional": None}
    return {
        "delta_fp_int_minus_conventional": fp_int - conventional,
        "ratio_fp_int_over_conventional": (
            fp_int / conventional if conventional != 0 else None
        ),
    }


def summarize_records(
    records: list[dict[str, Any]], finite_target: float
) -> dict[str, Any]:
    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        coverage_rows = [row["coverage"] for row in rows]
        total = sum(row["output_elements"] for row in coverage_rows)
        common = sum(row["common_finite_elements"] for row in coverage_rows)
        coverage = {
            "output_elements": total,
            "fp64_reference_nonfinite": sum(
                row["fp64_reference_nonfinite"] for row in coverage_rows
            ),
            "rounded_reference_nonfinite": sum(
                row.get(
                    "rounded_reference_nonfinite",
                    row.get(
                        "fp16_rounded_reference_nonfinite",
                        row.get("bf16_rounded_reference_nonfinite", 0),
                    ),
                )
                for row in coverage_rows
            ),
            "conventional_nonfinite": sum(
                row["conventional_nonfinite"] for row in coverage_rows
            ),
            "fp_int_nonfinite": sum(row["fp_int_nonfinite"] for row in coverage_rows),
            "common_finite_elements": common,
            "common_finite_fraction": common / total if total else None,
        }
        conventional = _aggregate_error(
            [row["comparisons"]["conventional_err"] for row in rows]
        )
        fp_int = _aggregate_error(
            [row["comparisons"]["fp_int_err"] for row in rows]
        )
        extra = {}
        if "conventional_full_precision_err" in rows[0]["comparisons"]:
            full_precision = _aggregate_error([
                row["comparisons"]["conventional_full_precision_err"] for row in rows
            ])
            coverage["conventional_full_precision_nonfinite"] = sum(
                row["conventional_full_precision_nonfinite"] for row in coverage_rows
            )
            extra = {
                "conventional_full_precision_err": full_precision,
                "full_precision_comparison": {
                    metric: _delta_ratio(fp_int[metric], full_precision[metric])
                    for metric in ("trial_rmse_mean", "trial_mean_ulp_mean")
                },
            }
        return {
            **extra,
            "cases": len(rows),
            "coverage": coverage,
            "finite_target_met": bool(
                coverage["common_finite_fraction"] is not None
                and coverage["common_finite_fraction"] >= finite_target
            ),
            "activation_sign_bits": {
                "positive": sum(row["input_stats"]["positive_sign_bits"] for row in rows),
                "negative": sum(row["input_stats"]["negative_sign_bits"] for row in rows),
            },
            "qcol_correctness": {
                "torch_allclose_cases": sum(
                    row["comparisons"]["fpint_torch_vs_qcol_reference"]["allclose"]
                    for row in rows
                ),
                "cuda_cases": sum(
                    "fpint_cuda_vs_qcol_reference" in row["comparisons"]
                    for row in rows
                ),
                "cuda_allclose_cases": sum(
                    row["comparisons"].get(
                        "fpint_cuda_vs_qcol_reference", {"allclose": False}
                    )["allclose"]
                    for row in rows
                ),
            },
            "conventional_err": conventional,
            "fp_int_err": fp_int,
            "comparison": {
                "trial_rmse_mean": _delta_ratio(
                    fp_int["trial_rmse_mean"], conventional["trial_rmse_mean"]
                ),
                "trial_mean_ulp_mean": _delta_ratio(
                    fp_int["trial_mean_ulp_mean"],
                    conventional["trial_mean_ulp_mean"],
                ),
            },
        }

    k_values = sorted({int(record["k"]) for record in records})
    return {
        "overall": aggregate(records),
        "by_k": {
            str(k): aggregate([record for record in records if record["k"] == k])
            for k in k_values
        },
    }


def flatten_rows(records: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for record in records:
        common = {
            key: record[key]
            for key in ("trial", "seed", "m", "k", "n", "exponent_max")
        }
        for comparison, metrics in record["comparisons"].items():
            yield {**common, "comparison": comparison, **metrics}
        yield {
            **common,
            "comparison": "coverage",
            **record["coverage"],
            **record["input_stats"],
        }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    rows = list(flatten_rows(records))
    fieldnames = list(rows[0])
    extra_names = sorted({key for row in rows for key in row} - set(fieldnames))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra_names)
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    exponent_max_by_k = dict(args.exponent_max_by_k)
    missing = set(args.k_values) - set(exponent_max_by_k)
    extra = set(exponent_max_by_k) - set(args.k_values)
    if missing or extra:
        raise ValueError(
            f"exponent map must exactly match K values; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    activation_format = getattr(args, "activation_format", "fp16")
    scale_mode = getattr(args, "scale_mode", "identity")
    scale_format = getattr(args, "scale_format", "activation")
    if scale_format == "activation":
        scale_format = activation_format
    scale_log2_min = getattr(args, "scale_log2_min", -4.0)
    scale_log2_max = getattr(args, "scale_log2_max", 0.0)
    scale_exp_min = getattr(args, "scale_exp_min", 0)
    scale_exp_max = getattr(args, "scale_exp_max", None)
    if scale_exp_max is None:
        scale_exp_max = 15 if scale_format == "fp16" else 127
    config = FpIntConfig(
        weight_bits=args.bits,
        group_size=args.group_size,
        mxu_rows=args.mxu_rows,
        extra_bits=args.extra_bits,
        reduce_extra_bits=args.reduce_extra_bits,
        activation_format=activation_format,
    )
    records: list[dict[str, Any]] = []
    all_reference_close = True
    all_signs_present = True
    case_index = 0
    total_cases = args.trials * len(args.k_values)
    for trial in range(args.trials):
        for k in args.k_values:
            seed = args.base_seed + case_index
            case_index += 1
            exponent_max = exponent_max_by_k[k]
            activation, weight, scale, zero = make_case(
                m=args.m,
                k=k,
                n=args.n,
                config=config,
                seed=seed,
                exponent_max=exponent_max,
                scale_mode=scale_mode, scale_format=scale_format,
                scale_log2_min=scale_log2_min, scale_log2_max=scale_log2_max,
                scale_exp_min=scale_exp_min, scale_exp_max=scale_exp_max,
            )
            input_stats = activation_field_stats(activation)
            signs_present = (
                input_stats["positive_sign_bits"] > 0
                and input_stats["negative_sign_bits"] > 0
            )
            all_signs_present &= signs_present

            qcol_reference = torch.from_numpy(
                qcol_real_2scomp_reference(activation, weight, scale, zero, config)
            )
            activation_tensor = (
                activation.to(device)
                if isinstance(activation, torch.Tensor)
                else torch.from_numpy(activation).to(device)
            )
            tensors = (
                activation_tensor,
                torch.from_numpy(weight).to(device),
                (scale if isinstance(scale, torch.Tensor) else torch.from_numpy(scale)).to(device),
                torch.from_numpy(zero).to(device),
            )
            actual_torch = fpint_linear(
                *tensors, config, backend="fpint_torch", has_zero=False
            )
            comparisons: dict[str, Any] = {
                "fpint_torch_vs_qcol_reference": qcol_allclose_metrics(
                    actual_torch, qcol_reference, args.atol, args.rtol
                )
            }
            if device.type == "cuda":
                actual_fp_int = fpint_linear(
                    *tensors,
                    config,
                    backend="fpint_cuda",
                    has_zero=False,
                )
                comparisons["fpint_cuda_vs_qcol_reference"] = qcol_allclose_metrics(
                    actual_fp_int, qcol_reference, args.atol, args.rtol
                )
            else:
                actual_fp_int = actual_torch

            reference_fp64 = fp64_reference_linear(*tensors[:3], config.group_size)
            conventional = conventional_linear(
                *tensors[:3], config.group_size, reduced_precision=True
            )
            conventional_full_precision = conventional_linear(
                *tensors[:3], config.group_size, reduced_precision=False
            )
            numerical = paired_error_metrics(
                conventional,
                actual_fp_int,
                reference_fp64,
                activation_format=activation_format,
                conventional_full_precision=conventional_full_precision,
            )
            comparisons["conventional_err"] = numerical["conventional_err"]
            comparisons["fp_int_err"] = numerical["fp_int_err"]
            comparisons["conventional_full_precision_err"] = numerical["conventional_full_precision_err"]
            all_reference_close &= all(
                metrics["allclose"]
                for name, metrics in comparisons.items()
                if name.endswith("vs_qcol_reference")
            )
            records.append(
                {
                    "trial": trial,
                    "seed": seed,
                    "m": args.m,
                    "k": k,
                    "n": args.n,
                    "exponent_max": exponent_max,
                    "input_stats": input_stats,
                    "input_sha256": hashlib.sha256(
                        tensors[0].cpu().contiguous().view(torch.int16).numpy().tobytes()
                        + tensors[1].cpu().numpy().tobytes()
                    ).hexdigest(),
                    "scale_stats": {
                        "min": float(tensors[2].min()),
                        "max": float(tensors[2].max()),
                        "sha256": hashlib.sha256(
                            tensors[2].cpu().contiguous().view(torch.int16).numpy().tobytes()
                        ).hexdigest(),
                    },
                    "coverage": numerical["coverage"],
                    "comparisons": comparisons,
                }
            )
            print(
                f"[{case_index}/{total_cases}] K={k} EXP<={exponent_max}: "
                "qcol_allclose="
                f"{all(metrics['allclose'] for name, metrics in comparisons.items() if name.endswith('vs_qcol_reference'))}, "
                f"common_finite={numerical['coverage']['common_finite_elements']}/"
                f"{numerical['coverage']['output_elements']}",
                flush=True,
            )

    summary = summarize_records(records, args.finite_target)
    finite_target_met = all(
        row["finite_target_met"] for row in summary["by_k"].values()
    )
    passed = all_reference_close and all_signs_present and finite_target_met
    kernel_path = Path(__file__).parent / "fpint_emul/csrc/fpint_cuda_kernel.cu"
    return {
        "status": "pass" if passed else "fail",
        "evaluation_policy": {
            "fp64_gpu_ground_truth": device.type == "cuda",
            "qcol_reference_must_be_allclose": True,
            "common_finite_mask_for_paired_errors": True,
            "minimum_common_finite_fraction_per_k": args.finite_target,
            "candidate_error_ranking_is_diagnostic_only": True,
        },
        "environment": {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "allow_fp16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
            ),
            "allow_bf16_reduced_precision_reduction": (
                torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
            ),
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
            "fpint_cuda_kernel_sha256": sha256_file(kernel_path),
            "measurement_script_sha256": sha256_file(Path(__file__)),
            "source_sha256": {
                name: sha256_file(Path(__file__).parent / name)
                for name in ("fpint_emul/reference.py", "fpint_emul/torch_backend.py", "fpint_emul/config.py")
            },
        },
        "config": {
            **vars(config),
            "sampler_version": (
                SAMPLER_VERSION
                if activation_format == "fp16"
                else BF16_SAMPLER_VERSION
            ),
            "operation": (
                f"raw_{activation_format}_times_signed_int" if scale_mode == "identity"
                else f"{activation_format}_times_scaled_signed_int"
            ),
            "identity_scale": 1.0 if scale_mode == "identity" else None,
            "scale_sampling": {
                "mode": scale_mode, "format": scale_format,
                "log2_min": scale_log2_min if scale_mode == "log-uniform" else None,
                "log2_max": scale_log2_max if scale_mode == "log-uniform" else None,
                "exponent_min": scale_exp_min if scale_mode == "raw-fields" else None,
                "exponent_max": scale_exp_max if scale_mode == "raw-fields" else None,
                "sign": [0, 1] if scale_mode == "raw-fields" else None,
                "mantissa": [0, 1023 if scale_format == "fp16" else 127] if scale_mode == "raw-fields" else None,
                "granularity": "output_channel_x_k_group",
                "rounding": "assemble raw fields" if scale_mode == "raw-fields" else "sample in float64 then round to scale dtype",
                "rng": "third child of numpy SeedSequence(case_seed)",
            },
            "baseline_reduced_precision_modes": [True, False],
            "baseline_metric_keys": {
                "true": "conventional_err", "false": "conventional_full_precision_err",
            },
            "zero_point": 0,
            "m": args.m,
            "n": args.n,
            "atol": args.atol,
            "rtol": args.rtol,
            "trials": args.trials,
            "base_seed": args.base_seed,
            "k_values": list(args.k_values),
            "finite_target": args.finite_target,
            f"{activation_format}_fields": {
                "sign": [FP16_SIGN_MIN, FP16_SIGN_MAX],
                "exponent_min": FP16_EXPONENT_MIN,
                "exponent_max_by_k": {
                    str(k): exponent_max_by_k[k] for k in args.k_values
                },
                "mantissa": [0, (1 << config.mantissa_bits) - 1],
                "sampling": "independent_uniform_raw_fields",
            },
            "integer_range": [
                -(1 << (config.weight_bits - 1)),
                (1 << (config.weight_bits - 1)) - 1,
            ],
            "paths": {
                "fp64_reference": "FP64 activation x (FP64 integer weight * FP64 scale)",
                "conventional": (
                    f"{activation_format.upper()} activation x "
                    f"{activation_format.upper()}-cast (FP32 integer weight * FP32 scale)"
                ),
                "fp_int": "QCOL_REAL_2SCOMP FPINT CUDA emulation",
            },
        },
        "summary": summary,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mxu-rows", type=int, default=128)
    parser.add_argument("--extra-bits", type=int, default=19)
    parser.add_argument("--reduce-extra-bits", type=int, default=10)
    parser.add_argument(
        "--activation-format", choices=("fp16", "bf16"), default="fp16"
    )
    parser.add_argument("--scale-mode", choices=("identity", "log-uniform", "raw-fields"), default="raw-fields")
    parser.add_argument("--scale-exp-min", type=int, default=0)
    parser.add_argument("--scale-exp-max", type=int)
    parser.add_argument("--scale-format", choices=("activation", "fp16", "bf16"), default="activation")
    parser.add_argument("--scale-log2-min", type=float, default=-4.0)
    parser.add_argument("--scale-log2-max", type=float, default=0.0)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument(
        "--exp-max-by-k",
        dest="exponent_max_by_k",
        type=parse_exponent_max_map,
        default=None,
    )
    parser.add_argument("--finite-target", type=float, default=0.999)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260915)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scale_format = args.activation_format if args.scale_format == "activation" else args.scale_format
    if args.scale_exp_max is None:
        args.scale_exp_max = 15 if scale_format == "fp16" else 127
    largest_exponent = 30 if scale_format == "fp16" else 254
    if not 0 <= args.scale_exp_min <= args.scale_exp_max <= largest_exponent:
        parser.error("scale exponent range must contain finite fields")
    scale_lower, scale_upper = (-24, 15) if scale_format == "fp16" else (-133, 127)
    if not scale_lower <= args.scale_log2_min <= args.scale_log2_max <= scale_upper:
        parser.error(f"scale log2 bounds must satisfy {scale_lower} <= min <= max <= {scale_upper}")
    if args.exponent_max_by_k is None:
        args.exponent_max_by_k = (
            DEFAULT_EXPONENT_MAX_BY_K
            if args.activation_format == "fp16"
            else DEFAULT_BF16_EXPONENT_MAX_BY_K
        )
    if args.m <= 0 or args.n <= 0 or args.trials <= 0:
        parser.error("--m, --n and --trials must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    if not 0.0 <= args.finite_target <= 1.0:
        parser.error("--finite-target must be in [0, 1]")
    missing = set(args.k_values) - set(args.exponent_max_by_k)
    extra = set(args.exponent_max_by_k) - set(args.k_values)
    if missing or extra:
        parser.error(
            "--exp-max-by-k must contain exactly one entry for every --k-values entry"
        )
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), result["records"])
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
