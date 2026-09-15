#!/usr/bin/env python3
"""FP16-INT4 real two's-complement numerical-accuracy fan-in sweep."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Sequence

import numpy as np

from fpint_emul import (
    EXTRA_BIT,
    EXTRA_BIT_FOR_REDUCE,
    EXTRA_BIT_FOR_REDUCE_QROW,
    IN_EXP_BIAS,
    IN_MAN_WIDTH,
    MXU_K,
    MXU_N,
    QBLOCK,
    QCOL,
    QROW,
)
from visualize import ulp_diff_fp16_bits


DEFAULT_K_VALUES = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
METHOD_GPU = "gpu_tensor_core"
METHOD_GPU_MODEL = "gpu_model"
METHOD_OURS = "ours"
METHODS = (METHOD_GPU, METHOD_GPU_MODEL, METHOD_OURS)
REFERENCE_CPU = "cpu_fp64"
REFERENCE_GPU = "gpu_fp64"
REFERENCES = (REFERENCE_CPU, REFERENCE_GPU)
QDIR_NAMES = {QCOL: "qcol", QROW: "qrow"}
PLOT_LOG_FLOOR = 1e-5
COL_WIDTH = 3.5
FIG_HEIGHT = 1.12
DEFAULT_FONT_SIZE = 4.0
TITLE_FONT_SIZE = 4.2
LINE_WIDTH = 0.25
Y_TICK_WIDTH = 0.15
INPUT_EXP_MIN = -24
INPUT_EXP_MAX = 15
INPUT_MANTISSA_MEAN = 512.0
INPUT_MANTISSA_STD = 256.0
SCALE_MIN = 2.0**-14
SCALE_MAX = 2.0**-11


def bits_to_fp16(bits: np.ndarray) -> np.ndarray:
    """View uint16 IEEE-754 bit patterns as FP16 values."""
    return np.ascontiguousarray(bits, dtype=np.uint16).view(np.float16)


def fp16_to_bits(values: np.ndarray) -> np.ndarray:
    """Round values to FP16 and return their uint16 bit patterns."""
    with np.errstate(over="ignore", invalid="ignore"):
        half = np.ascontiguousarray(values, dtype=np.float16)
    return half.view(np.uint16).copy()


@dataclass(frozen=True)
class TrialData:
    input_bits: np.ndarray
    weights: np.ndarray
    qcol_scale_bits: np.ndarray
    qcol_zero: np.ndarray
    qrow_scale_bits: np.ndarray
    qrow_zero: np.ndarray


def _sample_componentwise_fp16(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> np.ndarray:
    """Sample FP16 values from independent sign, exponent, and mantissa fields."""
    sign = rng.integers(0, 2, size=shape, dtype=np.int8)
    exponent = rng.integers(
        INPUT_EXP_MIN, INPUT_EXP_MAX + 1, size=shape, dtype=np.int16
    )
    mantissa = np.clip(
        np.rint(rng.normal(INPUT_MANTISSA_MEAN, INPUT_MANTISSA_STD, size=shape)),
        0,
        1023,
    ).astype(np.int16)

    magnitude = np.ldexp(1.0 + mantissa.astype(np.float64) / 1024.0, exponent)
    values = np.where(sign == 0, magnitude, -magnitude)
    return fp16_to_bits(values)


def generate_trial_data(
    seed: int,
    m: int,
    n: int,
    max_k: int,
    qblock: int = QBLOCK,
) -> TrialData:
    """Generate deterministic signed-asymmetric FP16-INT4 operands."""
    seed_sequence = np.random.SeedSequence(seed)
    rng_input, rng_weight, rng_qcol, rng_qrow = [
        np.random.default_rng(child) for child in seed_sequence.spawn(4)
    ]

    input_bits = _sample_componentwise_fp16(rng_input, (m, max_k))
    weights = rng_weight.integers(-8, 8, size=(max_k, n), dtype=np.int8)

    qcol_shape = (max_k // qblock, n)
    qcol_scale_bits = fp16_to_bits(
        rng_qcol.uniform(SCALE_MIN, SCALE_MAX, size=qcol_shape)
    )
    qcol_zero = rng_qcol.integers(-4, 4, size=qcol_shape, dtype=np.int16)

    qrow_shape = (max_k, (n + qblock - 1) // qblock)
    qrow_scale_bits = fp16_to_bits(
        rng_qrow.uniform(SCALE_MIN, SCALE_MAX, size=qrow_shape)
    )
    qrow_zero = rng_qrow.integers(-4, 4, size=qrow_shape, dtype=np.int16)

    return TrialData(
        input_bits=input_bits,
        weights=weights,
        qcol_scale_bits=qcol_scale_bits,
        qcol_zero=qcol_zero,
        qrow_scale_bits=qrow_scale_bits,
        qrow_zero=qrow_zero,
    )


def _expand_qparams(
    scale_bits: np.ndarray,
    zero: np.ndarray,
    qdir: int,
    max_k: int,
    n: int,
    qblock: int,
) -> tuple[np.ndarray, np.ndarray]:
    scales = bits_to_fp16(scale_bits)
    if qdir == QCOL:
        scale_expanded = np.repeat(scales, qblock, axis=0)[:max_k, :n]
        zero_expanded = np.repeat(zero, qblock, axis=0)[:max_k, :n]
    elif qdir == QROW:
        scale_expanded = np.repeat(scales, qblock, axis=1)[:max_k, :n]
        zero_expanded = np.repeat(zero, qblock, axis=1)[:max_k, :n]
    else:
        raise ValueError(f"Unsupported quantization direction: {qdir}")
    return scale_expanded, zero_expanded


def reference_prefix_outputs(
    input_bits: np.ndarray,
    weights: np.ndarray,
    scale_bits: np.ndarray,
    zero: np.ndarray,
    qdir: int,
    k_values: Sequence[int],
    qblock: int = QBLOCK,
) -> Dict[int, np.ndarray]:
    """FP64 accurate operation, rounded to FP16 at each fan-in checkpoint."""
    max_k = max(k_values)
    n = weights.shape[1]
    scales, zeros = _expand_qparams(scale_bits, zero, qdir, max_k, n, qblock)
    dequant = (
        (weights[:max_k].astype(np.float64) - zeros.astype(np.float64))
        * scales.astype(np.float64)
    )
    contributions = input_bits_to_float64(input_bits[:, :max_k])[:, :, None] * dequant[None, :, :]
    accumulated = np.cumsum(contributions, axis=1, dtype=np.float64)
    return {k: fp16_to_bits(accumulated[:, k - 1, :]) for k in k_values}


def input_bits_to_float64(bits: np.ndarray) -> np.ndarray:
    return bits_to_fp16(bits).astype(np.float64)


def gpu_fp64_reference_outputs(
    input_bits: np.ndarray,
    weights: np.ndarray,
    scale_bits: np.ndarray,
    zero: np.ndarray,
    qdir: int,
    k_values: Sequence[int],
    device: str,
    qblock: int = QBLOCK,
) -> Dict[int, np.ndarray]:
    """Run the FP64 reference matmul on CUDA and round outputs to FP16."""
    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the vortex conda environment")

    max_k = max(k_values)
    n = weights.shape[1]
    scales, zeros = _expand_qparams(scale_bits, zero, qdir, max_k, n, qblock)
    input_gpu = torch.from_numpy(bits_to_fp16(input_bits[:, :max_k]).copy()).to(
        device=device,
        dtype=torch.float64,
    )
    weights_gpu = torch.from_numpy(weights[:max_k].copy()).to(
        device=device,
        dtype=torch.float64,
    )
    scales_gpu = torch.from_numpy(scales.copy()).to(
        device=device,
        dtype=torch.float64,
    )
    zeros_gpu = torch.from_numpy(zeros.copy()).to(
        device=device,
        dtype=torch.float64,
    )
    dequant_gpu = (weights_gpu - zeros_gpu) * scales_gpu

    outputs: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for k in k_values:
            output = torch.matmul(input_gpu[:, :k], dequant_gpu[:k, :])
            outputs[k] = fp16_to_bits(output.cpu().numpy())
    torch.cuda.synchronize()
    return outputs


def gpu_model_prefix_outputs(
    input_bits: np.ndarray,
    weights: np.ndarray,
    scale_bits: np.ndarray,
    zero: np.ndarray,
    qdir: int,
    k_values: Sequence[int],
    qblock: int = QBLOCK,
) -> Dict[int, np.ndarray]:
    """Legacy GPU model: FP16 products, sequential FP32 accumulation."""
    max_k = max(k_values)
    n = weights.shape[1]
    scales, zeros = _expand_qparams(scale_bits, zero, qdir, max_k, n, qblock)
    with np.errstate(over="ignore", invalid="ignore"):
        dequant = np.asarray(
            (weights[:max_k].astype(np.int16) - zeros.astype(np.int16)).astype(np.float16)
            * scales,
            dtype=np.float16,
        )
        products = np.asarray(
            bits_to_fp16(input_bits[:, :max_k])[:, :, None] * dequant[None, :, :],
            dtype=np.float16,
        )
    accumulated = np.cumsum(products.astype(np.float32), axis=1, dtype=np.float32)
    return {k: fp16_to_bits(accumulated[:, k - 1, :]) for k in k_values}


def _prealign_grouped(bits: np.ndarray, extra_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized equivalent of prealign() for arrays ending in an MXU_K axis."""
    bits = np.asarray(bits, dtype=np.uint16)
    if bits.shape[-1] != MXU_K:
        raise ValueError(f"Last prealign axis must be MXU_K={MXU_K}")

    sign = (bits & np.uint16(0x8000)) != 0
    exponent = ((bits >> np.uint16(10)) & np.uint16(0x1F)).astype(np.int16)
    mantissa = (bits & np.uint16(0x03FF)).astype(np.int64)
    effective_exponent = np.maximum(exponent, 1)
    max_exponent = np.max(effective_exponent, axis=-1).astype(np.uint8)
    hidden = np.where(exponent == 0, 0, 1).astype(np.int64)
    significand = (hidden << IN_MAN_WIDTH) | mantissa
    shift = max_exponent.astype(np.int16)[..., None] - effective_exponent
    aligned = np.right_shift(significand << extra_bits, shift)
    aligned = np.where(sign, -aligned, aligned).astype(np.int64)
    return aligned, max_exponent


def qcol_real_2scomp_prefix_outputs(
    input_bits: np.ndarray,
    weights: np.ndarray,
    scale_bits: np.ndarray,
    zero: np.ndarray,
    k_values: Sequence[int],
    qblock: int = QBLOCK,
) -> Dict[int, np.ndarray]:
    """Vectorized QCOL real two's-complement emulation for all K prefixes."""
    if qblock != MXU_K:
        raise ValueError(f"QCOL prefix evaluator requires qblock=MXU_K={MXU_K}")
    max_k = max(k_values)
    m = input_bits.shape[0]
    n = weights.shape[1]
    groups = max_k // qblock

    grouped_input = input_bits[:, :max_k].reshape(m, groups, qblock)
    aligned, max_exponent = _prealign_grouped(grouped_input, EXTRA_BIT)
    aligned_reduce, _ = _prealign_grouped(grouped_input, EXTRA_BIT_FOR_REDUCE)
    grouped_weights = weights[:max_k].reshape(groups, qblock, n).astype(np.int64)

    inner = np.einsum("mgq,gqn->mgn", aligned, grouped_weights, optimize=True)
    reduce_sum = np.sum(aligned_reduce, axis=-1, dtype=np.int64)
    correction = zero[:groups].astype(np.int64)[None, :, :] * reduce_sum[:, :, None]
    post = inner - (correction << (EXTRA_BIT - EXTRA_BIT_FOR_REDUCE))
    # Match the QCOL FPINT pipeline: restore the post-inner-product in FP32,
    # multiply it by the FP16 scale promoted to FP32, and accumulate in FP32.
    factor = np.exp2(
        max_exponent.astype(np.float32) - IN_EXP_BIAS - (IN_MAN_WIDTH + EXTRA_BIT)
    ).astype(np.float32)
    scales = bits_to_fp16(scale_bits[:groups]).astype(np.float32)
    post_fp32 = post.astype(np.float32) * factor[:, :, None]
    tile_values = post_fp32 * scales[None, :, :]
    accumulated = np.cumsum(tile_values, axis=1, dtype=np.float32)
    return {k: fp16_to_bits(accumulated[:, k // qblock - 1, :]) for k in k_values}


def qrow_real_2scomp_prefix_outputs(
    input_bits: np.ndarray,
    weights: np.ndarray,
    scale_bits: np.ndarray,
    zero: np.ndarray,
    k_values: Sequence[int],
    qblock: int = QBLOCK,
) -> Dict[int, np.ndarray]:
    """Vectorized QROW real two's-complement emulation for all K prefixes."""
    if qblock != MXU_K:
        raise ValueError(f"QROW prefix evaluator requires qblock=MXU_K={MXU_K}")
    max_k = max(k_values)
    m = input_bits.shape[0]
    n = weights.shape[1]
    groups = max_k // qblock
    scales, zeros = _expand_qparams(scale_bits, zero, QROW, max_k, n, qblock)

    with np.errstate(over="ignore", invalid="ignore"):
        scaled_inputs = np.asarray(
            bits_to_fp16(input_bits[:, :max_k])[:, :, None] * scales[None, :, :],
            dtype=np.float16,
        )
    grouped_bits = fp16_to_bits(scaled_inputs).reshape(m, groups, qblock, n)
    grouped_bits = np.transpose(grouped_bits, (0, 3, 1, 2))
    aligned, max_exponent = _prealign_grouped(grouped_bits, EXTRA_BIT)
    aligned_reduce, _ = _prealign_grouped(grouped_bits, EXTRA_BIT_FOR_REDUCE_QROW)

    grouped_weights = weights[:max_k].reshape(groups, qblock, n).transpose(0, 2, 1)
    grouped_zeros = zeros.reshape(groups, qblock, n).transpose(0, 2, 1)
    inner = np.einsum(
        "mngq,gnq->mng", aligned, grouped_weights.astype(np.int64), optimize=True
    )
    reduce_zero = np.einsum(
        "mngq,gnq->mng", aligned_reduce, grouped_zeros.astype(np.int64), optimize=True
    )
    post = inner - (reduce_zero << (EXTRA_BIT - EXTRA_BIT_FOR_REDUCE_QROW))
    factor = np.exp2(
        max_exponent.astype(np.float64) - IN_EXP_BIAS - (IN_MAN_WIDTH + EXTRA_BIT)
    )
    tile_values = post.astype(np.float64) * factor
    accumulated = np.cumsum(tile_values, axis=2, dtype=np.float64)
    return {k: fp16_to_bits(accumulated[:, :, k // qblock - 1]) for k in k_values}


def _load_torch():
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    return torch


def tensor_core_outputs(
    input_bits: np.ndarray,
    weights: np.ndarray,
    scale_bits: np.ndarray,
    zero: np.ndarray,
    qdir: int,
    k_values: Sequence[int],
    device: str,
    qblock: int = QBLOCK,
) -> Dict[int, np.ndarray]:
    """Run the dequantized FP16 GEMM on the actual NVIDIA Tensor Core path."""
    torch = _load_torch()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the vortex conda environment")
    torch.use_deterministic_algorithms(True)

    max_k = max(k_values)
    n = weights.shape[1]
    scales, zeros = _expand_qparams(scale_bits, zero, qdir, max_k, n, qblock)
    with np.errstate(over="ignore", invalid="ignore"):
        dequant = np.asarray(
            (weights[:max_k].astype(np.int16) - zeros.astype(np.int16)).astype(np.float16)
            * scales,
            dtype=np.float16,
        )
    input_half = bits_to_fp16(input_bits[:, :max_k]).copy()
    input_gpu = torch.from_numpy(input_half).to(device)
    weight_gpu = torch.from_numpy(dequant.copy()).to(device)

    outputs: Dict[int, np.ndarray] = {}
    with torch.no_grad():
        for k in k_values:
            output = torch.matmul(input_gpu[:, :k], weight_gpu[:k, :])
            outputs[k] = fp16_to_bits(output.cpu().numpy())
    torch.cuda.synchronize()
    return outputs


def verify_tensor_core(device: str, k_values: Iterable[int]) -> Dict[str, list[str]]:
    """Profile the smallest and largest K and require Tensor Core kernel selection."""
    torch = _load_torch()
    from torch.profiler import ProfilerActivity, profile

    selected = sorted({min(k_values), max(k_values)})
    kernel_names: Dict[str, list[str]] = {}
    for k in selected:
        a = torch.randn((MXU_N, k), dtype=torch.float16, device=device)
        b = torch.randn((k, MXU_N), dtype=torch.float16, device=device)
        for _ in range(3):
            torch.matmul(a, b)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], acc_events=True) as prof:
            torch.matmul(a, b)
            torch.cuda.synchronize()
        names = [
            event.key
            for event in prof.key_averages()
            if "gemm" in event.key.lower() or "mma" in event.key.lower()
        ]
        kernel_names[str(k)] = names
        if not any("wmma" in name.lower() or "tensorop" in name.lower() for name in names):
            raise RuntimeError(f"K={k} did not select an identifiable Tensor Core kernel: {names}")
    return kernel_names


def _assert_finite(bits: np.ndarray, context: str) -> None:
    if not np.all(np.isfinite(bits_to_fp16(bits))):
        raise RuntimeError(f"Non-finite FP16 output encountered: {context}")


def _metric_row(
    seed: int,
    qdir: int,
    k: int,
    method: str,
    reference_name: str,
    reference: np.ndarray,
    evaluated: np.ndarray,
) -> dict:
    _assert_finite(reference, f"seed={seed} qdir={qdir} K={k} ref")
    _assert_finite(evaluated, f"seed={seed} qdir={qdir} K={k} method={method}")
    error = ulp_diff_fp16_bits(reference, evaluated).astype(np.float64)
    return {
        "seed": seed,
        "qdir": QDIR_NAMES[qdir],
        "K": k,
        "method": method,
        "reference": reference_name,
        "mean_ulp": float(np.mean(error)),
        "max_ulp": float(np.max(error)),
        "p50_ulp": float(np.percentile(error, 50)),
        "p95_ulp": float(np.percentile(error, 95)),
        "sample_count": int(error.size),
    }


def summarize_results(raw):
    import pandas as pd
    from scipy.stats import t

    raw = raw.copy()
    if "reference" not in raw.columns:
        raw["reference"] = REFERENCE_CPU
    rows = []
    for (reference, qdir, k, method), group in raw.groupby(
        ["reference", "qdir", "K", "method"],
        sort=True,
    ):
        values = group["mean_ulp"].to_numpy(dtype=np.float64)
        count = len(values)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if count > 1 else 0.0
        half_width = float(t.ppf(0.975, count - 1) * std / np.sqrt(count)) if count > 1 else 0.0
        rows.append(
            {
                "reference": reference,
                "qdir": qdir,
                "K": int(k),
                "method": method,
                "mean_ulp": mean,
                "std_ulp": std,
                "ci95_low": max(0.0, mean - half_width),
                "ci95_high": mean + half_width,
                "trial_count": count,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["reference", "qdir", "K", "method"]
    ).reset_index(drop=True)


def _plot_reference_summary(
    summary,
    output_dir: Path,
    reference_suffix: str,
) -> tuple[Path, Path, Path]:
    """Render one reference-specific two-panel paper-column plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_values = sorted(summary["K"].unique())
    x = np.arange(len(k_values), dtype=np.float64)
    width = 0.36
    plot_methods = (
        (METHOD_GPU, "GPU", "#327da8", -width / 2),
        (METHOD_OURS, "FINISH", "#9bcbb5", width / 2),
    )
    rc = {
        "font.size": DEFAULT_FONT_SIZE,
        "axes.titlesize": TITLE_FONT_SIZE,
        "axes.labelsize": DEFAULT_FONT_SIZE,
        "xtick.labelsize": DEFAULT_FONT_SIZE,
        "ytick.labelsize": DEFAULT_FONT_SIZE,
        "legend.fontsize": DEFAULT_FONT_SIZE,
        "axes.linewidth": LINE_WIDTH,
        "grid.linewidth": LINE_WIDTH,
        "lines.linewidth": LINE_WIDTH,
        "patch.linewidth": LINE_WIDTH,
        "xtick.major.width": LINE_WIDTH,
        "ytick.major.width": Y_TICK_WIDTH,
        "ytick.minor.width": Y_TICK_WIDTH,
    }
    with plt.rc_context(rc):
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(COL_WIDTH, FIG_HEIGHT),
            sharex=True,
            sharey=True,
        )
        for column, (axis, qdir) in enumerate(zip(axes, ("qcol", "qrow"))):
            qdata = summary[summary["qdir"] == qdir]
            for method, label, color, offset in plot_methods:
                data = qdata[qdata["method"] == method].set_index("K").loc[k_values]
                means = data["mean_ulp"].to_numpy(dtype=np.float64)
                lows = data["ci95_low"].to_numpy(dtype=np.float64)
                highs = data["ci95_high"].to_numpy(dtype=np.float64)
                zero_mask = means == 0.0
                plot_means = np.maximum(means, PLOT_LOG_FLOOR)
                plot_means = np.where(zero_mask, 2.0 * PLOT_LOG_FLOOR, plot_means)
                plot_lows = np.minimum(plot_means, np.maximum(lows, PLOT_LOG_FLOOR))
                plot_highs = np.maximum(highs, plot_means)
                axis.bar(
                    x + offset,
                    plot_means,
                    width,
                    yerr=np.vstack((plot_means - plot_lows, plot_highs - plot_means)),
                    label=label,
                    color=color,
                    edgecolor="black",
                    linewidth=LINE_WIDTH,
                    capsize=1,
                    error_kw={"elinewidth": LINE_WIDTH, "capthick": LINE_WIDTH},
                )
                for position, mean, plot_mean in zip(x + offset, means, plot_means):
                    if mean == 0.0:
                        axis.annotate(
                            "0",
                            xy=(position, plot_mean),
                            xytext=(0, 2),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontweight="bold",
                        )
            axis.set_title(f"FP16-INT4 {qdir.upper()}", fontsize=TITLE_FONT_SIZE)
            axis.set_xticks(x)
            axis.set_xticklabels([str(k) for k in k_values], rotation=45, ha="right")
            axis.grid(axis="y", alpha=0.3, linewidth=LINE_WIDTH)
            axis.set_yscale("log")
            axis.set_ylim(bottom=PLOT_LOG_FLOOR)
            if column == 0:
                axis.set_ylabel("Mean ULP Error")
                axis.tick_params(axis="y", which="both", width=Y_TICK_WIDTH)
            else:
                axis.tick_params(axis="y", which="both", left=False, labelleft=False)
                axis.legend(loc="lower right")
        fig.supxlabel("Fan-in (K)", fontsize=DEFAULT_FONT_SIZE)
        fig.tight_layout(pad=0.3, w_pad=0.6)
        stem = f"fp16_int4_real2scomp_fanin_vs_ref_{reference_suffix}"
        png_path = output_dir / f"{stem}.png"
        svg_path = output_dir / f"{stem}.svg"
        pdf_path = output_dir / f"{stem}.pdf"
        fig.savefig(png_path, dpi=200)
        fig.savefig(svg_path)
        fig.savefig(pdf_path)
        plt.close(fig)
    return png_path, svg_path, pdf_path


def plot_summary(summary, output_dir: Path) -> tuple[Path, ...]:
    """Render CPU- and GPU-reference figures available in summarized metrics."""
    outputs: list[Path] = []
    for reference, suffix in (
        (REFERENCE_CPU, "CPU"),
        (REFERENCE_GPU, "GPU"),
    ):
        selected = summary[summary["reference"].eq(reference)]
        if not selected.empty:
            outputs.extend(_plot_reference_summary(selected, output_dir, suffix))
    return tuple(outputs)


def plot_csv(csv_path: Path) -> tuple[Path, ...]:
    """Load raw seed metrics and create reference-specific figures beside it."""
    import pandas as pd

    csv_path = csv_path.resolve()
    raw = pd.read_csv(csv_path)
    required = {"seed", "qdir", "K", "method", "mean_ulp"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
    if raw.empty:
        raise ValueError(f"CSV contains no rows: {csv_path}")

    summary = summarize_results(raw)
    summary.to_csv(csv_path.parent / "summary.csv", index=False)
    return plot_summary(summary, csv_path.parent)


def _system_metadata(device: str) -> dict:
    torch = _load_torch()
    props = torch.cuda.get_device_properties(device)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_memory_bytes": props.total_memory,
        "driver_version": driver,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "git_commit": commit,
    }


def export_data(args: argparse.Namespace) -> Path:
    """Run the numerical experiment and export raw seed metrics as CSV."""
    import pandas as pd

    k_values = tuple(sorted(set(args.k_values)))
    _validate_config(args, k_values)
    output_dir = _resolve_output_dir(args.output, args.resume)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    raw_path = output_dir / "raw_seed_metrics.csv"
    metadata_path = output_dir / "metadata.json"

    if raw_path.exists() and not args.resume:
        raise FileExistsError(f"Result already exists; use --resume: {raw_path}")
    raw = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    if not raw.empty and "reference" not in raw.columns:
        raw["reference"] = REFERENCE_CPU
    completed = set()
    if not raw.empty:
        completed = set(
            zip(
                raw["seed"],
                raw["qdir"],
                raw["K"],
                raw["method"],
                raw["reference"],
            )
        )

    kernels = verify_tensor_core(args.device, k_values) if args.verify_tensor_core else {}
    metadata = {
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(sys.argv),
        "config": {
            "k_values": list(k_values),
            "m": args.m,
            "n": args.n,
            "qblock": args.qblock,
            "trials": args.trials,
            "base_seed": args.base_seed,
            "quant": "signed-asymmetric",
            "references": list(REFERENCES),
            "weight_range": [-8, 7],
            "zero_range": [-4, 3],
            "scale_range": [SCALE_MIN, SCALE_MAX],
            "input_distribution": {
                "kind": "independent-sign-exponent-mantissa-rounded-to-fp16",
                "sign": "uniform-{0,1}",
                "exponent": {
                    "distribution": "discrete-uniform-inclusive",
                    "min": INPUT_EXP_MIN,
                    "max": INPUT_EXP_MAX,
                },
                "mantissa_field": {
                    "distribution": "normal-rounded-and-clipped",
                    "mean": INPUT_MANTISSA_MEAN,
                    "std": INPUT_MANTISSA_STD,
                    "min": 0,
                    "max": 1023,
                },
            },
        },
        "system": _system_metadata(args.device),
        "tensor_core_kernels": kernels,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    expected_per_seed = 2 * len(k_values) * len(METHODS) * len(REFERENCES)
    for trial in range(args.trials):
        seed = args.base_seed + trial
        seed_keys = {
            (seed, qdir, k, method, reference)
            for qdir in QDIR_NAMES.values()
            for k in k_values
            for method in METHODS
            for reference in REFERENCES
        }
        if len(seed_keys & completed) == expected_per_seed:
            print(f"[{trial + 1}/{args.trials}] seed={seed}: already complete")
            continue

        data = generate_trial_data(seed, args.m, args.n, max(k_values), args.qblock)
        new_rows = []
        for qdir in (QCOL, QROW):
            if qdir == QCOL:
                scales, zeros = data.qcol_scale_bits, data.qcol_zero
                ours = qcol_real_2scomp_prefix_outputs(
                    data.input_bits, data.weights, scales, zeros, k_values, args.qblock
                )
            else:
                scales, zeros = data.qrow_scale_bits, data.qrow_zero
                ours = qrow_real_2scomp_prefix_outputs(
                    data.input_bits, data.weights, scales, zeros, k_values, args.qblock
                )
            reference = reference_prefix_outputs(
                data.input_bits, data.weights, scales, zeros, qdir, k_values, args.qblock
            )
            gpu_reference = gpu_fp64_reference_outputs(
                data.input_bits,
                data.weights,
                scales,
                zeros,
                qdir,
                k_values,
                args.device,
                args.qblock,
            )
            gpu_model = gpu_model_prefix_outputs(
                data.input_bits, data.weights, scales, zeros, qdir, k_values, args.qblock
            )
            gpu = tensor_core_outputs(
                data.input_bits,
                data.weights,
                scales,
                zeros,
                qdir,
                k_values,
                args.device,
                args.qblock,
            )
            evaluations = {METHOD_GPU: gpu, METHOD_GPU_MODEL: gpu_model, METHOD_OURS: ours}
            for k in k_values:
                for reference_name, reference_outputs in (
                    (REFERENCE_CPU, reference),
                    (REFERENCE_GPU, gpu_reference),
                ):
                    for method, outputs in evaluations.items():
                        key = (
                            seed,
                            QDIR_NAMES[qdir],
                            k,
                            method,
                            reference_name,
                        )
                        if key not in completed:
                            new_rows.append(
                                _metric_row(
                                    seed,
                                    qdir,
                                    k,
                                    method,
                                    reference_name,
                                    reference_outputs[k],
                                    outputs[k],
                                )
                            )
                            completed.add(key)

        if new_rows:
            raw = pd.concat((raw, pd.DataFrame(new_rows)), ignore_index=True)
            raw = raw.sort_values(
                ["seed", "reference", "qdir", "K", "method"]
            ).reset_index(drop=True)
            temporary = raw_path.with_suffix(".csv.tmp")
            raw.to_csv(temporary, index=False)
            temporary.replace(raw_path)
        print(f"[{trial + 1}/{args.trials}] seed={seed}: wrote {len(new_rows)} metrics")

    metadata["status"] = "complete"
    metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
    metadata["raw_row_count"] = int(len(raw))
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return raw_path


def _validate_config(args: argparse.Namespace, k_values: Sequence[int]) -> None:
    if args.m % MXU_N != 0 or args.n % MXU_N != 0:
        raise ValueError(f"M and N must be multiples of {MXU_N} for Tensor Core coverage")
    if args.qblock != QBLOCK or args.qblock != MXU_K:
        raise ValueError(f"This experiment requires qblock=QBLOCK=MXU_K={QBLOCK}")
    if args.trials < 2:
        raise ValueError("At least two trials are required for a confidence interval")
    if not k_values or any(k <= 0 or k % args.qblock != 0 for k in k_values):
        raise ValueError(f"All K values must be positive multiples of {args.qblock}")


def _resolve_output_dir(output: Path | None, resume: bool) -> Path:
    if resume and output is None:
        raise ValueError("--resume requires an explicit --output directory")
    if output is not None:
        return output.resolve()
    repo_root = Path(__file__).resolve().parents[4]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return repo_root / "analysis_workspace/numerical_acc/results/fpint_real_2scomp" / stamp


def _parse_k_values(value: str) -> list[int]:
    try:
        return [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError("K values must be comma-separated integers") from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data_parser = commands.add_parser("data", help="run experiments and export raw CSV data")
    data_parser.add_argument(
        "--k-values", type=_parse_k_values, default=list(DEFAULT_K_VALUES), help="comma-separated fan-ins"
    )
    data_parser.add_argument("--m", type=int, default=16)
    data_parser.add_argument("--n", type=int, default=16)
    data_parser.add_argument("--qblock", type=int, default=QBLOCK)
    data_parser.add_argument("--trials", type=int, default=30)
    data_parser.add_argument("--base-seed", type=int, default=20260729)
    data_parser.add_argument(
        "--quant", choices=("signed-asymmetric",), default="signed-asymmetric"
    )
    data_parser.add_argument("--device", default="cuda:0")
    data_parser.add_argument("--output", type=Path)
    data_parser.add_argument("--resume", action="store_true")
    data_parser.add_argument(
        "--verify-tensor-core", action=argparse.BooleanOptionalAction, default=True
    )

    plot_parser = commands.add_parser("plot", help="plot an exported raw metrics CSV")
    plot_parser.add_argument("csv", type=Path, help="path to raw_seed_metrics.csv")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "data":
        csv_path = export_data(args)
        print(f"Exported fan-in data: {csv_path}")
    else:
        paths = plot_csv(args.csv)
        print(f"Created plots: {', '.join(str(path) for path in paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
