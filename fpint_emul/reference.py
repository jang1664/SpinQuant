from typing import Optional, Tuple

import numpy as np
import torch

from .config import FpIntConfig


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        if value.dtype == torch.bfloat16:
            return value.detach().cpu().contiguous().view(torch.uint16).numpy()
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _bf16_bits_to_float32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(
        np.float32
    )


def _round_float32_to_bf16(values: np.ndarray) -> np.ndarray:
    """Return float32 values rounded exactly to representable BF16 values."""

    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    upper = bits >> np.uint32(16)
    lsb = upper & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    exponent = bits & np.uint32(0x7F800000)
    mantissa = bits & np.uint32(0x007FFFFF)
    is_nan = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    rounded = np.where(is_nan, upper | np.uint32(0x40), rounded)
    return _bf16_bits_to_float32(rounded.astype(np.uint16))


def _validate_inputs(
    activation,
    weight,
    scale,
    zero,
    bias,
    config: FpIntConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    activation = _as_numpy(activation)
    weight = _as_numpy(weight)
    scale_is_bf16 = isinstance(scale, torch.Tensor) and scale.dtype == torch.bfloat16
    scale = _as_numpy(scale.float() if scale_is_bf16 else scale)
    zero = _as_numpy(zero)
    bias = None if bias is None else _as_numpy(bias)

    expected_activation_dtype = np.float16 if config.activation_format == "fp16" else np.uint16
    if activation.dtype != expected_activation_dtype:
        raise TypeError(
            f"activation must have dtype {config.activation_format}"
        )
    if activation.ndim < 1:
        raise ValueError("activation must have shape [..., K]")
    activation_values = (
        activation
        if config.activation_format == "fp16"
        else _bf16_bits_to_float32(activation)
    )
    if not np.isfinite(activation_values).all():
        raise ValueError("NaN and Inf activations are not supported")
    k = activation.shape[-1]
    if weight.dtype != np.int8 or weight.ndim != 2:
        raise TypeError("weight must be an int8 tensor with shape [N, K]")
    if weight.shape[1] != k:
        raise ValueError("activation K and weight K do not match")
    qmin = -(1 << (config.weight_bits - 1))
    qmax = (1 << (config.weight_bits - 1)) - 1
    if weight.size and (weight.min() < qmin or weight.max() > qmax):
        raise ValueError(
            f"weight values must be in [{qmin}, {qmax}] for INT{config.weight_bits}"
        )
    n = weight.shape[0]
    expected = (n, config.group_count(k))
    if (scale.dtype != np.float16 and not scale_is_bf16) or scale.shape != expected:
        raise TypeError(f"scale must be float16 or bfloat16 with shape {expected}")
    if not np.isfinite(scale).all():
        raise ValueError("scale must contain finite values")
    if zero.dtype not in (np.int16, np.int32, np.int64) or zero.shape != expected:
        raise TypeError(f"zero must be int16/int32/int64 with shape {expected}")
    maximum_zero = int(np.abs(zero.astype(object)).max()) if zero.size else 0
    config.validate_zero_bound(maximum_zero)
    expected_bias_dtype = np.float16 if config.activation_format == "fp16" else np.uint16
    if bias is not None and (bias.dtype != expected_bias_dtype or bias.shape != (n,)):
        raise TypeError(
            f"bias must be {config.activation_format} with shape {(n,)}"
        )
    return activation, weight, scale, zero, bias


def prealign_reference(
    activation: np.ndarray,
    extra_bits: int,
    mxu_rows: int,
    mantissa_bits: int = 10,
    exponent_bits: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split 16-bit floating fields and align each K tile to its maximum exponent."""

    rows, k = activation.shape
    tiles = (k + mxu_rows - 1) // mxu_rows
    padded_k = tiles * mxu_rows
    padded = np.zeros((rows, padded_k), dtype=activation.dtype)
    padded[:, :k] = activation
    bits = padded.view(np.uint16) if padded.dtype == np.float16 else padded
    sign = (bits >> 15) & 1
    exponent = (bits >> mantissa_bits) & ((1 << exponent_bits) - 1)
    mantissa = bits & ((1 << mantissa_bits) - 1)
    exponent_for_align = np.maximum(exponent, 1).astype(np.int64)
    maximum = exponent_for_align.reshape(rows, tiles, mxu_rows).max(axis=2)
    hidden = ((exponent != 0).astype(np.int64) << mantissa_bits) | mantissa
    shifts = (
        maximum[:, :, None]
        - exponent_for_align.reshape(rows, tiles, mxu_rows)
    ).reshape(rows, padded_k)
    shifted = hidden.astype(np.int64) << extra_bits
    aligned = np.where(shifts >= 63, 0, shifted >> np.minimum(shifts, 62))
    aligned = np.where(sign != 0, -aligned, aligned).astype(np.int64)
    return aligned, maximum.astype(np.int16)


def qcol_real_2scomp_reference(
    activation,
    weight,
    scale,
    zero,
    config: FpIntConfig,
    bias=None,
) -> np.ndarray:
    """Generalized CPU reference for QCOL_REAL_2SCOMP.

    FP16 output is returned as float16. BF16 output is returned as float32
    containing exactly BF16-representable values. Each MXU K tile is
    converted/scaled and then added to the accumulator in FP32 K order.
    """

    activation, weight, scale, zero, bias = _validate_inputs(
        activation, weight, scale, zero, bias, config
    )
    original_shape = activation.shape[:-1]
    k = activation.shape[-1]
    n = weight.shape[0]
    flat = np.ascontiguousarray(activation).reshape(-1, k)
    rows = flat.shape[0]
    if rows == 0:
        output_dtype = np.float16 if config.activation_format == "fp16" else np.float32
        return np.empty((*original_shape, n), dtype=output_dtype)

    aligned_main, maximum = prealign_reference(
        flat,
        config.extra_bits,
        config.mxu_rows,
        config.mantissa_bits,
        config.exponent_bits,
    )
    has_zero = bool(np.any(zero))
    aligned_reduce = None
    if has_zero:
        aligned_reduce, _ = prealign_reference(
            flat,
            config.reduce_extra_bits,
            config.mxu_rows,
            config.mantissa_bits,
            config.exponent_bits,
        )

    padded_k = aligned_main.shape[1]
    padded_weight = np.zeros((n, padded_k), dtype=np.int8)
    padded_weight[:, :k] = weight
    accumulator = np.zeros((rows, n), dtype=np.float32)
    shift_back = config.extra_bits - config.reduce_extra_bits
    binary_scale_exponent = -(config.mantissa_bits + config.extra_bits)

    for tile in range(config.tile_count(k)):
        start = tile * config.mxu_rows
        end = start + config.mxu_rows
        group = config.group_for_tile(tile, k)
        inner = aligned_main[:, start:end] @ padded_weight[:, start:end].T.astype(
            np.int64
        )
        post = inner
        if has_zero:
            reduction = aligned_reduce[:, start:end].sum(axis=1, dtype=np.int64)
            correction = (
                reduction[:, None]
                * zero[:, group].astype(np.int64)[None, :]
                * np.int64(1 << shift_back)
            )
            post = inner - correction
        # Use float64 for the integer-to-real expression, then perform the two
        # explicitly specified FP32 operations (cast and scale multiplication).
        exponent = maximum[:, tile].astype(np.int32) - config.exponent_bias
        with np.errstate(over="ignore", invalid="ignore"):
            restored = np.ldexp(
                post.astype(np.float64),
                exponent[:, None] + binary_scale_exponent,
            ).astype(np.float32)
            scaled = np.multiply(
                restored,
                scale[:, group].astype(np.float32)[None, :],
                dtype=np.float32,
            )
            accumulator = np.add(accumulator, scaled, dtype=np.float32)

    # Full-range FP16 tests intentionally allow finite accumulators to overflow
    # at the FP16 output cast; callers account for the resulting infinities.
    with np.errstate(over="ignore", invalid="ignore"):
        if config.activation_format == "fp16":
            output = accumulator.astype(np.float16)
            if bias is not None:
                output = np.add(output, bias, dtype=np.float16)
        else:
            output = _round_float32_to_bf16(accumulator)
            if bias is not None:
                bias_values = _bf16_bits_to_float32(bias)
                output = _round_float32_to_bf16(output + bias_values)
    return output.reshape(*original_shape, n)
