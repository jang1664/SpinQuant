from typing import Optional, Tuple

import numpy as np

from .config import FpIntConfig


_FP16_EXP_BIAS = 15
_FP16_MANTISSA_BITS = 10


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


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
    scale = _as_numpy(scale)
    zero = _as_numpy(zero)
    bias = None if bias is None else _as_numpy(bias)

    if activation.dtype != np.float16:
        raise TypeError("activation must have dtype float16")
    if activation.ndim < 1:
        raise ValueError("activation must have shape [..., K]")
    if not np.isfinite(activation).all():
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
    if scale.dtype != np.float16 or scale.shape != expected:
        raise TypeError(f"scale must be float16 with shape {expected}")
    if not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("scale must contain finite positive values")
    if zero.dtype not in (np.int16, np.int32, np.int64) or zero.shape != expected:
        raise TypeError(f"zero must be int16/int32/int64 with shape {expected}")
    maximum_zero = int(np.abs(zero.astype(object)).max()) if zero.size else 0
    config.validate_zero_bound(maximum_zero)
    if bias is not None and (bias.dtype != np.float16 or bias.shape != (n,)):
        raise TypeError(f"bias must be float16 with shape {(n,)}")
    return activation, weight, scale, zero, bias


def prealign_reference(
    activation: np.ndarray,
    extra_bits: int,
    mxu_rows: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split FP16 fields and align each K tile to its maximum exponent."""

    rows, k = activation.shape
    tiles = (k + mxu_rows - 1) // mxu_rows
    padded_k = tiles * mxu_rows
    padded = np.zeros((rows, padded_k), dtype=np.float16)
    padded[:, :k] = activation
    bits = padded.view(np.uint16)
    sign = (bits >> 15) & 1
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    exponent_for_align = np.maximum(exponent, 1).astype(np.int64)
    maximum = exponent_for_align.reshape(rows, tiles, mxu_rows).max(axis=2)
    hidden = ((exponent != 0).astype(np.int64) << _FP16_MANTISSA_BITS) | mantissa
    shifts = (
        maximum[:, :, None]
        - exponent_for_align.reshape(rows, tiles, mxu_rows)
    ).reshape(rows, padded_k)
    aligned = (hidden.astype(np.int64) << extra_bits) >> shifts
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

    The output is FP16. Each MXU K tile is converted/scaled and then added to
    the accumulator in FP32 K order, matching the hardware reference sequence.
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
        return np.empty((*original_shape, n), dtype=np.float16)

    aligned_main, maximum = prealign_reference(
        flat, config.extra_bits, config.mxu_rows
    )
    has_zero = bool(np.any(zero))
    aligned_reduce = None
    if has_zero:
        aligned_reduce, _ = prealign_reference(
            flat, config.reduce_extra_bits, config.mxu_rows
        )

    padded_k = aligned_main.shape[1]
    padded_weight = np.zeros((n, padded_k), dtype=np.int8)
    padded_weight[:, :k] = weight
    accumulator = np.zeros((rows, n), dtype=np.float32)
    shift_back = config.extra_bits - config.reduce_extra_bits
    binary_scale_exponent = -(_FP16_MANTISSA_BITS + config.extra_bits)

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
        exponent = maximum[:, tile].astype(np.int32) - _FP16_EXP_BIAS
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

    output = accumulator.astype(np.float16)
    if bias is not None:
        output = np.add(output, bias, dtype=np.float16)
    return output.reshape(*original_shape, n)
