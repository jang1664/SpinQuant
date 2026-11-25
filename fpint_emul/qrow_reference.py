"""Independent scalar QROW oracle (Python integer MAC, NumPy FP rounding)."""

import math

import numpy as np

from .reference import _validate_inputs, _bf16_bits_to_float32, _round_float32_to_bf16


def qrow_real_2scomp_reference(activation, weight, scale, zero, config, bias=None):
    if config.quantization_direction != "qrow":
        raise ValueError("QROW reference requires quantization_direction='qrow'")
    x, w, s, z, bias = _validate_inputs(activation, weight, scale, zero, bias, config)
    if x.shape[-1] <= 0:
        raise ValueError("K must be positive")
    if z.size and (z.min() < -(1 << 31) or z.max() > (1 << 31) - 1):
        raise ValueError("QROW zero points must fit int32")
    is_half = config.activation_format == "fp16"
    values = x.astype(np.float32) if is_half else _bf16_bits_to_float32(x)
    k, n = x.shape[-1], w.shape[0]
    flat = values.reshape(-1, k)
    result = np.zeros((flat.shape[0], n), dtype=np.float32)
    mb, eb = config.mantissa_bits, config.exponent_bits
    for row in range(flat.shape[0]):
        for column in range(n):
            group = 0 if config.group_size == -1 else column // config.group_size
            with np.errstate(over="ignore", invalid="ignore"):
                scaled = flat[row] * s[:, group].astype(np.float32)
                scaled = scaled.astype(np.float16) if is_half else _round_float32_to_bf16(scaled)
            if not np.isfinite(scaled).all():
                raise ValueError("QROW activation * scale must remain finite after rounding")
            bits = scaled.view(np.uint16) if is_half else (scaled.view(np.uint32) >> 16)
            acc = np.float32(0)
            for start in range(0, k, config.mxu_rows):
                end = min(start + config.mxu_rows, k)
                exponents = [max(1, (int(bits[i]) >> mb) & ((1 << eb) - 1)) for i in range(start, end)]
                maximum = max(exponents)
                inner, correction = 0, 0
                for i, exponent in zip(range(start, end), exponents):
                    b = int(bits[i])
                    hidden = (b & ((1 << mb) - 1)) | (int(((b >> mb) & ((1 << eb) - 1)) != 0) << mb)
                    sign = -1 if b & 0x8000 else 1
                    main = sign * ((hidden << config.extra_bits) >> (maximum - exponent))
                    reduce = sign * ((hidden << config.reduce_extra_bits) >> (maximum - exponent))
                    inner += main * int(w[column, i])
                    correction += reduce * int(z[i, group])
                post = inner - correction * (1 << (config.extra_bits - config.reduce_extra_bits))
                with np.errstate(over="ignore", invalid="ignore"):
                    tile = np.float32(math.ldexp(float(post), maximum - config.exponent_bias - mb - config.extra_bits))
                    acc = np.float32(acc + tile)
            result[row, column] = acc
    with np.errstate(over="ignore", invalid="ignore"):
        result = result.astype(np.float16) if is_half else _round_float32_to_bf16(result)
        if bias is not None:
            bv = bias if is_half else _bf16_bits_to_float32(bias)
            result = result + bv
            result = result.astype(np.float16) if is_half else _round_float32_to_bf16(result)
    return result.reshape(*x.shape[:-1], n)
