"""QROW: scale/zero [K, ceil(N/group_size)], weight [N, K].

Scale is applied and rounded to the activation dtype before prealignment.
Like the QCOL backend, restored tiles are rounded/accumulated in FP32.
"""

import torch

from .torch_backend import _validate_inputs, prealign_torch


def _validate_qrow(activation, weight, scale, zero, config, bias, validate_values):
    if config.quantization_direction != "qrow":
        raise ValueError("QROW backend requires quantization_direction='qrow'")
    if validate_values and zero.numel():
        # CUDA stores zero points as int32. Do not silently narrow int64 values.
        zero_wide = zero.to(torch.int64)
        if bool(((zero_wide < -(1 << 31)) | (zero_wide > (1 << 31) - 1)).any()):
            raise ValueError("QROW zero points must fit int32")
    _validate_inputs(activation, weight, scale, zero, bias, config,
                     check_values=validate_values)
    if validate_values:
        flat = activation.reshape(-1, activation.shape[-1])
        for group in range(scale.shape[1]):
            for start in range(0, flat.shape[0], 256):
                scaled = (flat[start:start + 256].float() * scale[:, group].float()).to(activation.dtype)
                if not bool(torch.isfinite(scaled).all()):
                    raise ValueError("QROW activation * scale must remain finite after rounding")


@torch.no_grad()
def qrow_real_2scomp_torch(activation, weight, scale, zero, config, bias=None,
                         *, validate_values=True, has_zero=None):
    """Torch QROW backend with integer contraction and FP32 tile accumulation."""
    _validate_qrow(activation, weight, scale, zero, config, bias, validate_values)
    k, n = activation.shape[-1], weight.shape[0]
    flat = activation.reshape(-1, k)
    acc = torch.zeros((flat.shape[0], n), dtype=torch.float32, device=activation.device)
    has_zero = bool(zero.any()) if has_zero is None else has_zero
    width = n if config.group_size == -1 else config.group_size
    if not flat.shape[0] or not n:
        return acc.to(activation.dtype).reshape(*activation.shape[:-1], n)
    for group in range(scale.shape[1]):
        n0, n1 = group * width, min((group + 1) * width, n)
        scaled = (flat.float() * scale[:, group].float()).to(activation.dtype)
        args = (config.mxu_rows, config.mantissa_bits, config.exponent_bits)
        aligned, maximum = prealign_torch(scaled, config.extra_bits, *args)
        reduced = prealign_torch(scaled, config.reduce_extra_bits, *args)[0] if has_zero else None
        for tile in range(config.tile_count(k)):
            start, end = tile * config.mxu_rows, min((tile + 1) * config.mxu_rows, k)
            exponent = maximum[:, tile].to(torch.int64) - config.exponent_bias - config.mantissa_bits - config.extra_bits
            factor = ((exponent + 1023) << 52).view(torch.float64)[:, None]
            correction = 0
            if has_zero:
                correction = (reduced[:, start:end] * zero[start:end, group].to(torch.int64)).sum(1)[:, None]
                correction = correction * (1 << (config.extra_bits - config.reduce_extra_bits))
            for c0 in range(n0, n1, config.n_chunk_size):
                c1 = min(c0 + config.n_chunk_size, n1)
                inner = (aligned[:, start:end].double() @ weight[c0:c1, start:end].double().T).round().to(torch.int64)
                contribution = ((inner - correction).double() * factor).float()
                acc[:, c0:c1] = acc[:, c0:c1] + contribution
    output = acc.to(activation.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*activation.shape[:-1], n)


@torch.no_grad()
def qrow_real_2scomp_cuda(activation, weight, scale, zero, config, bias=None,
                        *, validate_values=True, has_zero=None):
    """Fused CUDA QROW backend; supports FP16/BF16 and signed INT4/INT8."""
    from .cuda_backend import _load_extension

    _validate_qrow(activation, weight, scale, zero, config, bias, validate_values)
    if not activation.is_cuda:
        raise ValueError("fpint_cuda requires CUDA tensors")
    if config.mxu_rows > 256:
        raise ValueError("fpint_cuda supports mxu_rows <= 256")
    flat = activation.contiguous().reshape(-1, activation.shape[-1])
    output = _load_extension().qrow_real_2scomp(
        flat, weight.contiguous(), scale.contiguous(), zero.to(torch.int32).contiguous(),
        config.group_size, config.mxu_rows, config.extra_bits, config.reduce_extra_bits,
        bool(zero.any()) if has_zero is None else has_zero,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*activation.shape[:-1], weight.shape[0])
