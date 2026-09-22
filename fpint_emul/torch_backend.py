from typing import Optional, Tuple

import torch

from .config import FpIntConfig


def _validate_inputs(
    activation: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    bias: Optional[torch.Tensor],
    config: FpIntConfig,
    check_values: bool = True,
) -> None:
    activation_dtype = (
        torch.float16 if config.activation_format == "fp16" else torch.bfloat16
    )
    if activation.dtype != activation_dtype or activation.ndim < 1:
        raise TypeError(
            f"activation must be {config.activation_format} with shape [..., K]"
        )
    if check_values and not bool(torch.isfinite(activation).all()):
        raise ValueError("NaN and Inf activations are not supported")
    if weight.dtype != torch.int8 or weight.ndim != 2:
        raise TypeError("weight must be int8 with shape [N, K]")
    k = activation.shape[-1]
    if weight.shape[1] != k:
        raise ValueError("activation K and weight K do not match")
    qmin = -(1 << (config.weight_bits - 1))
    qmax = (1 << (config.weight_bits - 1)) - 1
    if check_values and weight.numel() and bool(((weight < qmin) | (weight > qmax)).any()):
        raise ValueError(
            f"weight values must be in [{qmin}, {qmax}] for INT{config.weight_bits}"
        )
    expected = (weight.shape[0], config.group_count(k))
    if scale.dtype not in (torch.float16, torch.bfloat16) or tuple(scale.shape) != expected:
        raise TypeError(f"scale must be float16 or bfloat16 with shape {expected}")
    if check_values and not bool(torch.isfinite(scale).all()):
        raise ValueError("scale must contain finite values")
    if zero.dtype not in (torch.int16, torch.int32, torch.int64) or tuple(
        zero.shape
    ) != expected:
        raise TypeError(f"zero must be int16/int32/int64 with shape {expected}")
    if check_values:
        maximum_zero = int(zero.to(torch.int64).abs().max()) if zero.numel() else 0
        config.validate_zero_bound(maximum_zero)
    if bias is not None and (
        bias.dtype != activation_dtype or tuple(bias.shape) != (weight.shape[0],)
    ):
        raise TypeError(
            f"bias must be {config.activation_format} with shape {(weight.shape[0],)}"
        )
    tensors = (weight, scale, zero) + (() if bias is None else (bias,))
    if any(t.device != activation.device for t in tensors):
        raise ValueError("activation, weight, scale, zero and bias must share a device")


def prealign_torch(
    activation: torch.Tensor,
    extra_bits: int,
    mxu_rows: int,
    mantissa_bits: int = 10,
    exponent_bits: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows, k = activation.shape
    tiles = (k + mxu_rows - 1) // mxu_rows
    padded_k = tiles * mxu_rows
    if padded_k != k:
        activation = torch.nn.functional.pad(activation, (0, padded_k - k))
    bits = activation.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign = (bits >> 15) & 1
    exponent_mask = (1 << exponent_bits) - 1
    mantissa_mask = (1 << mantissa_bits) - 1
    exponent = (bits >> mantissa_bits) & exponent_mask
    mantissa = bits & mantissa_mask
    exponent_for_align = torch.where(
        exponent == 0, torch.ones_like(exponent), exponent
    )
    maximum = exponent_for_align.view(rows, tiles, mxu_rows).max(dim=2).values
    hidden = ((exponent != 0).to(torch.int64) << mantissa_bits) | mantissa
    shifts = (
        maximum.unsqueeze(-1)
        - exponent_for_align.view(rows, tiles, mxu_rows)
    ).reshape(rows, padded_k)
    shifted = hidden.to(torch.int64) << extra_bits
    # PyTorch/CUDA shifts by >= 64 are backend-dependent. The mathematical
    # aligned value is zero once every significand bit has been discarded.
    aligned = torch.where(
        shifts >= 63,
        torch.zeros_like(shifted),
        shifted >> shifts.clamp(max=62).to(torch.int64),
    )
    aligned = torch.where(sign.bool(), -aligned, aligned)
    return aligned, maximum.to(torch.int16)


@torch.no_grad()
def qcol_real_2scomp_torch(
    activation: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    config: FpIntConfig,
    bias: Optional[torch.Tensor] = None,
    *,
    validate_values: bool = True,
    has_zero: Optional[bool] = None,
) -> torch.Tensor:
    """Torch QCOL_REAL_2SCOMP backend with sequential FP32 tile accumulation."""

    _validate_inputs(
        activation, weight, scale, zero, bias, config, check_values=validate_values
    )
    original_shape = activation.shape[:-1]
    k = activation.shape[-1]
    n = weight.shape[0]
    flat = activation.contiguous().reshape(-1, k)
    rows = flat.shape[0]
    if rows == 0:
        return torch.empty(
            (*original_shape, n), dtype=activation.dtype, device=activation.device
        )

    aligned_main, maximum = prealign_torch(
        flat,
        config.extra_bits,
        config.mxu_rows,
        config.mantissa_bits,
        config.exponent_bits,
    )
    has_zero = bool(zero.any()) if has_zero is None else has_zero
    aligned_reduce = None
    if has_zero:
        aligned_reduce, _ = prealign_torch(
            flat,
            config.reduce_extra_bits,
            config.mxu_rows,
            config.mantissa_bits,
            config.exponent_bits,
        )

    padded_k = aligned_main.shape[1]
    if padded_k != k:
        padded_weight = torch.nn.functional.pad(weight, (0, padded_k - k))
    else:
        padded_weight = weight
    accumulator = torch.zeros((rows, n), dtype=torch.float32, device=activation.device)
    shift_back = config.extra_bits - config.reduce_extra_bits
    binary_scale_exponent = -(config.mantissa_bits + config.extra_bits)

    for tile in range(config.tile_count(k)):
        start = tile * config.mxu_rows
        end = start + config.mxu_rows
        group = config.group_for_tile(tile, k)
        a_tile = aligned_main[:, start:end].to(torch.float64)
        exponent = maximum[:, tile].to(torch.int32) - config.exponent_bias
        # Build the exact normal FP64 power of two from its exponent field.
        # On some CUDA/PyTorch versions ldexp(1., -19) is one FP64 ULP
        # below 2**-19, changing FP32 tie rounding and eventually BF16 output.
        # FP16/BF16 exponents and the validated extra-bit bounds keep this
        # factor well inside the normal FP64 exponent range.
        factor = (
            (exponent.to(torch.int64) + binary_scale_exponent + 1023) << 52
        ).view(torch.float64).unsqueeze(1)
        reduction = None
        if has_zero:
            reduction = aligned_reduce[:, start:end].sum(dim=1, dtype=torch.int64)

        for n0 in range(0, n, config.n_chunk_size):
            n1 = min(n, n0 + config.n_chunk_size)
            # The configured bounds ensure every integer dot product is exactly
            # representable in float64. Round before converting back to int64 so
            # the expression below remains an integer operation.
            inner = torch.matmul(
                a_tile,
                padded_weight[n0:n1, start:end].to(torch.float64).T,
            ).round().to(torch.int64)
            post = inner
            if has_zero:
                post = inner - (
                    reduction[:, None]
                    * zero[n0:n1, group].to(torch.int64)[None, :]
                    * (1 << shift_back)
                )
            restored = (post.to(torch.float64) * factor).to(torch.float32)
            scaled = restored * scale[n0:n1, group].to(torch.float32)[None, :]
            accumulator[:, n0:n1] = accumulator[:, n0:n1] + scaled

    output = accumulator.to(activation.dtype)
    if bias is not None:
        output = output + bias
    return output.reshape(*original_shape, n)
