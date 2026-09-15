from typing import Optional

import torch

from .config import FpIntConfig
from .torch_backend import qcol_real_2scomp_torch


def dequantize_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    config: FpIntConfig,
) -> torch.Tensor:
    """Expand compact QCOL parameters and dequantize to FP16 [N, K]."""

    if weight.ndim != 2:
        raise ValueError("weight must have shape [N, K]")
    k = weight.shape[1]
    group_size = config.effective_group_size(k)
    group_index = torch.arange(k, device=weight.device) // group_size
    return (
        (weight.to(torch.int32) - zero[:, group_index].to(torch.int32))
        * scale[:, group_index].to(torch.float32)
    ).to(torch.float16)


def fpint_linear(
    activation: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    config: FpIntConfig,
    bias: Optional[torch.Tensor] = None,
    backend: str = "fpint_torch",
    *,
    validate_values: bool = True,
    has_zero=None,
) -> torch.Tensor:
    if backend == "fpint_torch":
        return qcol_real_2scomp_torch(
            activation,
            weight,
            scale,
            zero,
            config,
            bias=bias,
            validate_values=validate_values,
            has_zero=has_zero,
        )
    if backend == "standard":
        return torch.nn.functional.linear(
            activation,
            dequantize_weight(weight, scale, zero, config),
            bias,
        )
    if backend == "fpint_cuda":
        from .cuda_backend import qcol_real_2scomp_cuda

        return qcol_real_2scomp_cuda(
            activation,
            weight,
            scale,
            zero,
            config,
            bias=bias,
            validate_values=validate_values,
            has_zero=has_zero,
        )
    raise ValueError(f"unsupported FPINT linear backend: {backend!r}")
