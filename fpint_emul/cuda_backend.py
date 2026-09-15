import os
from pathlib import Path

import torch

from .config import FpIntConfig
from .torch_backend import _validate_inputs


_EXTENSION = None


def _load_extension():
    global _EXTENSION
    if _EXTENSION is None:
        from torch.utils.cpp_extension import load

        source_dir = Path(__file__).parent / "csrc"
        _EXTENSION = load(
            name="spinquant_fpint_cuda",
            sources=[
                str(source_dir / "fpint_cuda.cpp"),
                str(source_dir / "fpint_cuda_kernel.cu"),
            ],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=os.environ.get("SPINQUANT_FPINT_BUILD_VERBOSE", "0") == "1",
        )
    return _EXTENSION


@torch.no_grad()
def qcol_real_2scomp_cuda(
    activation: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    zero: torch.Tensor,
    config: FpIntConfig,
    bias: torch.Tensor = None,
    *,
    validate_values: bool = True,
    has_zero=None,
) -> torch.Tensor:
    """Run the fused CUDA QCOL kernel on arbitrary leading activation dims."""

    _validate_inputs(
        activation, weight, scale, zero, bias, config, check_values=validate_values
    )
    if not activation.is_cuda:
        raise ValueError("fpint_cuda requires CUDA tensors")
    if config.mxu_rows > 256:
        raise ValueError("fpint_cuda supports mxu_rows <= 256")
    original_shape = activation.shape[:-1]
    k = activation.shape[-1]
    flat = activation.contiguous().reshape(-1, k)
    output = _load_extension().qcol_real_2scomp(
        flat,
        weight.contiguous(),
        scale.contiguous(),
        zero.to(torch.int32).contiguous(),
        config.group_size,
        config.mxu_rows,
        config.extra_bits,
        config.reduce_extra_bits,
        bool(zero.any()) if has_zero is None else has_zero,
    )
    if bias is not None:
        output = output + bias
    return output.reshape(*original_shape, weight.shape[0])
