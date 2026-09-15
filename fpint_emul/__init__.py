"""FP16 x signed-integer hardware emulation.

The legacy verification code historically imported ``fpint_emul.py`` as a
top-level module. Keep those public symbols available while adding the new
generalized Linear API at the package root.
"""

from .py import fpint_emul as _legacy

_legacy_public = [name for name in vars(_legacy) if not name.startswith("_")]
globals().update({name: getattr(_legacy, name) for name in _legacy_public})

from .config import FpIntConfig
from .linear import dequantize_weight, fpint_linear
from .reference import qcol_real_2scomp_reference
from .torch_backend import qcol_real_2scomp_torch

__all__ = _legacy_public + [
    "FpIntConfig",
    "dequantize_weight",
    "fpint_linear",
    "qcol_real_2scomp_reference",
    "qcol_real_2scomp_torch",
]
