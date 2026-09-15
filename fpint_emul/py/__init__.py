"""
FPINT GEMM Emulation Package

Python implementation of FPINT GEMM verification code
ported from SystemVerilog.
"""

from .fpint_emul import (
    # Main functions
    prealign,
    fpint_gemm_ref,
    fpint_gemm_qcol_2scomp,
    fpint_gemm_qrow_2scomp,
    
    # Utility functions
    fp16_to_components,
    fp16_bit_to_float,
    float_to_fp16_bit,
    
    # Classes
    FixedPointArray,
    
    # Constants
    IN_WIDTH,
    W_WIDTH,
    MAX_W_WIDTH,
    O_WIDTH,
    S_WIDTH,
    Z_WIDTH,
    QBLOCK,
    MAX_M,
    MAX_N,
    MAX_K,
    MAX_KG,
    MAX_NG,
    MXU_K,
    MXU_N,
    MAX_ALIGN_WIDTH,
    MAX_EXP_WIDTH,
    POST_RESULT_WIDTH,
    EXTRA_BIT,
    EXTRA_BIT_FOR_REDUCE,
    IN_MAN_WIDTH,
    MAX_EXTRA_WIDTH,
    IN_EXP_BIAS,
    SIGN_WIDTH,
    EXP_WIDTH,
    MANTISSA_WIDTH,
    HIDDEN_WIDTH,
    QCOL,
    QROW,
)

__version__ = "0.1.0"
__all__ = [
    # Functions
    "prealign",
    "fpint_gemm_ref",
    "fpint_gemm_qcol_2scomp",
    "fpint_gemm_qrow_2scomp",
    "fp16_to_components",
    "fp16_bit_to_float",
    "float_to_fp16_bit",
    
    # Classes
    "FixedPointArray",
    
    # Constants
    "IN_WIDTH",
    "W_WIDTH",
    "MAX_W_WIDTH",
    "O_WIDTH",
    "S_WIDTH",
    "Z_WIDTH",
    "QBLOCK",
    "MAX_M",
    "MAX_N",
    "MAX_K",
    "MAX_KG",
    "MAX_NG",
    "MXU_K",
    "MXU_N",
    "MAX_ALIGN_WIDTH",
    "MAX_EXP_WIDTH",
    "POST_RESULT_WIDTH",
    "EXTRA_BIT",
    "EXTRA_BIT_FOR_REDUCE",
    "IN_MAN_WIDTH",
    "MAX_EXTRA_WIDTH",
    "IN_EXP_BIAS",
    "SIGN_WIDTH",
    "EXP_WIDTH",
    "MANTISSA_WIDTH",
    "HIDDEN_WIDTH",
    "QCOL",
    "QROW",
]
