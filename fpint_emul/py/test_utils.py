"""
Test utilities for FPINT GEMM Emulation

This module provides common helper functions for generating test data,
running tests, and visualizing results.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Callable, List, Tuple, Optional, Any, Union
from enum import Enum, IntEnum
from fpint_emul import FpIntImplType
import vsc

from fpint_emul import (
    float_to_fp16_bit, fp16_bit_to_float,
    fpint_gemm_ref, fpint_gemm_gpu,
    fpint_gemm_qcol_2scomp, fpint_gemm_qcol_zero_less, fpint_gemm_qcol_real_2scomp,
    fpint_gemm_qrow_2scomp, fpint_gemm_qrow_zero_less, fpint_gemm_qrow_real_2scomp,
    QCOL, QROW, QBLOCK, MXU_K
)
from visualize import (
    plot_heatmap_comparison, ulp_diff_fp16, relative_diff_percent, absolute_diff
)

# ===============================
# Enums
# ===============================
class ErrorMethod(Enum):
    REL_PERCENT = 'rel_percent'
    ABSOLUTE = 'absolute'
    ULP_FP16 = 'ulp_fp16'

class ErrorType(str, Enum):
    """Error types for FPINT comparison. Inherits from str for direct use as dict keys."""
    REF_VS_GPU = 'err_ref_vs_gpu'
    REF_VS_EMUL = 'err_ref_vs_emul'
    ULP_DIFF = 'err_ulp_diff'
    GPU_VS_EMUL = 'err_gpu_vs_emul'

class TagKey(str, Enum):
    """
    Keys for TestResult.tags dict.

    Inherits from str for easy use as dict keys and in filtering.
    Add new analysis categories here as needed.
    """
    # Input characteristics
    HAS_LARGE_EXP_DIFF = 'has_large_exp_diff'  # bool: exp diff > threshold
    HAS_SUBNORMAL = 'has_subnormal'            # bool: contains subnormal values
    MAX_EXP_DIFF = 'max_exp_diff'              # int: maximum exponent difference

    # Weight characteristics
    HAS_ZERO_WEIGHT = 'has_zero_weight'        # bool: contains zero weights
    HAS_MAX_WEIGHT = 'has_max_weight'          # bool: contains max value weights
    ZERO_WEIGHT_RATIO = 'zero_weight_ratio'    # float: ratio of zero weights

    # Cross characteristics (MXU group level)
    MAX_MEETS_ZERO = 'max_meets_zero'          # bool: max input meets zero weight

    # Sign combinations
    SIGN_PP = 'sign_pp'  # bool: positive input × positive weight
    SIGN_PN = 'sign_pn'  # bool: positive input × negative weight
    SIGN_NP = 'sign_np'  # bool: negative input × positive weight
    SIGN_NN = 'sign_nn'  # bool: negative input × negative weight

# Type alias for store predicate function
# Input: Dict[ErrorType, float] mapping error types to their max values
# Output: bool indicating whether to store data
StorePredicate = Callable[[Dict[ErrorType, float]], bool]

# =============================================================================
# FP16 Constants
# =============================================================================

FP16_EXP_BIAS = 15
FP16_MAX_EXP = 30  # 2^5 - 2 (excluding inf/nan)
FP16_SUBNORMAL_EXP = 0


# =============================================================================
# Data Generation Functions
# =============================================================================

def generate_random_fp16(size, value_range=(-10.0, 10.0), seed=None):
    """Generate random FP16 values as uint16 bit representation"""
    if seed is not None:
        np.random.seed(seed)

    # Generate random floats
    low, high = value_range
    float_vals = np.random.uniform(low, high, size=size)

    # Convert to FP16 bit representation
    fp16_vals = np.zeros(size, dtype=np.uint16)
    flat_float = float_vals.flatten()
    flat_fp16 = fp16_vals.ravel()

    for i, val in enumerate(flat_float):
        flat_fp16[i] = float_to_fp16_bit(val)

    return fp16_vals


def generate_normal_fp16(size, mean=0.0, std=1.0, seed=None):
    """Generate FP16 values from normal distribution"""
    if seed is not None:
        np.random.seed(seed)

    float_vals = np.random.normal(mean, std, size=size)
    fp16_vals = np.zeros(size, dtype=np.uint16)
    flat_float = float_vals.flatten()
    flat_fp16 = fp16_vals.ravel()

    for i, val in enumerate(flat_float):
        flat_fp16[i] = float_to_fp16_bit(val)

    return fp16_vals


def generate_mixed_exp_fp16(size, large_ratio=0.1, exp_diff=10, seed=None):
    """
    Generate FP16 values with mixed exponents (large and small values).

    Args:
        size: Output shape
        large_ratio: Ratio of large values (0.0 ~ 1.0)
        exp_diff: Exponent difference between large and small values
        seed: Random seed

    Returns:
        FP16 bit representation array
    """
    if seed is not None:
        np.random.seed(seed)

    total = np.prod(size)
    n_large = int(total * large_ratio)
    n_small = total - n_large

    # Small values: around 2^0 = 1
    small_vals = np.random.uniform(-1.0, 1.0, n_small)

    # Large values: around 2^exp_diff
    large_scale = 2.0 ** exp_diff
    large_vals = np.random.uniform(-large_scale, large_scale, n_large)

    # Combine and shuffle
    all_vals = np.concatenate([small_vals, large_vals])
    np.random.shuffle(all_vals)

    fp16_vals = np.zeros(total, dtype=np.uint16)
    for i, val in enumerate(all_vals):
        fp16_vals[i] = float_to_fp16_bit(val)

    return fp16_vals.reshape(size)


def generate_mixed_subnormal_fp16(size, subnormal_ratio=0.1, seed=None):
    """
    Generate FP16 values with mixed normal and subnormal numbers.

    Subnormal FP16: |value| < 2^(-14) ≈ 6.1e-5
    """
    if seed is not None:
        np.random.seed(seed)

    total = np.prod(size)
    n_subnormal = int(total * subnormal_ratio)
    n_normal = total - n_subnormal

    # Normal values
    normal_vals = np.random.uniform(-1.0, 1.0, n_normal)

    # Subnormal values: |x| < 2^(-14)
    subnormal_max = 2.0 ** (-14) * 0.99  # slightly less than min normal
    subnormal_vals = np.random.uniform(-subnormal_max, subnormal_max, n_subnormal)

    all_vals = np.concatenate([normal_vals, subnormal_vals])
    np.random.shuffle(all_vals)

    fp16_vals = np.zeros(total, dtype=np.uint16)
    for i, val in enumerate(all_vals):
        fp16_vals[i] = float_to_fp16_bit(val)

    return fp16_vals.reshape(size)


def generate_random_weights(size, w_width=4, seed=None):
    """Generate random quantized weights (0 to 2^w_width - 1)"""
    if seed is not None:
        np.random.seed(seed)

    max_val = (1 << w_width) - 1
    return np.random.randint(0, max_val + 1, size=size, dtype=np.uint8)


def generate_sparse_zero_weights(size, zero_ratio=0.3, w_width=4, seed=None):
    """Generate weights with specified ratio of zeros"""
    if seed is not None:
        np.random.seed(seed)

    max_val = (1 << w_width) - 1
    weights = np.random.randint(0, max_val + 1, size=size, dtype=np.uint8)

    # Set some to zero
    total = np.prod(size)
    n_zeros = int(total * zero_ratio)
    zero_indices = np.random.choice(total, n_zeros, replace=False)
    weights.ravel()[zero_indices] = 0

    return weights


def generate_weights_with_max(size, w_width=4, max_ratio=0.1, seed=None):
    """Generate weights with specified ratio of max values"""
    if seed is not None:
        np.random.seed(seed)

    max_val = (1 << w_width) - 1
    weights = np.random.randint(0, max_val + 1, size=size, dtype=np.uint8)

    # Set some to max
    total = np.prod(size)
    n_max = int(total * max_ratio)
    max_indices = np.random.choice(total, n_max, replace=False)
    weights.ravel()[max_indices] = max_val

    return weights


def generate_random_zero_points(size, z_range=(-8, 8), seed=None):
    """Generate random zero points"""
    if seed is not None:
        np.random.seed(seed)

    low, high = z_range
    return np.random.randint(low, high, size=size, dtype=np.int16)


# =============================================================================
# FP16 Analysis Functions
# =============================================================================

def get_fp16_exp(fp16_bits: int) -> int:
    """Extract exponent from FP16 bit representation"""
    return (fp16_bits >> 10) & 0x1F


def get_fp16_sign(fp16_bits: int) -> int:
    """Extract sign from FP16 bit representation (0=positive, 1=negative)"""
    return (fp16_bits >> 15) & 0x1


def is_subnormal(fp16_bits: int) -> bool:
    """Check if FP16 value is subnormal"""
    return get_fp16_exp(fp16_bits) == 0


def analyze_mxu_group(input_row: np.ndarray, weight_col: np.ndarray, mxu_k: int = MXU_K):
    """
    Analyze characteristics of an MXU_K group for coverage tracking.

    Returns dict with:
        - max_exp_diff: Max exponent difference within the group
        - has_subnormal: Whether group contains subnormal
        - max_input_idx: Index of max |input| in group
        - zero_weight_indices: Indices where weight is zero
        - max_meets_zero: Whether max input is multiplied by zero weight
        - sign_combinations: Set of (input_sign, weight_sign) pairs
    """
    K = len(input_row)
    n_groups = (K + mxu_k - 1) // mxu_k

    results = []
    for g in range(n_groups):
        start = g * mxu_k
        end = min(start + mxu_k, K)

        group_input = input_row[start:end]
        group_weight = weight_col[start:end]

        # Exponent analysis
        exps = [get_fp16_exp(int(x)) for x in group_input]
        max_exp = max(exps)
        min_exp = min(e for e in exps if e > 0) if any(e > 0 for e in exps) else 0
        max_exp_diff = max_exp - min_exp if min_exp > 0 else max_exp

        # Subnormal check
        has_subnormal = any(is_subnormal(int(x)) for x in group_input)

        # Find max |input| index
        float_vals = [abs(fp16_bit_to_float(int(x))) for x in group_input]
        max_input_idx = np.argmax(float_vals)

        # Zero weight indices
        zero_weight_indices = [i for i, w in enumerate(group_weight) if w == 0]

        # Check if max input meets zero weight
        max_meets_zero = max_input_idx in zero_weight_indices

        # Sign combinations
        sign_combinations = set()
        for i in range(len(group_input)):
            inp_sign = get_fp16_sign(int(group_input[i]))
            # Weight sign: assume 2's complement, MSB is sign
            w_val = int(group_weight[i])
            w_sign = 1 if w_val >= 8 else 0  # For 4-bit weight
            sign_combinations.add((inp_sign, w_sign))

        results.append({
            'group_idx': g,
            'max_exp_diff': max_exp_diff,
            'has_subnormal': has_subnormal,
            'max_input_idx': max_input_idx,
            'zero_weight_indices': zero_weight_indices,
            'max_meets_zero': max_meets_zero,
            'sign_combinations': sign_combinations
        })

    return results


# =============================================================================
# Error Function Configurations
# =============================================================================

ERROR_FUNCS = {
    'rel_percent': (relative_diff_percent, 'Relative Error (%)'),
    'absolute': (absolute_diff, 'Absolute Error'),
    'ulp_fp16': (ulp_diff_fp16, 'ULP Error (FP16)'),
}


# =============================================================================
# Helper Functions
# =============================================================================

def to_float_matrix(output: np.ndarray) -> np.ndarray:
    """Convert FP16 bit representation matrix to float matrix"""
    M, N = output.shape
    result = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            result[m, n] = fp16_bit_to_float(output[m, n])
    return result


def print_error_stats(
    name: str,
    output_float: np.ndarray,
    ref_float: np.ndarray,
    err_func: Callable,
    err_name: str
) -> np.ndarray:
    """Print error statistics for a single implementation"""
    error = err_func(ref_float, output_float)

    print(f"\n{name}:")
    print(f"  {err_name} - Max: {np.max(error):.6f}, Mean: {np.mean(error):.6f}, Median: {np.median(error):.6f}")

    return error


# =============================================================================
# Test Result Data Class (Unified)
# =============================================================================

@dataclass
class TestResult:
    """
    Unified container for test results.

    Stores 4 types of errors for comprehensive analysis:
    1. err_ref_vs_gpu: fpint_gemm_ref (FP64) vs fpint_gemm_gpu (FP16 bit-exact)
    2. err_ref_vs_emul: fpint_gemm_ref (FP64) vs fpint_emul implementation
    3. err_ulp_diff: err_ref_vs_gpu - err_ref_vs_emul (emul이 gpu보다 얼마나 정확한지)
    4. err_gpu_vs_emul: fpint_gemm_gpu vs fpint_emul (GPU를 ref로 하여 emul 직접 비교)

    Goal: Verify that fpint_emul is as accurate as (or better than) fpint_gemm_gpu
    compared to the FP64 reference, and understand the direct difference between GPU and emul.
    """
    # === Core fields (required) ===
    name: str                    # Implementation name (e.g., 'qcol_2scomp')
    M: int                       # Input rows
    K: int                       # Input cols / Weight rows
    N: int                       # Weight cols / Output cols
    seed: int                    # Random seed used
    passed: bool                 # Whether test passed threshold
    error_threshold: float       # Threshold used for pass/fail

    # === Four error types ===
    err_ref_vs_gpu: np.ndarray   # ULP error: fpint_gemm_ref vs fpint_gemm_gpu
    err_ref_vs_emul: np.ndarray  # ULP error: fpint_gemm_ref vs fpint_emul
    err_ulp_diff: np.ndarray     # Difference: err_ref_vs_gpu - err_ref_vs_emul
                                 # Positive = emul is MORE accurate than gpu
                                 # Negative = emul is LESS accurate than gpu
    err_gpu_vs_emul: np.ndarray  # ULP error: fpint_gemm_gpu vs fpint_emul
                                 # GPU as reference, direct comparison to emul

    # === Optional fields (for detailed analysis) ===
    qdir: Optional[int] = None   # QCOL or QROW

    # Distribution info (coverage test)
    input_dist: Optional[int] = None   # InputDistType
    weight_dist: Optional[int] = None  # WeightDistType

    # Raw data (for reproducibility)
    input_data: Optional[np.ndarray] = None
    weight_data: Optional[np.ndarray] = None
    scale_data: Optional[np.ndarray] = None
    zero_data: Optional[np.ndarray] = None

    # Output data (uint16 FP16 bits)
    output_ref: Optional[np.ndarray] = None    # fpint_gemm_ref output
    output_gpu: Optional[np.ndarray] = None    # fpint_gemm_gpu output
    output_emul: Optional[np.ndarray] = None   # fpint_emul output

    # Output data (float)
    output_ref_float: Optional[np.ndarray] = None
    output_gpu_float: Optional[np.ndarray] = None
    output_emul_float: Optional[np.ndarray] = None

    # Analysis tags (flexible dict for coverage analysis results)
    # Keys should be TagKey enum values for consistency
    # Values can be bool (flags) or numeric (measurements)
    tags: Dict[TagKey, Any] = field(default_factory=dict)

    # === Computed properties for err_ref_vs_gpu ===
    @property
    def gpu_max_error(self) -> float:
        """Max ULP error: ref vs gpu"""
        return float(np.max(self.err_ref_vs_gpu))

    @property
    def gpu_mean_error(self) -> float:
        """Mean ULP error: ref vs gpu"""
        return float(np.mean(self.err_ref_vs_gpu))
    
    def gpu_error_percentile(self, percentile:List[float]=[90.0, 95.0, 99.0]) -> List[float]:
        """Percentile ULP error: ref vs gpu"""
        return np.percentile(self.err_ref_vs_gpu, percentile).tolist()

    # === Computed properties for err_ref_vs_emul ===
    @property
    def emul_max_error(self) -> float:
        """Max ULP error: ref vs emul"""
        return float(np.max(self.err_ref_vs_emul))

    @property
    def emul_mean_error(self) -> float:
        """Mean ULP error: ref vs emul"""
        return float(np.mean(self.err_ref_vs_emul))

    def emul_error_percentile(self, percentile:List[float]=[90.0, 95.0, 99.0]) -> List[float]:
        """Percentile ULP error: ref vs emul"""
        return np.percentile(self.err_ref_vs_emul, percentile).tolist()

    # === Computed properties for err_ulp_diff (accuracy comparison) ===
    @property
    def ulp_diff_max(self) -> float:
        """Max ULP diff (positive = emul better)"""
        return float(np.max(self.err_ulp_diff))

    @property
    def ulp_diff_min(self) -> float:
        """Min ULP diff (negative = emul worse)"""
        return float(np.min(self.err_ulp_diff))

    @property
    def ulp_diff_mean(self) -> float:
        """Mean ULP diff"""
        return float(np.mean(self.err_ulp_diff))

    def ulp_diff_percentile(self, percentile:List[float]=[90.0, 95.0, 99.0]) -> List[float]:
        """Percentile ULP diff (positive = emul better)"""
        return np.percentile(self.err_ulp_diff, percentile).tolist()

    @property
    def emul_better_count(self) -> int:
        """Count of positions where emul is more accurate than gpu"""
        return int(np.sum(self.err_ulp_diff > 0))

    @property
    def emul_worse_count(self) -> int:
        """Count of positions where emul is less accurate than gpu"""
        return int(np.sum(self.err_ulp_diff < 0))

    @property
    def emul_same_count(self) -> int:
        """Count of positions where emul has same accuracy as gpu"""
        return int(np.sum(self.err_ulp_diff == 0))

    # === Computed properties for err_gpu_vs_emul (direct GPU vs Emul comparison) ===
    @property
    def gpu_vs_emul_max_error(self) -> float:
        """Max ULP error: gpu vs emul (GPU as reference)"""
        return float(np.max(self.err_gpu_vs_emul))

    @property
    def gpu_vs_emul_mean_error(self) -> float:
        """Mean ULP error: gpu vs emul (GPU as reference)"""
        return float(np.mean(self.err_gpu_vs_emul))

    def gpu_vs_emul_error_percentile(self, percentile:List[float]=[90.0, 95.0, 99.0]) -> List[float]:
        """Percentile ULP error: gpu vs emul (GPU as reference)"""
        return np.percentile(self.err_gpu_vs_emul, percentile).tolist()

    @property
    def gpu_emul_same_count(self) -> int:
        """Count of positions where gpu and emul produce exactly same result"""
        return int(np.sum(self.err_gpu_vs_emul == 0))

    @property
    def gpu_emul_same_pct(self) -> float:
        """Percentage of positions where gpu and emul produce exactly same result"""
        total = self.M * self.N
        return 100 * self.gpu_emul_same_count / total

    # === Legacy property for compatibility (uses err_ref_vs_emul) ===
    @property
    def error(self) -> np.ndarray:
        """Legacy: returns err_ref_vs_emul for backward compatibility"""
        return self.err_ref_vs_emul

    @property
    def max_error(self) -> float:
        """Legacy: returns emul_max_error"""
        return self.emul_max_error

    @property
    def mean_error(self) -> float:
        """Legacy: returns emul_mean_error"""
        return self.emul_mean_error

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        total = self.M * self.N
        better_pct = 100 * self.emul_better_count / total
        return (f"TestResult({status}, {self.name}, M={self.M}, K={self.K}, N={self.N}, "
                f"gpu_err={self.gpu_max_error:.1f}, emul_err={self.emul_max_error:.1f}, "
                f"gpu_vs_emul={self.gpu_vs_emul_max_error:.1f}, same={self.gpu_emul_same_pct:.0f}%)")

    def summary(self) -> Dict[str, Any]:
        """Return summary dict for easy inspection"""
        total = self.M * self.N
        return {
            'name': self.name,
            'M': self.M, 'K': self.K, 'N': self.N,
            'seed': self.seed,
            'passed': self.passed,
            'error_threshold': self.error_threshold,
            # GPU error stats (ref vs gpu)
            'gpu_max_error': self.gpu_max_error,
            'gpu_mean_error': self.gpu_mean_error,
            # Emul error stats (ref vs emul)
            'emul_max_error': self.emul_max_error,
            'emul_mean_error': self.emul_mean_error,
            # Comparison stats (ulp diff)
            'ulp_diff_max': self.ulp_diff_max,
            'ulp_diff_min': self.ulp_diff_min,
            'ulp_diff_mean': self.ulp_diff_mean,
            'emul_better_count': self.emul_better_count,
            'emul_worse_count': self.emul_worse_count,
            'emul_same_count': self.emul_same_count,
            'emul_better_pct': 100 * self.emul_better_count / total,
            # GPU vs Emul direct comparison
            'gpu_vs_emul_max_error': self.gpu_vs_emul_max_error,
            'gpu_vs_emul_mean_error': self.gpu_vs_emul_mean_error,
            'gpu_emul_same_count': self.gpu_emul_same_count,
            'gpu_emul_same_pct': self.gpu_emul_same_pct,
            # Distribution info
            'input_dist': self.input_dist,
            'weight_dist': self.weight_dist,
        }

    def print_comparison(self):
        """Print detailed comparison between gpu and emul accuracy"""
        total = self.M * self.N
        print(f"{'='*60}")
        print(f"Accuracy Comparison: {self.name}")
        print(f"{'='*60}")
        print(f"Dimensions: M={self.M}, K={self.K}, N={self.N} (total {total} elements)")
        print(f"\n1. GPU Error (ref vs gpu):")
        print(f"   Max: {self.gpu_max_error:.2f} ULP, Mean: {self.gpu_mean_error:.2f} ULP")
        print(f"\n2. Emul Error (ref vs emul):")
        print(f"   Max: {self.emul_max_error:.2f} ULP, Mean: {self.emul_mean_error:.2f} ULP")
        print(f"\n3. Accuracy Comparison (gpu_err - emul_err):")
        print(f"   Max diff: {self.ulp_diff_max:.2f}, Min diff: {self.ulp_diff_min:.2f}")
        print(f"   Mean diff: {self.ulp_diff_mean:.2f}")
        print(f"\n   Emul BETTER than GPU: {self.emul_better_count} ({100*self.emul_better_count/total:.1f}%)")
        print(f"   Emul WORSE than GPU:  {self.emul_worse_count} ({100*self.emul_worse_count/total:.1f}%)")
        print(f"   Emul SAME as GPU:     {self.emul_same_count} ({100*self.emul_same_count/total:.1f}%)")
        print(f"\n4. GPU vs Emul Direct Comparison (gpu as reference):")
        print(f"   Max: {self.gpu_vs_emul_max_error:.2f} ULP, Mean: {self.gpu_vs_emul_mean_error:.2f} ULP")
        print(f"   Exact match: {self.gpu_emul_same_count} ({self.gpu_emul_same_pct:.1f}%)")
        print(f"{'='*60}")


# =============================================================================
# Test Result Analyzer
# =============================================================================

class TestResultAnalyzer:
    """
    Analyzer for a collection of TestResult objects.

    Provides filtering, grouping, statistics, and visualization.

    Example:
        >>> results = [result1, result2, ...]
        >>> analyzer = TestResultAnalyzer(results)
        >>> analyzer.summary()
        >>> analyzer.by_impl()
        >>> analyzer.worst_cases(5)
        >>> analyzer.plot_summary()
    """

    def __init__(self, results: List[TestResult]):
        self.results = results

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, idx) -> TestResult:
        return self.results[idx]

    # -------------------------------------------------------------------------
    # Summary Statistics
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Get overall summary statistics"""
        if not self.results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0}

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        # GPU error stats
        all_gpu_max = [r.gpu_max_error for r in self.results]
        all_gpu_mean = [r.gpu_mean_error for r in self.results]

        # Emul error stats
        all_emul_max = [r.emul_max_error for r in self.results]
        all_emul_mean = [r.emul_mean_error for r in self.results]

        # GPU vs Emul direct comparison stats
        all_gpu_vs_emul_max = [r.gpu_vs_emul_max_error for r in self.results]
        all_gpu_vs_emul_mean = [r.gpu_vs_emul_mean_error for r in self.results]

        # Comparison stats
        total_elements = sum(r.M * r.N for r in self.results)
        total_better = sum(r.emul_better_count for r in self.results)
        total_worse = sum(r.emul_worse_count for r in self.results)
        total_same = sum(r.emul_same_count for r in self.results)
        total_gpu_emul_same = sum(r.gpu_emul_same_count for r in self.results)

        return {
            'total': len(self.results),
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / len(self.results),
            # GPU stats
            'gpu_max_error_worst': max(all_gpu_max),
            'gpu_max_error_avg': np.mean(all_gpu_max),
            'gpu_mean_error_avg': np.mean(all_gpu_mean),
            # Emul stats
            'emul_max_error_worst': max(all_emul_max),
            'emul_max_error_avg': np.mean(all_emul_max),
            'emul_mean_error_avg': np.mean(all_emul_mean),
            # GPU vs Emul direct comparison
            'gpu_vs_emul_max_error_worst': max(all_gpu_vs_emul_max),
            'gpu_vs_emul_max_error_avg': np.mean(all_gpu_vs_emul_max),
            'gpu_vs_emul_mean_error_avg': np.mean(all_gpu_vs_emul_mean),
            'gpu_emul_same_total': total_gpu_emul_same,
            'gpu_emul_same_pct': 100 * total_gpu_emul_same / total_elements if total_elements > 0 else 0,
            # Comparison
            'total_elements': total_elements,
            'emul_better_total': total_better,
            'emul_worse_total': total_worse,
            'emul_same_total': total_same,
            'emul_better_pct': 100 * total_better / total_elements if total_elements > 0 else 0,
        }

    def print_summary(self):
        """Print formatted summary"""
        s = self.summary()
        print("=" * 70)
        print(f"Test Results Summary (GPU vs Emul Accuracy Comparison)")
        print("=" * 70)
        print(f"  Total tests: {s['total']}")
        print(f"  Passed: {s['passed']} ({s['pass_rate']*100:.1f}%)")
        print(f"  Failed: {s['failed']}")
        print(f"\n  GPU Error (ref vs gpu):")
        print(f"    Worst max: {s['gpu_max_error_worst']:.2f} ULP")
        print(f"    Avg max:   {s['gpu_max_error_avg']:.2f} ULP")
        print(f"\n  Emul Error (ref vs emul):")
        print(f"    Worst max: {s['emul_max_error_worst']:.2f} ULP")
        print(f"    Avg max:   {s['emul_max_error_avg']:.2f} ULP")
        print(f"\n  GPU vs Emul Direct (gpu as reference):")
        print(f"    Worst max: {s['gpu_vs_emul_max_error_worst']:.2f} ULP")
        print(f"    Avg max:   {s['gpu_vs_emul_max_error_avg']:.2f} ULP")
        print(f"    Exact match: {s['gpu_emul_same_total']} ({s['gpu_emul_same_pct']:.1f}%)")
        print(f"\n  Accuracy Comparison ({s['total_elements']} total elements):")
        print(f"    Emul BETTER than GPU: {s['emul_better_total']} ({s['emul_better_pct']:.1f}%)")
        print(f"    Emul WORSE than GPU:  {s['emul_worse_total']} ({100*s['emul_worse_total']/s['total_elements']:.1f}%)")
        print(f"    Emul SAME as GPU:     {s['emul_same_total']} ({100*s['emul_same_total']/s['total_elements']:.1f}%)")
        print("=" * 70)

    # -------------------------------------------------------------------------
    # Grouping Methods
    # -------------------------------------------------------------------------

    def group_by(
        self,
        key: Union[str, Callable[[TestResult], Any]],
        skip_none: bool = True,
        sort_keys: bool = False
    ) -> Dict[Any, 'TestResultAnalyzer']:
        """
        Group results by arbitrary key.

        Args:
            key: Attribute name (str) or function that extracts grouping key from TestResult
            skip_none: If True, skip results where key value is None
            sort_keys: If True, sort the result dict by keys

        Returns:
            Dict mapping key values to TestResultAnalyzer for each group

        Examples:
            >>> analyzer.group_by('name')  # by attribute
            >>> analyzer.group_by(lambda r: r.M // 16)  # by custom function
            >>> analyzer.group_by('K', sort_keys=True)  # sorted by K
        """
        # Convert string key to attribute accessor
        if isinstance(key, str):
            attr_name = key
            key_func = lambda r: getattr(r, attr_name)
        else:
            key_func = key

        groups: Dict[Any, List[TestResult]] = {}
        for r in self.results:
            k = key_func(r)
            if skip_none and k is None:
                continue
            if k not in groups:
                groups[k] = []
            groups[k].append(r)

        result = {k: TestResultAnalyzer(v) for k, v in groups.items()}
        if sort_keys:
            result = dict(sorted(result.items()))
        return result

    # Predefined grouping methods (use group_by internally)
    def by_impl(self) -> Dict[str, 'TestResultAnalyzer']:
        """Group results by implementation name"""
        return self.group_by('name')

    def by_size(self, key: str = 'K') -> Dict[int, 'TestResultAnalyzer']:
        """Group results by matrix dimension (M, K, or N)"""
        return self.group_by(key, sort_keys=True)

    def by_input_dist(self) -> Dict[int, 'TestResultAnalyzer']:
        """Group results by input distribution type"""
        return self.group_by('input_dist', sort_keys=True)

    def by_weight_dist(self) -> Dict[int, 'TestResultAnalyzer']:
        """Group results by weight distribution type"""
        return self.group_by('weight_dist', sort_keys=True)

    # -------------------------------------------------------------------------
    # Filtering Methods
    # -------------------------------------------------------------------------

    def filter(self, condition: Callable[[TestResult], bool]) -> 'TestResultAnalyzer':
        """Filter results by arbitrary condition"""
        return TestResultAnalyzer([r for r in self.results if condition(r)])

    def passed_only(self) -> 'TestResultAnalyzer':
        """Get only passed results"""
        return self.filter(lambda r: r.passed)

    def failed_only(self) -> 'TestResultAnalyzer':
        """Get only failed results"""
        return self.filter(lambda r: not r.passed)

    def worst_cases(self, n: int = 10) -> List[TestResult]:
        """Get top N worst error cases"""
        return sorted(self.results, key=lambda r: r.max_error, reverse=True)[:n]

    def best_cases(self, n: int = 10) -> List[TestResult]:
        """Get top N best (lowest error) cases"""
        return sorted(self.results, key=lambda r: r.max_error)[:n]

    # -------------------------------------------------------------------------
    # Statistics Methods
    # -------------------------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        """
        Compute standardized statistics for this analyzer's results.

        Returns:
            Dict with standardized keys:
            - count: number of results
            - pass_rate: ratio of passed tests
            - gpu_max_error_avg, gpu_mean_error_avg: GPU error averages
            - gpu_error_percentiles: list of {percentile, value} for GPU errors
            - emul_max_error_avg, emul_mean_error_avg: Emul error averages
            - emul_error_percentiles: list of {percentile, value} for Emul errors
            - gpu_vs_emul_max_error_avg: direct comparison stat
            - gpu_vs_emul_mean_error_avg: direct comparison stat
        """
        percentile_boundaries = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 99.9, 99.99]
        if not self.results:
            return {
                'count': 0,
                'pass_rate': 0.0,
                'gpu_max_error_avg': 0.0,
                'gpu_mean_error_avg': 0.0,
                'gpu_error_percentiles': [],
                'emul_max_error_avg': 0.0,
                'emul_mean_error_avg': 0.0,
                'emul_error_percentiles': [],
                'gpu_vs_emul_max_error_avg': 0.0,
                'gpu_vs_emul_mean_error_avg': 0.0,
            }

        # Collect error values
        gpu_max_errors = [r.gpu_max_error for r in self.results]
        gpu_mean_errors = [r.gpu_mean_error for r in self.results]
        emul_max_errors = [r.emul_max_error for r in self.results]
        emul_mean_errors = [r.emul_mean_error for r in self.results]
        gpu_vs_emul_max_errors = [r.gpu_vs_emul_max_error for r in self.results]
        gpu_vs_emul_mean_errors = [r.gpu_vs_emul_mean_error for r in self.results]

        # Compute percentiles
        gpu_error_percentiles = [
            {'percentile': p, 'value': float(np.mean([np.percentile(r.err_ref_vs_gpu, p) for r in self.results]) )}
            for p in percentile_boundaries
        ]
        emul_error_percentiles = [
            {'percentile': p, 'value': float(np.mean([np.percentile(r.err_ref_vs_emul, p) for r in self.results]) )}
            for p in percentile_boundaries
        ]

        return {
            'count': len(self.results),
            'pass_rate': sum(1 for r in self.results if r.passed) / len(self.results),
            'gpu_max_error_avg': float(np.mean(gpu_max_errors)),
            'gpu_mean_error_avg': float(np.mean(gpu_mean_errors)),
            'gpu_error_percentiles': gpu_error_percentiles,
            'emul_max_error_avg': float(np.mean(emul_max_errors)),
            'emul_mean_error_avg': float(np.mean(emul_mean_errors)),
            'emul_error_percentiles': emul_error_percentiles,
            'gpu_vs_emul_max_error_avg': float(np.mean(gpu_vs_emul_max_errors)),
            'gpu_vs_emul_mean_error_avg': float(np.mean(gpu_vs_emul_mean_errors)),
        }

    def stats_by(
        self,
        key: Union[str, Callable[[TestResult], Any]],
        sort_keys: bool = False
    ) -> Dict[Any, Dict[str, Any]]:
        """
        Group by key and compute stats for each group.

        Args:
            key: Attribute name (str) or function that extracts grouping key
            sort_keys: If True, sort result dict by keys

        Returns:
            Dict mapping group keys to their stats dicts (standardized keys)

        Examples:
            >>> analyzer.stats_by('name')  # stats by implementation
            >>> analyzer.stats_by('K', sort_keys=True)  # stats by K dimension
            >>> analyzer.stats_by(lambda r: r.input_dist)  # stats by input distribution
        """
        groups = self.group_by(key, sort_keys=sort_keys)
        return {k: v.stats() for k, v in groups.items()}

    # Predefined stats methods (use stats_by with post-processing)
    def stats_by_impl(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics grouped by implementation"""
        return self.stats_by('name')

    def stats_by_input_dist(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics grouped by input distribution (with human-readable names)"""
        result = self.stats_by('input_dist', sort_keys=True)
        # Post-process: add human-readable name using IntEnum
        for dist, stats in result.items():
            try:
                stats['name'] = InputDistType(dist).label
            except ValueError:
                stats['name'] = f'dist_{dist}'
        return result

    def stats_by_weight_dist(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics grouped by weight distribution (with human-readable names)"""
        result = self.stats_by('weight_dist', sort_keys=True)
        # Post-process: add human-readable name using IntEnum
        for dist, stats in result.items():
            try:
                stats['name'] = WeightDistType(dist).label
            except ValueError:
                stats['name'] = f'dist_{dist}'
        return result

    # -------------------------------------------------------------------------
    # Visualization Methods
    # -------------------------------------------------------------------------

    def plot_by(
        self,
        key: Union[str, Callable[[TestResult], Any]],
        plot_type: str = 'boxplot',
        labels: Optional[Dict[Any, str]] = None,
        title: Optional[str] = None,
        ylabel: str = 'Max ULP Error',
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot error distribution grouped by arbitrary key.

        Args:
            key: Attribute name (str) or function that extracts grouping key
            plot_type: 'boxplot', 'violin', 'bar' (mean with std), or 'hist'
            labels: Optional dict mapping group keys to display labels
            title: Plot title (auto-generated if None)
            ylabel: Y-axis label
            figsize: Figure size

        Returns:
            matplotlib Figure

        Examples:
            >>> analyzer.plot_by('name')  # boxplot by implementation
            >>> analyzer.plot_by('K', plot_type='bar', title='Error by K')
            >>> analyzer.plot_by('input_dist', labels={0: 'normal', 1: 'mixed_exp', 2: 'mixed_subnormal'})
        """
        groups = self.group_by(key, sort_keys=True)

        if not groups:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No data to display', ha='center', va='center', transform=ax.transAxes)
            return fig

        # Prepare data
        group_keys = list(groups.keys())
        group_labels = [labels.get(k, str(k)) if labels else str(k) for k in group_keys]
        group_errors = [[r.max_error for r in groups[k].results] for k in group_keys]

        fig, ax = plt.subplots(figsize=figsize)

        if plot_type == 'boxplot':
            ax.boxplot(group_errors, labels=group_labels)
            ax.set_xticklabels(group_labels, rotation=45, ha='right')

        elif plot_type == 'violin':
            parts = ax.violinplot(group_errors, positions=range(len(group_keys)), showmeans=True, showmedians=True)
            ax.set_xticks(range(len(group_keys)))
            ax.set_xticklabels(group_labels, rotation=45, ha='right')

        elif plot_type == 'bar':
            stats = {k: groups[k].stats() for k in group_keys}
            means = [stats[k]['emul_max_error_avg'] for k in group_keys]
            x = np.arange(len(group_keys))
            ax.bar(x, means, capsize=5, alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(group_labels, rotation=45, ha='right')

        elif plot_type == 'hist':
            for k, label in zip(group_keys, group_labels):
                errors = [r.max_error for r in groups[k].results]
                ax.hist(errors, bins=20, alpha=0.5, label=label)
            ax.legend()

        else:
            raise ValueError(f"Unknown plot_type: {plot_type}. Use 'boxplot', 'violin', 'bar', or 'hist'")

        # Set labels and title
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        else:
            key_name = key if isinstance(key, str) else 'custom'
            ax.set_title(f'Error Distribution by {key_name}')

        plt.tight_layout()
        return fig

    def plot_summary(self, figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
        """Plot comprehensive summary of results"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1. Pass/Fail by implementation
        ax = axes[0, 0]
        impl_stats = self.stats_by_impl()
        impl_names = list(impl_stats.keys())
        pass_counts = [sum(1 for r in self.by_impl()[name].results if r.passed) for name in impl_names]
        fail_counts = [sum(1 for r in self.by_impl()[name].results if not r.passed) for name in impl_names]

        x = np.arange(len(impl_names))
        width = 0.35
        ax.bar(x - width/2, pass_counts, width, label='Pass', color='green', alpha=0.7)
        ax.bar(x + width/2, fail_counts, width, label='Fail', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(impl_names, rotation=45, ha='right')
        ax.set_ylabel('Count')
        ax.set_title('Pass/Fail by Implementation')
        ax.legend()

        # 2. Max error distribution
        ax = axes[0, 1]
        for name in impl_names:
            errors = [r.max_error for r in self.by_impl()[name].results]
            ax.hist(errors, bins=20, alpha=0.5, label=name)
        ax.set_xlabel('Max Error')
        ax.set_ylabel('Count')
        ax.set_title('Max Error Distribution')
        ax.legend(fontsize=8)

        # 3. Error by input distribution type
        ax = axes[1, 0]
        dist_stats = self.stats_by_input_dist()
        if dist_stats:
            dist_names = [dist_stats[d]['name'] for d in sorted(dist_stats.keys())]
            for i, dist in enumerate(sorted(dist_stats.keys())):
                errors = [r.max_error for r in self.by_input_dist()[dist].results]
                if errors:
                    ax.boxplot([errors], positions=[i], widths=0.6)
            ax.set_xticks(range(len(dist_names)))
            ax.set_xticklabels(dist_names)
            ax.set_xlabel('Input Distribution')
            ax.set_ylabel('Max Error')
            ax.set_title('Error by Input Distribution')
        else:
            ax.text(0.5, 0.5, 'No distribution info', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Error by Input Distribution (N/A)')

        # 4. Error statistics by implementation
        ax = axes[1, 1]
        means = [impl_stats[name]['emul_max_error_avg'] for name in impl_names]
        ax.bar(impl_names, means, capsize=5, alpha=0.7)
        ax.set_ylabel('Max Error (mean)')
        ax.set_title('Error Statistics by Implementation')
        ax.set_xticklabels(impl_names, rotation=45, ha='right')

        summary = self.summary()
        fig.suptitle(f'Test Results: {summary["passed"]}/{summary["total"]} passed ({summary["pass_rate"]*100:.1f}%)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_error_heatmap(self, impl_name: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Plot error heatmap for worst cases"""
        if impl_name:
            results = self.by_impl().get(impl_name, TestResultAnalyzer([])).results
        else:
            results = self.results

        worst = sorted(results, key=lambda r: r.max_error, reverse=True)[:6]

        n_plots = min(len(worst), 6)
        if n_plots == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center')
            return fig

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        axes_flat = axes.flatten()

        for idx, result in enumerate(worst):
            ax = axes_flat[idx]
            im = ax.imshow(result.error, aspect='auto', cmap='hot')
            ax.set_title(f'{result.name}\nmax={result.max_error:.2f}, mean={result.mean_error:.2f}')
            plt.colorbar(im, ax=ax)

        for idx in range(n_plots, 6):
            axes_flat[idx].axis('off')

        fig.suptitle('Worst Case Error Heatmaps', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_error_histogram(self, figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
        """Plot histogram of errors for all implementations"""
        impl_groups = self.by_impl()
        n_impls = len(impl_groups)

        if n_impls == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center')
            return fig

        cols = min(3, n_impls)
        rows = (n_impls + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize)

        if n_impls == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)

        axes_flat = axes.flatten()

        for idx, (name, analyzer) in enumerate(impl_groups.items()):
            ax = axes_flat[idx]
            all_errors = np.concatenate([r.error.flatten() for r in analyzer.results])

            ax.hist(all_errors, bins=50, edgecolor='black', alpha=0.7)
            ax.axvline(np.mean(all_errors), color='r', linestyle='--', label=f'Mean={np.mean(all_errors):.2f}')
            ax.axvline(np.median(all_errors), color='g', linestyle='--', label=f'Median={np.median(all_errors):.2f}')
            ax.set_title(f'{name}')
            ax.set_xlabel('Error')
            ax.set_ylabel('Count')
            ax.legend(fontsize=8)

        for idx in range(len(impl_groups), len(axes_flat)):
            axes_flat[idx].axis('off')

        fig.suptitle('Error Distribution by Implementation', fontsize=14, fontweight='bold')
        plt.tight_layout()

        return fig

    def plot_error_percentile(
        self,
        impl_types: List[FpIntImplType] = [FpIntImplType.QCOL_2SCOMP, FpIntImplType.QROW_2SCOMP],
        error_types: List[ErrorType] = [ErrorType.REF_VS_GPU, ErrorType.REF_VS_EMUL, ErrorType.ULP_DIFF, ErrorType.GPU_VS_EMUL],
        percentiles: List[float] = [90.0, 95.0, 99.0],
        figsize: Tuple[int, int] = (14, 5)
    ) -> plt.Figure:
        """
        Plot error percentiles for specified error types, separated by impl_type.

        Args:
            impl_types: List of FpIntImplType to include (one subplot per impl_type)
            error_types: List of ErrorType to plot (x-axis)
            percentiles: List of percentile values to compute (default: [90, 95, 99])
            figsize: Figure size

        Returns:
            matplotlib Figure

        Example:
            >>> analyzer.plot_error_percentile(
            ...     impl_types=[FpIntImplType.QCOL_2SCOMP, FpIntImplType.QROW_2SCOMP],
            ...     error_types=[ErrorType.REF_VS_GPU, ErrorType.REF_VS_EMUL],
            ...     percentiles=[90.0, 95.0, 99.0]
            ... )
        """
        if not self.results:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center')
            return fig

        # Map ErrorType to percentile method
        percentile_methods = {
            ErrorType.REF_VS_GPU: lambda r, p: r.gpu_error_percentile(p),
            ErrorType.REF_VS_EMUL: lambda r, p: r.emul_error_percentile(p),
            ErrorType.ULP_DIFF: lambda r, p: r.ulp_diff_percentile(p),
            ErrorType.GPU_VS_EMUL: lambda r, p: r.gpu_vs_emul_error_percentile(p),
        }

        # Create subplots: one per impl_type
        n_impl = len(impl_types)
        cols = min(n_impl, 3)
        rows = (n_impl + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
        axes_flat = axes.flatten()

        # Colors for percentiles
        n_percentiles = len(percentiles)
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_percentiles))

        for impl_idx, impl_type in enumerate(impl_types):
            ax = axes_flat[impl_idx]

            # Filter results for this impl_type
            impl_results = [r for r in self.results if r.name == impl_type.value]

            if not impl_results:
                ax.text(0.5, 0.5, f'No results for {impl_type.value}', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(impl_type.value)
                continue

            # Collect percentile values for this impl_type
            data = {et: {p: [] for p in percentiles} for et in error_types}
            for result in impl_results:
                for et in error_types:
                    pct_values = percentile_methods[et](result, percentiles)
                    for i, p in enumerate(percentiles):
                        data[et][p].append(pct_values[i])

            # Compute mean percentile
            mean_data = {et: {p: np.mean(data[et][p]) for p in percentiles} for et in error_types}

            # Build heatmap matrix: rows=percentiles, cols=error_types
            heatmap_data = np.zeros((n_percentiles, len(error_types)))
            for i, p in enumerate(percentiles):
                for j, et in enumerate(error_types):
                    heatmap_data[i, j] = mean_data[et][p]

            # Plot heatmap
            im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')

            # Add text annotations
            for i in range(n_percentiles):
                for j in range(len(error_types)):
                    val = heatmap_data[i, j]
                    # Choose text color based on background brightness
                    text_color = 'white' if val > np.max(heatmap_data) * 0.6 else 'black'
                    ax.text(j, i, f'{val:.1f}', ha='center', va='center', fontsize=8, color=text_color)

            # Set labels
            ax.set_xticks(np.arange(len(error_types)))
            ax.set_xticklabels([et.name for et in error_types], rotation=30, ha='right', fontsize=8)
            ax.set_yticks(np.arange(n_percentiles))
            y_labels = [f'P{int(p)}' if p == int(p) else f'P{p}' for p in percentiles]
            ax.set_yticklabels(y_labels, fontsize=8)
            ax.set_xlabel('Error Type')
            ax.set_ylabel('Percentile')
            ax.set_title(f'{impl_type.value} (n={len(impl_results)})')

        # Hide unused subplots
        for idx in range(n_impl, len(axes_flat)):
            axes_flat[idx].axis('off')

        fig.suptitle('Error Percentiles by Implementation', fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Add colorbar after tight_layout, with padding for title and colorbar
        fig.subplots_adjust(top=0.88, right=0.88)
        cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.65])  # [left, bottom, width, height]
        fig.colorbar(im, cax=cbar_ax, label='ULP Error')

        return fig

# # =============================================================================
# # Single Test Function
# # =============================================================================

# def run_single_test(
#     name: str,
#     impl_func: Callable,
#     qdir: int,
#     M: int = 8,
#     K: int = 32,
#     N: int = 32,
#     seed: int = 200,
#     err_func: Callable = None,
#     err_name: str = 'ULP Error (FP16)',
#     error_threshold: float = 500.0,
#     make_weight_odd: bool = False,
#     visualize: bool = False,
#     debug: bool = False
# ) -> TestResult:
#     """Generic test function for any FPINT GEMM implementation."""
#     if err_func is None:
#         err_func = ulp_diff_fp16

#     print("=" * 60)
#     print(f"Test: {name} (err={err_name})")
#     print("=" * 60)

#     if qdir == QCOL:
#         KG = K // QBLOCK
#         scale_shape = (KG, N)
#         zero_shape = (KG, N)
#         print(f"Dimensions: M={M}, K={K}, N={N}, KG={KG}")
#     else:
#         NG = (N + QBLOCK - 1) // QBLOCK
#         scale_shape = (K, NG)
#         zero_shape = (K, NG)
#         print(f"Dimensions: M={M}, K={K}, N={N}, NG={NG}")

#     input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
#     weight_data = generate_random_weights((K, N), w_width=4, seed=seed+1)
#     if make_weight_odd:
#         weight_data = weight_data * 2 + 1
#     scale_data = generate_random_fp16(scale_shape, value_range=(0.1, 1.0), seed=seed+2)
#     zero_data = generate_random_zero_points(zero_shape, z_range=(-4, 4), seed=seed+3)

#     output = impl_func(input_data, weight_data, scale_data, zero_data, M, N, K, debug=debug)
#     output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
#                                 M, N, K, qdir=qdir, debug=debug)

#     output_float = to_float_matrix(output)
#     ref_float = to_float_matrix(output_ref)

#     error = print_error_stats(f"{name} vs ref", output_float, ref_float, err_func, err_name)

#     max_err = float(np.max(error))
#     passed = max_err <= error_threshold

#     if visualize:
#         plot_heatmap_comparison(
#             ref_float, {name: output_float},
#             diff_func=err_func, diff_func_name=err_name,
#             title=f"{name} vs Reference)",
#             show=True, auto_scale=True, shared_scale=False
#         )

#     status = "PASSED" if passed else "FAILED"
#     print(f"  {status} (max_err={max_err:.2f}, threshold={error_threshold})")

#     return TestResult(
#         name=name, M=M, K=K, N=N, seed=seed,
#         error=error, passed=passed, error_threshold=error_threshold,
#         err_func_name=err_name, qdir=qdir,
#         input_data=input_data, weight_data=weight_data,
#         scale_data=scale_data, zero_data=zero_data,
#         output=output, output_ref=output_ref,
#         output_float=output_float, ref_float=ref_float,
#     )


# # =============================================================================
# # Run All Tests Function
# # =============================================================================

# def run_all_tests(
#     M: int = 8,
#     K: int = 32,
#     N: int = 32,
#     err_func: Callable = None,
#     err_name: str = 'ULP Error (FP16)',
#     visualize: bool = False,
#     debug: bool = False
# ) -> Dict[str, TestResult]:
#     """Run all 5 FPINT GEMM tests and return results."""
#     if err_func is None:
#         err_func = ulp_diff_fp16

#     print("\n" + "=" * 60)
#     print(f"FPINT GEMM INTEGRATION TEST SUITE")
#     print(f"  err_func={err_name}")
#     print("=" * 60)

#     results = {}
#     test_configs = [
#         (FpIntImplType.QCOL_2SCOMP, fpint_gemm_qcol_2scomp, QCOL, 200, False),
#         (FpIntImplType.QCOL_ZERO_LESS, fpint_gemm_qcol_zero_less, QCOL, 210, True),
#         (FpIntImplType.QCOL_REAL_2SCOMP, fpint_gemm_qcol_real_2scomp, QCOL, 250, False),
#         (FpIntImplType.QROW_2SCOMP, fpint_gemm_qrow_2scomp, QROW, 220, False),
#         (FpIntImplType.QROW_ZERO_LESS, fpint_gemm_qrow_zero_less, QROW, 230, True),
#         (FpIntImplType.QROW_REAL_2SCOMP, fpint_gemm_qrow_real_2scomp, QROW, 240, False),
#     ]

#     for impl_type, impl_func, qdir, seed, make_weight_odd in test_configs:
#         results[impl_type] = run_single_test(
#             name=impl_type.value, impl_func=impl_func, qdir=qdir,
#             M=M, K=K, N=N, seed=seed,
#             err_func=err_func, err_name=err_name,
#             make_weight_odd=make_weight_odd, visualize=visualize, debug=debug
#         )

#     print("\n" + "=" * 60)
#     print(f"Error Summary ({err_name})")
#     print("=" * 60)

#     for name, result in results.items():
#         print(f"{name:20s}: Max={np.max(result.error):10.4f}, Mean={np.mean(result.error):10.4f}, Median={np.median(result.error):10.4f}")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED")
#     print("=" * 60)

#     return results


# # =============================================================================
# # Visualization Functions
# # =============================================================================

# def visualize_all_results(
#     results: Dict[str, TestResult],
#     title_suffix: str = "",
#     cmap: str = "hot"
# ) -> plt.Figure:
#     """Plot all errors in one figure."""
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes_flat = axes.flatten()

#     all_errors = [result.error for result in results.values()]
#     global_vmin = min(np.min(e) for e in all_errors)
#     global_vmax = max(np.max(e) for e in all_errors)

#     for idx, (name, result) in enumerate(results.items()):
#         ax = axes_flat[idx]
#         err = result.error

#         im = ax.imshow(err, aspect='auto', cmap=cmap, vmin=global_vmin, vmax=global_vmax)
#         ax.set_title(f'{name}\n{result.err_func_name}: max={np.max(err):.2f}, mean={np.mean(err):.2f}')
#         ax.set_xlabel('Column')
#         ax.set_ylabel('Row')
#         plt.colorbar(im, ax=ax, label=result.err_func_name)

#     axes_flat[-1].axis('off')

#     first_result = next(iter(results.values()))
#     M, K, N = first_result.M, first_result.K, first_result.N
#     err_name = first_result.err_func_name

#     fig.suptitle(f'All FPINT GEMM Implementations vs Reference\n'
#                  f'(M={M}, K={K}, N={N}) - {err_name}{title_suffix}',
#                 fontsize=14, fontweight='bold')
#     plt.tight_layout()

#     return fig


# def plot_error_histogram(
#     results: Dict[str, TestResult],
#     title_suffix: str = ""
# ) -> plt.Figure:
#     """Plot histogram of errors for all implementations."""
#     fig, axes = plt.subplots(2, 3, figsize=(15, 8))
#     axes_flat = axes.flatten()

#     for idx, (name, result) in enumerate(results.items()):
#         ax = axes_flat[idx]
#         err_flat = result.error.flatten()

#         ax.hist(err_flat, bins=50, edgecolor='black', alpha=0.7)
#         ax.axvline(np.mean(err_flat), color='r', linestyle='--', label=f'Mean={np.mean(err_flat):.2f}')
#         ax.axvline(np.median(err_flat), color='g', linestyle='--', label=f'Median={np.median(err_flat):.2f}')
#         ax.set_title(f'{name}')
#         ax.set_xlabel(result.err_func_name)
#         ax.set_ylabel('Count')
#         ax.legend(fontsize=8)

#     axes_flat[-1].axis('off')

#     first_result = next(iter(results.values()))
#     err_name = first_result.err_func_name

#     fig.suptitle(f'{err_name} Distribution{title_suffix}',
#                 fontsize=14, fontweight='bold')
#     plt.tight_layout()

#     return fig

# =============================================================================
# PyVSC Coverage-Driven Testing
# =============================================================================

# -----------------------------------------------------------------------------
# Distribution Types (IntEnum for type safety and easy int/name conversion)
# -----------------------------------------------------------------------------

class InputDistType(IntEnum):
    """
    Input distribution type for data generation.

    Usage:
        InputDistType.NORMAL.name  -> 'NORMAL'
        InputDistType.NORMAL.value -> 0
        InputDistType(0)           -> InputDistType.NORMAL
    """
    NORMAL = 0           # 정규분포
    MIXED_EXP = 1        # 큰값/작은값 혼합 (exp 차이 큰 경우)
    MIXED_SUBNORMAL = 2  # normal/subnormal 혼합

    @property
    def label(self) -> str:
        """Human-readable label for display"""
        return _INPUT_DIST_LABELS[self]


class WeightDistType(IntEnum):
    """
    Weight distribution type.

    - UNIFORM, SPARSE_ZERO, HAS_MAX: For data generation
    - ZERO_AND_MAX, ZERO_ONLY, MAX_ONLY, NONE: From analysis (use from_flags)

    Usage:
        WeightDistType.UNIFORM.name  -> 'UNIFORM'
        WeightDistType.UNIFORM.value -> 0
        WeightDistType.from_flags(has_zero=True, has_max=True) -> ZERO_AND_MAX
    """
    # Generation types
    UNIFORM = 0          # 균일 분포 (생성용)
    SPARSE_ZERO = 1      # zero가 많은 경우 (생성용)
    HAS_MAX = 2          # max 값이 있는 경우 (생성용)

    # Analysis types (from actual data content)
    ZERO_AND_MAX = 3     # zero 있고 max 있고
    ZERO_ONLY = 4        # zero 있고 max 없고
    MAX_ONLY = 5         # zero 없고 max 있고
    NONE = 6             # zero 없고 max 없고

    @classmethod
    def from_flags(cls, has_zero: bool, has_max: bool) -> 'WeightDistType':
        """Determine category from analysis flags"""
        if has_zero and has_max:
            return cls.ZERO_AND_MAX
        elif has_zero:
            return cls.ZERO_ONLY
        elif has_max:
            return cls.MAX_ONLY
        else:
            return cls.NONE

    @property
    def label(self) -> str:
        """Human-readable label for display"""
        return _WEIGHT_DIST_LABELS[self]


# Human-readable labels for display
_INPUT_DIST_LABELS = {
    InputDistType.NORMAL: 'normal',
    InputDistType.MIXED_EXP: 'mixed_exp',
    InputDistType.MIXED_SUBNORMAL: 'mixed_subnormal',
}

_WEIGHT_DIST_LABELS = {
    WeightDistType.UNIFORM: 'uniform',
    WeightDistType.SPARSE_ZERO: 'sparse_zero',
    WeightDistType.HAS_MAX: 'has_max',
    WeightDistType.ZERO_AND_MAX: 'zero_and_max',
    WeightDistType.ZERO_ONLY: 'zero_only',
    WeightDistType.MAX_ONLY: 'max_only',
    WeightDistType.NONE: 'none',
}


# -----------------------------------------------------------------------------
# Constrained Random Test Vector
# -----------------------------------------------------------------------------

@vsc.randobj
class FpintTestVector:
    """Constrained random test vector for FPINT GEMM"""

    def __init__(self):
        # Matrix dimensions
        self.M = vsc.rand_uint16_t()
        self.K = vsc.rand_uint16_t()
        self.N = vsc.rand_uint16_t()

        # Input distribution type
        self.input_dist = vsc.rand_uint8_t()

        # Weight distribution type
        self.weight_dist = vsc.rand_uint8_t()

        # Parameters for mixed distributions
        self.large_ratio = vsc.rand_uint8_t()      # 0~100 -> 0.0~1.0
        self.exp_diff = vsc.rand_uint8_t()         # exponent difference
        self.subnormal_ratio = vsc.rand_uint8_t()  # 0~100 -> 0.0~1.0
        self.zero_ratio = vsc.rand_uint8_t()       # 0~100 -> 0.0~1.0
        self.max_ratio = vsc.rand_uint8_t()        # 0~100 -> 0.0~1.0

        # Random seed
        self.seed = vsc.rand_uint32_t()

    @vsc.constraint
    def dim_c(self):
        """Matrix dimension constraints"""
        self.M >= 1
        self.M <= 64
        self.K.inside(vsc.rangelist(32, 64, 128, 256))  # Multiple of MXU_K (32)
        self.N.inside(vsc.rangelist(16, 32, 48, 64))   # Multiple of MXU_N (16)

    @vsc.constraint
    def dist_type_c(self):
        """Distribution type constraints"""
        self.input_dist <= 2   # 0, 1, 2
        self.weight_dist <= 2  # 0, 1, 2

    @vsc.constraint
    def ratio_c(self):
        """Ratio constraints (0~50%)"""
        self.large_ratio <= 50
        self.subnormal_ratio <= 50
        self.zero_ratio <= 50
        self.max_ratio <= 50

    @vsc.constraint
    def exp_diff_c(self):
        """Exponent difference constraint"""
        self.exp_diff >= 5
        self.exp_diff <= 14  # FP16 max meaningful diff


# -----------------------------------------------------------------------------
# Coverage Groups
# -----------------------------------------------------------------------------

@vsc.covergroup
class FpintInputCoverage:
    """Coverage for input characteristics"""

    def __init__(self):
        self.with_sample(
            input_dist=vsc.uint8_t(),
            has_large_exp_diff=vsc.bit_t(1),  # exp diff > 10
            has_subnormal=vsc.bit_t(1)
        )

        # Input distribution type coverage
        self.cp_input_dist = vsc.coverpoint(self.input_dist, bins={
            "normal": vsc.bin(InputDistType.NORMAL),
            "mixed_exp": vsc.bin(InputDistType.MIXED_EXP),
            "mixed_subnormal": vsc.bin(InputDistType.MIXED_SUBNORMAL)
        })

        # Large exp diff coverage
        self.cp_large_exp_diff = vsc.coverpoint(self.has_large_exp_diff, bins={
            "no": vsc.bin(0),
            "yes": vsc.bin(1)
        })

        # Subnormal coverage
        self.cp_subnormal = vsc.coverpoint(self.has_subnormal, bins={
            "no": vsc.bin(0),
            "yes": vsc.bin(1)
        })

        # Cross: input_dist x exp_diff x subnormal
        self.cross_input = vsc.cross([self.cp_input_dist, self.cp_large_exp_diff, self.cp_subnormal])


@vsc.covergroup
class FpintWeightCoverage:
    """Coverage for weight characteristics"""

    def __init__(self):
        self.with_sample(
            weight_dist=vsc.uint8_t(),
            has_zero=vsc.bit_t(1),
            has_max=vsc.bit_t(1)
        )

        # Weight distribution type coverage (analysis-based)
        self.cp_weight_dist = vsc.coverpoint(self.weight_dist, bins={
            "zero_and_max": vsc.bin(WeightDistType.ZERO_AND_MAX),
            "zero_only": vsc.bin(WeightDistType.ZERO_ONLY),
            "max_only": vsc.bin(WeightDistType.MAX_ONLY),
            "none": vsc.bin(WeightDistType.NONE)
        })

        # Zero weight coverage
        self.cp_has_zero = vsc.coverpoint(self.has_zero, bins={
            "no": vsc.bin(0),
            "yes": vsc.bin(1)
        })

        # Max weight coverage
        self.cp_has_max = vsc.coverpoint(self.has_max, bins={
            "no": vsc.bin(0),
            "yes": vsc.bin(1)
        })

        # Cross coverage
        self.cross_weight = vsc.cross([self.cp_weight_dist, self.cp_has_zero, self.cp_has_max])


@vsc.covergroup
class FpintCrossCoverage:
    """Cross coverage for critical FPINT error cases"""

    def __init__(self):
        self.with_sample(
            max_meets_zero=vsc.bit_t(1),      # max input x zero weight (prealign worst case)
            sign_pp=vsc.bit_t(1),              # pos x pos
            sign_pn=vsc.bit_t(1),              # pos x neg
            sign_np=vsc.bit_t(1),              # neg x pos
            sign_nn=vsc.bit_t(1)               # neg x neg
        )

        # Prealign worst case: max input meets zero weight
        self.cp_max_meets_zero = vsc.coverpoint(self.max_meets_zero, bins={
            "no": vsc.bin(0),
            "yes": vsc.bin(1)
        })

        # Sign combinations
        self.cp_sign_pp = vsc.coverpoint(self.sign_pp, bins={"no": vsc.bin(0), "yes": vsc.bin(1)})
        self.cp_sign_pn = vsc.coverpoint(self.sign_pn, bins={"no": vsc.bin(0), "yes": vsc.bin(1)})
        self.cp_sign_np = vsc.coverpoint(self.sign_np, bins={"no": vsc.bin(0), "yes": vsc.bin(1)})
        self.cp_sign_nn = vsc.coverpoint(self.sign_nn, bins={"no": vsc.bin(0), "yes": vsc.bin(1)})


@vsc.covergroup
class FpintMatrixSizeCoverage:
    """Coverage for matrix dimensions"""

    def __init__(self):
        self.with_sample(
            M=vsc.uint16_t(),
            K=vsc.uint16_t(),
            N=vsc.uint16_t()
        )

        self.cp_M = vsc.coverpoint(self.M, bins={
            "tiny": vsc.bin([1, 4]),
            "small": vsc.bin([5, 16]),
            "medium": vsc.bin([17, 32]),
            "large": vsc.bin([33, 64])
        })

        self.cp_K = vsc.coverpoint(self.K, bins={
            "k16": vsc.bin(16),
            "k32": vsc.bin(32),
            "k64": vsc.bin(64),
            "k128": vsc.bin(128)
        })

        self.cp_N = vsc.coverpoint(self.N, bins={
            "n16": vsc.bin(16),
            "n32": vsc.bin(32),
            "n48": vsc.bin(48),
            "n64": vsc.bin(64)
        })

        self.cross_MN = vsc.cross([self.cp_M, self.cp_N])


# -----------------------------------------------------------------------------
# Coverage-Driven Test Runner
# -----------------------------------------------------------------------------

class FpintCoverageTestRunner:
    """Coverage-driven test runner for FPINT GEMM"""

    # All available test configurations
    ALL_TEST_CONFIGS = [
        (FpIntImplType.QCOL_2SCOMP, fpint_gemm_qcol_2scomp, QCOL, False),
        (FpIntImplType.QCOL_ZERO_LESS, fpint_gemm_qcol_zero_less, QCOL, True),
        (FpIntImplType.QCOL_REAL_2SCOMP, fpint_gemm_qcol_real_2scomp, QCOL, False),
        (FpIntImplType.QROW_2SCOMP, fpint_gemm_qrow_2scomp, QROW, False),
        (FpIntImplType.QROW_ZERO_LESS, fpint_gemm_qrow_zero_less, QROW, True),
        (FpIntImplType.QROW_REAL_2SCOMP, fpint_gemm_qrow_real_2scomp, QROW, False),
    ]

    def __init__(
        self,
        err_func: Callable = None,
        error_threshold: float = 500.0
    ):
        self.err_func = err_func or ulp_diff_fp16
        self.error_threshold = error_threshold

        # Coverage groups per impl_type (initialized in run_coverage_test)
        self.coverage: Dict[FpIntImplType, Dict[str, Any]] = {}

        # Test vector
        self.tv = FpintTestVector()

        # Results
        self.results: List[TestResult] = []

    def _create_coverage_for_impl(self, impl_type: FpIntImplType) -> Dict[str, Any]:
        """Create coverage groups for a specific implementation type"""
        impl_name = impl_type.value

        input_cov = FpintInputCoverage()
        input_cov.set_name(f"{impl_name}_input")

        weight_cov = FpintWeightCoverage()
        weight_cov.set_name(f"{impl_name}_weight")

        cross_cov = FpintCrossCoverage()
        cross_cov.set_name(f"{impl_name}_cross")

        size_cov = FpintMatrixSizeCoverage()
        size_cov.set_name(f"{impl_name}_size")

        return {
            'input': input_cov,
            'weight': weight_cov,
            'cross': cross_cov,
            'size': size_cov,
        }

    def generate_input_data(self, M: int, K: int, seed: int) -> np.ndarray:
        """Generate input data based on test vector distribution type"""
        dist = int(self.tv.input_dist)

        if dist == InputDistType.NORMAL:
            return generate_normal_fp16((M, K), mean=0.0, std=1.0, seed=seed)
        elif dist == InputDistType.MIXED_EXP:
            large_ratio = int(self.tv.large_ratio) / 100.0
            exp_diff = int(self.tv.exp_diff)
            return generate_mixed_exp_fp16((M, K), large_ratio=large_ratio, exp_diff=exp_diff, seed=seed)
        else:  # MIXED_SUBNORMAL
            subnormal_ratio = int(self.tv.subnormal_ratio) / 100.0
            return generate_mixed_subnormal_fp16((M, K), subnormal_ratio=subnormal_ratio, seed=seed)

    def generate_weight_data(self, K: int, N: int, seed: int) -> np.ndarray:
        """Generate weight data based on test vector distribution type"""
        dist = int(self.tv.weight_dist)

        if dist == WeightDistType.UNIFORM:
            return generate_random_weights((K, N), w_width=4, seed=seed)
        elif dist == WeightDistType.SPARSE_ZERO:
            zero_ratio = int(self.tv.zero_ratio) / 100.0
            return generate_sparse_zero_weights((K, N), zero_ratio=zero_ratio, seed=seed)
        else:  # HAS_MAX
            max_ratio = int(self.tv.max_ratio) / 100.0
            return generate_weights_with_max((K, N), max_ratio=max_ratio, seed=seed)

    def analyze_and_sample_coverage(
        self,
        impl_type: FpIntImplType,
        input_data: np.ndarray,
        weight_data: np.ndarray
    ) -> Dict[TagKey, Any]:
        """
        Analyze data and sample coverage for a specific implementation type.

        Returns:
            Dict[TagKey, Any]: Analysis results to be stored in TestResult.tags
        """
        M, K = input_data.shape
        _, N = weight_data.shape

        # Get coverage groups for this impl_type
        cov = self.coverage[impl_type]

        # Analyze input characteristics
        has_large_exp_diff = False
        has_subnormal = False
        max_exp_diff = 0

        for m in range(M):
            exps = [get_fp16_exp(int(x)) for x in input_data[m, :]]
            max_exp = max(exps)
            min_exp = min(e for e in exps if e > 0) if any(e > 0 for e in exps) else 0
            exp_diff = max_exp - min_exp
            if exp_diff > max_exp_diff:
                max_exp_diff = exp_diff
            if exp_diff > 10:
                has_large_exp_diff = True
            if any(is_subnormal(int(x)) for x in input_data[m, :]):
                has_subnormal = True

        cov['input'].sample(int(self.tv.input_dist), int(has_large_exp_diff), int(has_subnormal))

        # Analyze weight characteristics
        has_zero_weight = bool(np.any(weight_data == 0))
        has_max_weight = bool(np.any(weight_data == 15))
        zero_weight_ratio = float(np.sum(weight_data == 0)) / weight_data.size

        analysis_weight_dist = WeightDistType.from_flags(has_zero_weight, has_max_weight)
        cov['weight'].sample(int(analysis_weight_dist), int(has_zero_weight), int(has_max_weight))

        # Analyze cross coverage (MXU group level)
        max_meets_zero = False
        sign_pp = sign_pn = sign_np = sign_nn = False

        for m in range(M):
            for n in range(N):
                analysis = analyze_mxu_group(input_data[m, :], weight_data[:, n])
                for group in analysis:
                    if group['max_meets_zero']:
                        max_meets_zero = True
                    for (inp_sign, w_sign) in group['sign_combinations']:
                        if inp_sign == 0 and w_sign == 0:
                            sign_pp = True
                        elif inp_sign == 0 and w_sign == 1:
                            sign_pn = True
                        elif inp_sign == 1 and w_sign == 0:
                            sign_np = True
                        else:
                            sign_nn = True

        cov['cross'].sample(int(max_meets_zero), int(sign_pp), int(sign_pn), int(sign_np), int(sign_nn))

        # Size coverage
        cov['size'].sample(M, K, N)

        # Return analysis results as tags dict
        return {
            # Input characteristics
            TagKey.HAS_LARGE_EXP_DIFF: has_large_exp_diff,
            TagKey.HAS_SUBNORMAL: has_subnormal,
            TagKey.MAX_EXP_DIFF: max_exp_diff,
            # Weight characteristics
            TagKey.HAS_ZERO_WEIGHT: has_zero_weight,
            TagKey.HAS_MAX_WEIGHT: has_max_weight,
            TagKey.ZERO_WEIGHT_RATIO: zero_weight_ratio,
            # Cross characteristics
            TagKey.MAX_MEETS_ZERO: max_meets_zero,
            # Sign combinations
            TagKey.SIGN_PP: sign_pp,
            TagKey.SIGN_PN: sign_pn,
            TagKey.SIGN_NP: sign_np,
            TagKey.SIGN_NN: sign_nn,
        }

    def run_single_test(
        self,
        impl_type: FpIntImplType,
        impl_func: Callable,
        qdir: int,
        make_weight_odd: bool = False,
        verbose: bool = False,
        store_preds: Optional[List[StorePredicate]] = None
    ) -> TestResult:
        """
        Run a single test with current test vector.

        Computes 4 errors:
        1. err_ref_vs_gpu: fpint_gemm_ref (FP64) vs fpint_gemm_gpu (FP16 bit-exact)
        2. err_ref_vs_emul: fpint_gemm_ref (FP64) vs fpint_emul implementation
        3. err_ulp_diff: err_ref_vs_gpu - err_ref_vs_emul (positive = emul better)
        4. err_gpu_vs_emul: fpint_gemm_gpu vs fpint_emul (GPU as reference)

        Args:
            impl_type: Implementation type enum
            store_preds: List of predicate functions. Each takes Dict[ErrorType, float]
                         and returns bool. Data is stored if ANY predicate returns True (OR).
                         Use AND logic inside a single predicate if needed.
                         If None or empty, data is not stored.
        """
        M = int(self.tv.M)
        K = int(self.tv.K)
        N = int(self.tv.N)
        seed = int(self.tv.seed)
        input_dist = int(self.tv.input_dist)
        weight_dist = int(self.tv.weight_dist)

        # Generate data
        input_data = self.generate_input_data(M, K, seed)
        weight_data = self.generate_weight_data(K, N, seed + 1)
        if make_weight_odd:
            weight_data = weight_data * 2 + 1

        # Analyze and sample coverage (returns tags dict)
        tags = self.analyze_and_sample_coverage(impl_type, input_data, weight_data)

        # Generate scale and zero
        if qdir == QCOL:
            KG = K // QBLOCK
            scale_shape = (KG, N)
            zero_shape = (KG, N)
        else:
            NG = (N + QBLOCK - 1) // QBLOCK
            scale_shape = (K, NG)
            zero_shape = (K, NG)

        scale_data = generate_random_fp16(scale_shape, value_range=(0.1, 1.0), seed=seed + 2)
        zero_data = generate_random_zero_points(zero_shape, z_range=(-4, 4), seed=seed + 3)

        # Run all three implementations
        # 1. Reference (FP64 high precision)
        output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                     M, N, K, qdir=qdir, debug=False)
        # 2. GPU (FP16 bit-exact)
        output_gpu = fpint_gemm_gpu(input_data, weight_data, scale_data, zero_data,
                                     M, N, K, qdir=qdir, debug=False)
        # 3. Emulation implementation
        output_emul = impl_func(input_data, weight_data, scale_data, zero_data, M, N, K, debug=False)

        # Convert to float for error calculation
        ref_float = to_float_matrix(output_ref)
        gpu_float = to_float_matrix(output_gpu)
        emul_float = to_float_matrix(output_emul)

        # Calculate 4 errors
        err_ref_vs_gpu = ulp_diff_fp16(ref_float, gpu_float)
        err_ref_vs_emul = ulp_diff_fp16(ref_float, emul_float)
        err_ulp_diff = err_ref_vs_gpu - err_ref_vs_emul  # positive = emul is better
        err_gpu_vs_emul = ulp_diff_fp16(gpu_float, emul_float)  # GPU as reference

        # Pass/fail based on emul error
        max_emul_err = float(np.max(err_ref_vs_emul))
        passed = max_emul_err <= self.error_threshold

        # Determine if data should be stored (OR condition: ANY predicate returns True)
        should_store = False
        if store_preds:
            # Map ErrorType to their max values
            err_max_values = {
                ErrorType.REF_VS_GPU: float(np.max(err_ref_vs_gpu)),
                ErrorType.REF_VS_EMUL: max_emul_err,
                ErrorType.ULP_DIFF: float(np.max(np.abs(err_ulp_diff))),
                ErrorType.GPU_VS_EMUL: float(np.max(err_gpu_vs_emul)),
            }
            # Check if ANY predicate returns True (OR)
            should_store = any(pred(err_max_values) for pred in store_preds)

        # Create TestResult
        result = TestResult(
            name=impl_type.value,
            M=M, K=K, N=N,
            seed=seed,
            passed=passed,
            error_threshold=self.error_threshold,
            # Four error types
            err_ref_vs_gpu=err_ref_vs_gpu,
            err_ref_vs_emul=err_ref_vs_emul,
            err_ulp_diff=err_ulp_diff,
            err_gpu_vs_emul=err_gpu_vs_emul,
            # Config
            qdir=qdir,
            input_dist=input_dist,
            weight_dist=int(WeightDistType.from_flags(
                tags[TagKey.HAS_ZERO_WEIGHT],
                tags[TagKey.HAS_MAX_WEIGHT]
            )),
            # Store data only if error thresholds exceeded (saves memory)
            input_data=input_data if should_store else None,
            weight_data=weight_data if should_store else None,
            scale_data=scale_data if should_store else None,
            zero_data=zero_data if should_store else None,
            output_ref=output_ref if should_store else None,
            output_gpu=output_gpu if should_store else None,
            output_emul=output_emul if should_store else None,
            output_ref_float=ref_float if should_store else None,
            output_gpu_float=gpu_float if should_store else None,
            output_emul_float=emul_float if should_store else None,
            # Analysis tags (always stored for filtering)
            tags=tags,
        )

        self.results.append(result)

        if verbose:
            status = "PASS" if passed else "FAIL"
            gpu_max = float(np.max(err_ref_vs_gpu))
            gpu_vs_emul_max = float(np.max(err_gpu_vs_emul))
            print(f"  [{status}] {impl_type.value}: M={M}, K={K}, N={N}, "
                  f"gpu_err={gpu_max:.1f}, emul_err={max_emul_err:.1f}, "
                  f"gpu_vs_emul={gpu_vs_emul_max:.1f}, same={result.gpu_emul_same_pct:.0f}%")

        return result

    def run_coverage_test(
        self,
        n_tests: int = 100,
        impl_types: Optional[List[FpIntImplType]] = None,
        verbose: bool = True,
        store_preds: Optional[List[StorePredicate]] = None
    ) -> TestResultAnalyzer:
        """
        Run coverage-driven tests.

        Args:
            n_tests: Number of tests per implementation
            impl_types: List of implementation types to test. If None, tests all.
            verbose: Print progress
            store_preds: List of predicate functions for conditional data storage.
                         Data is stored if ANY predicate returns True (OR condition).

        Returns:
            TestResultAnalyzer with all results
        """
        # Clear all previous coverage instances
        vsc.CoverageRegistry.inst().clear()

        # Reset results
        self.results = []

        # Filter test configs based on impl_types
        if impl_types is None:
            test_configs = self.ALL_TEST_CONFIGS
        else:
            impl_type_set = set(impl_types)
            test_configs = [cfg for cfg in self.ALL_TEST_CONFIGS if cfg[0] in impl_type_set]

        # Create coverage groups for each impl_type
        self.coverage = {}
        for impl_type, _, _, _ in test_configs:
            self.coverage[impl_type] = self._create_coverage_for_impl(impl_type)

        if verbose:
            print("=" * 70)
            print(f"FPINT Coverage-Driven Test")
            print(f"  n_tests={n_tests}, error_threshold={self.error_threshold}")
            print(f"  impl_types={[t.value for t, _, _, _ in test_configs]}")
            print("=" * 70)

        for impl_type, impl_func, qdir, make_weight_odd in test_configs:
            if verbose:
                print(f"\n{impl_type.value}:")

            for _ in range(n_tests):
                self.tv.randomize()
                self.run_single_test(impl_type, impl_func, qdir, make_weight_odd, verbose, store_preds)

        # Summary
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)

        if verbose:
            print("\n" + "=" * 70)
            if total > 0:
                print(f"Summary: {passed}/{total} passed ({100*passed/total:.1f}%)")
            else:
                print("Summary: No tests run")
            print("=" * 70)

            # Print coverage
            print("\nCoverage Report:")
            vsc.report_coverage(details=True)

        return self.get_analyzer()

    def get_coverage(self, impl_type: Optional[FpIntImplType] = None) -> Dict[str, Any]:
        """
        Get coverage percentages.

        Args:
            impl_type: If specified, returns coverage for that impl_type only.
                       If None, returns coverage for all impl_types.

        Returns:
            If impl_type is specified: Dict[str, float] with coverage percentages
            If impl_type is None: Dict[FpIntImplType, Dict[str, float]]
        """
        if impl_type is not None:
            if impl_type not in self.coverage:
                return {'input': 0.0, 'weight': 0.0, 'cross': 0.0, 'size': 0.0}
            cov = self.coverage[impl_type]
            return {
                'input': cov['input'].get_inst_coverage(),
                'weight': cov['weight'].get_inst_coverage(),
                'cross': cov['cross'].get_inst_coverage(),
                'size': cov['size'].get_inst_coverage()
            }
        else:
            result = {}
            for it, cov in self.coverage.items():
                result[it] = {
                    'input': cov['input'].get_inst_coverage(),
                    'weight': cov['weight'].get_inst_coverage(),
                    'cross': cov['cross'].get_inst_coverage(),
                    'size': cov['size'].get_inst_coverage()
                }
            return result

    def get_analyzer(self) -> TestResultAnalyzer:
        """Get TestResultAnalyzer for the results"""
        return TestResultAnalyzer(self.results)

# -----------------------------------------------------------------------------
# Convenience Function
# -----------------------------------------------------------------------------

def run_fpint_coverage_test(
    n_tests: int = 20,
    impl_types: Optional[List[FpIntImplType]] = None,
    error_threshold: float = 500.0,
    verbose: bool = True,
    store_preds: Optional[List[StorePredicate]] = None
) -> Tuple[FpintCoverageTestRunner, TestResultAnalyzer]:
    """
    Run FPINT coverage-driven tests.

    Args:
        n_tests: Number of tests per implementation
        impl_types: List of implementation types to test. If None, tests all.
        error_threshold: Max allowed ULP error
        verbose: Print progress
        store_preds: List of predicate functions for conditional data storage.
                     Data is stored if ANY predicate returns True (OR condition).
                     Each predicate receives Dict[ErrorType, float] with max error values.

    Returns:
        Tuple of (FpintCoverageTestRunner, TestResultAnalyzer)

    Example:
        >>> # Test only QCOL_2SCOMP
        >>> runner, analyzer = run_fpint_coverage_test(
        ...     n_tests=50,
        ...     impl_types=[FpIntImplType.QCOL_2SCOMP]
        ... )
        >>> # Store when gpu_vs_emul > 10 OR ref_vs_emul > 100
        >>> runner, analyzer = run_fpint_coverage_test(
        ...     n_tests=50,
        ...     store_preds=[
        ...         lambda e: e[ErrorType.GPU_VS_EMUL] > 10,
        ...         lambda e: e[ErrorType.REF_VS_EMUL] > 100,
        ...     ]
        ... )
        >>> # Store when (gpu_vs_emul > 10 AND ref_vs_emul > 50)
        >>> runner, analyzer = run_fpint_coverage_test(
        ...     n_tests=50,
        ...     store_preds=[
        ...         lambda e: e[ErrorType.GPU_VS_EMUL] > 10 and e[ErrorType.REF_VS_EMUL] > 50,
        ...     ]
        ... )
        >>> runner.plot_results()
        >>> analyzer.print_summary()
    """
    runner = FpintCoverageTestRunner(
        error_threshold=error_threshold
    )
    analyzer = runner.run_coverage_test(
        n_tests=n_tests,
        impl_types=impl_types,
        verbose=verbose,
        store_preds=store_preds
    )
    return runner, analyzer
