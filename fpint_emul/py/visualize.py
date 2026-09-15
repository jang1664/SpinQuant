"""
Visualization utilities for FPINT GEMM Emulation

This module provides visualization functions for comparing reference and
evaluated GEMM results using heatmaps.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Dict, Union, Optional, Tuple


def calculate_subplot_grid(n: int, max_cols: int = 4) -> Tuple[int, int]:
    """
    Calculate optimal subplot grid dimensions (rows, cols) for n items.

    Uses sqrt-based approach to get close to square, with max column limit.

    Args:
        n: Number of subplots needed
        max_cols: Maximum number of columns allowed

    Returns:
        Tuple of (rows, cols)
    """
    if n <= 0:
        return (1, 1)

    cols = min(math.ceil(math.sqrt(n)), max_cols)
    rows = math.ceil(n / cols)
    return (rows, cols)


# Predefined difference functions
def absolute_diff(ref: np.ndarray, eval: np.ndarray) -> np.ndarray:
    """Calculate absolute difference between reference and evaluation"""
    return np.abs(ref - eval)


def relative_diff(ref: np.ndarray, eval: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """
    Calculate relative difference between reference and evaluation
    
    Args:
        ref: Reference values
        eval: Evaluation values
        epsilon: Small value to prevent division by zero
    
    Returns:
        Relative difference as percentage
    """
    return np.abs(ref - eval) / (np.abs(ref) + epsilon)


def relative_diff_percent(ref: np.ndarray, eval: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Calculate relative difference as percentage"""
    return 100.0 * relative_diff(ref, eval, epsilon)


def squared_diff(ref: np.ndarray, eval: np.ndarray) -> np.ndarray:
    """Calculate squared difference (useful for MSE visualization)"""
    return (ref - eval) ** 2


def log_absolute_diff(ref: np.ndarray, eval: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """Calculate log of absolute difference (useful for large value ranges)"""
    abs_diff = np.abs(ref - eval)
    return np.log10(abs_diff + epsilon)


def _float32_to_ulp_int(val: np.float32) -> int:
    """Convert float32 to its ULP integer representation (for comparison)"""
    # Reinterpret float32 bits as int32, then convert to Python int
    int_val = int(np.frombuffer(np.float32(val).tobytes(), dtype=np.int32)[0])
    # Handle negative numbers: convert to two's complement-like representation
    # This maps negative floats to negative integers in a way that preserves ordering
    if int_val < 0:
        int_val = -int_val ^ 0x80000000
    return int_val


def ulp_diff(ref: np.ndarray, eval: np.ndarray) -> np.ndarray:
    """
    Calculate ULP (Units in Last Place) difference between reference and evaluation.

    ULP measures the number of representable floating-point numbers between two values.
    This is more meaningful for floating-point comparison than relative error,
    especially when values are close to zero.

    Args:
        ref: Reference values (float32)
        eval: Evaluation values (float32)

    Returns:
        ULP difference as integer array
    """
    ref_flat = ref.flatten().astype(np.float32)
    eval_flat = eval.flatten().astype(np.float32)

    result = np.zeros(ref_flat.shape, dtype=np.float64)

    for i in range(len(ref_flat)):
        ref_ulp = _float32_to_ulp_int(ref_flat[i])
        eval_ulp = _float32_to_ulp_int(eval_flat[i])
        result[i] = abs(ref_ulp - eval_ulp)

    return result.reshape(ref.shape)


def ulp_diff_fp16(ref: np.ndarray, eval: np.ndarray) -> np.ndarray:
    """
    Calculate ULP difference for FP16 values stored as float32.

    Since FP16 has fewer mantissa bits (10 vs 23), we scale the ULP
    to be meaningful for FP16 precision.

    For FP16: 1 ULP ≈ 2^(-10) * 2^exp relative error

    Args:
        ref: Reference values (float32, but representing FP16 precision)
        eval: Evaluation values (float32, but representing FP16 precision)

    Returns:
        ULP difference scaled for FP16
    """
    # FP16 has 10 mantissa bits, FP32 has 23
    # Scale factor: 2^(23-10) = 2^13 = 8192
    FP16_TO_FP32_MANTISSA_SHIFT = 13

    fp32_ulp = ulp_diff(ref, eval)

    # Scale down to FP16 ULPs
    return fp32_ulp / (1 << FP16_TO_FP32_MANTISSA_SHIFT)


def _fp16_bits_to_ordered(bits: np.ndarray) -> np.ndarray:
    """Map finite FP16 bit patterns to monotonically ordered integers."""
    bits = np.asarray(bits, dtype=np.uint16)
    exponent = bits & np.uint16(0x7C00)
    mantissa = bits & np.uint16(0x03FF)
    if np.any((exponent == 0x7C00) & (mantissa != 0)):
        raise ValueError("FP16 ULP distance is undefined for NaN")

    magnitude = (bits & np.uint16(0x7FFF)).astype(np.int32)
    negative = (bits & np.uint16(0x8000)) != 0
    # Collapse -0 and +0 to the same point while preserving FP16 ordering.
    return np.where(negative, 0x8000 - magnitude, 0x8000 + magnitude)


def ulp_diff_fp16_bits(ref_bits: np.ndarray, eval_bits: np.ndarray) -> np.ndarray:
    """Return exact FP16 representable-value distance from uint16 bit patterns."""
    ref_bits = np.asarray(ref_bits, dtype=np.uint16)
    eval_bits = np.asarray(eval_bits, dtype=np.uint16)
    if ref_bits.shape != eval_bits.shape:
        raise ValueError(
            f"FP16 ULP inputs must have the same shape: "
            f"{ref_bits.shape} != {eval_bits.shape}"
        )
    ref_ordered = _fp16_bits_to_ordered(ref_bits)
    eval_ordered = _fp16_bits_to_ordered(eval_bits)
    return np.abs(ref_ordered - eval_ordered).astype(np.int32)


def plot_heatmap_comparison(
    ref_value: np.ndarray,
    eval_val: Union[np.ndarray, Dict[str, np.ndarray]],
    diff_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = absolute_diff,
    diff_func_name: str = "Absolute Difference",
    title: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    cmap: str = "RdYlGn_r",
    save_path: Optional[str] = None,
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    max_cols: int = 4,
    auto_scale: bool = True,
    shared_scale: bool = True
) -> plt.Figure:
    """
    Plot heatmap comparison between reference and evaluation values

    Args:
        ref_value: Reference matrix values (M x N)
        eval_val: Evaluation matrix values or dict of evaluation matrices
        diff_func: Function to calculate difference (signature: func(ref, eval) -> diff)
        diff_func_name: Name of the difference function for labeling
        title: Overall title for the figure
        figsize: Figure size (width, height), auto-calculated if None
        cmap: Colormap for heatmap
        save_path: Path to save the figure (if None, won't save)
        show: Whether to display the plot
        vmin: Minimum value for colorbar (overrides auto_scale)
        vmax: Maximum value for colorbar (overrides auto_scale)
        max_cols: Maximum number of columns in subplot grid
        auto_scale: If True and vmin/vmax not provided, use data min/max for colorbar
        shared_scale: If True, use same vmin/vmax across all subplots

    Returns:
        matplotlib Figure object
    """
    # Handle dict input for eval_val
    if not isinstance(eval_val, dict):
        eval_val = {"Evaluation": eval_val}

    n_evals = len(eval_val)

    # Calculate subplot grid
    rows, cols = calculate_subplot_grid(n_evals, max_cols)

    # Auto-calculate figure size if not provided
    if figsize is None:
        figsize = (5 * cols, 4 * rows)

    # Pre-compute all diff matrices for auto-scaling
    diff_matrices = {}
    for name, eval_matrix in eval_val.items():
        diff_matrices[name] = diff_func(ref_value, eval_matrix)

    # Auto-scale: compute global vmin/vmax from all diff matrices
    if auto_scale and (vmin is None or vmax is None):
        if shared_scale:
            all_diffs = np.concatenate([d.flatten() for d in diff_matrices.values()])
            computed_vmin = np.min(all_diffs)
            computed_vmax = np.max(all_diffs)
        else:
            computed_vmin = None
            computed_vmax = None

        if vmin is None:
            vmin = computed_vmin
        if vmax is None:
            vmax = computed_vmax

    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

    # Flatten axes for easy iteration
    axes_flat = axes.flatten()

    # Plot each evaluation's difference heatmap
    for idx, (name, eval_matrix) in enumerate(eval_val.items()):
        ax = axes_flat[idx]
        diff_matrix = diff_matrices[name]

        # Per-subplot scaling if not shared
        if not shared_scale and auto_scale:
            plot_vmin = np.min(diff_matrix)
            plot_vmax = np.max(diff_matrix)
        else:
            plot_vmin = vmin
            plot_vmax = vmax

        im = ax.imshow(diff_matrix, aspect='auto', cmap=cmap, vmin=plot_vmin, vmax=plot_vmax)
        ax.set_title(f'{name}\n{diff_func_name}')
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        cbar = plt.colorbar(im, ax=ax)
        # Add min/max annotation to colorbar
        cbar.ax.set_ylabel(f'min={np.min(diff_matrix):.2e}\nmax={np.max(diff_matrix):.2e}',
                          rotation=270, labelpad=15, fontsize=8)

    # Hide unused subplots
    for idx in range(n_evals, len(axes_flat)):
        axes_flat[idx].axis('off')

    # Overall title
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig

def print_diff_statistics(
    ref_value: np.ndarray,
    eval_val: Union[np.ndarray, Dict[str, np.ndarray]],
    diff_func: Callable[[np.ndarray, np.ndarray], np.ndarray] = absolute_diff,
    diff_func_name: str = "Absolute Difference"
) -> None:
    """
    Print statistical summary of differences
    
    Args:
        ref_value: Reference matrix values
        eval_val: Evaluation matrix values or dict of evaluation matrices
        diff_func: Function to calculate difference
        diff_func_name: Name of the difference function for labeling
    """
    print("\n" + "=" * 60)
    print(f"Difference Statistics: {diff_func_name}")
    print("=" * 60)
    
    # Handle dict input
    if isinstance(eval_val, dict):
        for name, eval_matrix in eval_val.items():
            diff_matrix = diff_func(ref_value, eval_matrix)
            print(f"\n{name}:")
            print(f"  Max:  {np.max(diff_matrix):.6e}")
            print(f"  Min:  {np.min(diff_matrix):.6e}")
            print(f"  Mean: {np.mean(diff_matrix):.6e}")
            print(f"  Std:  {np.std(diff_matrix):.6e}")
            print(f"  Median: {np.median(diff_matrix):.6e}")
    else:
        diff_matrix = diff_func(ref_value, eval_val)
        print(f"  Max:  {np.max(diff_matrix):.6e}")
        print(f"  Min:  {np.min(diff_matrix):.6e}")
        print(f"  Mean: {np.mean(diff_matrix):.6e}")
        print(f"  Std:  {np.std(diff_matrix):.6e}")
        print(f"  Median: {np.median(diff_matrix):.6e}")
    
    print("=" * 60)
