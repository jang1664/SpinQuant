"""
FPINT GEMM Integration Tests

This module provides integration tests for FPINT GEMM implementations.
Tests compare all 5 implementations against reference and visualize results.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from fpint_emul import *
from visualize import *
from test_utils import *


def to_float_matrix(output: np.ndarray) -> np.ndarray:
    """Convert FP16 bit representation matrix to float matrix"""
    M, N = output.shape
    result = np.zeros((M, N), dtype=np.float32)
    for m in range(M):
        for n in range(N):
            result[m, n] = fp16_bit_to_float(output[m, n])
    return result


def print_error_stats(name: str, output_float: np.ndarray, ref_float: np.ndarray):
    """Print error statistics for a single implementation"""
    abs_error = np.abs(output_float - ref_float)
    rel_error = abs_error / (np.abs(ref_float) + 1e-8) * 100  # percentage

    print(f"\n{name}:")
    print(f"  Absolute Error - Max: {np.max(abs_error):.6f}, Mean: {np.mean(abs_error):.6f}")
    print(f"  Relative Error - Max: {np.max(rel_error):.4f}%, Mean: {np.mean(rel_error):.4f}%")


def test_fpint_gemm_qcol_2scomp(M=8, K=32, N=32, seed=200, visualize=False):
    """Test QCOL 2's complement GEMM implementation vs reference"""
    print("\n" + "=" * 60)
    print("Test: fpint_gemm_qcol_2scomp")
    print("=" * 60)

    KG = K // QBLOCK

    # Generate test data
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_data = generate_random_weights((K, N), w_width=4, seed=seed+1)
    scale_data = generate_random_fp16((KG, N), value_range=(0.1, 1.0), seed=seed+2)
    zero_data = generate_random_zero_points((KG, N), z_range=(-4, 4), seed=seed+3)

    print(f"Dimensions: M={M}, K={K}, N={N}, KG={KG}")

    # Run implementation
    output = fpint_gemm_qcol_2scomp(input_data, weight_data, scale_data, zero_data,
                                    M, N, K, debug=False)

    # Run reference
    output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                M, N, K, qdir=QCOL, debug=False)

    # Convert to float
    output_float = to_float_matrix(output)
    ref_float = to_float_matrix(output_ref)

    # Print error stats
    print_error_stats("qcol_2scomp vs ref", output_float, ref_float)

    if visualize:
        plot_heatmap_comparison(
            ref_float,
            {"qcol_2scomp": output_float},
            diff_func=relative_diff_percent,
            diff_func_name="Relative Error (%)",
            title="fpint_gemm_qcol_2scomp vs Reference",
            show=True,
            save_path="test_qcol_2scomp.png",
            auto_scale=True,
            shared_scale=False
        )

    print("  PASSED")
    return output_float, ref_float


def test_fpint_gemm_qcol_zero_less(M=8, K=32, N=32, seed=210, visualize=False):
    """Test QCOL zero-less GEMM implementation vs reference"""
    print("\n" + "=" * 60)
    print("Test: fpint_gemm_qcol_zero_less")
    print("=" * 60)

    KG = K // QBLOCK

    # Generate test data (weights must be odd for zero-less encoding)
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_data = generate_random_weights((K, N), w_width=4, seed=seed+1)
    weight_data = weight_data * 2 + 1  # Make odd: 1, 3, 5, ..., 31
    scale_data = generate_random_fp16((KG, N), value_range=(0.1, 1.0), seed=seed+2)
    zero_data = generate_random_zero_points((KG, N), z_range=(-4, 4), seed=seed+3)

    print(f"Dimensions: M={M}, K={K}, N={N}, KG={KG}")

    # Run implementation
    output = fpint_gemm_qcol_zero_less(input_data, weight_data, scale_data, zero_data,
                                       M, N, K, debug=False)

    # Run reference
    output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                M, N, K, qdir=QCOL, debug=False)

    # Convert to float
    output_float = to_float_matrix(output)
    ref_float = to_float_matrix(output_ref)

    # Print error stats
    print_error_stats("qcol_zero_less vs ref", output_float, ref_float)

    if visualize:
        plot_heatmap_comparison(
            ref_float,
            {"qcol_zero_less": output_float},
            diff_func=relative_diff_percent,
            diff_func_name="Relative Error (%)",
            title="fpint_gemm_qcol_zero_less vs Reference",
            show=True,
            save_path="test_qcol_zero_less.png",
            auto_scale=True,
            shared_scale=False
        )

    print("  PASSED")
    return output_float, ref_float


def test_fpint_gemm_qrow_2scomp(M=8, K=32, N=32, seed=220, visualize=False):
    """Test QROW 2's complement GEMM implementation vs reference"""
    print("\n" + "=" * 60)
    print("Test: fpint_gemm_qrow_2scomp")
    print("=" * 60)

    NG = N // QBLOCK

    # Generate test data
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_data = generate_random_weights((K, N), w_width=4, seed=seed+1)
    scale_data = generate_random_fp16((K, NG), value_range=(0.1, 1.0), seed=seed+2)
    zero_data = generate_random_zero_points((K, NG), z_range=(-4, 4), seed=seed+3)

    print(f"Dimensions: M={M}, K={K}, N={N}, NG={NG}")

    # Run implementation
    output = fpint_gemm_qrow_2scomp(input_data, weight_data, scale_data, zero_data,
                                    M, N, K, debug=False)

    # Run reference
    output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                M, N, K, qdir=QROW, debug=False)

    # Convert to float
    output_float = to_float_matrix(output)
    ref_float = to_float_matrix(output_ref)

    # Print error stats
    print_error_stats("qrow_2scomp vs ref", output_float, ref_float)

    if visualize:
        plot_heatmap_comparison(
            ref_float,
            {"qrow_2scomp": output_float},
            diff_func=relative_diff_percent,
            diff_func_name="Relative Error (%)",
            title="fpint_gemm_qrow_2scomp vs Reference",
            show=True,
            save_path="test_qrow_2scomp.png",
            auto_scale=True,
            shared_scale=False
        )

    print("  PASSED")
    return output_float, ref_float


def test_fpint_gemm_qrow_zero_less(M=8, K=32, N=32, seed=230, visualize=False):
    """Test QROW zero-less GEMM implementation vs reference"""
    print("\n" + "=" * 60)
    print("Test: fpint_gemm_qrow_zero_less")
    print("=" * 60)

    NG = N // QBLOCK

    # Generate test data (weights must be odd for zero-less encoding)
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_data = generate_random_weights((K, N), w_width=4, seed=seed+1)
    weight_data = weight_data * 2 + 1  # Make odd
    scale_data = generate_random_fp16((K, NG), value_range=(0.1, 1.0), seed=seed+2)
    zero_data = generate_random_zero_points((K, NG), z_range=(-4, 4), seed=seed+3)

    print(f"Dimensions: M={M}, K={K}, N={N}, NG={NG}")

    # Run implementation
    output = fpint_gemm_qrow_zero_less(input_data, weight_data, scale_data, zero_data,
                                       M, N, K, debug=False)

    # Run reference
    output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                M, N, K, qdir=QROW, debug=False)

    # Convert to float
    output_float = to_float_matrix(output)
    ref_float = to_float_matrix(output_ref)

    # Print error stats
    print_error_stats("qrow_zero_less vs ref", output_float, ref_float)

    if visualize:
        plot_heatmap_comparison(
            ref_float,
            {"qrow_zero_less": output_float},
            diff_func=relative_diff_percent,
            diff_func_name="Relative Error (%)",
            title="fpint_gemm_qrow_zero_less vs Reference",
            show=True,
            save_path="test_qrow_zero_less.png",
            auto_scale=True,
            shared_scale=False
        )

    print("  PASSED")
    return output_float, ref_float


def test_fpint_gemm_qrow_real_2scomp(M=8, K=32, N=32, seed=240, visualize=False):
    """Test QROW real 2's complement GEMM implementation vs reference"""
    print("\n" + "=" * 60)
    print("Test: fpint_gemm_qrow_real_2scomp")
    print("=" * 60)

    NG = N // QBLOCK

    # Generate test data
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_data = generate_random_weights((K, N), w_width=4, seed=seed+1)
    scale_data = generate_random_fp16((K, NG), value_range=(0.1, 1.0), seed=seed+2)
    zero_data = generate_random_zero_points((K, NG), z_range=(-4, 4), seed=seed+3)

    print(f"Dimensions: M={M}, K={K}, N={N}, NG={NG}")

    # Run implementation
    output = fpint_gemm_qrow_real_2scomp(input_data, weight_data, scale_data, zero_data,
                                         M, N, K, debug=False)

    # Run reference
    output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                M, N, K, qdir=QROW, debug=False)

    # Convert to float
    output_float = to_float_matrix(output)
    ref_float = to_float_matrix(output_ref)

    # Print error stats
    print_error_stats("qrow_real_2scomp vs ref", output_float, ref_float)

    if visualize:
        plot_heatmap_comparison(
            ref_float,
            {"qrow_real_2scomp": output_float},
            diff_func=relative_diff_percent,
            diff_func_name="Relative Error (%)",
            title="fpint_gemm_qrow_real_2scomp vs Reference",
            show=True,
            save_path="test_qrow_real_2scomp.png",
            auto_scale=True,
            shared_scale=False
        )

    print("  PASSED")
    return output_float, ref_float


def test_all_implementations_comparison(M=8, K=32, N=32, visualize=True):
    """Run all tests and compare all implementations in one visualization"""
    print("\n" + "=" * 60)
    print("Test: All Implementations Comparison")
    print("=" * 60)

    # Run all tests and collect results
    results = {}

    # QCOL tests
    KG = K // QBLOCK
    seed = 200

    input_qcol = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_qcol = generate_random_weights((K, N), w_width=4, seed=seed+1)
    weight_qcol_odd = weight_qcol * 2 + 1
    scale_qcol = generate_random_fp16((KG, N), value_range=(0.1, 1.0), seed=seed+2)
    zero_qcol = generate_random_zero_points((KG, N), z_range=(-4, 4), seed=seed+3)

    ref_qcol = to_float_matrix(fpint_gemm_ref(input_qcol, weight_qcol, scale_qcol, zero_qcol, M, N, K, qdir=QCOL))
    ref_qcol_odd = to_float_matrix(fpint_gemm_ref(input_qcol, weight_qcol_odd, scale_qcol, zero_qcol, M, N, K, qdir=QCOL))

    results["qcol_2scomp"] = {
        "output": to_float_matrix(fpint_gemm_qcol_2scomp(input_qcol, weight_qcol, scale_qcol, zero_qcol, M, N, K)),
        "ref": ref_qcol
    }
    results["qcol_zero_less"] = {
        "output": to_float_matrix(fpint_gemm_qcol_zero_less(input_qcol, weight_qcol_odd, scale_qcol, zero_qcol, M, N, K)),
        "ref": ref_qcol_odd
    }

    # QROW tests
    NG = N // QBLOCK
    seed = 220

    input_qrow = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=seed)
    weight_qrow = generate_random_weights((K, N), w_width=4, seed=seed+1)
    weight_qrow_odd = weight_qrow * 2 + 1
    scale_qrow = generate_random_fp16((K, NG), value_range=(0.1, 1.0), seed=seed+2)
    zero_qrow = generate_random_zero_points((K, NG), z_range=(-4, 4), seed=seed+3)

    ref_qrow = to_float_matrix(fpint_gemm_ref(input_qrow, weight_qrow, scale_qrow, zero_qrow, M, N, K, qdir=QROW))
    ref_qrow_odd = to_float_matrix(fpint_gemm_ref(input_qrow, weight_qrow_odd, scale_qrow, zero_qrow, M, N, K, qdir=QROW))

    results["qrow_2scomp"] = {
        "output": to_float_matrix(fpint_gemm_qrow_2scomp(input_qrow, weight_qrow, scale_qrow, zero_qrow, M, N, K)),
        "ref": ref_qrow
    }
    results["qrow_zero_less"] = {
        "output": to_float_matrix(fpint_gemm_qrow_zero_less(input_qrow, weight_qrow_odd, scale_qrow, zero_qrow, M, N, K)),
        "ref": ref_qrow_odd
    }
    results["qrow_real_2scomp"] = {
        "output": to_float_matrix(fpint_gemm_qrow_real_2scomp(input_qrow, weight_qrow, scale_qrow, zero_qrow, M, N, K)),
        "ref": ref_qrow
    }

    # Print summary
    print("\n" + "-" * 60)
    print("Error Summary (Relative Error %)")
    print("-" * 60)

    for name, data in results.items():
        rel_err = np.abs(data["output"] - data["ref"]) / (np.abs(data["ref"]) + 1e-8) * 100
        print(f"{name:20s}: Max={np.max(rel_err):8.4f}%, Mean={np.mean(rel_err):8.4f}%")

    # Visualize all in one figure
    if visualize:
        # Create dict for visualization
        eval_dict = {}
        for name, data in results.items():
            # Compute relative error for each
            rel_err = relative_diff_percent(data["ref"], data["output"])
            eval_dict[name] = data["output"]

        # Use first reference as baseline for visualization structure
        # But we need to show relative error, so create a combined view

        # Plot all relative errors
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes_flat = axes.flatten()

        cmap = "hot"  # Better for error visualization

        # Find global min/max for consistent colormap
        all_rel_errs = []
        for name, data in results.items():
            rel_err = np.abs(data["output"] - data["ref"]) / (np.abs(data["ref"]) + 1e-8) * 100
            all_rel_errs.append(rel_err)

        global_vmin = min(np.min(e) for e in all_rel_errs)
        global_vmax = max(np.max(e) for e in all_rel_errs)

        for idx, (name, data) in enumerate(results.items()):
            ax = axes_flat[idx]
            rel_err = np.abs(data["output"] - data["ref"]) / (np.abs(data["ref"]) + 1e-8) * 100

            im = ax.imshow(rel_err, aspect='auto', cmap=cmap, vmin=global_vmin, vmax=global_vmax)
            ax.set_title(f'{name}\nRel Err: max={np.max(rel_err):.2f}%, mean={np.mean(rel_err):.2f}%')
            ax.set_xlabel('Column')
            ax.set_ylabel('Row')
            plt.colorbar(im, ax=ax, label='Rel Error (%)')

        # Hide last subplot if odd number
        axes_flat[-1].axis('off')

        fig.suptitle(f'All FPINT GEMM Implementations vs Reference\n(M={M}, K={K}, N={N})',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('test_all_implementations.png', dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: test_all_implementations.png")
        plt.show()

    print("\n  ALL TESTS PASSED")


def run_all_tests(visualize=False):
    """Run all integration tests"""
    print("\n" + "=" * 60)
    print("FPINT GEMM INTEGRATION TEST SUITE")
    print("=" * 60)

    try:
        test_fpint_gemm_qcol_2scomp(visualize=visualize)
        test_fpint_gemm_qcol_zero_less(visualize=visualize)
        test_fpint_gemm_qrow_2scomp(visualize=visualize)
        test_fpint_gemm_qrow_zero_less(visualize=visualize)
        test_fpint_gemm_qrow_real_2scomp(visualize=visualize)

        if visualize:
            test_all_implementations_comparison(visualize=True)

        print("\n" + "=" * 60)
        print("ALL INTEGRATION TESTS PASSED")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run FPINT GEMM integration tests')
    parser.add_argument('--visualize', '-v', action='store_true',
                       help='Generate visualization plots')
    parser.add_argument('--test', '-t', type=str, default='all',
                       choices=['all', 'qcol_2scomp', 'qcol_zero_less',
                               'qrow_2scomp', 'qrow_zero_less', 'qrow_real_2scomp', 'compare'],
                       help='Which test to run')
    args = parser.parse_args()

    if args.test == 'all':
        run_all_tests(visualize=args.visualize)
    elif args.test == 'qcol_2scomp':
        test_fpint_gemm_qcol_2scomp(visualize=args.visualize)
    elif args.test == 'qcol_zero_less':
        test_fpint_gemm_qcol_zero_less(visualize=args.visualize)
    elif args.test == 'qrow_2scomp':
        test_fpint_gemm_qrow_2scomp(visualize=args.visualize)
    elif args.test == 'qrow_zero_less':
        test_fpint_gemm_qrow_zero_less(visualize=args.visualize)
    elif args.test == 'qrow_real_2scomp':
        test_fpint_gemm_qrow_real_2scomp(visualize=args.visualize)
    elif args.test == 'compare':
        test_all_implementations_comparison(visualize=True)
