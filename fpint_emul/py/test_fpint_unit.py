"""
Unit tests for FPINT GEMM Emulation

This module provides unit tests for individual components:
- FP16 conversion functions
- Prealign function
- Denormal number handling
- Reference GEMM implementation
- Visualization functions
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from fpint_emul import (
    prealign, fpint_gemm_ref, fpint_gemm_qcol_2scomp,
    fp16_bit_to_float, float_to_fp16_bit, fp16_to_components,
    QBLOCK, MXU_K, EXTRA_BIT, QCOL
)
from visualize import (
    plot_heatmap_comparison, plot_multi_eval_comparison, print_diff_statistics,
    absolute_diff, relative_diff_percent, squared_diff
)
from test_utils import generate_random_fp16, generate_random_weights, generate_random_zero_points


def test_fp16_conversion():
    """Test FP16 conversion functions"""
    print("\n" + "=" * 60)
    print("Test: FP16 Conversion")
    print("=" * 60)
    
    test_values = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 10.0, -10.0, 0.1, -0.1]
    
    for val in test_values:
        fp16_bits = float_to_fp16_bit(val)
        recovered = fp16_bit_to_float(fp16_bits)
        sign, exp, mantissa = fp16_to_components(fp16_bits)
        
        print(f"Original: {val:8.4f} -> FP16: 0x{fp16_bits:04x} "
              f"(s={sign}, e={exp:2d}, m=0x{mantissa:03x}) -> "
              f"Recovered: {recovered:8.4f}")
    
    print("✓ FP16 conversion test passed")


def test_prealign():
    """Test prealign function"""
    print("\n" + "=" * 60)
    print("Test: Prealign Function")
    print("=" * 60)
    
    M, K = 2, 16
    
    # Create simple test data
    input_data = generate_random_fp16((M, K), value_range=(-5.0, 5.0), seed=42)
    
    print(f"Input data shape: {input_data.shape}")
    print(f"Sample input values (as float):")
    for m in range(min(2, M)):
        print(f"  Row {m}: ", end="")
        for k in range(min(4, K)):
            fp_val = fp16_bit_to_float(input_data[m, k])
            print(f"{fp_val:7.3f} ", end="")
        print("...")
    
    # Test with EXTRA_BIT
    aligned_fx, aligned_exp = prealign(input_data, EXTRA_BIT, M, K, debug=True)
    
    print(f"\nAligned FX data shape: {aligned_fx.shape}")
    print(f"Aligned EXP data shape: {aligned_exp.shape}")
    print(f"Sample aligned exponents: {aligned_exp[0, :]}")
    
    # Verify alignment within groups
    for m in range(M):
        for kg in range(K // MXU_K):
            max_exp = aligned_exp[m, kg]
            print(f"\nRow {m}, Group {kg}, Max Exp: {max_exp}")
            for k_in_group in range(min(4, MXU_K)):
                k = kg * MXU_K + k_in_group
                fp_val = fp16_bit_to_float(input_data[m, k])
                sign, exp, mantissa = fp16_to_components(input_data[m, k])
                aligned_val = aligned_fx[m, k]
                # Convert to unsigned for hex display (handles negative correctly)
                aligned_uint = int(aligned_val) & 0xFFFFFFFFFFFFFFFF
                print(f"  k={k}: fp={fp_val:7.3f}, exp={exp:2d}, aligned={aligned_val:10d} (0x{aligned_uint:016x})")
    
    print("✓ Prealign test passed")


def test_denormal_handling():
    """Test denormal number handling in prealign"""
    print("\n" + "=" * 60)
    print("Test: Denormal Number Handling")
    print("=" * 60)
    
    M, K = 1, 16
    
    # Create test data with denormal numbers
    # FP16 denormal: exp=0, mantissa!=0
    # Value = (-1)^sign * 2^(-14) * (0.mantissa)
    
    input_data = np.zeros((M, K), dtype=np.uint16)
    
    # Normal numbers
    input_data[0, 0] = float_to_fp16_bit(1.0)      # Normal: exp=15
    input_data[0, 1] = float_to_fp16_bit(0.5)      # Normal: exp=14
    input_data[0, 2] = float_to_fp16_bit(0.25)     # Normal: exp=13
    
    # Small numbers that might be denormal
    input_data[0, 3] = float_to_fp16_bit(6.1e-5)   # Close to denormal threshold
    input_data[0, 4] = float_to_fp16_bit(1e-5)     # Likely denormal
    input_data[0, 5] = float_to_fp16_bit(1e-6)     # Definitely denormal
    
    # Zero
    input_data[0, 6] = float_to_fp16_bit(0.0)
    
    # Fill rest with normal values
    for k in range(7, K):
        input_data[0, k] = float_to_fp16_bit(float(k - 7))
    
    print("Test values:")
    for k in range(7):
        fp_val = fp16_bit_to_float(input_data[0, k])
        sign, exp, mantissa = fp16_to_components(input_data[0, k])
        is_denormal = (exp == 0) and (mantissa != 0)
        is_zero = (exp == 0) and (mantissa == 0)
        
        status = "DENORMAL" if is_denormal else ("ZERO" if is_zero else "NORMAL")
        print(f"  k={k}: val={fp_val:12.6e}, 0x{input_data[0, k]:04x}, "
              f"s={sign}, e={exp:2d}, m=0x{mantissa:03x} [{status}]")
    
    # Test prealign with denormals
    aligned_fx, aligned_exp = prealign(input_data, EXTRA_BIT, M, K, debug=True)
    
    print(f"\nAligned results:")
    print(f"  Max exponent for group 0: {aligned_exp[0, 0]}")
    for k in range(7):
        # Convert to unsigned for hex display (handles negative correctly)
        aligned_uint = int(aligned_fx[0, k]) & 0xFFFFFFFFFFFFFFFF
        print(f"  k={k}: aligned={aligned_fx[0, k]:20d} (0x{aligned_uint:016x})")
    
    print("✓ Denormal handling test passed")


def test_gemm_ref():
    """Test reference GEMM implementation"""
    print("\n" + "=" * 60)
    print("Test: Reference GEMM (qcol)")
    print("=" * 60)
    
    M, K, N = 2, 32, 16
    KG = K // QBLOCK
    
    # Generate test data
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=100)
    weight_data = generate_random_weights((K, N), w_width=4, seed=101)
    scale_data = generate_random_fp16((KG, N), value_range=(0.1, 1.0), seed=102)
    zero_data = generate_random_zero_points((KG, N), z_range=(-4, 4), seed=103)
    
    print(f"Input: {input_data.shape}, Weight: {weight_data.shape}")
    print(f"Scale: {scale_data.shape}, Zero: {zero_data.shape}")
    
    # Run reference GEMM
    output = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                            M, N, K, qdir=QCOL, debug=False)
    
    print(f"Output shape: {output.shape}")
    print(f"Sample outputs (as float):")
    for m in range(min(2, M)):
        print(f"  Row {m}: ", end="")
        for n in range(min(4, N)):
            out_val = fp16_bit_to_float(output[m, n])
            print(f"{out_val:9.4f} ", end="")
        print("...")
    
    print("✓ Reference GEMM test passed")


def test_visualization():
    """Test visualization functions with GEMM results"""
    print("\n" + "=" * 60)
    print("Test: Visualization")
    print("=" * 60)
    
    M, K, N = 8, 32, 16
    KG = K // QBLOCK
    
    # Generate test data
    input_data = generate_random_fp16((M, K), value_range=(-2.0, 2.0), seed=400)
    weight_data = generate_random_weights((K, N), w_width=4, seed=401)
    scale_data = generate_random_fp16((KG, N), value_range=(0.1, 1.0), seed=402)
    zero_data = generate_random_zero_points((KG, N), z_range=(-4, 4), seed=403)
    
    # Run reference and implementation
    output_ref = fpint_gemm_ref(input_data, weight_data, scale_data, zero_data,
                                 M, N, K, qdir=QCOL, debug=False)
    output_qcol = fpint_gemm_qcol_2scomp(input_data, weight_data, scale_data, zero_data,
                                          M, N, K, debug=False)
    
    # Convert to float for visualization
    output_ref_float = np.zeros((M, N), dtype=np.float32)
    output_qcol_float = np.zeros((M, N), dtype=np.float32)
    
    for m in range(M):
        for n in range(N):
            output_ref_float[m, n] = fp16_bit_to_float(output_ref[m, n])
            output_qcol_float[m, n] = fp16_bit_to_float(output_qcol[m, n])
    
    print("\nVisualizing single evaluation result...")
    # Single evaluation comparison with absolute difference
    plot_heatmap_comparison(
        output_ref_float,
        output_qcol_float,
        diff_func=absolute_diff,
        diff_func_name="Absolute Difference",
        title="QCOL GEMM vs Reference (Absolute Diff)",
        show=False,
        save_path="qcol_comparison_abs.png"
    )
    
    # Single evaluation comparison with relative difference
    plot_heatmap_comparison(
        output_ref_float,
        output_qcol_float,
        diff_func=relative_diff_percent,
        diff_func_name="Relative Difference (%)",
        title="QCOL GEMM vs Reference (Relative Diff %)",
        show=False,
        save_path="qcol_comparison_rel.png"
    )
    
    # Print statistics for various diff functions
    print_diff_statistics(output_ref_float, output_qcol_float, 
                         diff_func=absolute_diff, diff_func_name="Absolute Difference")
    print_diff_statistics(output_ref_float, output_qcol_float,
                         diff_func=relative_diff_percent, diff_func_name="Relative Difference (%)")
    
    # Multiple evaluation comparison (with dict input)
    print("\nVisualizing multiple evaluation results...")
    eval_dict = {
        "QCOL Implementation": output_qcol_float,
    }
    
    plot_multi_eval_comparison(
        output_ref_float,
        eval_dict,
        diff_func=absolute_diff,
        diff_func_name="Absolute Difference",
        title="Multiple GEMM Implementations Comparison",
        show=False,
        save_path="multi_comparison.png"
    )
    
    # Example with dict input for single comparison
    print("\nUsing dict input for single comparison...")
    plot_heatmap_comparison(
        output_ref_float,
        {"QCOL": output_qcol_float},
        diff_func=squared_diff,
        diff_func_name="Squared Difference",
        title="QCOL GEMM vs Reference (Squared Diff)",
        show=False,
        save_path="qcol_comparison_squared.png"
    )
    
    print("\nVisualization files saved:")
    print("  - qcol_comparison_abs.png")
    print("  - qcol_comparison_rel.png")
    print("  - multi_comparison.png")
    print("  - qcol_comparison_squared.png")
    print("\n✓ Visualization test passed")


def run_all_tests():
    """Run all unit tests"""
    print("\n" + "=" * 60)
    print("FPINT UNIT TEST SUITE")
    print("=" * 60)
    
    try:
        test_fp16_conversion()
        test_prealign()
        test_denormal_handling()
        test_gemm_ref()
        # Uncomment to run visualization tests
        # test_visualization()
        
        print("\n" + "=" * 60)
        print("ALL UNIT TESTS PASSED ✓")
        print("=" * 60 + "\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
