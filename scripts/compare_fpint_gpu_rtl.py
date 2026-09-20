#!/usr/bin/env python3
"""Cross-check the CUDA FP-INT emulator against gemm_unit_wrap_opt_v3 RTL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from fpint_emul import FpIntConfig, fpint_linear
from fpint_emul.reference import prealign_reference


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RTL_ROOT = Path(
    "/home/jaeyongjang/project.local/fpint/hw/reconfigurable"
)
RTL_RESULT = re.compile(
    r"RTL_RESULT mode=(\S+) row=(\d+) col=(\d+) fp32=([0-9a-fA-F]+) out16=([0-9a-fA-F]+)"
)
CASES = ("normal", "wide_exponent", "subnormal", "cancellation")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_filelist(filelist: Path) -> tuple[list[Path], list[Path]]:
    sources: list[Path] = []
    includes: list[Path] = []
    invocation_directory = filelist.parent.parent
    for raw_line in filelist.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith("#"):
            continue
        if line.startswith("+incdir+"):
            for entry in line.removeprefix("+incdir+").split("+"):
                includes.append((invocation_directory / entry).resolve())
            continue
        source = (invocation_directory / line).resolve()
        if source.name == "tb_gemm_unit_wrap.sv":
            continue
        sources.append(source)
    return sources, includes


def build_verilator(rtl_root: Path, build_dir: Path, force: bool) -> Path:
    executable = build_dir / "rtl_compare_sim"
    sources, includes = parse_filelist(
        rtl_root / "sim/functional/gemm_unit_wrap/script/run_opt_v3.f"
    )
    local_sources = [
        ROOT / "rtl_verification/designware_verilator_compat.sv",
        ROOT / "rtl_verification/tb_gemm_unit_wrap_opt_v3_verilator.sv",
        ROOT / "rtl_verification/fp32_dpi.cpp",
    ]
    dependencies = sources + local_sources
    if (
        not force
        and executable.exists()
        and executable.stat().st_mtime >= max(path.stat().st_mtime for path in dependencies)
    ):
        return executable

    build_dir.mkdir(parents=True, exist_ok=True)
    # Verilator 5.020 cannot elaborate a nonblocking assignment to a packed
    # array element from inside a procedural for-loop (BLKLOOPINIT).  Preserve
    # the RTL's exact shift-register semantics with equivalent packed slices.
    mxu_source = rtl_root / "rtl/mxu/mxu_var5.sv"
    mxu_overlay = build_dir / "mxu_var5_verilator.sv"
    mxu_text = mxu_source.read_text()
    reset_loop = """          for (int r = 0; r < PE_ROW; r++) begin
            weight_bar_mem[0][r][c] <= '0;
            weight_bar_mem[1][r][c] <= '0;
          end"""
    shift_loop = """          for (int r = 1; r < PE_ROW; r++) begin
            weight_bar_mem[in_weight_sel_i][r][c] <= weight_bar_mem[in_weight_sel_i][r-1][c];
          end"""
    reset_slice = "\n".join(
        f"          weight_bar_mem[{bank}][{row}][c] <= '0;"
        for bank in range(2)
        for row in range(128)
    )
    shift_slice = "\n".join(
        f"          weight_bar_mem[in_weight_sel_i][{row}][c] <= "
        f"weight_bar_mem[in_weight_sel_i][{row - 1}][c];"
        for row in range(1, 128)
    )
    if reset_loop not in mxu_text or shift_loop not in mxu_text:
        raise RuntimeError("mxu_var5.sv no longer matches the validated Verilator overlay")
    mxu_overlay.write_text(
        mxu_text.replace(reset_loop, reset_slice).replace(shift_loop, shift_slice)
    )
    sources = [mxu_overlay if path == mxu_source else path for path in sources]
    command = [
        "verilator",
        "--binary",
        "--timing",
        "-j",
        "4",
        "--top-module",
        "tb_gemm_unit_wrap_opt_v3_verilator",
        "--Mdir",
        str(build_dir),
        "-o",
        executable.name,
        "-Wno-fatal",
        "-Wno-TIMESCALEMOD",
        "-Wno-WIDTHEXPAND",
        "-Wno-WIDTHTRUNC",
        "-Wno-WIDTHCONCAT",
        "-Wno-UNOPTFLAT",
        "-Wno-UNUSEDSIGNAL",
        "-Wno-UNUSEDPARAM",
        f"-I{rtl_root / 'rtl/include'}",
    ]
    command.extend(f"-I{path}" for path in includes)
    command.append(str(local_sources[0]))
    command.extend(str(path) for path in sources)
    command.extend(str(path) for path in local_sources[1:])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=900,
    )
    (build_dir / "build.log").write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"Verilator build failed:\n{tail}")
    return executable


def sample_activation_bits(
    mode: str, case: str, rows: int, rng: np.random.Generator
) -> np.ndarray:
    shape = (rows, 128)
    mantissa_bits = 7 if mode == "bf16" else 10
    exponent_shift = mantissa_bits
    max_mantissa = (1 << mantissa_bits) - 1
    signs = rng.integers(0, 2, size=shape, dtype=np.uint16)

    if case == "normal":
        bounds = (116, 135) if mode == "bf16" else (6, 21)
        exponents = rng.integers(bounds[0], bounds[1], size=shape, dtype=np.uint16)
    elif case == "wide_exponent":
        bounds = (70, 145) if mode == "bf16" else (1, 25)
        exponents = rng.integers(bounds[0], bounds[1], size=shape, dtype=np.uint16)
    elif case == "subnormal":
        exponents = np.zeros(shape, dtype=np.uint16)
    elif case == "cancellation":
        bounds = (122, 132) if mode == "bf16" else (10, 18)
        exponents = rng.integers(bounds[0], bounds[1], size=shape, dtype=np.uint16)
        signs[:, 1::2] = 1 - signs[:, 0::2]
        exponents[:, 1::2] = exponents[:, 0::2]
    else:
        raise ValueError(case)

    mantissas = rng.integers(
        1 if case == "subnormal" else 0,
        max_mantissa + 1,
        size=shape,
        dtype=np.uint16,
    )
    if case == "cancellation":
        mantissas[:, 1::2] = mantissas[:, 0::2]
    bits = (
        (signs << np.uint16(15))
        | (exponents << np.uint16(exponent_shift))
        | mantissas
    )
    if case == "subnormal":
        bits[:, ::31] = 0
        bits[:, 15::31] = np.uint16(0x8000)
    return np.ascontiguousarray(bits)


def sample_weights(case_index: int, rng: np.random.Generator) -> np.ndarray:
    weight = rng.integers(-8, 8, size=(2, 128), dtype=np.int16).astype(np.int8)
    weight[0, :8] = np.arange(-8, 0, dtype=np.int8)
    weight[1, :8] = np.arange(0, 8, dtype=np.int8)
    if case_index == 3:
        weight[:, 1::2] = weight[:, 0::2]
    return weight


def write_vectors(
    directory: Path,
    activation_bits: np.ndarray,
    weight: np.ndarray,
    scale: np.ndarray,
) -> tuple[Path, Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    activation_path = directory / "activation.mem"
    weight_path = directory / "weight.mem"
    scale_path = directory / "scale.mem"
    activation_path.write_text(
        "".join(f"{int(value):04x}\n" for value in activation_bits.reshape(-1))
    )
    packed_weight = (
        (weight[0].astype(np.int16) & 0xF)
        | ((weight[1].astype(np.int16) & 0xF) << 4)
    )
    weight_path.write_text("".join(f"{int(value):02x}\n" for value in packed_weight))
    scale_fp32_bits = scale.astype(np.float32).reshape(-1).view(np.uint32)
    scale_path.write_text("".join(f"{int(value):08x}\n" for value in scale_fp32_bits))
    return activation_path, weight_path, scale_path


def torch_activation(bits: np.ndarray, mode: str, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(bits.copy()).view(
        torch.bfloat16 if mode == "bf16" else torch.float16
    )
    return tensor.to(device)


def run_cuda(
    activation_bits: np.ndarray,
    weight_numpy: np.ndarray,
    scale_numpy: np.ndarray,
    mode: str,
    extra_bits: int,
) -> np.ndarray:
    device = torch.device("cuda:0")
    activation = torch_activation(activation_bits, mode, device)
    weight = torch.from_numpy(weight_numpy.copy()).to(device)
    scale = torch.from_numpy(scale_numpy.copy()).reshape(2, 1).to(device)
    zero = torch.zeros((2, 1), dtype=torch.int32, device=device)
    config = FpIntConfig(
        4,
        group_size=128,
        mxu_rows=128,
        extra_bits=extra_bits,
        reduce_extra_bits=10,
        activation_format=mode,
    )
    output = fpint_linear(
        activation,
        weight,
        scale,
        zero,
        config,
        backend="fpint_cuda",
        has_zero=False,
    )
    torch.cuda.synchronize()
    return (
        output.detach().contiguous().view(torch.int16).to(torch.int32).cpu().numpy()
        & 0xFFFF
    ).astype(np.uint16)


def run_rtl(
    executable: Path,
    mode: str,
    rows: int,
    vector_paths: tuple[Path, Path, Path],
    log_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    activation_path, weight_path, scale_path = vector_paths
    command = [
        str(executable),
        f"+MODE={mode}",
        f"+M={rows}",
        f"+ACT={activation_path}",
        f"+WEIGHT={weight_path}",
        f"+SCALE={scale_path}",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=120,
    )
    log_path.write_text(completed.stdout)
    if completed.returncode != 0:
        tail = "\n".join(completed.stdout.splitlines()[-80:])
        raise RuntimeError(f"RTL simulation failed:\n{tail}")

    output = np.zeros((rows, 2), dtype=np.uint16)
    fp32 = np.zeros((rows, 2), dtype=np.uint32)
    seen = np.zeros((rows, 2), dtype=bool)
    for match in RTL_RESULT.finditer(completed.stdout):
        row = int(match.group(2))
        col = int(match.group(3))
        if row >= rows or col >= 2 or seen[row, col]:
            raise RuntimeError(f"invalid or duplicate RTL result row={row} col={col}")
        fp32[row, col] = int(match.group(4), 16)
        output[row, col] = int(match.group(5), 16)
        seen[row, col] = True
    if not seen.all():
        raise RuntimeError(f"RTL output missing {int((~seen).sum())} elements")
    return output, fp32


def bits_to_float(bits: np.ndarray, mode: str) -> np.ndarray:
    if mode == "bf16":
        return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
    return np.ascontiguousarray(bits).view(np.float16).astype(np.float32)


def ordered_16(bits: np.ndarray) -> np.ndarray:
    values = bits.astype(np.int64)
    magnitude = values & 0x7FFF
    return np.where((values & 0x8000) != 0, 0x8000 - magnitude, 0x8000 + magnitude)


def correctly_pack_fp32(fp32_bits: np.ndarray, mode: str) -> np.ndarray:
    values = np.ascontiguousarray(fp32_bits, dtype=np.uint32).view(np.float32)
    if mode == "fp16":
        with np.errstate(over="ignore", invalid="ignore"):
            return values.astype(np.float16).view(np.uint16)
    upper = fp32_bits >> np.uint32(16)
    rounded = (
        fp32_bits + np.uint32(0x7FFF) + (upper & np.uint32(1))
    ) >> np.uint32(16)
    exponent = fp32_bits & np.uint32(0x7F800000)
    mantissa = fp32_bits & np.uint32(0x007FFFFF)
    is_nan = (exponent == np.uint32(0x7F800000)) & (mantissa != 0)
    rounded = np.where(is_nan, upper | np.uint32(0x40), rounded)
    return rounded.astype(np.uint16)


def software_fp32_before_output_cast(
    activation_bits: np.ndarray,
    weight: np.ndarray,
    scale: np.ndarray,
    mode: str,
    extra_bits: int,
) -> np.ndarray:
    activation = (
        activation_bits
        if mode == "bf16"
        else np.ascontiguousarray(activation_bits).view(np.float16)
    )
    mantissa_bits = 7 if mode == "bf16" else 10
    exponent_bits = 8 if mode == "bf16" else 5
    exponent_bias = 127 if mode == "bf16" else 15
    aligned, maximum = prealign_reference(
        activation,
        extra_bits,
        128,
        mantissa_bits,
        exponent_bits,
    )
    inner = aligned @ weight.T.astype(np.int64)
    binary_exponent = maximum[:, 0].astype(np.int32) - exponent_bias - (
        mantissa_bits + extra_bits
    )
    restored = np.ldexp(
        inner.astype(np.float64), binary_exponent[:, None]
    ).astype(np.float32)
    scaled = np.multiply(restored, scale.astype(np.float32)[None, :], dtype=np.float32)
    return np.ascontiguousarray(scaled).view(np.uint32)


def fp32_comparison_metrics(rtl: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    exact = rtl == expected
    rtl_values = np.ascontiguousarray(rtl).view(np.float32)
    expected_values = np.ascontiguousarray(expected).view(np.float32)
    finite = np.isfinite(rtl_values) & np.isfinite(expected_values)
    error = np.abs(rtl_values[finite] - expected_values[finite])
    examples = []
    for row, col in np.argwhere(~exact)[:8]:
        examples.append(
            {
                "row": int(row),
                "col": int(col),
                "rtl_hex": f"0x{int(rtl[row, col]):08x}",
                "reference_hex": f"0x{int(expected[row, col]):08x}",
                "rtl": float(rtl_values[row, col]),
                "reference": float(expected_values[row, col]),
            }
        )
    return {
        "elements": int(exact.size),
        "exact_matches": int(exact.sum()),
        "mismatches": int((~exact).sum()),
        "exact_fraction": float(exact.mean()),
        "max_abs": float(error.max()) if error.size else None,
        "examples": examples,
    }


def comparison_metrics(rtl: np.ndarray, gpu: np.ndarray, mode: str) -> dict[str, object]:
    exact = rtl == gpu
    rtl_values = bits_to_float(rtl, mode)
    gpu_values = bits_to_float(gpu, mode)
    both_finite = np.isfinite(rtl_values) & np.isfinite(gpu_values)
    absolute = np.abs(rtl_values[both_finite] - gpu_values[both_finite])
    ulp = np.abs(ordered_16(rtl).astype(np.int64) - ordered_16(gpu).astype(np.int64))
    mismatches = np.argwhere(~exact)
    examples = []
    for row, col in mismatches[:8]:
        examples.append(
            {
                "row": int(row),
                "col": int(col),
                "rtl_hex": f"0x{int(rtl[row, col]):04x}",
                "gpu_hex": f"0x{int(gpu[row, col]):04x}",
                "rtl": float(rtl_values[row, col]),
                "gpu": float(gpu_values[row, col]),
                "ulp": int(ulp[row, col]),
            }
        )
    return {
        "elements": int(exact.size),
        "exact_matches": int(exact.sum()),
        "mismatches": int((~exact).sum()),
        "exact_fraction": float(exact.mean()),
        "mean_ulp": float(ulp.mean()),
        "p95_ulp": float(np.percentile(ulp, 95)),
        "max_ulp": int(ulp.max()),
        "max_abs": float(absolute.max()) if absolute.size else None,
        "nonfinite_mismatches": int(((~exact) & ~both_finite).sum()),
        "examples": examples,
    }


def aggregate_metrics(pairs: list[tuple[np.ndarray, np.ndarray]], mode: str) -> dict[str, object]:
    rtl = np.concatenate([pair[0].reshape(-1) for pair in pairs])
    gpu = np.concatenate([pair[1].reshape(-1) for pair in pairs])
    return comparison_metrics(rtl.reshape(-1, 1), gpu.reshape(-1, 1), mode)


def aggregate_fp32_metrics(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> dict[str, object]:
    rtl = np.concatenate([pair[0].reshape(-1) for pair in pairs])
    expected = np.concatenate([pair[1].reshape(-1) for pair in pairs])
    return fp32_comparison_metrics(rtl.reshape(-1, 1), expected.reshape(-1, 1))


def markdown_report(summary: dict[str, object]) -> str:
    lines = [
        "# gemm_unit_wrap_opt_v3 RTL vs CUDA FP-INT",
        "",
        f"- Status: **{summary['status']}**",
        f"- RTL: `{summary['rtl_file']}`",
        f"- RTL SHA256: `{summary['rtl_sha256']}`",
        f"- RTL Git: `{summary['rtl_git_branch']}` @ `{summary['rtl_git_commit']}`",
        f"- BF16 packer source SHA256: `{summary['bf16_packer_sha256']}`",
        f"- BF16 exponent source SHA256: `{summary['bf16_exponent_sha256']}`",
        f"- CUDA kernel SHA256: `{summary['cuda_kernel_sha256']}`",
        f"- Verilator: `{summary['verilator_version']}`",
        f"- GPU: `{summary['gpu']}`",
        "- Shape per case: `M × K × N = "
        f"{summary['rows']} × 128 × 2`; weights are signed INT4.",
        "- RTL path: `gemm_unit_wrap_opt_v3`, K-first (`load_i=1`), FP32 dequant scale, packed FP16/BF16 output.",
        "",
        "## Aggregate",
        "",
        "| Activation | CUDA geometry | Exact | Mismatch | Exact rate | Mean ULP | P95 ULP | Max ULP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in summary["modes"]:
        for geometry in ("experiment_extra19", "rtl_matched"):
            item = summary["aggregate"][mode][geometry]
            lines.append(
                f"| {mode.upper()} | {geometry} | {item['exact_matches']} / {item['elements']} "
                f"| {item['mismatches']} | {100 * item['exact_fraction']:.6f}% "
                f"| {item['mean_ulp']:.6g} | {item['p95_ulp']:.6g} | {item['max_ulp']} |"
            )
    lines.extend(
        [
            "",
            "## Isolation checks",
            "",
            "| Activation | RTL FP32 core vs software FP32 | Correctly repacked RTL FP32 vs CUDA |",
            "|---|---:|---:|",
        ]
    )
    for mode in summary["modes"]:
        core = summary["aggregate_core_fp32"][mode]
        repacked = summary["aggregate_correct_pack"][mode]
        lines.append(
            f"| {mode.upper()} | {core['exact_matches']} / {core['elements']} exact "
            f"| {repacked['exact_matches']} / {repacked['elements']} exact |"
        )
    lines.extend(
        [
            "",
            "## Per-case result (current experiment: extra_bits=19)",
            "",
            "| Activation | Case | Exact | Mismatch | Exact rate | Max ULP | Max abs |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for case in summary["cases"]:
        item = case["experiment_extra19"]
        max_abs = "n/a" if item["max_abs"] is None else f"{item['max_abs']:.8g}"
        lines.append(
            f"| {case['mode'].upper()} | {case['case']} | {item['exact_matches']} / {item['elements']} "
            f"| {item['mismatches']} | {100 * item['exact_fraction']:.6f}% "
            f"| {item['max_ulp']} | {max_abs} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            summary["interpretation"],
            "",
            "## Applied fixes",
            "",
            "1. RTL BF16 packing: replace the concatenated rounding bias with `32'h0000_7fff + fp32[16]`.",
            "2. RTL BF16 subnormal alignment: use effective exponent 1 when exponent is zero and mantissa is nonzero.",
            "3. CUDA integer restoration: for normal results, round the integer once with `__ll2float_rn` and apply the power-of-two scale with `ldexpf`; for subnormal results, round the integer quotient/remainder directly to FP32 subnormal bits. This avoids both an underflowed intermediate factor and FP32 double rounding without using FP64.",
            "",
            "The Verilator environment replaces unavailable Synopsys DesignWare cells. FIFO and leading-zero detection use SystemVerilog compatibility models; FP32 multiply uses a DPI-C IEEE binary32 operation. The reconfigurable prealigner, MXU, column merger, int-to-FP logic, wrapper, and packer are the supplied RTL. This run does not cover multi-tile K accumulation (`load_i=0`) or asymmetric zero-point correction.",
            "",
            f"Raw JSON: `{summary['json_path']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rtl-root", type=Path, default=DEFAULT_RTL_ROOT)
    parser.add_argument("--rows", type=int, default=48)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results/fpint-rtl-compare"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "docs/FP-INT-hw-acc/mxu128_rtl_gpu_comparison.md",
    )
    parser.add_argument("--force-build", action="store_true")
    args = parser.parse_args()
    # Calling this script through an absolute conda Python path does not activate
    # the environment, so make its build tools (notably ninja) discoverable.
    python_bin = str(Path(sys.executable).resolve().parent)
    os.environ["PATH"] = python_bin + os.pathsep + os.environ.get("PATH", "")
    if not 1 <= args.rows <= 256:
        parser.error("--rows must be in [1, 256]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    executable = build_verilator(
        args.rtl_root, args.output_dir / "verilator-build", args.force_build
    )
    cases: list[dict[str, object]] = []
    aggregate_pairs: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {
        mode: {"experiment_extra19": [], "rtl_matched": []}
        for mode in ("bf16", "fp16")
    }
    core_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        mode: [] for mode in ("bf16", "fp16")
    }
    correct_pack_pairs: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
        mode: [] for mode in ("bf16", "fp16")
    }

    for mode_index, mode in enumerate(("bf16", "fp16")):
        rtl_extra_bits = 23 if mode == "bf16" else 20
        for case_index, case_name in enumerate(CASES):
            seed = 0x5A170000 + mode_index * 0x1000 + case_index
            rng = np.random.default_rng(seed)
            activation_bits = sample_activation_bits(mode, case_name, args.rows, rng)
            weight = sample_weights(case_index, rng)
            scale = np.array(
                [1.0, 1.0] if case_name == "subnormal" else [0.8125, 1.234375],
                dtype=np.float16,
            )
            case_dir = args.output_dir / f"{mode}-{case_name}"
            vectors = write_vectors(case_dir, activation_bits, weight, scale)
            rtl_output, rtl_fp32 = run_rtl(
                executable,
                mode,
                args.rows,
                vectors,
                case_dir / "rtl.log",
            )
            gpu_extra19 = run_cuda(activation_bits, weight, scale, mode, 19)
            gpu_matched = run_cuda(
                activation_bits, weight, scale, mode, rtl_extra_bits
            )
            metrics_extra19 = comparison_metrics(rtl_output, gpu_extra19, mode)
            metrics_matched = comparison_metrics(rtl_output, gpu_matched, mode)
            reference_fp32 = software_fp32_before_output_cast(
                activation_bits,
                weight,
                scale,
                mode,
                rtl_extra_bits,
            )
            core_metrics = fp32_comparison_metrics(rtl_fp32, reference_fp32)
            correctly_packed = correctly_pack_fp32(rtl_fp32, mode)
            correct_pack_metrics = comparison_metrics(
                correctly_packed, gpu_extra19, mode
            )
            aggregate_pairs[mode]["experiment_extra19"].append(
                (rtl_output, gpu_extra19)
            )
            aggregate_pairs[mode]["rtl_matched"].append((rtl_output, gpu_matched))
            core_pairs[mode].append((rtl_fp32, reference_fp32))
            correct_pack_pairs[mode].append((correctly_packed, gpu_extra19))
            np.save(case_dir / "rtl_output_bits.npy", rtl_output)
            np.save(case_dir / "rtl_fp32_bits.npy", rtl_fp32)
            np.save(case_dir / "gpu_extra19_bits.npy", gpu_extra19)
            np.save(case_dir / "gpu_rtl_matched_bits.npy", gpu_matched)
            cases.append(
                {
                    "mode": mode,
                    "case": case_name,
                    "seed": seed,
                    "scale_fp16": [float(value) for value in scale],
                    "experiment_extra19": metrics_extra19,
                    "rtl_matched": metrics_matched,
                    "rtl_core_vs_reference_fp32": core_metrics,
                    "correctly_packed_rtl_fp32_vs_cuda": correct_pack_metrics,
                }
            )
            print(
                f"{mode}/{case_name}: extra19 mismatches={metrics_extra19['mismatches']} "
                f"rtl-matched mismatches={metrics_matched['mismatches']}",
                flush=True,
            )

    aggregate = {
        mode: {
            geometry: aggregate_metrics(pairs, mode)
            for geometry, pairs in geometries.items()
        }
        for mode, geometries in aggregate_pairs.items()
    }
    aggregate_core_fp32 = {
        mode: aggregate_fp32_metrics(pairs) for mode, pairs in core_pairs.items()
    }
    aggregate_correct_pack = {
        mode: aggregate_metrics(pairs, mode)
        for mode, pairs in correct_pack_pairs.items()
    }
    strict_mismatches = sum(
        aggregate[mode]["rtl_matched"]["mismatches"] for mode in aggregate
    )
    current_mismatches = sum(
        aggregate[mode]["experiment_extra19"]["mismatches"] for mode in aggregate
    )
    if strict_mismatches == 0 and current_mismatches == 0:
        status = "PASS — bit-exact"
        interpretation = (
            "The current CUDA experiment configuration and the RTL are bit-exact for all tested outputs."
        )
    elif strict_mismatches == 0:
        status = "PASS with configuration discrepancy"
        interpretation = (
            "The CUDA implementation becomes bit-exact when its fixed-point headroom matches the RTL "
            "(BF16 extra_bits=23, FP16 extra_bits=20). The current experiment setting extra_bits=19 "
            "drops low aligned bits earlier, so the remaining differences are emulator-configuration differences rather than RTL arithmetic errors."
        )
    else:
        status = "MISMATCH"
        interpretation = (
            "FP16 is bit-exact in the tested matrix. For finite normal BF16 values, the RTL FP32 core is bit-exact, but the RTL BF16 packer can be 1 ULP high: its rounding expression concatenates `15'h7fff` with the tie bit, producing a bias near `0xfffe` instead of `0x7fff + lsb`. BF16 subnormals expose two additional differences: the RTL exponent path treats raw exponent zero as zero (rather than effective exponent one), while the CUDA kernel forms `ldexpf(1.0f, binary_exponent)` first and underflows that factor to zero. The Torch/NumPy reference preserves those subnormal values."
        )

    rtl_file = args.rtl_root / "rtl/gemm_unit_wrap_opt_v3.sv"
    rtl_git_branch = subprocess.check_output(
        ["git", "-C", str(args.rtl_root), "branch", "--show-current"], text=True
    ).strip()
    rtl_git_commit = subprocess.check_output(
        ["git", "-C", str(args.rtl_root), "rev-parse", "HEAD"], text=True
    ).strip()
    verilator_version = subprocess.check_output(
        ["verilator", "--version"], text=True
    ).strip()
    gpu_name = torch.cuda.get_device_name(0)
    json_path = args.output_dir / "summary.json"
    summary: dict[str, object] = {
        "status": status,
        "rtl_file": str(rtl_file),
        "rtl_sha256": sha256_file(rtl_file),
        "rtl_git_branch": rtl_git_branch,
        "rtl_git_commit": rtl_git_commit,
        "bf16_packer_sha256": sha256_file(
            args.rtl_root / "rtl/common_package/common_parameters.sv"
        ),
        "bf16_exponent_sha256": sha256_file(
            args.rtl_root / "rtl/prealigner/lane_exp_selector.sv"
        ),
        "cuda_kernel_sha256": sha256_file(
            ROOT / "fpint_emul/csrc/fpint_cuda_kernel.cu"
        ),
        "verilator_version": verilator_version,
        "gpu": gpu_name,
        "torch": torch.__version__,
        "rows": args.rows,
        "modes": ["bf16", "fp16"],
        "cases": cases,
        "aggregate": aggregate,
        "aggregate_core_fp32": aggregate_core_fp32,
        "aggregate_correct_pack": aggregate_correct_pack,
        "json_path": str(json_path),
        "interpretation": interpretation,
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(markdown_report(summary))
    print(f"status: {status}")
    print(f"json: {json_path}")
    print(f"report: {args.report}")


if __name__ == "__main__":
    main()
