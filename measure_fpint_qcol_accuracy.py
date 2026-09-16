#!/usr/bin/env python3
"""Sweep random QCOL_REAL_2SCOMP inputs against the independent reference."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from fpint_emul import (
    FpIntConfig,
    dequantize_weight,
    fpint_linear,
    qcol_real_2scomp_reference,
)


DEFAULT_K_VALUES = (127, 128, 129, 255, 256, 257, 1024, 4096, 14336)
ZERO_MODES = ("symmetric", "asymmetric")
DISTRIBUTIONS = ("gaussian", "componentwise")


def parse_int_list(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item) for item in value.split(",") if item)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("all values must be positive")
    return values


def componentwise_fp16(rng: np.random.Generator, shape: tuple[int, ...]) -> np.ndarray:
    """Sample sign, exponent and mantissa independently, then round to FP16."""

    sign = rng.integers(0, 2, size=shape, dtype=np.int8)
    exponent = rng.integers(-24, 16, size=shape, dtype=np.int16)
    mantissa = np.clip(np.rint(rng.normal(512.0, 256.0, size=shape)), 0, 1023)
    value = (1.0 + mantissa / 1024.0) * np.exp2(exponent.astype(np.float64))
    return np.where(sign != 0, -value, value).astype(np.float16)


def make_case(
    *,
    m: int,
    k: int,
    n: int,
    config: FpIntConfig,
    zero_mode: str,
    distribution: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    if distribution == "componentwise":
        activation = componentwise_fp16(rng, (m, k))
    elif distribution == "gaussian":
        activation = rng.normal(size=(m, k)).astype(np.float16)
    else:
        raise ValueError(f"unknown distribution: {distribution}")

    qmin = -(1 << (config.weight_bits - 1))
    qmax = (1 << (config.weight_bits - 1)) - 1
    weight = rng.integers(qmin, qmax + 1, size=(n, k), dtype=np.int16).astype(
        np.int8
    )
    groups = config.group_count(k)
    scale = rng.uniform(2.0**-14, 2.0**-11, size=(n, groups)).astype(np.float16)
    if zero_mode == "symmetric":
        zero = np.zeros((n, groups), dtype=np.int32)
    elif zero_mode == "asymmetric":
        zero = rng.integers(-4, 4, size=(n, groups), dtype=np.int32)
        if not np.any(zero):
            zero[0, 0] = 1
    else:
        raise ValueError(f"unknown zero mode: {zero_mode}")
    return activation, weight, scale, zero


def error_metrics(actual: torch.Tensor, expected: torch.Tensor, atol: float, rtol: float) -> dict:
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    difference = (actual - expected).abs()
    threshold = atol + rtol * expected.abs()
    outside = difference > threshold
    return {
        "allclose": not bool(outside.any()),
        "elements": difference.numel(),
        "outside_tolerance": int(outside.sum()),
        "outside_tolerance_fraction": float(outside.float().mean()),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rms": float(difference.square().mean().sqrt()),
    }


def flatten_rows(records: Iterable[dict]) -> Iterable[dict]:
    for record in records:
        common = {
            key: record[key]
            for key in ("trial", "seed", "distribution", "zero_mode", "m", "k", "n")
        }
        for comparison, metrics in record["comparisons"].items():
            yield {**common, "comparison": comparison, **metrics}


def write_csv(path: Path, records: list[dict]) -> None:
    rows = list(flatten_rows(records))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    config = FpIntConfig(
        weight_bits=args.bits,
        group_size=args.group_size,
        mxu_rows=args.mxu_rows,
        extra_bits=args.extra_bits,
        reduce_extra_bits=args.reduce_extra_bits,
    )
    records: list[dict] = []
    all_reference_close = True
    case_index = 0
    for trial in range(args.trials):
        for distribution in args.distributions:
            for zero_mode in args.zero_modes:
                for k in args.k_values:
                    seed = args.base_seed + case_index
                    case_index += 1
                    activation, weight, scale, zero = make_case(
                        m=args.m,
                        k=k,
                        n=args.n,
                        config=config,
                        zero_mode=zero_mode,
                        distribution=distribution,
                        seed=seed,
                    )
                    reference = torch.from_numpy(
                        qcol_real_2scomp_reference(
                            activation, weight, scale, zero, config
                        )
                    )
                    tensors = tuple(
                        torch.from_numpy(value).to(device)
                        for value in (activation, weight, scale, zero)
                    )
                    actual_torch = fpint_linear(
                        *tensors, config, backend="fpint_torch"
                    )
                    comparisons = {
                        "fpint_torch_vs_reference": error_metrics(
                            actual_torch, reference, args.atol, args.rtol
                        )
                    }
                    if device.type == "cuda":
                        actual_cuda = fpint_linear(
                            *tensors, config, backend="fpint_cuda"
                        )
                        comparisons["fpint_cuda_vs_reference"] = error_metrics(
                            actual_cuda, reference, args.atol, args.rtol
                        )
                    else:
                        actual_cuda = actual_torch
                    qdq = torch.nn.functional.linear(
                        tensors[0],
                        dequantize_weight(tensors[1], tensors[2], tensors[3], config),
                    )
                    comparisons["fpint_vs_standard_qdq"] = error_metrics(
                        actual_cuda, qdq, args.atol, args.rtol
                    )
                    for name, metrics in comparisons.items():
                        if name.endswith("vs_reference"):
                            all_reference_close &= metrics["allclose"]
                    records.append(
                        {
                            "trial": trial,
                            "seed": seed,
                            "distribution": distribution,
                            "zero_mode": zero_mode,
                            "m": args.m,
                            "k": k,
                            "n": args.n,
                            "comparisons": comparisons,
                        }
                    )
                    print(
                        f"[{case_index}/{args.trials * len(args.distributions) * len(args.zero_modes) * len(args.k_values)}] "
                        f"{distribution} {zero_mode} K={k}: "
                        f"reference_allclose={all(m['allclose'] for name, m in comparisons.items() if name.endswith('vs_reference'))}",
                        flush=True,
                    )

    return {
        "status": "pass" if all_reference_close else "fail",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
        },
        "config": {
            **vars(config),
            "atol": args.atol,
            "rtol": args.rtol,
            "trials": args.trials,
            "base_seed": args.base_seed,
            "k_values": list(args.k_values),
            "distributions": list(args.distributions),
            "zero_modes": list(args.zero_modes),
        },
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bits", type=int, choices=(4, 8), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mxu-rows", type=int, default=128)
    parser.add_argument("--extra-bits", type=int, default=19)
    parser.add_argument("--reduce-extra-bits", type=int, default=10)
    parser.add_argument("--m", type=int, default=4)
    parser.add_argument("--n", type=int, default=37)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--base-seed", type=int, default=20260915)
    parser.add_argument(
        "--distributions", nargs="+", choices=DISTRIBUTIONS, default=DISTRIBUTIONS
    )
    parser.add_argument("--zero-modes", nargs="+", choices=ZERO_MODES, default=ZERO_MODES)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.m <= 0 or args.n <= 0 or args.trials <= 0:
        parser.error("--m, --n and --trials must be positive")
    if args.atol < 0 or args.rtol < 0:
        parser.error("tolerances must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output.with_suffix(".csv"), result["records"])
    print(f"Saved {args.output} and {args.output.with_suffix('.csv')}")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
