#!/usr/bin/env python3
"""Sweep signed raw scale exponent fields with fixed GEMM activation inputs."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from fpint_emul import FpIntConfig, fpint_linear, qcol_real_2scomp_reference
from measure_fpint_qcol_accuracy import (
    DEFAULT_K_VALUES,
    DEFAULT_EXPONENT_MAX_BY_K,
    DEFAULT_BF16_EXPONENT_MAX_BY_K,
    conventional_linear,
    fp64_reference_linear,
    make_case,
    sample_scale_fields,
    parse_int_list,
    qcol_allclose_metrics,
    sha256_file,
)


DEFAULT_SCALE_MAXIMA = {
    "fp16": tuple(range(11, 31)),
    "bf16": (123, 127, 143, 159, 175, 191, 207, 223, 231, 239, *range(240, 255)),
}
CANDIDATES = ("fp64", "rounded_reference", "gpu_true", "gpu_false", "fpint")


def tensor_counts(tensor):
    tensor = tensor.detach()
    return {
        "total": tensor.numel(),
        "finite": int(torch.isfinite(tensor).sum().item()),
        "nan": int(torch.isnan(tensor).sum().item()),
        "positive_inf": int(torch.isposinf(tensor).sum().item()),
        "negative_inf": int(torch.isneginf(tensor).sum().item()),
    }


def output_coverage(outputs):
    cpu = {name: value.detach().cpu() for name, value in outputs.items()}
    common = torch.ones_like(cpu["fp64"], dtype=torch.bool)
    for value in cpu.values():
        common &= torch.isfinite(value)
    return {
        "outputs": {name: tensor_counts(value) for name, value in cpu.items()},
        "common_finite": int(common.sum().item()),
        "total": common.numel(),
    }


def aggregate(records, target):
    total = sum(row["total"] for row in records)
    common = sum(row["common_finite"] for row in records)
    outputs = {}
    for name in CANDIDATES:
        counts = {
            key: sum(row["outputs"][name][key] for row in records)
            for key in ("total", "finite", "nan", "positive_inf", "negative_inf")
        }
        outputs[name] = {**counts, "finite_fraction": counts["finite"] / total}
    checked = [row["reference_check"] for row in records if row["reference_check"] is not None]
    return {
        "cases": len(records), "total": total, "common_finite": common,
        "common_finite_fraction": common / total,
        "finite_target_met": common / total >= target,
        "outputs": outputs,
        "dequant_weight_nonfinite": sum(row["dequant_weight_nonfinite"] for row in records),
        "reference_checks": len(checked),
        "reference_checks_passed": sum(row["allclose"] for row in checked),
    }


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    dtype = args.activation_format
    config = FpIntConfig(4, 128, 128, activation_format=dtype)
    activation_maxima = (
        DEFAULT_EXPONENT_MAX_BY_K if dtype == "fp16" else DEFAULT_BF16_EXPONENT_MAX_BY_K
    )
    records = []
    for trial in range(args.trials):
        for k_index, k in enumerate(args.k_values):
            seed = args.base_seed + trial * len(args.k_values) + k_index
            activation, weight, _, zero = make_case(
                m=args.m, k=k, n=args.n, config=config, seed=seed,
                exponent_max=activation_maxima[k],
            )
            activation = torch.as_tensor(activation)
            weight = torch.from_numpy(weight)
            zero = torch.from_numpy(zero)
            a, w, z = (value.to(device) for value in (activation, weight, zero))
            group_indices = torch.arange(k, device=device) // config.group_size
            input_hash = hashlib.sha256(
                activation.contiguous().view(torch.int16).numpy().tobytes()
                + weight.numpy().tobytes()
            ).hexdigest()
            for exponent_max in args.scale_exp_max_values:
                scale = sample_scale_fields(
                    seed, (args.n, config.group_count(k)), dtype, exponent_max,
                    args.scale_exp_min,
                )
                scale_device = scale.to(device)
                reference = fp64_reference_linear(a, w, scale_device, config.group_size)
                true = conventional_linear(a, w, scale_device, config.group_size, True)
                false = conventional_linear(a, w, scale_device, config.group_size, False)
                fpint = fpint_linear(
                    a, w, scale_device, z, config,
                    backend="fpint_cuda" if device.type == "cuda" else "fpint_torch",
                    has_zero=False,
                )
                outputs = {
                    "fp64": reference, "rounded_reference": reference.to(a.dtype),
                    "gpu_true": true, "gpu_false": false, "fpint": fpint,
                }
                coverage = output_coverage(outputs)
                dequant = (w.float() * scale_device[:, group_indices].float()).to(a.dtype)
                reference_check = None
                if trial < args.reference_trials:
                    expected = torch.from_numpy(qcol_real_2scomp_reference(
                        activation, weight, scale, zero, config
                    ))
                    reference_check = qcol_allclose_metrics(fpint, expected, 0., 0.)
                bits = scale.view(torch.int16).numpy().view(np.uint16)
                records.append({
                    "trial": trial, "seed": seed, "k": k,
                    "activation_exp_max": activation_maxima[k],
                    "scale_exp_max": exponent_max, **coverage,
                    "input_sha256": input_hash,
                    "scale_sha256": hashlib.sha256(bits.tobytes()).hexdigest(),
                    "scale_positive_sign": int((bits >> 15 == 0).sum()),
                    "scale_negative_sign": int((bits >> 15 == 1).sum()),
                    "scale_numeric_zeros": int((scale == 0).sum()),
                    "dequant_weight_nonfinite": int((~torch.isfinite(dequant)).sum().item()),
                    "reference_check": reference_check,
                })
        print(f"{dtype}: trial {trial + 1}/{args.trials} complete ({len(records)} cases)", flush=True)
    summary = {}
    for exponent_max in args.scale_exp_max_values:
        rows = [row for row in records if row["scale_exp_max"] == exponent_max]
        by_k = {
            str(k): aggregate([row for row in rows if row["k"] == k], args.finite_target)
            for k in args.k_values
        }
        summary[str(exponent_max)] = {
            "overall": aggregate(rows, args.finite_target), "by_k": by_k,
            "all_k_meet_target": all(row["finite_target_met"] for row in by_k.values()),
        }
    checks_passed = all(
        row["reference_check"] is None or row["reference_check"]["allclose"]
        for row in records
    )
    sources = (
        "measure_fpint_scale_sweep.py", "measure_fpint_qcol_accuracy.py",
        "fpint_emul/reference.py", "fpint_emul/torch_backend.py",
        "fpint_emul/csrc/fpint_cuda_kernel.cu",
    )
    return {
        "status": "completed" if checks_passed else "reference_check_failed",
        "config": {
            **vars(config), "m": args.m, "n": args.n, "trials": args.trials,
            "k_values": list(args.k_values), "base_seed": args.base_seed,
            "activation_exp_max_by_k": {str(k): activation_maxima[k] for k in args.k_values},
            "scale_exp_min": args.scale_exp_min,
            "scale_exp_max_values": list(args.scale_exp_max_values),
            "scale_sign": [0, 1], "scale_mantissa": [0, (1 << config.mantissa_bits) - 1],
            "scale_format": dtype, "scale_sampling": "independent_uniform_raw_fields_v1",
            "reference_trials": args.reference_trials, "finite_target": args.finite_target,
            "zero_point": 0, "baseline_reduced_precision_modes": [True, False],
        },
        "policy": {
            "finite_target_is_diagnostic_only": True,
            "reference_checks": "exact finite values; NaNs equal; Inf signs must match",
            "paired_across_scale_ranges": "same activation/weight/sign/mantissa/exponent-uniform draws",
            "scale_exponent": "exp_min + floor(U * (exp_max - exp_min + 1))",
            "zero_and_subnormal_scales_included": True,
        },
        "environment": {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
            "torch": torch.__version__, "numpy": np.__version__, "cuda": torch.version.cuda,
            "source_sha256": {
                source: sha256_file(Path(__file__).parent / source) for source in sources
            },
        },
        "summary": summary, "records": records,
    }


def write_outputs(result, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    rows = []
    for record in result["records"]:
        row = {key: record[key] for key in (
            "trial", "seed", "k", "activation_exp_max", "scale_exp_max", "common_finite", "total",
            "scale_positive_sign", "scale_negative_sign", "scale_numeric_zeros",
            "dequant_weight_nonfinite", "input_sha256", "scale_sha256",
        )}
        for name, counts in record["outputs"].items():
            row.update({f"{name}_{key}": value for key, value in counts.items()})
        check = record["reference_check"]
        row["reference_check"] = None if check is None else check["allclose"]
        rows.append(row)
    with path.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_rows = []
    for exponent, data in result["summary"].items():
        for k, row in data["by_k"].items():
            summary_rows.append({
                "scale_exp_max": int(exponent), "k": int(k),
                "common_finite_fraction": row["common_finite_fraction"],
                **{f"{name}_finite_fraction": row["outputs"][name]["finite_fraction"] for name in CANDIDATES},
            })
    with path.with_name(path.stem + "-summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-format", choices=("fp16", "bf16"), required=True)
    parser.add_argument("--scale-exp-min", type=int, default=0)
    parser.add_argument("--scale-exp-max-values", type=parse_int_list)
    parser.add_argument("--k-values", type=parse_int_list, default=DEFAULT_K_VALUES)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--reference-trials", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=20260915)
    parser.add_argument("--finite-target", type=float, default=.999)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.scale_exp_max_values is None:
        args.scale_exp_max_values = DEFAULT_SCALE_MAXIMA[args.activation_format]
    largest = 30 if args.activation_format == "fp16" else 254
    if not all(0 <= args.scale_exp_min <= value <= largest for value in args.scale_exp_max_values):
        parser.error("scale exponent range includes nonfinite fields or is reversed")
    if len(set(args.scale_exp_max_values)) != len(args.scale_exp_max_values):
        parser.error("duplicate scale exponent maxima")
    if min(args.m, args.n, args.trials) <= 0 or not 0 <= args.reference_trials <= args.trials:
        parser.error("invalid shape or trial count")
    if not 0 <= args.finite_target <= 1:
        parser.error("finite target must be in [0, 1]")
    if not set(args.k_values) <= set(DEFAULT_K_VALUES):
        parser.error("K values must use the existing activation exponent map")
    return args


def main():
    args = parse_args()
    result = run(args)
    write_outputs(result, args.output)
    print(f"Saved {args.output}: {result['status']}", flush=True)
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
