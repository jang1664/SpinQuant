#!/usr/bin/env python3
"""Render the MXU ROW 128 JSON artifacts as a compact Markdown report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


TASK_METRICS = {
    "wikitext": ("word_perplexity,none", "perplexity"),
    "hellaswag": ("acc_norm,none", "accuracy"),
    "arc_easy": ("acc_norm,none", "accuracy"),
    "arc_challenge": ("acc_norm,none", "accuracy"),
    "winogrande": ("acc,none", "accuracy"),
    "openbookqa": ("acc_norm,none", "accuracy"),
}


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def _coverage_is_valid(metadata: dict[str, Any], backend: str) -> bool:
    coverage = metadata.get("fpint_linear_coverage", [])
    decoder = [item for item in coverage if item.get("name") != "lm_head"]
    lm_head = [item for item in coverage if item.get("name") == "lm_head"]
    return (
        len(decoder) == 224
        and all(item.get("backend") == backend for item in decoder)
        and len(lm_head) == 1
        and lm_head[0].get("backend") == "standard"
    )


def _collect_results(
    payloads: list[dict[str, Any]], backend: str
) -> tuple[dict[str, dict[str, Any]], str, float]:
    collected: dict[str, dict[str, Any]] = {}
    checkpoint_sha: str | None = None
    evaluation_seconds = 0.0
    for payload in payloads:
        metadata = payload.get("spinquant_quantization", {})
        expected = {
            "linear_backend": backend,
            "weight_bits": 4,
            "weight_groupsize": 128,
            "weight_symmetric": True,
            "fpint_mxu_rows": 128,
        }
        mismatches = {
            name: {"expected": value, "actual": metadata.get(name)}
            for name, value in expected.items()
            if metadata.get(name) != value
        }
        if mismatches:
            raise ValueError(f"{backend} result metadata mismatch: {mismatches}")
        if not _coverage_is_valid(metadata, backend):
            raise ValueError(f"{backend} result has invalid Linear coverage")
        observed_sha = metadata.get("quantized_checkpoint_sha256")
        if not observed_sha:
            raise ValueError(f"{backend} result has no checkpoint SHA256")
        if checkpoint_sha is None:
            checkpoint_sha = observed_sha
        elif checkpoint_sha != observed_sha:
            raise ValueError(f"{backend} shards use different checkpoints")
        elapsed = metadata.get("evaluation_seconds")
        if not isinstance(elapsed, (int, float)) or elapsed < 0:
            raise ValueError(f"{backend} result has invalid evaluation_seconds")
        evaluation_seconds += float(elapsed)
        payload_tasks = set(payload.get("results", {}))
        if set(metadata.get("eval_tasks", [])) != payload_tasks:
            raise ValueError(f"{backend} result task metadata does not match payload")
        for task, metrics in payload.get("results", {}).items():
            if task not in TASK_METRICS:
                raise ValueError(f"unexpected task in {backend} result: {task}")
            if task in collected:
                raise ValueError(f"duplicate {backend} task result: {task}")
            sample_count = payload.get("n-samples", {}).get(task, {}).get("effective")
            if not isinstance(sample_count, int) or sample_count <= 0:
                raise ValueError(f"{backend} {task} has invalid sample count")
            expected_batch = "1" if task == "wikitext" else "32"
            observed_batch = str(metadata.get("lm_eval_batch_size"))
            if observed_batch != expected_batch:
                raise ValueError(
                    f"{backend} {task} uses batch {observed_batch}, "
                    f"expected {expected_batch}"
                )
            collected[task] = {
                "metrics": metrics,
                "samples": sample_count,
                "batch_size": observed_batch,
            }
    if checkpoint_sha is None:
        raise ValueError(f"no {backend} results supplied")
    missing = set(TASK_METRICS) - set(collected)
    if missing:
        raise ValueError(f"missing {backend} tasks: {sorted(missing)}")
    return collected, checkpoint_sha, evaluation_seconds


def aggregate_full_results(
    standard_payloads: list[dict[str, Any]],
    fpint_payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    standard, standard_sha, standard_seconds = _collect_results(
        standard_payloads, "standard"
    )
    fpint, fpint_sha, fpint_seconds = _collect_results(
        fpint_payloads, "fpint_cuda"
    )
    if standard_sha != fpint_sha:
        raise ValueError("standard and FPINT results use different checkpoints")

    tasks: dict[str, Any] = {}
    standard_correct = 0.0
    fpint_correct = 0.0
    accuracy_samples = 0
    for task, (metric_name, kind) in TASK_METRICS.items():
        standard_task = standard[task]
        fpint_task = fpint[task]
        if standard_task["samples"] != fpint_task["samples"]:
            raise ValueError(f"{task} sample counts differ between backends")
        if standard_task["batch_size"] != fpint_task["batch_size"]:
            raise ValueError(f"{task} batch sizes differ between backends")
        standard_value = standard_task["metrics"].get(metric_name)
        fpint_value = fpint_task["metrics"].get(metric_name)
        if not isinstance(standard_value, (int, float)) or not math.isfinite(
            standard_value
        ):
            raise ValueError(f"standard {task} has invalid {metric_name}")
        if not isinstance(fpint_value, (int, float)) or not math.isfinite(fpint_value):
            raise ValueError(f"FPINT {task} has invalid {metric_name}")
        delta = float(fpint_value - standard_value)
        row = {
            "kind": kind,
            "metric": metric_name,
            "samples": standard_task["samples"],
            "standard": float(standard_value),
            "fpint_cuda": float(fpint_value),
            "delta_fpint_minus_standard": delta,
        }
        if kind == "accuracy":
            stderr_name = metric_name.replace(",none", "_stderr,none")
            standard_stderr = standard_task["metrics"].get(stderr_name)
            fpint_stderr = fpint_task["metrics"].get(stderr_name)
            if not isinstance(standard_stderr, (int, float)) or not math.isfinite(
                standard_stderr
            ):
                raise ValueError(f"standard {task} has invalid {stderr_name}")
            if not isinstance(fpint_stderr, (int, float)) or not math.isfinite(
                fpint_stderr
            ):
                raise ValueError(f"FPINT {task} has invalid {stderr_name}")
            row["standard_stderr"] = float(standard_stderr)
            row["fpint_cuda_stderr"] = float(fpint_stderr)
            row["delta_percentage_points"] = 100.0 * delta
            samples = standard_task["samples"]
            standard_correct += float(standard_value) * samples
            fpint_correct += float(fpint_value) * samples
            accuracy_samples += samples
        else:
            row["relative_delta"] = delta / max(abs(float(standard_value)), 1e-30)
        tasks[task] = row
    return {
        "status": "complete",
        "quality_gate": None,
        "checkpoint_sha256": standard_sha,
        "tasks": tasks,
        "accuracy_micro": {
            "samples": accuracy_samples,
            "standard": standard_correct / accuracy_samples,
            "fpint_cuda": fpint_correct / accuracy_samples,
            "delta_fpint_minus_standard": (
                fpint_correct - standard_correct
            )
            / accuracy_samples,
        },
        "evaluation_seconds_sum": {
            "standard": standard_seconds,
            "fpint_cuda": fpint_seconds,
        },
    }


def render_report(random: dict[str, Any]) -> str:
    """Render the paired FP64-reference numerical experiment."""

    config = random["config"]
    summary = random["summary"]
    overall = summary["overall"]
    fields = config["fp16_fields"]
    exp_by_k = fields["exponent_max_by_k"]
    lines = [
        "# MXU ROW 128 FP64-reference FP×INT 오차 비교",
        "",
        "## 설정",
        "",
        f"- 상태: `{random['status']}`",
        f"- Shape: M={config['m']}, N={config['n']}",
        f"- K: {', '.join(str(value) for value in config['k_values'])}",
        f"- Trials: {config['trials']}",
        f"- Finite target per K: {number(config['finite_target'])}",
        f"- FP16 raw field uniform sampling: sign {fields['sign']}, "
        f"exponent min {fields['exponent_min']}, mantissa {fields['mantissa']}",
        f"- Signed INT{config['weight_bits']} uniform sampling: {config['integer_range']}",
        "- Scale=1, zero-point=0 (raw FP×INT)",
        f"- MXU row / group size: {config['mxu_rows']} / {config['group_size']}",
        "- FP64 reference: FP64 activation × FP64-cast integer weight on GPU",
        "- Conventional: FP16 activation × FP16-cast integer weight on GPU",
        "- FPINT: QCOL_REAL_2SCOMP CUDA emulation",
        "- RMSE는 unrounded FP64 reference, ULP는 correctly-rounded FP16 reference 기준",
        "- 세 output이 모두 finite인 common mask에서 두 error를 paired 비교",
        "",
        "## Finite coverage",
        "",
        "| K | EXP max | Common finite | Fraction | FP16-ref non-finite | Conventional non-finite | FPINT non-finite | Target |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for k in config["k_values"]:
        row = summary["by_k"][str(k)]
        coverage = row["coverage"]
        lines.append(
            f"| {k} | {exp_by_k[str(k)]} | {coverage['common_finite_elements']} / "
            f"{coverage['output_elements']} | {number(coverage['common_finite_fraction'])} | "
            f"{coverage['fp16_rounded_reference_nonfinite']} | "
            f"{coverage['conventional_nonfinite']} | {coverage['fp_int_nonfinite']} | "
            f"{'pass' if row['finite_target_met'] else 'fail'} |"
        )
    lines.extend(
        [
            "",
            "## RMSE / signed error",
            "",
            "아래 `mean ± std`는 30개 trial metric의 평균과 sample std다.",
            "",
            "| K | Conventional RMSE mean ± std | FPINT RMSE mean ± std | FPINT-Conv | FPINT/Conv | Conventional signed-error std | FPINT signed-error std |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in config["k_values"]:
        row = summary["by_k"][str(k)]
        conv = row["conventional_err"]
        fp_int = row["fp_int_err"]
        comparison = row["comparison"]["trial_rmse_mean"]
        lines.append(
            f"| {k} | {number(conv['trial_rmse_mean'])} ± "
            f"{number(conv['trial_rmse_sample_std'])} | "
            f"{number(fp_int['trial_rmse_mean'])} ± "
            f"{number(fp_int['trial_rmse_sample_std'])} | "
            f"{number(comparison['delta_fp_int_minus_conventional'])} | "
            f"{number(comparison['ratio_fp_int_over_conventional'])} | "
            f"{number(conv['global_signed_error_std'])} | "
            f"{number(fp_int['global_signed_error_std'])} |"
        )
    lines.extend(
        [
            "",
            "## FP16 ULP error",
            "",
            "| K | Conventional mean ULP ± std | FPINT mean ULP ± std | FPINT-Conv | FPINT/Conv | Conventional ULP std | FPINT ULP std | Conventional / FPINT p95 |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for k in config["k_values"]:
        row = summary["by_k"][str(k)]
        conv = row["conventional_err"]
        fp_int = row["fp_int_err"]
        comparison = row["comparison"]["trial_mean_ulp_mean"]
        lines.append(
            f"| {k} | {number(conv['trial_mean_ulp_mean'])} ± "
            f"{number(conv['trial_mean_ulp_sample_std'])} | "
            f"{number(fp_int['trial_mean_ulp_mean'])} ± "
            f"{number(fp_int['trial_mean_ulp_sample_std'])} | "
            f"{number(comparison['delta_fp_int_minus_conventional'])} | "
            f"{number(comparison['ratio_fp_int_over_conventional'])} | "
            f"{number(conv['global_std_ulp'])} | {number(fp_int['global_std_ulp'])} | "
            f"{number(conv['trial_p95_ulp_mean'])} / "
            f"{number(fp_int['trial_p95_ulp_mean'])} |"
        )
    qcol = overall["qcol_correctness"]
    sign = overall["activation_sign_bits"]
    lines.extend(
        [
            "",
            "## Correctness / reproducibility",
            "",
            f"- Activation sign-bit positive / negative: {sign['positive']} / {sign['negative']}",
            f"- FPINT Torch vs QCOL reference all-close: "
            f"{qcol['torch_allclose_cases']}/{overall['cases']}",
            f"- FPINT CUDA vs QCOL reference all-close: "
            f"{qcol['cuda_allclose_cases']}/{qcol['cuda_cases']}",
            f"- Conventional reduced-precision reduction: "
            f"`{random['environment']['allow_fp16_reduced_precision_reduction']}`",
            "",
            f"CUDA kernel SHA256: `{random['environment']['fpint_cuda_kernel_sha256']}`",
            "",
            "원본 JSON/CSV에는 case별 seed, field 분포, common finite mask, "
            "Conventional_err와 FP_INT_err가 기록되어 있다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--standard-results", type=Path, nargs="+")
    parser.add_argument("--fpint-results", type=Path, nargs="+")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random = load(args.random)
    if bool(args.standard_results) != bool(args.fpint_results):
        parser.error("--standard-results and --fpint-results must be supplied together")
    full = None
    if args.standard_results:
        full = aggregate_full_results(
            [load(path) for path in args.standard_results],
            [load(path) for path in args.fpint_results],
        )
        if args.metrics_output is None:
            parser.error("--metrics-output is required for full results")
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(
            json.dumps(full, indent=2) + "\n", encoding="utf-8"
        )
    elif args.metrics_output is not None:
        parser.error("--metrics-output requires full results")
    report = render_report(random)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
