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


def render_report(
    random: dict[str, Any],
    smoke: dict[str, Any],
    full: dict[str, Any] | None = None,
) -> str:
    random_reference = [
        metrics
        for record in random["records"]
        for name, metrics in record["comparisons"].items()
        if name.endswith("vs_reference")
    ]
    random_qdq = [
        record["comparisons"]["fpint_vs_standard_qdq"]
        for record in random["records"]
    ]
    linear = smoke["linear_outputs"]["aggregate"]
    smoke_logits = smoke["logits"]["allclose_metrics"]
    smoke_distribution = smoke["logits"]["distribution_metrics"]
    failed_layers = smoke["linear_outputs"]["failed_layers"]
    per_layer = smoke["linear_outputs"]["per_layer"]
    worst_layers = sorted(
        per_layer.items(),
        key=lambda item: item[1]["relative_l2_error"],
        reverse=True,
    )[:5]
    lines = [
        "# MXU ROW 128 QCOL / Llama 3.1 8B 실험 결과",
        "",
        "## 설정",
        "",
        "- QCOL_REAL_2SCOMP, INT4, weight group size 128, MXU ROW 128",
        "- Llama 3.1 8B W4 GPTQ symmetric, A/Q/K/V/P FP16",
        "- Reference backend: standard QDQ Linear",
        "- Candidate backend: FPINT CUDA",
        "",
        "## Random QCOL",
        "",
        f"- 상태: `{random['status']}`",
        f"- Reference 비교 수: {len(random_reference)}",
        f"- 실패 수: {sum(not row['allclose'] for row in random_reference)}",
        f"- 최대 절대 오차: {number(max(row['max_abs'] for row in random_reference))}",
        f"- Standard QDQ all-close: {sum(row['allclose'] for row in random_qdq)}/{len(random_qdq)} "
        f"(최대 절대 오차 {number(max(row['max_abs'] for row in random_qdq))})",
        "",
        "## 실제 Linear smoke",
        "",
        f"- 상태: `{smoke['status']}` (`{smoke['evaluation_policy']['name']}`)",
        "- All-close는 diagnostic이며 smoke gate에 사용하지 않음",
        f"- 관측 Linear: {len(smoke['linear_outputs']['per_layer'])}",
        f"- 통과 / 실패 Linear: {len(per_layer) - len(failed_layers)} / {len(failed_layers)}",
        f"- All-close: `{linear['allclose']}` (`atol={linear['atol']}`, `rtol={linear['rtol']}`)",
        f"- Max abs / MAE / RMSE: {number(linear['max_abs_error'])} / "
        f"{number(linear['mae'])} / {number(linear['rmse'])}",
        f"- Relative L2 / cosine: {number(linear['relative_l2_error'])} / "
        f"{number(linear['cosine_similarity'])}",
        f"- 허용 오차 밖 element: {linear['outside_tolerance']}/{linear['elements']} "
        f"({number(linear['outside_tolerance_fraction'])})",
        "- Relative L2가 큰 Linear:",
        *[
            f"  - `{name}`: {number(metrics['relative_l2_error'])}"
            for name, metrics in worst_layers
        ],
        "",
        "## Smoke logits",
        "",
        f"- 문서 / token: {smoke['scope']['documents']} / {smoke['scope']['tokens']}",
        f"- Logit all-close: `{smoke_logits['allclose']}`",
        f"- Logit max abs / RMSE: {number(smoke_logits['max_abs_error'])} / "
        f"{number(smoke_logits['rmse'])}",
        f"- Symmetric KL: {number(smoke_distribution['symmetric_kl_nats'])}",
        f"- JS divergence: {number(smoke_distribution['js_divergence_nats'])}",
        f"- Top-1 agreement: {number(smoke_distribution['top1_agreement'])}",
        f"- Standard / FPINT perplexity: {number(smoke_distribution.get('fp_perplexity'))} / "
        f"{number(smoke_distribution.get('quant_perplexity'))}",
        "",
        "## 성능 참고",
        "",
        f"- Standard tokens/s: {number(smoke['timing']['standard_tokens_per_second'])}",
        f"- FPINT CUDA tokens/s: {number(smoke['timing']['fpint_cuda_tokens_per_second'])}",
        "",
    ]
    if full is None:
        lines.extend(
            [
                "## Full workload",
                "",
                "- 아직 실행 결과가 없다.",
            ]
        )
    else:
        wikitext = full["tasks"]["wikitext"]
        micro = full["accuracy_micro"]
        lines.extend(
            [
                "## Full WikiText",
                "",
                f"- Standard / FPINT PPL: {number(wikitext['standard'])} / "
                f"{number(wikitext['fpint_cuda'])}",
                f"- Absolute / relative delta: {number(wikitext['delta_fpint_minus_standard'])} / "
                f"{number(wikitext['relative_delta'])}",
                "",
                "## Downstream tasks",
                "",
                "| Task | Samples | Standard | FPINT | Delta (pp) | Standard stderr | FPINT stderr |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in TASK_METRICS:
            metrics = full["tasks"][name]
            if metrics["kind"] != "accuracy":
                continue
            lines.append(
                f"| {name} | {metrics['samples']} | {number(metrics['standard'])} | "
                f"{number(metrics['fpint_cuda'])} | {number(metrics['delta_percentage_points'])} | "
                f"{number(metrics['standard_stderr'])} | {number(metrics['fpint_cuda_stderr'])} |"
            )
        lines.extend(
            [
                "",
                f"- Micro accuracy standard / FPINT: {number(micro['standard'])} / "
                f"{number(micro['fpint_cuda'])}",
                f"- Micro accuracy delta: {number(100.0 * micro['delta_fpint_minus_standard'])} pp",
                "",
                "## 전체 workload 성능",
                "",
                f"- Standard lm-eval seconds 합: {number(full['evaluation_seconds_sum']['standard'])}",
                f"- FPINT CUDA lm-eval seconds 합: {number(full['evaluation_seconds_sum']['fpint_cuda'])}",
                "- 자동 품질 pass/fail 기준 없음",
            ]
        )
    lines.extend(
        [
            "",
            f"Checkpoint SHA256: `{smoke['conditions']['quantized_checkpoint_sha256']}`",
            "",
            "원본 JSON/CSV에 환경, coverage, layer별 오차와 checkpoint 경로가 기록되어 있다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--standard-results", type=Path, nargs="+")
    parser.add_argument("--fpint-results", type=Path, nargs="+")
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    random = load(args.random)
    smoke = load(args.smoke)
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
    report = render_report(random, smoke, full)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()
