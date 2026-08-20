#!/usr/bin/env python3
"""Summarize FP/AQP hard-workload lm-eval JSON results."""
import argparse
import json
from pathlib import Path

CONDITIONS = ("fp_base", "aqp16", "aqp8")
METRIC_PREFERENCES = {
    "mmlu": ("acc,none", "acc_norm,none"),
    "gsm8k_cot": ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none"),
    "bbh_cot_zeroshot": ("exact_match,strict-match", "exact_match,none"),
    "gpqa_diamond_zeroshot": ("acc_norm,none", "acc,none"),
}


def read(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("results"), dict):
        raise ValueError(f"{path}: missing results object")
    return payload["results"]


def metric(task, result):
    values = result.get(task)
    # lm-eval strips the `mmlu_` prefix from single MMLU task result keys.
    if values is None and task.startswith("mmlu_"):
        values = result.get(task[len("mmlu_"):])
    if not isinstance(values, dict):
        raise ValueError(f"missing task {task}")
    preferences = METRIC_PREFERENCES.get(task)
    if preferences is None and task.startswith("bbh"):
        preferences = ("exact_match,strict-match", "exact_match,flexible-extract", "exact_match,none")
    if preferences is None:
        preferences = ("acc,none", "acc_norm,none", "exact_match,none")
    for key in preferences:
        if key in values:
            return float(values[key]), key
    raise ValueError(f"{task}: no supported metric; available={sorted(values)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--tasks", default="mmlu,gsm8k_cot,bbh_cot_zeroshot,gpqa_diamond_zeroshot")
    args = parser.parse_args()
    payloads = {name: read(args.results_dir / f"{name}.json") for name in CONDITIONS}
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    lines = [
        "| Task | Metric | FP base | AQP16 | AQP8 | AQP8-FP (pp) | AQP8-AQP16 (pp) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for task in tasks:
        values = {name: metric(task, payloads[name])[0] for name in CONDITIONS}
        metric_name = metric(task, payloads["fp_base"])[1]
        lines.append(
            f"| {task} | {metric_name} | {values['fp_base']:.6f} | "
            f"{values['aqp16']:.6f} | {values['aqp8']:.6f} | "
            f"{(values['aqp8'] - values['fp_base']) * 100:+.3f} | "
            f"{(values['aqp8'] - values['aqp16']) * 100:+.3f} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
