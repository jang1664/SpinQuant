#!/usr/bin/env python3
"""Render legacy WikiText and manifest-driven matrix-output results together."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row(model: str, payload: dict) -> str:
    workload = payload.get("workload", {"name": "wikitext", "family": "ppl"})
    cells = []
    for group, values in payload["groups"].items():
        a = values["error_A_fp16_vs_aqp8"]
        b = values["error_B_fp16_vs_aqp16"]
        ab = values["A_vs_B"]
        cells.append(
            f"| {workload['family']} | {workload['name']} | {model} | {group} | "
            f"{a['mae']:.6g} | {b['mae']:.6g} | {ab['mae']['ratio']:.5g} | "
            f"{a['rmse']:.6g} | {b['rmse']:.6g} | {ab['rmse']['ratio']:.5g} |"
        )
    return "\n".join(cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-dir", type=Path, required=True)
    parser.add_argument("--workload-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lines = [
        "# AQP8 vs AQP16 matrix-output accuracy",
        "",
        "`error_A` is FP16 vs AQP8 W4/KV4; `error_B` is FP16 vs AQP16 W4/KV4. "
        "A/Q/P are 8 bit for AQP8 and 16 bit for AQP16.  Positive A−B or "
        "A/B above 1 means AQP8 has additional error.",
        "",
        "Only Linear, raw QK, and PV matrix outputs are measured.  No logits, "
        "generation, or task-score metrics are included.",
        "",
        "| Family | Workload | Model | Group | error_A MAE | error_B MAE | MAE A/B | error_A RMSE | error_B RMSE | RMSE A/B |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in args.models:
        legacy_path = args.legacy_dir / model / "matrix-output-accuracy.json"
        if legacy_path.is_file():
            lines.append(row(model, load(legacy_path)))
        model_dir = args.workload_dir / model
        for path in sorted(model_dir.glob("*.json")):
            lines.append(row(model, load(path)))

    lines += [
        "",
        "## Extended-workload provenance",
        "",
        "- Zero-shot rows are fixed representative subsets; each answer choice is a "
        "separate prompt-plus-continuation forward request.",
        "- The second PPL row is `allenai/c4` validation because the PG19 loader "
        "was unavailable in the experiment environment; its manifest pins the "
        "dataset/configuration and source-document count.",
        "- RULER uses the official NVIDIA RULER `niah_single_1` configuration, "
        "seed 42, four generated 2K prompts, and the recorded upstream revision. "
        "It is an input stress workload only; retrieval correctness is not reported.",
        "",
    ]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved summary: {args.output}")


if __name__ == "__main__":
    main()
