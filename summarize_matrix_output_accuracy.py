"""Render the three permitted matrix-output groups from result JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def format_error(metrics: dict[str, float | int]) -> str:
    return "<br>".join(
        f"{label}={metrics[key]:.6g}"
        for key, label in (
            ("mae", "MAE"),
            ("rmse", "RMSE"),
            ("relative_l2_error", "relL2"),
            ("max_abs_error", "maxAbs"),
            ("cosine_similarity", "cos"),
        )
    )


def format_comparison(metrics: dict[str, dict[str, float]]) -> tuple[str, str]:
    difference = []
    ratio = []
    for key, label in (
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("relative_l2_error", "relL2"),
        ("max_abs_error", "maxAbs"),
    ):
        difference.append(f"{label}={metrics[key]['difference']:+.6g}")
        ratio.append(f"{label}={metrics[key]['ratio']:.6g}")
    difference.append(
        f"cos={metrics['cosine_similarity']['difference']:+.6g}"
    )
    return "<br>".join(difference), "<br>".join(ratio)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = [
        "# AQP8 vs AQP16 matrix-output accuracy",
        "",
        "`error_A` is FP16 vs AQP8 W4/KV4; `error_B` is FP16 vs AQP16 W4/KV4.",
        "Positive `A−B` or ratio above 1 means AQP8 has the additional error.",
        "",
        "| Model / group | error_A (FP16 vs AQP8) | error_B (FP16 vs AQP16) | A−B | A/B |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for path in args.inputs:
        data = json.loads(path.read_text(encoding="utf-8"))
        model = Path(data.get("conditions", {}).get("quantized_checkpoint", path)).parent.name
        for group in ("Linear", "QK", "PV"):
            values = data["groups"][group]
            error_a = format_error(values["error_A_fp16_vs_aqp8"])
            error_b = format_error(values["error_B_fp16_vs_aqp16"])
            difference, ratio = format_comparison(values["A_vs_B"])
            lines.append(
                f"| {model} / {group} | {error_a} | {error_b} | "
                f"{difference} | {ratio} |"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved summary: {args.output}")


if __name__ == "__main__":
    main()
