#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = {
    "HellaSwag": ("hellaswag", "acc_norm,none"),
    "ARC Easy": ("arc_easy", "acc_norm,none"),
    "ARC Challenge": ("arc_challenge", "acc_norm,none"),
    "Winogrande": ("winogrande", "acc,none"),
    "OpenBookQA": ("openbookqa", "acc_norm,none"),
    "WikiText PPL": ("wikitext", "word_perplexity,none"),
}
DEFAULT_MODELS = ["llama2-7b", "llama3.1-8b", "llama3.2-3b"]


def load_metrics(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Result JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path}: missing object field 'results'")

    values = {}
    for label, (task, field) in METRICS.items():
        try:
            values[label] = float(results[task][field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: missing numeric metric results.{task}.{field}"
            ) from exc
    return values


def build_rows(factorized_dir, zero_padding_dir, models):
    rows = []
    for model in models:
        factorized = load_metrics(Path(factorized_dir) / f"{model}.json")
        zero_padding = load_metrics(
            Path(zero_padding_dir) / f"{model}.json"
        )
        for metric in METRICS:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "factorized": factorized[metric],
                    "zero_padding": zero_padding[metric],
                    "delta": zero_padding[metric] - factorized[metric],
                }
            )
    return rows


def markdown_table(rows):
    lines = [
        "| Model | Metric | Factorized | Zero padding | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | "
            f"{row['factorized']:.6f} | {row['zero_padding']:.6f} | "
            f"{row['delta']:+.6f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorized-dir", type=Path, required=True)
    parser.add_argument("--zero-padding-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    print(
        markdown_table(
            build_rows(
                args.factorized_dir,
                args.zero_padding_dir,
                args.models,
            )
        )
    )


if __name__ == "__main__":
    main()
