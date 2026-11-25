#!/usr/bin/env python3
"""Validate the paired Llama-3-8B experiment before reporting its full-task scores."""
import argparse
import json
import math
from pathlib import Path

TASKS = {
    "hellaswag": ("acc_norm,none", 10042),
    "arc_easy": ("acc_norm,none", 2376),
    "arc_challenge": ("acc_norm,none", 1172),
    "winogrande": ("acc,none", 1267),
    "openbookqa": ("acc_norm,none", 500),
    "wikitext": ("word_perplexity,none", 62),
}
CONDITIONS = ("fp_base", "aqp16", "aqp8")


def collect(root):
    payloads = {}
    for name in CONDITIONS:
        data = json.loads((root / "results" / f"{name}.json").read_text())
        meta = data["spinquant_quantization"]
        aqp = 8 if name == "aqp8" else 16
        weight_kv = 16 if name == "fp_base" else 4
        expected = {
            "weight_bits": weight_kv, "key_bits": weight_kv, "value_bits": weight_kv,
            "activation_bits": aqp, "query_bits": aqp, "probability_bits": aqp,
            "rotate": name != "fp_base", "compute_dtype": "fp16",
            "attention_backend": "eager", "linear_backend": "standard",
            "model_max_length": 2048, "num_fewshot": 0, "seed": 0, "eval_limit": None,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"{name}: {key}={meta.get(key)!r}, expected {value!r}")
        if set(data["results"]) != set(TASKS):
            raise ValueError(f"{name}: unexpected task set")
        for task, (metric, count) in TASKS.items():
            samples = data["n-samples"][task]
            if samples["original"] != count or samples["effective"] != count:
                raise ValueError(f"{name}: incomplete {task}")
            if not math.isfinite(data["results"][task][metric]):
                raise ValueError(f"{name}: nonfinite {task} score")
        payloads[name] = data
    for key in ("model", "compute_dtype", "model_max_length", "lm_eval_batch_size"):
        if len({data["spinquant_quantization"][key] for data in payloads.values()}) != 1:
            raise ValueError(f"Conditions differ in {key}")
    for key in ("quantized_checkpoint_sha256", "rotation_checkpoint_sha256"):
        value = payloads["aqp16"]["spinquant_quantization"][key]
        if not value or value != payloads["aqp8"]["spinquant_quantization"][key]:
            raise ValueError(f"AQP16 and AQP8 must share {key}")
    # Hashes verify that the full evaluations used the same documents and prompts.
    for task in TASKS:
        reference = payloads["fp_base"]["samples"][task]
        keys = [(s["doc_id"], s["doc_hash"], s["prompt_hash"]) for s in reference]
        for name in ("aqp16", "aqp8"):
            current = payloads[name]["samples"][task]
            if keys != [(s["doc_id"], s["doc_hash"], s["prompt_hash"]) for s in current]:
                raise ValueError(f"{name}: different {task} documents or prompts")
    return payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    data = collect(args.root)
    meta = data["aqp16"]["spinquant_quantization"]
    lines = [
        "# Llama-3-8B: FP16 baseline and AQP16/AQP8", "",
        "Original Meta-Llama-3-8B base model (not Llama-3.1 or Instruct).",
        "All conditions use FP16, eager attention, length 2048, zero-shot, seed 0, and full datasets.",
        f"lm-eval batch size: {meta['lm_eval_batch_size']}.",
        "FP base disables quantization and rotation. AQP16/AQP8 share W4/K4/V4 and weight/rotation checkpoints.",
        "AQP16 is 16-bit bypass for A/Q/P; AQP8 quantizes all three to 8 bits.", "",
        "| Task | Metric | Samples | FP base | AQP16 | AQP8 | AQP8 − AQP16 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    summary = {"conditions": {name: d["spinquant_quantization"] for name, d in data.items()}, "tasks": {}}
    for task, (metric, count) in TASKS.items():
        values = [data[name]["results"][task][metric] for name in CONDITIONS]
        if task == "wikitext":
            cells = [f"{value:.6f}" for value in values]
            delta = f"{values[2] - values[1]:+.6f} ({100 * (values[2] / values[1] - 1):+.3f}%)"
        else:
            cells = [f"{100 * value:.4f}%" for value in values]
            delta = f"{100 * (values[2] - values[1]):+.4f} pp"
        lines.append(f"| {task} | {metric} | {count} | " + " | ".join(cells) + f" | {delta} |")
        summary["tasks"][task] = {"metric": metric, "samples": count, **dict(zip(CONDITIONS, values))}
    count = sum(n for task, (_, n) in TASKS.items() if task != "wikitext")
    micro = {
        name: sum(d["results"][task][metric] * n for task, (metric, n) in TASKS.items() if task != "wikitext") / count
        for name, d in data.items()
    }
    summary["micro_accuracy"] = micro
    lines += [
        f"| Micro average | sample-weighted accuracy | {count} | "
        + " | ".join(f"{100 * micro[name]:.4f}%" for name in CONDITIONS)
        + f" | {100 * (micro['aqp8'] - micro['aqp16']):+.4f} pp |",
        "", "## Provenance", "",
        f"- Model: `{meta['model']}`",
        f"- Weight checkpoint SHA-256: `{meta['quantized_checkpoint_sha256']}`",
        f"- Rotation SHA-256: `{meta['rotation_checkpoint_sha256']}`",
        "- Full task counts and document/prompt hashes match across all three conditions.",
        "- Raw JSON files retain task configuration, sample scores, standard errors, and environment metadata.",
        "- These are task-quality results; intermediate QK/PV errors are not measured in this run.", "",
        "## Reproduction", "", "```bash",
        "conda activate spinquant", "bash scripts/run_llama3_8b_accuracy.sh", "```", "",
    ]
    (args.root / "SUMMARY.md").write_text("\n".join(lines))
    (args.root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Validated and summarized", args.root)


if __name__ == "__main__":
    main()
