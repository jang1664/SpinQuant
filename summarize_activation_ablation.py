#!/usr/bin/env python3
"""Summarize activation-ablation metrics with a paired document bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_260_720)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    return args


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def bootstrap_contrast(
    values: np.ndarray,
    weights: np.ndarray,
    bootstrap_indices: np.ndarray,
) -> dict[str, Any]:
    samples = np.sum(
        weights[bootstrap_indices] * values[bootstrap_indices], axis=1
    ) / np.sum(weights[bootstrap_indices], axis=1)
    low, high = np.quantile(samples, (0.025, 0.975))
    return {
        "estimate": weighted_mean(values, weights),
        "ci95_document_bootstrap": [float(low), float(high)],
        "positive_documents": int(np.sum(values > 0)),
        "negative_documents": int(np.sum(values < 0)),
        "zero_documents": int(np.sum(values == 0)),
        "bootstrap_probability_le_zero": float(np.mean(samples <= 0)),
        "bootstrap_probability_ge_zero": float(np.mean(samples >= 0)),
    }


def main() -> None:
    args = parse_args()
    with args.input.open(encoding="utf-8") as handle:
        source = json.load(handle)
    configurations = source["configurations"]
    document_keys = [
        (doc["doc_id"], doc["doc_hash"])
        for doc in configurations["a16_all"]["documents"]
    ]
    weights = np.asarray(
        [doc["tokens"] for doc in configurations["a16_all"]["documents"]],
        dtype=np.float64,
    )
    metrics: dict[str, dict[str, np.ndarray]] = {}
    metric_names = (
        "expected_acceptance_rate_full_vocab",
        "fraction_tokens_ear_below_0.50",
        "fraction_tokens_ear_below_0.80",
        "fraction_tokens_ear_below_0.90",
        "kl_fp_to_quant_nats",
        "nll_delta_quant_minus_fp_nats",
        "top1_agreement",
    )
    for configuration, result in configurations.items():
        keys = [(doc["doc_id"], doc["doc_hash"]) for doc in result["documents"]]
        if keys != document_keys:
            raise ValueError(f"document mismatch for {configuration}")
        metrics[configuration] = {
            metric: np.asarray(
                [doc[metric] for doc in result["documents"]], dtype=np.float64
            )
            for metric in metric_names
        }

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0,
        len(weights),
        size=(args.bootstrap_samples, len(weights)),
    )
    ear = {
        name: values["expected_acceptance_rate_full_vocab"]
        for name, values in metrics.items()
    }
    contrasts = {
        "a16_minus_a8_ear": ear["a16_all"] - ear["a8_all"],
        "a8_except_down_proj_minus_a8_all_ear": (
            ear["a8_except_down_proj"] - ear["a8_all"]
        ),
        "attention_mlp_interaction_ear": (
            ear["a8_all"]
            - ear["a8_attention_only"]
            - ear["a8_mlp_only"]
            + ear["a16_all"]
        ),
        "down_proj_rest_interaction_ear": (
            ear["a8_all"]
            - ear["a8_except_down_proj"]
            - ear["a8_down_proj_only"]
            + ear["a16_all"]
        ),
        "a8_minus_a16_fraction_tokens_ear_below_0.50": (
            metrics["a8_all"]["fraction_tokens_ear_below_0.50"]
            - metrics["a16_all"]["fraction_tokens_ear_below_0.50"]
        ),
        "a8_minus_a16_fraction_tokens_ear_below_0.80": (
            metrics["a8_all"]["fraction_tokens_ear_below_0.80"]
            - metrics["a16_all"]["fraction_tokens_ear_below_0.80"]
        ),
        "a8_minus_a16_fraction_tokens_ear_below_0.90": (
            metrics["a8_all"]["fraction_tokens_ear_below_0.90"]
            - metrics["a16_all"]["fraction_tokens_ear_below_0.90"]
        ),
        "a8_minus_a16_kl_fp_to_quant_nats": (
            metrics["a8_all"]["kl_fp_to_quant_nats"]
            - metrics["a16_all"]["kl_fp_to_quant_nats"]
        ),
        "a8_minus_a16_nll_delta_nats": (
            metrics["a8_all"]["nll_delta_quant_minus_fp_nats"]
            - metrics["a16_all"]["nll_delta_quant_minus_fp_nats"]
        ),
        "a16_minus_a8_top1_agreement": (
            metrics["a16_all"]["top1_agreement"]
            - metrics["a8_all"]["top1_agreement"]
        ),
    }
    summary = {
        "source": str(args.input),
        "documents": len(weights),
        "tokens": int(np.sum(weights)),
        "bootstrap": {
            "unit": "WikiText document",
            "paired": True,
            "samples": args.bootstrap_samples,
            "seed": args.seed,
        },
        "aggregates": {
            name: result["aggregate"] for name, result in configurations.items()
        },
        "contrasts": {
            name: bootstrap_contrast(values, weights, bootstrap_indices)
            for name, values in contrasts.items()
        },
    }
    output = args.output or args.input.with_name(
        f"{args.input.stem}-summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary["contrasts"], indent=2))
    print(f"Saved activation-ablation summary: {output}")


if __name__ == "__main__":
    main()
