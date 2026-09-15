#!/usr/bin/env python3
"""Compare FP16/AQP matrix outputs over pinned, non-WikiText workloads only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from measure_matrix_output_accuracy import (
    attach_observer,
    detach_observer,
    load_condition,
    load_rotated_fp16_reference,
    sha256_file,
    validate_same_element_counts,
)
from utils.matrix_output_metrics import GROUPS, MatrixOutputObserver, comparison
from utils.matrix_workloads import MatrixWorkload, load_workloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--load-qmodel-path", required=True)
    parser.add_argument("--rotation-path", required=True)
    parser.add_argument("--workload-manifest", required=True)
    parser.add_argument("--workloads", required=True, help="Comma-separated manifest workload names")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fp-results-path", help="Required only when workloads includes legacy wikitext")
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-tokens-per-example", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.sequence_length < 2 or args.max_tokens_per_example < 2:
        parser.error("sequence lengths must be at least 2")
    args.workload_names = [name.strip() for name in args.workloads.split(",") if name.strip()]
    if not args.workload_names:
        parser.error("--workloads must contain at least one name")
    return args


def manifest_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure_workload(
    workload: MatrixWorkload,
    tokenizer: Any,
    fp_model: torch.nn.Module,
    quant_model: torch.nn.Module,
    args: argparse.Namespace,
) -> tuple[dict[str, dict[str, float | int]], dict[str, int]]:
    """Pair every FP reference forward with precisely one candidate forward."""
    observer = MatrixOutputObserver()
    fp_handles = attach_observer(fp_model, observer, quantized=True)
    quant_handles = attach_observer(quant_model, observer, quantized=True)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    chunks = 0
    tokens = 0
    measured_examples = 0
    try:
        with torch.inference_mode():
            for example in workload.examples:
                token_ids = tokenizer(
                    example.text, return_tensors="pt", add_special_tokens=False
                ).input_ids[0][: args.max_tokens_per_example]
                if token_ids.numel() < 2:
                    continue
                measured_examples += 1
                tokens += token_ids.numel()
                for start in range(0, token_ids.numel(), args.sequence_length):
                    targets = token_ids[start : start + args.sequence_length]
                    if targets.numel() < 2:
                        continue
                    context_id = eos_token_id if start == 0 else int(token_ids[start - 1])
                    input_ids = torch.cat((torch.tensor([context_id]), targets))[:-1]
                    input_ids = input_ids.unsqueeze(0).to(args.device)
                    observer.begin_reference()
                    fp_model(input_ids=input_ids, use_cache=False, return_dict=True)
                    observer.begin_candidate()
                    quant_model(input_ids=input_ids, use_cache=False, return_dict=True)
                    observer.finish_candidate()
                    chunks += 1
    finally:
        detach_observer(fp_model, fp_handles)
        detach_observer(quant_model, quant_handles)
    return observer.results(), {
        "source_documents": len({example.example_id.rsplit(":choice-", 1)[0] for example in workload.examples}),
        "request_examples": len(workload.examples),
        "measured_examples": measured_examples,
        "chunks": chunks,
        "input_tokens_after_cap": tokens,
    }


def build_result(
    workload: MatrixWorkload,
    error_a: dict[str, dict[str, float | int]],
    error_b: dict[str, dict[str, float | int]],
    counts: dict[str, int],
    args: argparse.Namespace,
    manifest: dict[str, Any],
    aqp8_config: dict[str, Any],
    aqp16_config: dict[str, Any],
) -> dict[str, Any]:
    validate_same_element_counts(error_a, error_b)
    groups = {
        group: {
            "error_A_fp16_vs_aqp8": error_a[group],
            "error_B_fp16_vs_aqp16": error_b[group],
            "A_vs_B": comparison(error_a[group], error_b[group]),
        }
        for group in GROUPS
    }
    return {
        "workload": {
            "name": workload.name,
            "family": workload.family,
            "adapter_config": workload.config,
            "manifest_version": manifest["version"],
            "manifest_sha256": manifest_sha256(args.workload_manifest),
        },
        "conditions": {
            "AQP8": {"w_bits": 4, "k_bits": 4, "v_bits": 4, "a_bits": 8, "q_bits": 8, "p_bits": 8},
            "AQP16": {"w_bits": 4, "k_bits": 4, "v_bits": 4, "a_bits": 16, "q_bits": 16, "p_bits": 16},
            "attention_backend": "eager",
            "quantized_checkpoint": args.load_qmodel_path,
            "rotation_checkpoint": args.rotation_path,
            "quantized_checkpoint_sha256": args.quantized_checkpoint_sha256,
            "rotation_checkpoint_sha256": args.rotation_checkpoint_sha256,
            "verified_configuration": {"AQP8": aqp8_config, "AQP16": aqp16_config},
        },
        "scope": {
            **counts,
            "sequence_length": args.sequence_length,
            "max_tokens_per_example": args.max_tokens_per_example,
            "input_kinds": dict(Counter(example.kind for example in workload.examples)),
            "matrix_outputs_only": list(GROUPS),
        },
        "groups": groups,
    }


def main() -> None:
    args = parse_args()
    for path in (
        args.input_model,
        args.load_qmodel_path,
        args.rotation_path,
        args.workload_manifest,
    ):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if "wikitext" in args.workload_names and not args.fp_results_path:
        raise ValueError("wikitext workload requires --fp-results-path")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")

    manifest, workloads = load_workloads(
        args.workload_manifest, args.workload_names,
        legacy_fp_results_path=args.fp_results_path,
    )
    args.quantized_checkpoint_sha256 = sha256_file(args.load_qmodel_path)
    args.rotation_checkpoint_sha256 = sha256_file(args.rotation_path)
    fp_model = load_rotated_fp16_reference(args)
    aqp16, tokenizer, aqp16_config = load_condition(args, 16)
    error_b = {}
    counts = {}
    for workload in workloads:
        error_b[workload.name], counts[workload.name] = measure_workload(
            workload, tokenizer, fp_model, aqp16, args
        )
    del aqp16
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    aqp8, _, aqp8_config = load_condition(args, 8)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for workload in workloads:
        error_a, counts_a = measure_workload(workload, tokenizer, fp_model, aqp8, args)
        if counts_a != counts[workload.name]:
            raise ValueError(f"{workload.name}: AQP8/AQP16 scope differs: {counts_a} != {counts[workload.name]}")
        result = build_result(
            workload, error_a, error_b[workload.name], counts_a, args,
            manifest, aqp8_config, aqp16_config,
        )
        output = output_dir / f"{workload.name}.json"
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {workload.name}: {output}", flush=True)


if __name__ == "__main__":
    main()
