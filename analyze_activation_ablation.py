#!/usr/bin/env python3
"""Localize the output-distribution error caused by A8 activation quantization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from measure_logit_divergence import load_fp_reference, read_json
from result_analysis.load_model import load_model
from utils import quant_utils
from utils.logit_metrics import LogitMetricAccumulator


CONFIGURATIONS = (
    "a16_all",
    "a8_all",
    "a8_attention_only",
    "a8_mlp_only",
    "a8_down_proj_only",
    "a8_except_down_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--load-qmodel-path", required=True)
    parser.add_argument("--fp-results-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--k-groupsize", type=int, default=128)
    parser.add_argument("--v-groupsize", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument("--ear-top-k", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-tokens-per-document", type=int)
    args = parser.parse_args()
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if args.token_chunk_size <= 0:
        parser.error("--token-chunk-size must be positive")
    if args.ear_top_k <= 0:
        parser.error("--ear-top-k must be positive")
    if args.max_documents is not None and args.max_documents <= 0:
        parser.error("--max-documents must be positive")
    if (
        args.max_tokens_per_document is not None
        and args.max_tokens_per_document <= 0
    ):
        parser.error("--max-tokens-per-document must be positive")
    return args


def extract_wikitext_documents(fp_results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "doc_id": sample["doc_id"],
            "doc_hash": sample["doc_hash"],
            "target": sample["target"],
        }
        for sample in fp_results["samples"]["wikitext"]
    ]


def activation_is_enabled(name: str, configuration: str) -> bool:
    if name.endswith("lm_head"):
        return False
    is_attention = ".self_attn." in name
    is_mlp = ".mlp." in name
    is_down_proj = name.endswith(".down_proj")
    if configuration == "a16_all":
        return False
    if configuration == "a8_all":
        return True
    if configuration == "a8_attention_only":
        return is_attention
    if configuration == "a8_mlp_only":
        return is_mlp
    if configuration == "a8_down_proj_only":
        return is_down_proj
    if configuration == "a8_except_down_proj":
        return not is_down_proj
    raise ValueError(f"unknown configuration: {configuration}")


def set_activation_configuration(
    qlayers: dict[str, quant_utils.ActQuantWrapper], configuration: str
) -> int:
    enabled = 0
    for name, layer in qlayers.items():
        use_a8 = activation_is_enabled(name, configuration)
        layer.quantizer.bits = 8 if use_a8 else 16
        enabled += int(use_a8)
    return enabled


def new_metric(args: argparse.Namespace) -> LogitMetricAccumulator:
    return LogitMetricAccumulator(
        temperature=args.temperature,
        token_chunk_size=args.token_chunk_size,
        ear_top_k=args.ear_top_k,
    )


def add_relative_metrics(results: dict[str, Any]) -> None:
    a16 = results["a16_all"]["aggregate"]
    a8 = results["a8_all"]["aggregate"]
    a16_rejection = a16["rejection_rate_full_vocab"]
    a8_rejection = a8["rejection_rate_full_vocab"]
    full_a8_penalty = a8_rejection - a16_rejection

    for configuration, configuration_result in results.items():
        metrics = configuration_result["aggregate"]
        rejection = metrics["rejection_rate_full_vocab"]
        relative = {
            "ear_delta_vs_a16": (
                metrics["expected_acceptance_rate_full_vocab"]
                - a16["expected_acceptance_rate_full_vocab"]
            ),
            "rejection_increase_vs_a16": rejection - a16_rejection,
            "kl_fp_to_quant_increase_vs_a16_nats": (
                metrics["kl_fp_to_quant_nats"] - a16["kl_fp_to_quant_nats"]
            ),
            "top1_agreement_delta_vs_a16": (
                metrics["top1_agreement"] - a16["top1_agreement"]
            ),
        }
        if full_a8_penalty > 0:
            relative["fraction_of_full_a8_rejection_penalty"] = (
                rejection - a16_rejection
            ) / full_a8_penalty
            relative["fraction_of_full_a8_rejection_penalty_recovered"] = (
                a8_rejection - rejection
            ) / full_a8_penalty
        configuration_result["relative_to_a16"] = relative


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    fp_results = read_json(args.fp_results_path)
    documents = extract_wikitext_documents(fp_results)
    del fp_results
    if args.max_documents is not None:
        documents = documents[: args.max_documents]

    quant_model, tokenizer = load_model(
        input_model=args.input_model,
        load_qmodel_path=args.load_qmodel_path,
        optimized_rotation_path=None,
        w_bits=args.w_bits,
        a_bits=8,
        k_bits=args.k_bits,
        v_bits=args.v_bits,
        k_groupsize=args.k_groupsize,
        v_groupsize=args.v_groupsize,
        w_clip=True,
        a_asym=True,
        k_asym=True,
        v_asym=True,
        rotate=True,
        model_max_length=args.sequence_length,
        device=args.device,
    )
    quant_model.config.use_cache = False
    quant_model.eval()
    fp_model = load_fp_reference(args.input_model, args.device)
    tokenizer.model_max_length = 1_000_000_000

    qlayers = quant_utils.find_qlayers(
        quant_model, layers=[quant_utils.ActQuantWrapper]
    )
    enabled_layer_counts = {
        configuration: set_activation_configuration(qlayers, configuration)
        for configuration in CONFIGURATIONS
    }
    print(
        "Activation-quantized linear layers: "
        + ", ".join(
            f"{name}={count}" for name, count in enabled_layer_counts.items()
        ),
        flush=True,
    )

    aggregate_metrics = {name: new_metric(args) for name in CONFIGURATIONS}
    document_results = {name: [] for name in CONFIGURATIONS}
    prefix_token_id = tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError("tokenizer has no EOS token for first-token context")
    total_chunks = 0

    with torch.inference_mode():
        for document_index, document in enumerate(documents, start=1):
            token_ids = tokenizer(
                document["target"],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            if args.max_tokens_per_document is not None:
                token_ids = token_ids[: args.max_tokens_per_document]
            per_document = {name: new_metric(args) for name in CONFIGURATIONS}
            document_chunks = 0

            for start in range(0, token_ids.numel(), args.sequence_length):
                targets = token_ids[start : start + args.sequence_length]
                context_id = (
                    prefix_token_id if start == 0 else int(token_ids[start - 1])
                )
                context = torch.tensor([context_id], dtype=torch.long)
                input_ids = torch.cat((context, targets))[:-1].unsqueeze(0)
                labels = targets.unsqueeze(0)
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)
                fp_logits = fp_model(
                    input_ids=input_ids, use_cache=False, return_dict=True
                ).logits

                for configuration in CONFIGURATIONS:
                    set_activation_configuration(qlayers, configuration)
                    quant_logits = quant_model(
                        input_ids=input_ids, use_cache=False, return_dict=True
                    ).logits
                    per_document[configuration].update(
                        fp_logits, quant_logits, labels
                    )
                    aggregate_metrics[configuration].update(
                        fp_logits, quant_logits, labels
                    )
                    del quant_logits

                document_chunks += 1
                total_chunks += 1
                del input_ids, labels, fp_logits

            if token_ids.numel():
                for configuration in CONFIGURATIONS:
                    document_results[configuration].append({
                        "doc_id": document["doc_id"],
                        "doc_hash": document["doc_hash"],
                        "chunks": document_chunks,
                        **per_document[configuration].compute(),
                    })
            print(
                f"WikiText document {document_index}/{len(documents)}: "
                f"{token_ids.numel()} tokens, {document_chunks} chunks",
                flush=True,
            )

    results = {
        configuration: {
            "activation_quantized_linear_layers": enabled_layer_counts[configuration],
            "aggregate": aggregate_metrics[configuration].compute(),
            "documents": document_results[configuration],
        }
        for configuration in CONFIGURATIONS
    }
    add_relative_metrics(results)
    return {
        "metric_definition": {
            "purpose": (
                "causal ablation of A8 linear-input quantization while holding "
                "the W4 checkpoint, KV4 quantization, rotations, prompts, and FP reference fixed"
            ),
            "a8": "asymmetric per-token dynamic activation quantization",
            "tail_ear": (
                "histogram-estimated per-token full-vocabulary probability-overlap quantiles"
            ),
            "warning": (
                "module-only penalties need not add linearly because transformer layers interact"
            ),
        },
        "model": args.input_model,
        "quantized_checkpoint": args.load_qmodel_path,
        "fp_results": args.fp_results_path,
        "quantization": {
            "w_bits": args.w_bits,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "k_groupsize": args.k_groupsize,
            "v_groupsize": args.v_groupsize,
        },
        "scope": {
            "documents": len(documents),
            "chunks": total_chunks,
            "sequence_length": args.sequence_length,
            "full_corpus": (
                args.max_documents is None
                and args.max_tokens_per_document is None
                and len(documents) == 62
            ),
        },
        "configurations": results,
    }


def main() -> None:
    args = parse_args()
    for path in (args.input_model, args.load_qmodel_path, args.fp_results_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    result = evaluate(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {
        name: {
            "ear": values["aggregate"]["expected_acceptance_rate_full_vocab"],
            "ear_p01": values["aggregate"]["expected_acceptance_rate_full_vocab_p01"],
            "fraction_ear_below_0.80": values["aggregate"][
                "fraction_tokens_ear_below_0.80"
            ],
            **values["relative_to_a16"],
        }
        for name, values in result["configurations"].items()
    }
    print(json.dumps(summary, indent=2))
    print(f"Saved activation ablation: {output}")


if __name__ == "__main__":
    main()
