"""Measure FP16-relative matrix-output error for AQP8 and AQP16 W4/KV4.

Only Linear projection, raw QK, and post-P-QDQ raw PV outputs are observed.
No logits or non-matmul tensors are collected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from measure_logit_divergence import read_json
from result_analysis.load_model import load_model
from utils import quant_utils
from utils.matrix_output_metrics import GROUPS, MatrixOutputObserver, comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--load-qmodel-path", required=True)
    parser.add_argument("--fp-results-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rotation-path", required=True)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-tokens-per-document", type=int)
    args = parser.parse_args()
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    for name in ("max_documents", "max_tokens_per_document"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def wikitext_documents(results: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"doc_id": sample["doc_id"], "doc_hash": sample["doc_hash"], "target": sample["target"]}
        for sample in results["samples"]["wikitext"]
    ]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def linear_modules(model: torch.nn.Module, quantized: bool) -> dict[str, torch.nn.Module]:
    if quantized:
        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, quant_utils.ActQuantWrapper) and name != "lm_head"
        }
    else:
        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear)
            and name != "lm_head"
            and not ".module" in name
        }
    if not modules:
        raise ValueError("no eligible linear projection modules found")
    return modules


def attach_observer(
    model: torch.nn.Module, observer: MatrixOutputObserver, *, quantized: bool
) -> list[Any]:
    handles = []
    for name, module in linear_modules(model, quantized).items():
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, name: str = name) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"linear projection {name} returned {type(output)!r}")
            observer.record("Linear", name, output)

        handles.append(module.register_forward_hook(hook))
    for layer in model.model.layers:
        layer.self_attn.matrix_output_observer = observer
    return handles


def detach_observer(model: torch.nn.Module, handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()
    for layer in model.model.layers:
        layer.self_attn.matrix_output_observer = None


def validate_quantized_configuration(model: torch.nn.Module, bits: int) -> dict[str, Any]:
    qlayers = linear_modules(model, quantized=True)
    wrong_linear = {
        name: layer.quantizer.bits
        for name, layer in qlayers.items()
        if layer.quantizer.bits != bits
    }
    if wrong_linear:
        raise ValueError(f"unexpected linear activation bits: {wrong_linear}")
    qk_bits = []
    p_bits = []
    for index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        wrapper = getattr(attention, "apply_rotary_pos_emb_qk_rotation_wrapper", None)
        observed_q_bits = 16 if wrapper is None else wrapper.q_bits
        observed_p_bits = 16 if attention.p_quantizer is None else attention.p_quantizer.bits
        qk_bits.append(observed_q_bits)
        p_bits.append(observed_p_bits)
    if set(qk_bits) != {bits} or set(p_bits) != {bits}:
        raise ValueError(f"unexpected Q/P bits: Q={set(qk_bits)}, P={set(p_bits)}, expected={bits}")
    return {
        "linear_activation_bits": bits,
        "q_bits_by_layer": qk_bits,
        "p_bits_by_layer": p_bits,
        "attention_backend": getattr(model.config, "_attn_implementation", None),
    }


def load_condition(args: argparse.Namespace, bits: int) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    model, tokenizer = load_model(
        input_model=args.input_model,
        load_qmodel_path=args.load_qmodel_path,
        optimized_rotation_path=None,
        w_bits=4,
        a_bits=bits,
        q_bits=bits,
        q_groupsize=128,
        # Keep Q quantization identical to run_hard_workload_comparison.sh:
        # Q uses the parser's symmetric default, while A and P are asymmetric.
        q_asym=False,
        p_bits=bits,
        # Hard-workload P quantization is per probability row. Unlike Q/K
        # head groups, a fixed width would reject short final chunks.
        p_groupsize=-1,
        p_asym=True,
        k_bits=4,
        v_bits=4,
        k_groupsize=128,
        v_groupsize=128,
        w_clip=True,
        a_asym=True,
        k_asym=True,
        v_asym=True,
        rotate=True,
        model_max_length=args.sequence_length,
        attention_backend="eager",
        device=args.device,
    )
    model.config.use_cache = False
    model.eval()
    return model, tokenizer, validate_quantized_configuration(model, bits)


def load_rotated_fp16_reference(args: argparse.Namespace) -> torch.nn.Module:
    """Use the same rotation basis as the W4 checkpoint, while retaining FP16 operands."""
    model, _ = load_model(
        input_model=args.input_model,
        load_qmodel_path=None,
        optimized_rotation_path=args.rotation_path,
        w_bits=16,
        a_bits=16,
        q_bits=16,
        p_bits=16,
        k_bits=16,
        v_bits=16,
        rotate=True,
        model_max_length=args.sequence_length,
        attention_backend="eager",
        device=args.device,
    )
    model.config.use_cache = False
    return model.eval()


def measure_condition(
    documents: list[dict[str, Any]], tokenizer: Any, fp_model: torch.nn.Module,
    quant_model: torch.nn.Module, args: argparse.Namespace,
) -> tuple[dict[str, dict[str, float | int]], int]:
    observer = MatrixOutputObserver()
    fp_handles = attach_observer(fp_model, observer, quantized=True)
    quant_handles = attach_observer(quant_model, observer, quantized=True)
    prefix_token_id = tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError("tokenizer has no EOS token")
    chunks = 0
    try:
        with torch.inference_mode():
            for document in documents:
                token_ids = tokenizer(document["target"], return_tensors="pt", add_special_tokens=False).input_ids[0]
                if args.max_tokens_per_document is not None:
                    token_ids = token_ids[: args.max_tokens_per_document]
                for start in range(0, token_ids.numel(), args.sequence_length):
                    targets = token_ids[start : start + args.sequence_length]
                    context_id = prefix_token_id if start == 0 else int(token_ids[start - 1])
                    input_ids = torch.cat((torch.tensor([context_id]), targets))[:-1].unsqueeze(0).to(args.device)
                    observer.begin_reference()
                    fp_model(input_ids=input_ids, use_cache=False, return_dict=True)
                    observer.begin_candidate()
                    quant_model(input_ids=input_ids, use_cache=False, return_dict=True)
                    observer.finish_candidate()
                    chunks += 1
    finally:
        detach_observer(fp_model, fp_handles)
        detach_observer(quant_model, quant_handles)
    return observer.results(), chunks


def validate_same_element_counts(
    error_a: dict[str, dict[str, float | int]],
    error_b: dict[str, dict[str, float | int]],
) -> None:
    for group in GROUPS:
        if error_a[group]["elements"] != error_b[group]["elements"]:
            raise ValueError(
                f"{group}: AQP8/AQP16 element counts differ: "
                f"{error_a[group]['elements']} != {error_b[group]['elements']}"
            )


def main() -> None:
    args = parse_args()
    for path in (args.input_model, args.load_qmodel_path, args.fp_results_path, args.rotation_path):
        if not Path(path).exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but unavailable")
    fp_results = read_json(args.fp_results_path)
    documents = wikitext_documents(fp_results)
    if args.max_documents is not None:
        documents = documents[: args.max_documents]
    fp_model = load_rotated_fp16_reference(args)
    aqp16, tokenizer, aqp16_config = load_condition(args, 16)
    error_b, chunks_b = measure_condition(documents, tokenizer, fp_model, aqp16, args)
    # Keep only FP16 plus one fake-quantized model resident. SpinQuant keeps
    # fake-quantized weights in floating point, so retaining both conditions
    # would needlessly consume another full model's GPU memory.
    del aqp16
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    aqp8, _, aqp8_config = load_condition(args, 8)
    error_a, chunks_a = measure_condition(documents, tokenizer, fp_model, aqp8, args)
    if chunks_a != chunks_b:
        raise ValueError(f"AQP8/AQP16 chunk counts differ: {chunks_a} != {chunks_b}")
    validate_same_element_counts(error_a, error_b)
    groups = {
        group: {
            "error_A_fp16_vs_aqp8": error_a[group],
            "error_B_fp16_vs_aqp16": error_b[group],
            "A_vs_B": comparison(error_a[group], error_b[group]),
        }
        for group in GROUPS
    }
    result = {
        "conditions": {
            "AQP8": {"w_bits": 4, "k_bits": 4, "v_bits": 4, "a_bits": 8, "q_bits": 8, "p_bits": 8},
            "AQP16": {"w_bits": 4, "k_bits": 4, "v_bits": 4, "a_bits": 16, "q_bits": 16, "p_bits": 16},
            "attention_backend": "eager",
            "quantized_checkpoint": args.load_qmodel_path,
            "rotation_checkpoint": args.rotation_path,
            "quantized_checkpoint_sha256": sha256_file(args.load_qmodel_path),
            "rotation_checkpoint_sha256": sha256_file(args.rotation_path),
            "verified_configuration": {"AQP8": aqp8_config, "AQP16": aqp16_config},
        },
        "scope": {
            "dataset": "WikiText lm-eval samples",
            "documents": len(documents),
            "chunks": chunks_a,
            "sequence_length": args.sequence_length,
            "full_corpus": args.max_documents is None
            and args.max_tokens_per_document is None
            and len(documents) == 62,
        },
        "groups": groups,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(groups, indent=2))
    print(f"Saved matrix-output accuracy metrics: {output}")


if __name__ == "__main__":
    main()
