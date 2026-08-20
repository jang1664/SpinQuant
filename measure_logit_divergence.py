#!/usr/bin/env python3
"""Compare FP16 and SpinQuant outputs over the complete evaluation workload."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import transformers

from eval_utils.modeling_llama import LlamaForCausalLM
from result_analysis.load_model import load_model
from utils.logit_metrics import LogitMetricAccumulator


ACCURACY_TASKS = (
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "openbookqa",
    "winogrande",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--load-qmodel-path", required=True)
    parser.add_argument("--fp-results-path", required=True)
    parser.add_argument("--quant-results-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--a-bits", type=int, choices=(8, 16), required=True)
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--k-bits", type=int, default=4)
    parser.add_argument("--v-bits", type=int, default=4)
    parser.add_argument("--k-groupsize", type=int, default=128)
    parser.add_argument("--v-groupsize", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument(
        "--ear-top-k",
        type=int,
        default=10,
        help="Number of FP top tokens used by the paper-style EAR (default: 10).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-documents",
        type=int,
        help="Testing only: limit the number of WikiText documents.",
    )
    parser.add_argument(
        "--max-tokens-per-document",
        type=int,
        help="Testing only: limit tokens evaluated from each WikiText document.",
    )
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


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


class ChoiceMetricAccumulator:
    """Aggregate differences between distributions over answer choices."""

    def __init__(self, temperature: float) -> None:
        self.temperature = temperature
        self.samples = 0
        self.choices = 0
        self.kl_fp_to_quant = 0.0
        self.kl_quant_to_fp = 0.0
        self.js_divergence = 0.0
        self.score_squared_error = 0.0
        self.score_absolute_error = 0.0
        self.top1_agreement = 0

    def update(self, fp_scores: Iterable[float], quant_scores: Iterable[float]) -> None:
        fp = torch.tensor(list(fp_scores), dtype=torch.float64)
        quant = torch.tensor(list(quant_scores), dtype=torch.float64)
        if fp.ndim != 1 or fp.numel() == 0 or fp.shape != quant.shape:
            raise ValueError("choice scores must be non-empty vectors with equal shape")
        if not torch.isfinite(fp).all() or not torch.isfinite(quant).all():
            raise ValueError("choice scores must be finite")

        fp_log_prob = F.log_softmax(fp / self.temperature, dim=0)
        quant_log_prob = F.log_softmax(quant / self.temperature, dim=0)
        fp_prob = fp_log_prob.exp()
        quant_prob = quant_log_prob.exp()
        mixture_log_prob = torch.logaddexp(fp_log_prob, quant_log_prob) - math.log(2.0)
        diff = quant - fp

        self.samples += 1
        self.choices += fp.numel()
        self.kl_fp_to_quant += (
            fp_prob * (fp_log_prob - quant_log_prob)
        ).sum().item()
        self.kl_quant_to_fp += (
            quant_prob * (quant_log_prob - fp_log_prob)
        ).sum().item()
        self.js_divergence += 0.5 * (
            (fp_prob * (fp_log_prob - mixture_log_prob)).sum()
            + (quant_prob * (quant_log_prob - mixture_log_prob)).sum()
        ).item()
        self.score_squared_error += diff.square().sum().item()
        self.score_absolute_error += diff.abs().sum().item()
        self.top1_agreement += int(fp.argmax().item() == quant.argmax().item())

    def compute(self) -> dict[str, float | int]:
        if not self.samples:
            raise ValueError("no choice scores were accumulated")
        kl_fp_to_quant = max(0.0, self.kl_fp_to_quant / self.samples)
        kl_quant_to_fp = max(0.0, self.kl_quant_to_fp / self.samples)
        score_mse = self.score_squared_error / self.choices
        return {
            "samples": self.samples,
            "choices": self.choices,
            "temperature": self.temperature,
            "kl_fp_to_quant_nats_per_sample": kl_fp_to_quant,
            "kl_quant_to_fp_nats_per_sample": kl_quant_to_fp,
            "symmetric_kl_nats_per_sample": (
                kl_fp_to_quant + kl_quant_to_fp
            ) / 2.0,
            "js_divergence_nats_per_sample": max(
                0.0, self.js_divergence / self.samples
            ),
            "score_mse": score_mse,
            "score_rmse": math.sqrt(score_mse),
            "score_mae": self.score_absolute_error / self.choices,
            "top1_agreement": self.top1_agreement / self.samples,
        }


class FlipMetricAccumulator:
    """Count paper-style correctness flips and MCQ answer changes."""

    def __init__(self) -> None:
        self.samples = 0
        self.both_correct = 0
        self.both_incorrect = 0
        self.correct_to_incorrect = 0
        self.incorrect_to_correct = 0
        self.answer_changes = 0
        self.incorrect_to_different_incorrect = 0

    def update(
        self,
        fp_correct: bool,
        quant_correct: bool,
        fp_prediction: int,
        quant_prediction: int,
    ) -> None:
        self.samples += 1
        if fp_correct and quant_correct:
            self.both_correct += 1
        elif fp_correct and not quant_correct:
            self.correct_to_incorrect += 1
        elif not fp_correct and quant_correct:
            self.incorrect_to_correct += 1
        else:
            self.both_incorrect += 1

        if fp_prediction != quant_prediction:
            self.answer_changes += 1
            if not fp_correct and not quant_correct:
                self.incorrect_to_different_incorrect += 1

    def compute(self) -> dict[str, float | int]:
        if not self.samples:
            raise ValueError("no flip samples were accumulated")
        flips = self.correct_to_incorrect + self.incorrect_to_correct
        fp_correct = self.both_correct + self.correct_to_incorrect
        quant_correct = self.both_correct + self.incorrect_to_correct
        return {
            "samples": self.samples,
            "flips": flips,
            "flip_rate": flips / self.samples,
            "correctness_agreement": 1.0 - flips / self.samples,
            "correct_to_incorrect": self.correct_to_incorrect,
            "correct_to_incorrect_rate": self.correct_to_incorrect / self.samples,
            "incorrect_to_correct": self.incorrect_to_correct,
            "incorrect_to_correct_rate": self.incorrect_to_correct / self.samples,
            "both_correct": self.both_correct,
            "both_incorrect": self.both_incorrect,
            "fp_correct": fp_correct,
            "quant_correct": quant_correct,
            "fp_correct_retention_rate": (
                self.both_correct / fp_correct if fp_correct else float("nan")
            ),
            "net_accuracy_change": (
                self.incorrect_to_correct - self.correct_to_incorrect
            ) / self.samples,
            "all_answer_flips": self.answer_changes,
            "all_answer_flip_rate": self.answer_changes / self.samples,
            "answer_agreement": 1.0 - self.answer_changes / self.samples,
            "incorrect_to_different_incorrect": self.incorrect_to_different_incorrect,
            "incorrect_to_different_incorrect_rate": (
                self.incorrect_to_different_incorrect / self.samples
            ),
        }


def sample_key(sample: dict[str, Any]) -> tuple[Any, ...]:
    return (
        sample.get("doc_id"),
        sample.get("doc_hash"),
        sample.get("prompt_hash"),
        sample.get("filter", "none"),
    )


def extract_choice_scores(sample: dict[str, Any]) -> list[float]:
    scores = []
    for response in sample["filtered_resps"]:
        value = response[0] if isinstance(response, (list, tuple)) else response
        scores.append(float(value))
    return scores


def completion_lengths(sample: dict[str, Any]) -> list[int]:
    # lm-eval's multiple-choice acc_norm divides by the character length of the
    # choice. The stored continuation includes the target delimiter (normally
    # one leading space), so remove only that delimiter here.
    return [max(1, len(arguments[1].lstrip())) for arguments in sample["arguments"]]


def compare_accuracy_tasks(
    fp_results: dict[str, Any],
    quant_results: dict[str, Any],
    temperature: float,
) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    overall_decision = ChoiceMetricAccumulator(temperature)
    overall_flips = FlipMetricAccumulator()
    total_fp_correct = 0.0
    total_quant_correct = 0.0
    total_samples = 0

    for task in ACCURACY_TASKS:
        fp_samples = fp_results["samples"][task]
        quant_by_key = {
            sample_key(sample): sample for sample in quant_results["samples"][task]
        }
        if len(quant_by_key) != len(fp_samples):
            raise ValueError(f"{task}: FP and quantized sample counts differ")

        raw = ChoiceMetricAccumulator(temperature)
        normalized = ChoiceMetricAccumulator(temperature)
        task_decision = ChoiceMetricAccumulator(temperature)
        task_flips = FlipMetricAccumulator()
        uses_normalized_score = "acc_norm" in fp_samples[0].get("metrics", [])
        metric_name = "acc_norm" if uses_normalized_score else "acc"
        fp_correct = 0.0
        quant_correct = 0.0

        for fp_sample in fp_samples:
            key = sample_key(fp_sample)
            if key not in quant_by_key:
                raise ValueError(f"{task}: missing matching quantized sample {key}")
            quant_sample = quant_by_key[key]
            if fp_sample["arguments"] != quant_sample["arguments"]:
                raise ValueError(f"{task}: prompt/choice mismatch for sample {key}")

            fp_scores = extract_choice_scores(fp_sample)
            quant_scores = extract_choice_scores(quant_sample)
            if len(fp_scores) != len(quant_scores):
                raise ValueError(f"{task}: choice count mismatch for sample {key}")
            lengths = completion_lengths(fp_sample)
            fp_normalized = [s / length for s, length in zip(fp_scores, lengths)]
            quant_normalized = [
                s / length for s, length in zip(quant_scores, lengths)
            ]

            raw.update(fp_scores, quant_scores)
            normalized.update(fp_normalized, quant_normalized)
            decision_fp = fp_normalized if uses_normalized_score else fp_scores
            decision_quant = (
                quant_normalized if uses_normalized_score else quant_scores
            )
            task_decision.update(decision_fp, decision_quant)
            overall_decision.update(decision_fp, decision_quant)
            fp_is_correct = bool(fp_sample[metric_name])
            quant_is_correct = bool(quant_sample[metric_name])
            fp_prediction = max(
                range(len(decision_fp)), key=decision_fp.__getitem__
            )
            quant_prediction = max(
                range(len(decision_quant)), key=decision_quant.__getitem__
            )
            task_flips.update(
                fp_is_correct,
                quant_is_correct,
                fp_prediction,
                quant_prediction,
            )
            overall_flips.update(
                fp_is_correct,
                quant_is_correct,
                fp_prediction,
                quant_prediction,
            )
            fp_correct += float(fp_is_correct)
            quant_correct += float(quant_is_correct)

        sample_count = len(fp_samples)
        total_fp_correct += fp_correct
        total_quant_correct += quant_correct
        total_samples += sample_count
        per_task[task] = {
            "decision_score": "length_normalized" if uses_normalized_score else "raw",
            "accuracy_metric": metric_name,
            "fp_accuracy": fp_correct / sample_count,
            "quant_accuracy": quant_correct / sample_count,
            "accuracy_delta_quant_minus_fp": (
                quant_correct - fp_correct
            ) / sample_count,
            "flips": task_flips.compute(),
            "task_decision_distribution": task_decision.compute(),
            "raw_loglikelihood_distribution": raw.compute(),
            "length_normalized_loglikelihood_distribution": normalized.compute(),
        }

    return {
        "definition": {
            "distribution": "softmax over all answer-choice loglikelihood scores",
            "direction": "KL(FP16 || quantized)",
            "aggregation": "unweighted mean over every sample",
            "task_decision": (
                "acc_norm tasks use character-length-normalized scores; "
                "other tasks use raw scores"
            ),
        },
        "overall": {
            **overall_decision.compute(),
            "fp_micro_accuracy": total_fp_correct / total_samples,
            "quant_micro_accuracy": total_quant_correct / total_samples,
            "accuracy_delta_quant_minus_fp": (
                total_quant_correct - total_fp_correct
            ) / total_samples,
            "flips": overall_flips.compute(),
        },
        "tasks": per_task,
    }


def wikitext_documents(
    fp_results: dict[str, Any], quant_results: dict[str, Any]
) -> list[dict[str, Any]]:
    fp_samples = fp_results["samples"]["wikitext"]
    quant_by_key = {
        sample_key(sample): sample for sample in quant_results["samples"]["wikitext"]
    }
    documents = []
    for sample in fp_samples:
        key = sample_key(sample)
        if key not in quant_by_key:
            raise ValueError(f"wikitext: missing matching quantized document {key}")
        quant_sample = quant_by_key[key]
        if sample["target"] != quant_sample["target"]:
            raise ValueError(f"wikitext: target mismatch for document {key}")
        documents.append({
            "doc_id": sample["doc_id"],
            "doc_hash": sample["doc_hash"],
            "target": sample["target"],
        })
    return documents


def load_fp_reference(input_model: str, device: str) -> LlamaForCausalLM:
    config = transformers.AutoConfig.from_pretrained(
        input_model, attn_implementation="eager"
    )
    clone_lm_head = bool(config.tie_word_embeddings)
    if clone_lm_head:
        config.tie_word_embeddings = False
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=input_model,
        config=config,
        torch_dtype=torch.float16,
    )
    if clone_lm_head:
        model.lm_head.weight.data = model.model.embed_tokens.weight.data.clone()
    model.config.use_cache = False
    return model.eval().to(device)


def compare_wikitext_logits(
    documents: list[dict[str, Any]],
    tokenizer: Any,
    fp_model: LlamaForCausalLM,
    quant_model: LlamaForCausalLM,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    prefix_token_id = tokenizer.eos_token_id
    if prefix_token_id is None:
        raise ValueError("tokenizer has no EOS token for the first-token context")
    aggregate = LogitMetricAccumulator(
        temperature=args.temperature,
        token_chunk_size=args.token_chunk_size,
        ear_top_k=args.ear_top_k,
    )
    document_results = []
    total_chunks = 0

    selected_documents = documents[: args.max_documents]
    with torch.inference_mode():
        for document_index, document in enumerate(selected_documents, start=1):
            token_ids = tokenizer(
                document["target"],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids[0]
            if args.max_tokens_per_document is not None:
                token_ids = token_ids[: args.max_tokens_per_document]
            document_metric = LogitMetricAccumulator(
                temperature=args.temperature,
                token_chunk_size=args.token_chunk_size,
                ear_top_k=args.ear_top_k,
            )
            document_chunks = 0

            for start in range(0, token_ids.numel(), args.sequence_length):
                targets = token_ids[start : start + args.sequence_length]
                context_id = (
                    prefix_token_id if start == 0 else int(token_ids[start - 1])
                )
                context = torch.tensor([context_id], dtype=torch.long)
                # Match lm-eval rolling likelihood: one context token followed
                # by up to model_max_length continuation tokens. Removing the
                # last target forms inputs whose logits predict every target.
                input_ids = torch.cat((context, targets))[:-1].unsqueeze(0)
                labels = targets.unsqueeze(0)
                input_ids = input_ids.to(args.device)
                labels = labels.to(args.device)

                fp_logits = fp_model(
                    input_ids=input_ids, use_cache=False, return_dict=True
                ).logits
                quant_logits = quant_model(
                    input_ids=input_ids, use_cache=False, return_dict=True
                ).logits
                document_metric.update(fp_logits, quant_logits, labels)
                aggregate.update(fp_logits, quant_logits, labels)
                document_chunks += 1
                total_chunks += 1
                del input_ids, labels, fp_logits, quant_logits

            if token_ids.numel():
                document_results.append({
                    "doc_id": document["doc_id"],
                    "doc_hash": document["doc_hash"],
                    "chunks": document_chunks,
                    **document_metric.compute(),
                })
            print(
                f"WikiText document {document_index}/{len(selected_documents)}: "
                f"{token_ids.numel()} tokens, {document_chunks} chunks",
                flush=True,
            )

    return (
        {
            "documents": len(document_results),
            "chunks": total_chunks,
            "sequence_length": args.sequence_length,
            "full_corpus": (
                args.max_documents is None
                and args.max_tokens_per_document is None
                and len(selected_documents) == len(documents)
            ),
            **aggregate.compute(),
        },
        document_results,
    )


def main() -> None:
    args = parse_args()
    for path in (
        args.load_qmodel_path,
        args.fp_results_path,
        args.quant_results_path,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but is not available")

    fp_results = read_json(args.fp_results_path)
    quant_results = read_json(args.quant_results_path)
    accuracy_metrics = compare_accuracy_tasks(
        fp_results, quant_results, args.temperature
    )
    documents = wikitext_documents(fp_results, quant_results)
    del fp_results, quant_results

    quant_model, tokenizer = load_model(
        input_model=args.input_model,
        load_qmodel_path=args.load_qmodel_path,
        optimized_rotation_path=None,
        w_bits=args.w_bits,
        a_bits=args.a_bits,
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
    # Documents can be longer than one model window; chunking below enforces
    # the actual sequence length without triggering a tokenizer warning.
    tokenizer.model_max_length = 1_000_000_000

    perplexity_metrics, document_metrics = compare_wikitext_logits(
        documents, tokenizer, fp_model, quant_model, args
    )
    result = {
        "metric_definition": {
            "distribution_direction": "KL(FP16 || quantized)",
            "temperature": args.temperature,
            "ear": (
                "mean next-token probability-mass overlap; paper-style top-k "
                "and exact full-vocabulary variants are both reported"
            ),
            "accuracy_scope": "all samples and all choices in five zero-shot tasks",
            "perplexity_scope": (
                "full-vocabulary next-token distributions for every token in "
                "all 62 lm-eval WikiText documents"
            ),
        },
        "model": args.input_model,
        "quantized_checkpoint": args.load_qmodel_path,
        "source_results": {
            "fp": args.fp_results_path,
            "quantized": args.quant_results_path,
        },
        "quantization": {
            "w_bits": args.w_bits,
            "a_bits": args.a_bits,
            "k_bits": args.k_bits,
            "v_bits": args.v_bits,
            "k_groupsize": args.k_groupsize,
            "v_groupsize": args.v_groupsize,
        },
        "accuracy": accuracy_metrics,
        "perplexity": {
            "definition": {
                "dataset": "lm-eval WikiText test documents from FP result JSON",
                "logit_scope": "full vocabulary",
                "unit": "nats per next-token position unless otherwise stated",
                "first_token_context": "tokenizer EOS; reset for every document",
            },
            "aggregate": perplexity_metrics,
            "documents": document_metrics,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "accuracy": result["accuracy"]["overall"],
        "perplexity": result["perplexity"]["aggregate"],
    }, indent=2))
    print(f"Saved complete workload metrics: {output}")


if __name__ == "__main__":
    main()
