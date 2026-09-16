#!/usr/bin/env python3
"""Compare standard QDQ Linear and FPINT CUDA on one quantized Llama workload."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from measure_logit_divergence import compare_accuracy_tasks, read_json
from result_analysis.load_model import load_model
from utils import quant_utils
from utils.logit_metrics import LogitMetricAccumulator


SANITY_POLICY = "sanity_v1"


class TensorErrorAccumulator:
    def __init__(self, atol: float, rtol: float) -> None:
        self.atol = atol
        self.rtol = rtol
        self.elements = 0
        self.outside_tolerance = 0
        self.abs_error_sum = 0.0
        self.error_sq_sum = 0.0
        self.reference_sq_sum = 0.0
        self.candidate_sq_sum = 0.0
        self.reference_candidate_dot = 0.0
        self.max_abs_error = 0.0
        self.nonfinite = 0

    def update(self, reference: torch.Tensor, candidate: torch.Tensor) -> None:
        if reference.shape != candidate.shape:
            raise ValueError(
                f"tensor shape mismatch: {tuple(reference.shape)} != {tuple(candidate.shape)}"
            )
        device = candidate.device
        reference = reference.detach().to(device=device, dtype=torch.float64)
        candidate = candidate.detach().to(device=device, dtype=torch.float64)
        finite = torch.isfinite(reference) & torch.isfinite(candidate)
        self.nonfinite += int((~finite).sum().item())
        difference = candidate - reference
        absolute = difference.abs()
        threshold = self.atol + self.rtol * reference.abs()
        self.elements += difference.numel()
        self.outside_tolerance += int(((absolute > threshold) | ~finite).sum().item())
        self.abs_error_sum += float(absolute.sum().item())
        self.error_sq_sum += float(difference.square().sum().item())
        self.reference_sq_sum += float(reference.square().sum().item())
        self.candidate_sq_sum += float(candidate.square().sum().item())
        self.reference_candidate_dot += float((reference * candidate).sum().item())
        if difference.numel():
            self.max_abs_error = max(self.max_abs_error, float(absolute.max().item()))

    def compute(self) -> dict[str, float | int | bool]:
        if not self.elements:
            raise ValueError("no tensor elements were accumulated")
        eps = 1e-30
        cosine_denominator = math.sqrt(
            self.reference_sq_sum * self.candidate_sq_sum
        )
        return {
            "allclose": self.outside_tolerance == 0,
            "atol": self.atol,
            "rtol": self.rtol,
            "elements": self.elements,
            "outside_tolerance": self.outside_tolerance,
            "outside_tolerance_fraction": self.outside_tolerance / self.elements,
            "nonfinite": self.nonfinite,
            "mae": self.abs_error_sum / self.elements,
            "rmse": math.sqrt(self.error_sq_sum / self.elements),
            "relative_l2_error": math.sqrt(self.error_sq_sum)
            / max(math.sqrt(self.reference_sq_sum), eps),
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.reference_candidate_dot
            / max(cosine_denominator, eps),
        }


class PairedLinearObserver:
    """Pair named Linear wrapper outputs across two model forwards."""

    def __init__(self, atol: float, rtol: float) -> None:
        self.atol = atol
        self.rtol = rtol
        self.reference: dict[str, torch.Tensor] = {}
        self.per_layer: dict[str, TensorErrorAccumulator] = {}
        self.aggregate = TensorErrorAccumulator(atol, rtol)
        self.mode: str | None = None
        self.reference_keys: set[str] = set()
        self.candidate_keys: set[str] = set()

    def begin_reference(self) -> None:
        if self.reference:
            raise RuntimeError("unconsumed reference Linear outputs")
        self.mode = "reference"
        self.reference_keys.clear()

    def begin_candidate(self) -> None:
        if self.mode != "reference" or not self.reference:
            raise RuntimeError("candidate forward requires reference outputs")
        self.mode = "candidate"
        self.candidate_keys.clear()

    def record(self, name: str, output: Any) -> None:
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"Linear {name} produced {type(output)!r}")
        # Hooks are installed once. Repeatability and warmup forwards run while
        # the observer is intentionally idle and must not create paired state.
        if self.mode is None:
            return
        if self.mode == "reference":
            if name in self.reference:
                raise ValueError(f"duplicate reference Linear output: {name}")
            self.reference[name] = output.detach().to("cpu", copy=True)
            self.reference_keys.add(name)
            return
        if self.mode != "candidate":
            raise RuntimeError("Linear observer used outside a paired forward")
        reference = self.reference.pop(name, None)
        if reference is None:
            raise ValueError(f"missing reference Linear output: {name}")
        accumulator = self.per_layer.setdefault(
            name, TensorErrorAccumulator(self.atol, self.rtol)
        )
        accumulator.update(reference, output)
        self.aggregate.update(reference, output)
        self.candidate_keys.add(name)

    def finish_candidate(self) -> None:
        if self.reference:
            missing = sorted(self.reference)[:3]
            self.reference.clear()
            raise ValueError(f"candidate missed reference Linear outputs: {missing}")
        if self.reference_keys != self.candidate_keys:
            raise ValueError("reference and candidate Linear keys differ")
        self.mode = None

    def results(self) -> dict[str, Any]:
        per_layer = {
            name: accumulator.compute()
            for name, accumulator in sorted(self.per_layer.items())
        }
        return {
            "aggregate": self.aggregate.compute(),
            "per_layer": per_layer,
            "failed_layers": [
                name for name, metrics in per_layer.items() if not metrics["allclose"]
            ],
        }


def evaluate_sanity(
    *,
    standard_config: dict[str, Any],
    fpint_config: dict[str, Any],
    expected_linears: int,
    chunks: int,
    tokens: int,
    repeatability: dict[str, Any] | None,
    logit_results: dict[str, Any],
    linear_results: dict[str, Any] | None,
    require_linears: bool,
) -> dict[str, bool]:
    """Evaluate execution sanity without treating numerical closeness as a gate."""

    return {
        "backend_coverage_valid": (
            standard_config["fpint_linears"] == expected_linears
            and fpint_config["fpint_linears"] == expected_linears
            and standard_config["lm_head_backend"] == "standard"
            and fpint_config["lm_head_backend"] == "standard"
        ),
        "workload_nonempty": chunks > 0 and tokens > 0,
        "standard_repeatable": bool(
            repeatability
            and repeatability["allclose"]
            and repeatability["nonfinite"] == 0
        ),
        "logits_finite": logit_results["nonfinite"] == 0,
        "linear_outputs_complete": (
            not require_linears
            or (
                linear_results is not None
                and len(linear_results["per_layer"]) == expected_linears
            )
        ),
        "linear_outputs_finite": (
            not require_linears
            or (
                linear_results is not None
                and linear_results["aggregate"]["nonfinite"] == 0
            )
        ),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def documents_from_results(path: str | Path) -> list[dict[str, Any]]:
    results = read_json(path)
    try:
        samples = results["samples"]["wikitext"]
    except KeyError as error:
        raise ValueError(f"{path} has no logged WikiText samples") from error
    return [
        {
            "doc_id": sample["doc_id"],
            "doc_hash": sample["doc_hash"],
            "target": sample["target"],
        }
        for sample in samples
    ]


def fpint_wrappers(model: torch.nn.Module) -> dict[str, quant_utils.ActQuantWrapper]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, quant_utils.ActQuantWrapper)
        and module.has_fpint_metadata
        and name != "lm_head"
    }


def validate_model(
    model: torch.nn.Module,
    *,
    backend: str,
    bits: int,
    group_size: int,
    mxu_rows: int,
    expected_linears: int,
) -> dict[str, Any]:
    wrappers = fpint_wrappers(model)
    if len(wrappers) != expected_linears:
        raise ValueError(
            f"expected {expected_linears} FPINT Linear layers, found {len(wrappers)}"
        )
    invalid = {}
    for name, wrapper in wrappers.items():
        version, observed_bits, observed_group, observed_mxu, _, _ = (
            wrapper.fpint_meta.tolist()
        )
        if (
            version != 1
            or observed_bits != bits
            or observed_group != group_size
            or observed_mxu != mxu_rows
            or wrapper.linear_backend != backend
        ):
            invalid[name] = {
                "meta": wrapper.fpint_meta.tolist(),
                "backend": wrapper.linear_backend,
            }
    if invalid:
        raise ValueError(f"invalid FPINT layer configuration: {invalid}")
    coverage = getattr(model, "fpint_linear_coverage", [])
    fallbacks = [
        item
        for item in coverage
        if item["backend"] != backend and item["name"] != "lm_head"
    ]
    if fallbacks:
        raise ValueError(f"unexpected Linear backend fallbacks: {fallbacks}")
    return {
        "backend": backend,
        "fpint_linears": len(wrappers),
        "expected_fpint_linears": expected_linears,
        "lm_head_backend": next(
            (item["backend"] for item in coverage if item["name"] == "lm_head"),
            "unknown",
        ),
        "fallbacks": fallbacks,
    }


def attach_linear_hooks(
    model: torch.nn.Module, observer: PairedLinearObserver
) -> list[Any]:
    handles = []
    for name, module in fpint_wrappers(model).items():
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, name=name: observer.record(name, output)
            )
        )
    return handles


def make_input(
    tokenizer: Any,
    document: dict[str, Any],
    start: int,
    sequence_length: int,
    max_tokens: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    token_ids = tokenizer(
        document["target"], return_tensors="pt", add_special_tokens=False
    ).input_ids[0]
    if max_tokens is not None:
        token_ids = token_ids[:max_tokens]
    targets = token_ids[start : start + sequence_length]
    if targets.numel() == 0:
        return None
    context_id = tokenizer.eos_token_id if start == 0 else int(token_ids[start - 1])
    if context_id is None:
        raise ValueError("tokenizer has no EOS token")
    input_ids = torch.cat((torch.tensor([context_id]), targets))[:-1].unsqueeze(0)
    return input_ids.to(device), targets.unsqueeze(0).to(device)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_forward(model: torch.nn.Module, input_ids: torch.Tensor, device: torch.device):
    synchronize(device)
    started = time.perf_counter()
    output = model(input_ids=input_ids, use_cache=False, return_dict=True).logits
    synchronize(device)
    return output, time.perf_counter() - started


def load_condition(args: argparse.Namespace, backend: str):
    model, tokenizer = load_model(
        input_model=args.input_model,
        load_qmodel_path=args.load_qmodel_path,
        optimized_rotation_path=None,
        w_bits=args.w_bits,
        a_bits=16,
        a_asym=False,
        k_bits=16,
        k_groupsize=-1,
        k_asym=False,
        v_bits=16,
        v_groupsize=-1,
        v_asym=False,
        rotate=True,
        model_max_length=args.sequence_length,
        device=args.device,
        linear_backend=backend,
        fpint_mxu_rows=args.mxu_rows,
        fpint_extra_bits=args.extra_bits,
        fpint_reduce_extra_bits=args.reduce_extra_bits,
        w_groupsize=args.group_size,
        q_bits=16,
        p_bits=16,
        attention_backend="eager",
    )
    model.config.use_cache = False
    return model.eval(), tokenizer


def run(args: argparse.Namespace) -> tuple[dict[str, Any], bool]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    documents = documents_from_results(args.source_results_path)
    if args.max_documents is not None:
        documents = documents[: args.max_documents]
    if not documents:
        raise ValueError("no WikiText documents selected")

    standard_model, tokenizer = load_condition(args, "standard")
    fpint_model, _ = load_condition(args, "fpint_cuda")
    standard_config = validate_model(
        standard_model,
        backend="standard",
        bits=args.w_bits,
        group_size=args.group_size,
        mxu_rows=args.mxu_rows,
        expected_linears=args.expected_fpint_linears,
    )
    fpint_config = validate_model(
        fpint_model,
        backend="fpint_cuda",
        bits=args.w_bits,
        group_size=args.group_size,
        mxu_rows=args.mxu_rows,
        expected_linears=args.expected_fpint_linears,
    )
    tokenizer.model_max_length = 1_000_000_000

    observer = PairedLinearObserver(args.atol, args.rtol) if args.capture_linears else None
    handles: list[Any] = []
    if observer is not None:
        handles.extend(attach_linear_hooks(standard_model, observer))
        handles.extend(attach_linear_hooks(fpint_model, observer))

    logit_distribution = LogitMetricAccumulator(
        temperature=args.temperature,
        token_chunk_size=args.token_chunk_size,
        ear_top_k=args.ear_top_k,
    )
    logit_error = TensorErrorAccumulator(args.atol, args.rtol)
    standard_seconds = 0.0
    fpint_seconds = 0.0
    tokens = 0
    chunks = 0
    repeatability = None
    try:
        with torch.inference_mode():
            for document_index, document in enumerate(documents, start=1):
                token_count = tokenizer(
                    document["target"], return_tensors="pt", add_special_tokens=False
                ).input_ids.shape[1]
                if args.max_tokens_per_document is not None:
                    token_count = min(token_count, args.max_tokens_per_document)
                for start in range(0, token_count, args.sequence_length):
                    prepared = make_input(
                        tokenizer,
                        document,
                        start,
                        args.sequence_length,
                        args.max_tokens_per_document,
                        device,
                    )
                    if prepared is None:
                        continue
                    input_ids, labels = prepared
                    if repeatability is None:
                        first, _ = timed_forward(standard_model, input_ids, device)
                        second, _ = timed_forward(standard_model, input_ids, device)
                        repeat = TensorErrorAccumulator(0.0, 0.0)
                        repeat.update(first, second)
                        repeatability = repeat.compute()
                        del first, second
                    if observer is not None:
                        observer.begin_reference()
                    standard_logits, elapsed = timed_forward(
                        standard_model, input_ids, device
                    )
                    standard_seconds += elapsed
                    if observer is not None:
                        observer.begin_candidate()
                    fpint_logits, elapsed = timed_forward(fpint_model, input_ids, device)
                    fpint_seconds += elapsed
                    if observer is not None:
                        observer.finish_candidate()
                    logit_distribution.update(standard_logits, fpint_logits, labels)
                    logit_error.update(standard_logits, fpint_logits)
                    tokens += labels.numel()
                    chunks += 1
                    del input_ids, labels, standard_logits, fpint_logits
                print(
                    f"WikiText document {document_index}/{len(documents)} complete",
                    flush=True,
                )
    finally:
        for handle in handles:
            handle.remove()

    linear_results = observer.results() if observer is not None else None
    logit_results = logit_error.compute()
    sanity_checks = evaluate_sanity(
        standard_config=standard_config,
        fpint_config=fpint_config,
        expected_linears=args.expected_fpint_linears,
        chunks=chunks,
        tokens=tokens,
        repeatability=repeatability,
        logit_results=logit_results,
        linear_results=linear_results,
        require_linears=args.capture_linears,
    )
    sanity_passed = all(sanity_checks.values())
    task_comparison = None
    if args.standard_eval_results and args.fpint_eval_results:
        task_comparison = compare_accuracy_tasks(
            read_json(args.standard_eval_results),
            read_json(args.fpint_eval_results),
            args.temperature,
        )
    result = {
        "status": "pass" if sanity_passed else "fail",
        "evaluation_policy": {
            "name": SANITY_POLICY,
            "allclose_is_diagnostic_only": True,
            "checks": sanity_checks,
        },
        "conditions": {
            "reference": "standard_qdq_linear",
            "candidate": "fpint_cuda",
            "input_model": args.input_model,
            "quantized_checkpoint": args.load_qmodel_path,
            "quantized_checkpoint_sha256": sha256_file(args.load_qmodel_path),
            "w_bits": args.w_bits,
            "weight_symmetric": True,
            "weight_group_size": args.group_size,
            "mxu_rows": args.mxu_rows,
            "extra_bits": args.extra_bits,
            "reduce_extra_bits": args.reduce_extra_bits,
            "activation_query_key_value_probability_bits": 16,
            "attention_backend": "eager",
            "standard_coverage": standard_config,
            "fpint_coverage": fpint_config,
        },
        "scope": {
            "dataset": "lm-eval WikiText samples",
            "documents": len(documents),
            "chunks": chunks,
            "tokens": tokens,
            "sequence_length": args.sequence_length,
            "max_tokens_per_document": args.max_tokens_per_document,
            "full_corpus": args.max_documents is None
            and args.max_tokens_per_document is None
            and len(documents) == 62,
        },
        "repeatability": {"standard_vs_standard": repeatability},
        "linear_outputs": linear_results,
        "logits": {
            "allclose_metrics": logit_results,
            "distribution_metrics": logit_distribution.compute(),
            "legacy_metric_names": {
                "fp": "standard_qdq_linear",
                "quant": "fpint_cuda",
            },
        },
        "task_comparison": task_comparison,
        "timing": {
            "standard_seconds": standard_seconds,
            "fpint_cuda_seconds": fpint_seconds,
            "standard_tokens_per_second": tokens / standard_seconds,
            "fpint_cuda_tokens_per_second": tokens / fpint_seconds,
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
            "git_commit": git_commit(),
            "fpint_cuda_kernel_sha256": sha256_file(
                Path(__file__).parent / "fpint_emul/csrc/fpint_cuda_kernel.cu"
            ),
        },
    }
    return result, sanity_passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-model", required=True)
    parser.add_argument("--load-qmodel-path", required=True)
    parser.add_argument("--source-results-path", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standard-eval-results")
    parser.add_argument("--fpint-eval-results")
    parser.add_argument("--w-bits", type=int, default=4, choices=(4, 8))
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--mxu-rows", type=int, default=128)
    parser.add_argument("--extra-bits", type=int, default=19)
    parser.add_argument("--reduce-extra-bits", type=int, default=10)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-tokens-per-document", type=int)
    parser.add_argument("--capture-linears", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expected-fpint-linears", type=int, default=224)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--token-chunk-size", type=int, default=16)
    parser.add_argument("--ear-top-k", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    for name in (
        "input_model",
        "load_qmodel_path",
        "source_results_path",
        "standard_eval_results",
        "fpint_eval_results",
    ):
        value = getattr(args, name)
        if value and not Path(value).exists():
            parser.error(f"--{name.replace('_', '-')} does not exist: {value}")
    if bool(args.standard_eval_results) != bool(args.fpint_eval_results):
        parser.error("standard and FPINT eval results must be supplied together")
    if args.group_size != 128 or args.mxu_rows != 128:
        parser.error("this experiment requires --group-size=128 and --mxu-rows=128")
    if args.sequence_length < 2 or args.expected_fpint_linears <= 0:
        parser.error("sequence length and expected Linear count must be positive")
    for name in ("max_documents", "max_tokens_per_document"):
        if getattr(args, name) is not None and getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    result, passed = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "scope": result["scope"],
        "linear": None if result["linear_outputs"] is None else result["linear_outputs"]["aggregate"],
        "logits": result["logits"]["allclose_metrics"],
        "timing": result["timing"],
    }, indent=2))
    print(f"Saved paired backend metrics: {args.output}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
