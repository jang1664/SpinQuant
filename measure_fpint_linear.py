#!/usr/bin/env python3
"""Reproducible accuracy, latency, memory and tiny-model FPINT benchmark."""

import argparse
import json
import platform
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fpint_emul import (
    FpIntConfig,
    dequantize_weight,
    fpint_linear,
    qcol_real_2scomp_reference,
)


def parse_shape(value):
    fields = value.lower().split("x")
    if len(fields) != 3:
        raise argparse.ArgumentTypeError("shape must be MxKxN")
    shape = tuple(int(field) for field in fields)
    if any(dimension <= 0 for dimension in shape):
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return shape


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def error_metrics(actual, expected, atol=1e-3, rtol=1e-3):
    actual = actual.float().cpu()
    expected = expected.float().cpu()
    difference = (actual - expected).abs()
    threshold = atol + rtol * expected.abs()
    return {
        "allclose": bool(torch.all(difference <= threshold)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs": float(difference.mean()) if difference.numel() else 0.0,
        "rms": float(torch.sqrt(torch.mean(difference.square())))
        if difference.numel()
        else 0.0,
        "outside_tolerance_fraction": float((difference > threshold).float().mean())
        if difference.numel()
        else 0.0,
        "atol": atol,
        "rtol": rtol,
    }


def make_case(m, k, n, config, asymmetric, device, seed):
    generator = torch.Generator(device=device).manual_seed(seed)
    activation = torch.randn(
        (m, k), generator=generator, device=device, dtype=torch.float16
    )
    qmin = -(1 << (config.weight_bits - 1))
    qmax = (1 << (config.weight_bits - 1)) - 1
    weight = torch.randint(
        qmin,
        qmax + 1,
        (n, k),
        generator=generator,
        device=device,
        dtype=torch.int8,
    )
    groups = config.group_count(k)
    scale = (
        torch.rand((n, groups), generator=generator, device=device) * 0.078 + 0.002
    ).half()
    zero = (
        torch.randint(
            qmin,
            qmax + 1,
            (n, groups),
            generator=generator,
            device=device,
            dtype=torch.int32,
        )
        if asymmetric
        else torch.zeros((n, groups), device=device, dtype=torch.int32)
    )
    return activation, weight, scale, zero


def measure_call(call, device, warmup, iterations):
    for _ in range(warmup):
        output = call()
    synchronize(device)
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        baseline = 0
    started = time.perf_counter()
    for _ in range(iterations):
        output = call()
    synchronize(device)
    elapsed = time.perf_counter() - started
    peak = (
        max(0, torch.cuda.max_memory_allocated(device) - baseline)
        if device.type == "cuda"
        else None
    )
    return {
        "latency_ms": elapsed * 1000.0 / iterations,
        "peak_increment_bytes": peak,
        "finite": bool(torch.isfinite(output).all()),
    }


def accuracy_check(config, asymmetric, device, seed):
    k = min(2 * config.mxu_rows + 3, 131)
    case = make_case(4, k, 37, config, asymmetric, device, seed)
    activation, weight, scale, zero = case
    bias = torch.linspace(-0.25, 0.25, weight.shape[0], device=device).half()
    reference = torch.from_numpy(
        qcol_real_2scomp_reference(
            activation.cpu().numpy(),
            weight.cpu().numpy(),
            scale.cpu().numpy(),
            zero.cpu().numpy(),
            config,
            bias=bias.cpu().numpy(),
        )
    )
    results = {}
    for backend in ("fpint_torch", "fpint_cuda"):
        if backend == "fpint_cuda" and device.type != "cuda":
            continue
        actual = fpint_linear(
            activation, weight, scale, zero, config, bias=bias, backend=backend
        )
        results[f"{backend}_vs_reference"] = error_metrics(actual, reference)
    qdq = torch.nn.functional.linear(
        activation, dequantize_weight(weight, scale, zero, config), bias
    )
    results["fpint_cuda_vs_qdq"] = (
        error_metrics(
            fpint_linear(
                activation,
                weight,
                scale,
                zero,
                config,
                bias=bias,
                backend="fpint_cuda" if device.type == "cuda" else "fpint_torch",
            ),
            qdq,
        )
    )
    return results


def benchmark_shape(shape, config, asymmetric, device, seed, warmup, iterations, backends):
    m, k, n = shape
    activation, weight, scale, zero = make_case(
        m, k, n, config, asymmetric, device, seed
    )
    synchronize(device)
    started = time.perf_counter()
    dequantized = dequantize_weight(weight, scale, zero, config)
    synchronize(device)
    preparation_ms = (time.perf_counter() - started) * 1000.0
    calls = {
        "standard_predequantized": lambda: torch.nn.functional.linear(
            activation, dequantized
        ),
        "dequantize_plus_linear": lambda: fpint_linear(
            activation, weight, scale, zero, config, backend="standard"
        ),
        "fpint_torch": lambda: fpint_linear(
            activation,
            weight,
            scale,
            zero,
            config,
            backend="fpint_torch",
            validate_values=False,
            has_zero=asymmetric,
        ),
        "fpint_cuda": lambda: fpint_linear(
            activation,
            weight,
            scale,
            zero,
            config,
            backend="fpint_cuda",
            validate_values=False,
            has_zero=asymmetric,
        ),
    }
    result = {
        "shape": {"m": m, "k": k, "n": n},
        "weight_preparation_ms": preparation_ms,
        "methods": {},
    }
    for backend in backends:
        if backend == "fpint_cuda" and device.type != "cuda":
            continue
        result["methods"][backend] = measure_call(
            calls[backend], device, warmup, iterations
        )
    return result


def run_tiny_model(device, backend, asymmetric, seed, warmup, iterations):
    from transformers import LlamaConfig

    from eval_utils.gptq_utils import rtn_fwrd
    from eval_utils.modeling_llama import LlamaForCausalLM
    from utils.quant_utils import add_actquant, configure_fpint_linears

    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    model = LlamaForCausalLM(config).half().eval()
    add_actquant(model)
    quant_args = SimpleNamespace(
        w_bits=4,
        w_groupsize=32,
        w_asym=asymmetric,
        w_clip=False,
        int8_down_proj=False,
        export_to_et=False,
    )
    rtn_fwrd(model, device, quant_args)
    selection_args = SimpleNamespace(
        linear_backend=backend,
        fpint_mxu_rows=32,
        fpint_extra_bits=19,
        fpint_reduce_extra_bits=10,
    )
    coverage = configure_fpint_linears(model, selection_args)
    model.to(device)
    input_ids = torch.arange(32, device=device).remainder(config.vocab_size).unsqueeze(0)
    measurement = measure_call(
        lambda: model(input_ids=input_ids, use_cache=False).logits,
        device,
        warmup,
        iterations,
    )
    measurement["coverage"] = coverage
    measurement["sequence_length"] = input_ids.shape[1]
    return measurement


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shape",
        action="append",
        type=parse_shape,
        default=None,
        help="repeatable MxKxN shape (default: decode and prefill-like Llama-3.2 projections)",
    )
    parser.add_argument("--bits", type=int, choices=[4, 8], default=4)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--mxu-rows", type=int, default=32)
    parser.add_argument("--extra-bits", type=int, default=19)
    parser.add_argument("--reduce-extra-bits", type=int, default=10)
    parser.add_argument("--asymmetric", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=[
            "standard_predequantized",
            "dequantize_plus_linear",
            "fpint_torch",
            "fpint_cuda",
        ],
        default=["standard_predequantized", "dequantize_plus_linear", "fpint_cuda"],
    )
    parser.add_argument("--tiny-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available")
    shapes = args.shape or [(1, 3072, 3072), (32, 3072, 3072)]
    config = FpIntConfig(
        args.bits,
        args.group_size,
        args.mxu_rows,
        args.extra_bits,
        args.reduce_extra_bits,
    )
    result = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device),
        },
        "seed": args.seed,
        "config": vars(config),
        "asymmetric": args.asymmetric,
        "accuracy": accuracy_check(config, args.asymmetric, device, args.seed),
        "benchmarks": [
            benchmark_shape(
                shape,
                config,
                args.asymmetric,
                device,
                args.seed,
                args.warmup,
                args.iterations,
                args.backends,
            )
            for shape in shapes
        ],
    }
    if args.tiny_model:
        model_backend = "fpint_cuda" if device.type == "cuda" else "fpint_torch"
        result["tiny_model"] = run_tiny_model(
            device,
            model_backend,
            args.asymmetric,
            args.seed,
            args.warmup,
            args.iterations,
        )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
