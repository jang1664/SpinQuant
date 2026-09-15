"""Streaming FP16-reference metrics for selected matrix multiplication outputs."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import torch


GROUPS = ("Linear", "QK", "PV")
ERROR_NAMES = ("error_A_fp16_vs_aqp8", "error_B_fp16_vs_aqp16")


class MatrixMetricAccumulator:
    """Accumulate element-weighted output error metrics in float64."""

    def __init__(self) -> None:
        self.elements = 0
        self.abs_error_sum = 0.0
        self.error_sq_sum = 0.0
        self.reference_sq_sum = 0.0
        self.quant_sq_sum = 0.0
        self.reference_quant_dot = 0.0
        self.max_abs_error = 0.0

    def update(self, reference: torch.Tensor, quantized: torch.Tensor) -> None:
        if reference.shape != quantized.shape:
            raise ValueError(
                f"matrix output shape mismatch: FP16={tuple(reference.shape)}, "
                f"quantized={tuple(quantized.shape)}"
            )
        # Reference tensors are staged on CPU between forwards. Transfer only
        # the currently matched record back to the candidate's GPU for FP64
        # scalar reductions; converting all long-sequence QK records on CPU
        # makes a full corpus run prohibitively slow.
        device = quantized.device
        reference = reference.detach().to(device=device, dtype=torch.float64)
        quantized = quantized.detach().to(device=device, dtype=torch.float64)
        error = quantized - reference
        self.elements += error.numel()
        self.abs_error_sum += error.abs().sum().item()
        self.error_sq_sum += error.square().sum().item()
        self.reference_sq_sum += reference.square().sum().item()
        self.quant_sq_sum += quantized.square().sum().item()
        self.reference_quant_dot += (reference * quantized).sum().item()
        if error.numel():
            self.max_abs_error = max(self.max_abs_error, error.abs().max().item())

    def compute(self) -> dict[str, float | int]:
        if not self.elements:
            raise ValueError("no matrix outputs were observed")
        eps = 1e-12
        cosine_denominator = math.sqrt(self.reference_sq_sum * self.quant_sq_sum)
        return {
            "elements": self.elements,
            "mae": self.abs_error_sum / self.elements,
            "rmse": math.sqrt(self.error_sq_sum / self.elements),
            "relative_l2_error": math.sqrt(self.error_sq_sum)
            / max(math.sqrt(self.reference_sq_sum), eps),
            "max_abs_error": self.max_abs_error,
            "cosine_similarity": self.reference_quant_dot
            / max(cosine_denominator, eps),
        }


class MatrixOutputObserver:
    """Pair a reference forward with one quantized forward by stable record key."""

    def __init__(self) -> None:
        self._reference: dict[tuple[str, str], torch.Tensor] = {}
        self._reference_keys: set[tuple[str, str]] = set()
        self._candidate_keys: set[tuple[str, str]] = set()
        self._mode: str | None = None
        self._accumulators = {group: MatrixMetricAccumulator() for group in GROUPS}

    def begin_reference(self) -> None:
        if self._reference:
            raise RuntimeError("unconsumed reference matrix outputs")
        self._mode = "reference"
        self._reference_keys.clear()

    def begin_candidate(self) -> None:
        if self._mode != "reference" or not self._reference:
            raise RuntimeError("candidate forward requires a completed reference forward")
        self._mode = "candidate"
        self._candidate_keys.clear()

    def record(self, group: str, key: str, tensor: torch.Tensor) -> None:
        if group not in self._accumulators:
            raise ValueError(f"unsupported matrix output group: {group}")
        record_key = (group, key)
        if self._mode == "reference":
            if record_key in self._reference:
                raise ValueError(f"duplicate reference matrix output: {record_key}")
            # CPU storage permits the quantized forward to run without retaining
            # FP16 activations on GPU. Values are released after this candidate run.
            self._reference[record_key] = tensor.detach().to("cpu", copy=True)
            self._reference_keys.add(record_key)
        elif self._mode == "candidate":
            reference = self._reference.pop(record_key, None)
            if reference is None:
                raise ValueError(f"missing reference matrix output: {record_key}")
            self._accumulators[group].update(reference, tensor)
            self._candidate_keys.add(record_key)
        else:
            raise RuntimeError("matrix observer used outside a paired forward")

    def finish_candidate(self) -> None:
        if self._mode != "candidate":
            raise RuntimeError("no candidate forward is active")
        if self._reference:
            missing = sorted(self._reference)[:3]
            self._reference.clear()
            raise ValueError(f"quantized forward missed reference outputs: {missing}")
        if self._candidate_keys != self._reference_keys:
            raise ValueError("FP16 and quantized matrix-output keys differ")
        self._mode = None

    def results(self) -> dict[str, dict[str, float | int]]:
        return {group: accumulator.compute() for group, accumulator in self._accumulators.items()}


def comparison(error_a: dict[str, Any], error_b: dict[str, Any]) -> dict[str, Any]:
    """Return the requested AQP8-versus-AQP16 error deltas and ratios."""
    eps = 1e-12
    result: dict[str, Any] = {}
    for metric in ("mae", "rmse", "relative_l2_error", "max_abs_error"):
        result[metric] = {
            "difference": error_a[metric] - error_b[metric],
            "ratio": error_a[metric] / max(error_b[metric], eps),
        }
    result["cosine_similarity"] = {
        "difference": error_a["cosine_similarity"] - error_b["cosine_similarity"]
    }
    return result
