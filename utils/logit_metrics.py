"""Streaming metrics for comparing reference and quantized next-token logits."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class LogitMetricAccumulator:
    """Accumulate full-vocabulary logit and distribution differences by token."""

    def __init__(
        self,
        temperature: float = 1.0,
        token_chunk_size: int = 16,
        ear_top_k: int = 10,
        ear_histogram_bins: int = 1000,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if token_chunk_size <= 0:
            raise ValueError("token_chunk_size must be positive")
        if ear_top_k <= 0:
            raise ValueError("ear_top_k must be positive")
        if ear_histogram_bins <= 0:
            raise ValueError("ear_histogram_bins must be positive")
        self.temperature = float(temperature)
        self.token_chunk_size = int(token_chunk_size)
        self.ear_top_k = int(ear_top_k)
        self.ear_histogram_bins = int(ear_histogram_bins)
        self.tokens = 0
        self.elements = 0
        self.logit_squared_error = 0.0
        self.centered_logit_squared_error = 0.0
        self.logit_absolute_error = 0.0
        self.logit_bias = 0.0
        self.logit_max_absolute_error = 0.0
        self.cosine_similarity = 0.0
        self.kl_fp_to_quant = 0.0
        self.kl_quant_to_fp = 0.0
        self.js_divergence = 0.0
        self.probability_overlap = 0.0
        self.probability_overlap_squared = 0.0
        self.minimum_probability_overlap = 1.0
        self.probability_overlap_histogram = torch.zeros(
            self.ear_histogram_bins, dtype=torch.int64
        )
        self.ear_thresholds = (0.50, 0.80, 0.90, 0.95, 0.99)
        self.probability_overlap_below = {
            threshold: 0 for threshold in self.ear_thresholds
        }
        self.topk_probability_overlap = 0.0
        self.fp_topk_probability_mass = 0.0
        self.quant_probability_mass_on_fp_topk = 0.0
        self.top1_agreement = 0
        self.fp_nll = 0.0
        self.quant_nll = 0.0
        self.label_tokens = 0

    def update(
        self,
        fp_logits: torch.Tensor,
        quant_logits: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> None:
        if fp_logits.shape != quant_logits.shape:
            raise ValueError(
                f"logit shape mismatch: {tuple(fp_logits.shape)} != {tuple(quant_logits.shape)}"
            )
        if fp_logits.ndim != 3:
            raise ValueError("expected logits with shape [batch, sequence, vocabulary]")

        vocab_size = fp_logits.shape[-1]
        fp_flat = fp_logits.reshape(-1, vocab_size)
        quant_flat = quant_logits.reshape(-1, vocab_size)
        label_flat = None if labels is None else labels.reshape(-1)
        if label_flat is not None and label_flat.numel() != fp_flat.shape[0]:
            raise ValueError("labels must have one entry per token")

        for start in range(0, fp_flat.shape[0], self.token_chunk_size):
            stop = min(start + self.token_chunk_size, fp_flat.shape[0])
            fp = fp_flat[start:stop].float()
            quant = quant_flat[start:stop].float()
            diff = quant - fp
            token_count = fp.shape[0]

            self.tokens += token_count
            self.elements += diff.numel()
            self.logit_squared_error += diff.square().sum().item()
            centered_diff = diff - diff.mean(dim=-1, keepdim=True)
            self.centered_logit_squared_error += centered_diff.square().sum().item()
            self.logit_absolute_error += diff.abs().sum().item()
            self.logit_bias += diff.sum().item()
            self.logit_max_absolute_error = max(
                self.logit_max_absolute_error, diff.abs().max().item()
            )
            self.cosine_similarity += F.cosine_similarity(fp, quant, dim=-1).sum().item()
            self.top1_agreement += (fp.argmax(dim=-1) == quant.argmax(dim=-1)).sum().item()

            fp_log_prob = F.log_softmax(fp / self.temperature, dim=-1)
            quant_log_prob = F.log_softmax(quant / self.temperature, dim=-1)
            fp_prob = fp_log_prob.exp()
            quant_prob = quant_log_prob.exp()
            self.kl_fp_to_quant += (
                fp_prob * (fp_log_prob - quant_log_prob)
            ).sum().item()
            self.kl_quant_to_fp += (
                quant_prob * (quant_log_prob - fp_log_prob)
            ).sum().item()
            mixture_log_prob = torch.logaddexp(fp_log_prob, quant_log_prob) - math.log(2.0)
            self.js_divergence += 0.5 * (
                (fp_prob * (fp_log_prob - mixture_log_prob)).sum()
                + (quant_prob * (quant_log_prob - mixture_log_prob)).sum()
            ).item()
            probability_overlap = torch.minimum(fp_prob, quant_prob).sum(dim=-1)
            self.probability_overlap += probability_overlap.sum().item()
            self.probability_overlap_squared += probability_overlap.square().sum().item()
            self.minimum_probability_overlap = min(
                self.minimum_probability_overlap, probability_overlap.min().item()
            )
            for threshold in self.ear_thresholds:
                self.probability_overlap_below[threshold] += (
                    probability_overlap < threshold
                ).sum().item()
            self.probability_overlap_histogram += torch.histc(
                probability_overlap,
                bins=self.ear_histogram_bins,
                min=0.0,
                max=1.0,
            ).to(dtype=torch.int64, device="cpu")
            top_k = min(self.ear_top_k, vocab_size)
            fp_topk_indices = fp_prob.topk(top_k, dim=-1).indices
            fp_topk_prob = fp_prob.gather(dim=-1, index=fp_topk_indices)
            quant_on_fp_topk = quant_prob.gather(dim=-1, index=fp_topk_indices)
            self.topk_probability_overlap += torch.minimum(
                fp_topk_prob, quant_on_fp_topk
            ).sum().item()
            self.fp_topk_probability_mass += fp_topk_prob.sum().item()
            self.quant_probability_mass_on_fp_topk += quant_on_fp_topk.sum().item()

            if label_flat is not None:
                chunk_labels = label_flat[start:stop].to(fp.device)
                valid = chunk_labels != -100
                if valid.any():
                    self.fp_nll += F.cross_entropy(
                        fp[valid], chunk_labels[valid], reduction="sum"
                    ).item()
                    self.quant_nll += F.cross_entropy(
                        quant[valid], chunk_labels[valid], reduction="sum"
                    ).item()
                    self.label_tokens += valid.sum().item()

    def compute(self) -> dict[str, float | int]:
        if self.tokens == 0 or self.elements == 0:
            raise ValueError("no logits were accumulated")
        logit_mse = self.logit_squared_error / self.elements
        kl_fp_to_quant = max(0.0, self.kl_fp_to_quant / self.tokens)
        kl_quant_to_fp = max(0.0, self.kl_quant_to_fp / self.tokens)
        js_divergence = max(0.0, self.js_divergence / self.tokens)
        mean_probability_overlap = self.probability_overlap / self.tokens
        overlap_variance = max(
            0.0,
            self.probability_overlap_squared / self.tokens
            - mean_probability_overlap**2,
        )

        cumulative_histogram = self.probability_overlap_histogram.cumsum(dim=0)

        def overlap_quantile(quantile: float) -> float:
            rank = max(1, math.ceil(quantile * self.tokens))
            bin_index = int(
                torch.searchsorted(
                    cumulative_histogram,
                    torch.tensor(rank, dtype=cumulative_histogram.dtype),
                ).item()
            )
            bin_index = min(bin_index, self.ear_histogram_bins - 1)
            return (bin_index + 0.5) / self.ear_histogram_bins

        result: dict[str, float | int] = {
            "tokens": self.tokens,
            "vocabulary_elements": self.elements,
            "temperature": self.temperature,
            "kl_fp_to_quant_nats": kl_fp_to_quant,
            "kl_quant_to_fp_nats": kl_quant_to_fp,
            "symmetric_kl_nats": (kl_fp_to_quant + kl_quant_to_fp) / 2.0,
            "js_divergence_nats": js_divergence,
            "ear_top_k": self.ear_top_k,
            "expected_acceptance_rate_topk": self.topk_probability_overlap
            / self.tokens,
            "expected_acceptance_rate_full_vocab": mean_probability_overlap,
            "expected_acceptance_rate_full_vocab_std": math.sqrt(overlap_variance),
            "expected_acceptance_rate_full_vocab_min": (
                self.minimum_probability_overlap
            ),
            "expected_acceptance_rate_full_vocab_p01": overlap_quantile(0.01),
            "expected_acceptance_rate_full_vocab_p05": overlap_quantile(0.05),
            "expected_acceptance_rate_full_vocab_p10": overlap_quantile(0.10),
            "expected_acceptance_rate_full_vocab_p50": overlap_quantile(0.50),
            "rejection_rate_full_vocab": 1.0 - mean_probability_overlap,
            "total_variation_distance_full_vocab": 1.0 - mean_probability_overlap,
            "fp_topk_probability_mass": self.fp_topk_probability_mass / self.tokens,
            "quant_probability_mass_on_fp_topk": (
                self.quant_probability_mass_on_fp_topk / self.tokens
            ),
            "kl_fp_to_quant_bits": kl_fp_to_quant / math.log(2.0),
            "logit_mse": logit_mse,
            "logit_rmse": math.sqrt(logit_mse),
            "centered_logit_mse": self.centered_logit_squared_error / self.elements,
            "logit_mae": self.logit_absolute_error / self.elements,
            "logit_mean_bias_quant_minus_fp": self.logit_bias / self.elements,
            "logit_max_absolute_error": self.logit_max_absolute_error,
            "logit_cosine_similarity": self.cosine_similarity / self.tokens,
            "top1_agreement": self.top1_agreement / self.tokens,
        }
        for threshold in self.ear_thresholds:
            result[
                f"fraction_tokens_ear_below_{threshold:.2f}"
            ] = self.probability_overlap_below[threshold] / self.tokens
        if self.label_tokens:
            fp_nll = self.fp_nll / self.label_tokens
            quant_nll = self.quant_nll / self.label_tokens
            result.update({
                "label_tokens": self.label_tokens,
                "fp_nll_nats": fp_nll,
                "quant_nll_nats": quant_nll,
                "nll_delta_quant_minus_fp_nats": quant_nll - fp_nll,
                "fp_perplexity": math.exp(fp_nll),
                "quant_perplexity": math.exp(quant_nll),
            })
        return result
