import argparse
import subprocess
from pathlib import Path

import pytest
import torch

from measure_fpint_backend_accuracy import (
    PairedLinearObserver,
    TensorErrorAccumulator,
    evaluate_sanity,
)
from measure_fpint_qcol_accuracy import run
from summarize_fpint_mxu128 import TASK_METRICS, aggregate_full_results, render_report


def test_random_qcol_experiment_smoke_cpu():
    args = argparse.Namespace(
        bits=4,
        group_size=128,
        mxu_rows=128,
        extra_bits=19,
        reduce_extra_bits=10,
        m=2,
        n=5,
        k_values=(127, 128, 129),
        trials=1,
        base_seed=7,
        distributions=("gaussian", "componentwise"),
        zero_modes=("symmetric", "asymmetric"),
        atol=1e-3,
        rtol=1e-3,
        device="cpu",
    )
    result = run(args)
    assert result["status"] == "pass"
    assert len(result["records"]) == 12
    for record in result["records"]:
        assert record["comparisons"]["fpint_torch_vs_reference"]["allclose"]


def test_tensor_error_accumulator_tracks_allclose_failures():
    accumulator = TensorErrorAccumulator(atol=1e-3, rtol=1e-3)
    accumulator.update(
        torch.tensor([1.0, 0.0]), torch.tensor([1.001, 0.01])
    )
    result = accumulator.compute()
    assert not result["allclose"]
    assert result["outside_tolerance"] == 1
    assert result["nonfinite"] == 0


def test_paired_observer_ignores_repeatability_forward_while_idle():
    observer = PairedLinearObserver(atol=1e-3, rtol=1e-3)
    observer.record("proj", torch.ones(2))
    observer.begin_reference()
    observer.record("proj", torch.ones(2))
    observer.begin_candidate()
    observer.record("proj", torch.ones(2))
    observer.finish_candidate()
    assert observer.results()["aggregate"]["allclose"]


def test_mxu128_runner_has_valid_shell_syntax():
    script = Path(__file__).parents[1] / "scripts" / "run_fpint_mxu128_llama31_8b.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)


def test_sanity_policy_does_not_gate_on_allclose():
    coverage = {"fpint_linears": 224, "lm_head_backend": "standard"}
    diagnostic = {"allclose": False, "nonfinite": 0}
    linear = {
        "aggregate": diagnostic,
        "per_layer": {f"proj.{index}": diagnostic for index in range(224)},
    }
    checks = evaluate_sanity(
        standard_config=coverage,
        fpint_config=coverage,
        expected_linears=224,
        chunks=1,
        tokens=128,
        repeatability={"allclose": True, "nonfinite": 0},
        logit_results=diagnostic,
        linear_results=linear,
        require_linears=True,
    )
    assert all(checks.values())


def test_sanity_policy_rejects_nonfinite_outputs():
    coverage = {"fpint_linears": 224, "lm_head_backend": "standard"}
    checks = evaluate_sanity(
        standard_config=coverage,
        fpint_config=coverage,
        expected_linears=224,
        chunks=1,
        tokens=128,
        repeatability={"allclose": True, "nonfinite": 0},
        logit_results={"allclose": False, "nonfinite": 1},
        linear_results=None,
        require_linears=False,
    )
    assert not checks["logits_finite"]


def _full_payload(backend, tasks, checkpoint="checkpoint-sha"):
    coverage = [
        {"name": f"model.proj.{index}", "backend": backend}
        for index in range(224)
    ] + [{"name": "lm_head", "backend": "standard"}]
    results = {}
    sample_counts = {}
    for index, task in enumerate(tasks):
        metric, kind = TASK_METRICS[task]
        value = 9.0 if kind == "perplexity" else 0.5 + index * 0.01
        results[task] = {metric: value}
        if kind == "accuracy":
            results[task][metric.replace(",none", "_stderr,none")] = 0.02
        sample_counts[task] = {"original": 10, "effective": 10}
    return {
        "results": results,
        "n-samples": sample_counts,
        "spinquant_quantization": {
            "linear_backend": backend,
            "weight_bits": 4,
            "weight_groupsize": 128,
            "weight_symmetric": True,
            "fpint_mxu_rows": 128,
            "quantized_checkpoint_sha256": checkpoint,
            "fpint_linear_coverage": coverage,
            "evaluation_seconds": 12.5,
            "eval_tasks": tasks,
            "lm_eval_batch_size": "1" if tasks == ["wikitext"] else "32",
        },
    }


def test_full_result_aggregation_reports_accuracy_and_ppl_only():
    shards = [
        ["wikitext"],
        ["hellaswag"],
        ["arc_easy", "openbookqa"],
        ["arc_challenge", "winogrande"],
    ]
    standard = [_full_payload("standard", tasks) for tasks in shards]
    fpint = [_full_payload("fpint_cuda", tasks) for tasks in shards]
    fpint[0]["results"]["wikitext"]["word_perplexity,none"] = 9.09
    fpint[1]["results"]["hellaswag"]["acc_norm,none"] = 0.49
    result = aggregate_full_results(standard, fpint)
    assert result["status"] == "complete"
    assert result["quality_gate"] is None
    assert result["tasks"]["wikitext"]["relative_delta"] == pytest.approx(0.01)
    assert result["tasks"]["hellaswag"]["delta_percentage_points"] == pytest.approx(-1.0)
    assert result["accuracy_micro"]["samples"] == 50


def test_full_result_aggregation_rejects_checkpoint_mismatch():
    shards = [
        ["wikitext"],
        ["hellaswag"],
        ["arc_easy", "openbookqa"],
        ["arc_challenge", "winogrande"],
    ]
    standard = [
        _full_payload("standard", tasks, checkpoint="standard-sha")
        for tasks in shards
    ]
    fpint = [
        _full_payload("fpint_cuda", tasks, checkpoint="fpint-sha")
        for tasks in shards
    ]
    with pytest.raises(ValueError, match="different checkpoints"):
        aggregate_full_results(standard, fpint)


def test_partial_report_marks_full_workload_as_pending():
    metrics = {
        "allclose": False,
        "atol": 1e-3,
        "rtol": 1e-3,
        "elements": 2,
        "outside_tolerance": 1,
        "outside_tolerance_fraction": 0.5,
        "nonfinite": 0,
        "mae": 0.01,
        "rmse": 0.02,
        "relative_l2_error": 0.03,
        "max_abs_error": 0.04,
        "cosine_similarity": 0.99,
    }
    random = {
        "status": "pass",
        "records": [
            {
                "comparisons": {
                    "fpint_torch_vs_reference": {"allclose": True, "max_abs": 0.0},
                    "fpint_vs_standard_qdq": {"allclose": False, "max_abs": 0.1},
                }
            }
        ],
    }
    smoke = {
        "status": "fail",
        "evaluation_policy": {"name": "sanity_v1"},
        "scope": {"documents": 1, "tokens": 2},
        "linear_outputs": {
            "aggregate": metrics,
            "per_layer": {"model.layers.0.mlp.down_proj": metrics},
            "failed_layers": ["model.layers.0.mlp.down_proj"],
        },
        "logits": {
            "allclose_metrics": metrics,
            "distribution_metrics": {
                "symmetric_kl_nats": 0.01,
                "js_divergence_nats": 0.001,
                "top1_agreement": 1.0,
                "fp_perplexity": 10.0,
                "quant_perplexity": 10.1,
            },
        },
        "timing": {
            "standard_tokens_per_second": 100.0,
            "fpint_cuda_tokens_per_second": 10.0,
        },
        "conditions": {"quantized_checkpoint_sha256": "abc"},
    }
    report = render_report(random, smoke)
    assert "아직 실행 결과가 없다" in report
    assert "All-close는 diagnostic" in report
    assert "model.layers.0.mlp.down_proj" in report
