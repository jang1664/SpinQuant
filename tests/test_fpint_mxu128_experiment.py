import argparse
import subprocess
from pathlib import Path

import pytest
import numpy as np
import torch

from fpint_emul import FpIntConfig, fpint_linear, qcol_real_2scomp_reference
from measure_fpint_backend_accuracy import (
    PairedLinearObserver,
    TensorErrorAccumulator,
    evaluate_sanity,
)
from measure_fpint_qcol_accuracy import (
    activation_field_stats,
    conventional_linear,
    fp16_ulp_distance,
    fp64_reference_linear,
    make_case,
    paired_error_metrics,
    parse_args,
    parse_exponent_max_map,
    qcol_allclose_metrics,
    run,
    sample_finite_bf16_fields,
    sample_finite_fp16_fields,
    summarize_records,
)
from summarize_fpint_mxu128 import TASK_METRICS, aggregate_full_results, render_report


def test_random_qcol_experiment_smoke_cpu():
    args = argparse.Namespace(
        bits=4,
        group_size=128,
        mxu_rows=128,
        extra_bits=19,
        reduce_extra_bits=10,
        m=32,
        n=32,
        k_values=(128,),
        exponent_max_by_k={128: 15},
        finite_target=1.0,
        trials=1,
        base_seed=7,
        atol=1e-3,
        rtol=1e-3,
        device="cpu",
    )
    result = run(args)
    assert result["status"] == "pass"
    assert len(result["records"]) == 1
    row = result["summary"]["by_k"]["128"]
    assert row["coverage"]["common_finite_fraction"] == 1.0
    assert row["conventional_err"]["global_rmse"] is not None
    assert row["fp_int_err"]["global_mean_ulp"] is not None
    for record in result["records"]:
        assert record["comparisons"]["fpint_torch_vs_qcol_reference"]["allclose"]


def test_raw_fp16_sampler_uses_all_finite_fields_uniformly():
    values = sample_finite_fp16_fields(
        np.random.default_rng(0), (200_000,), exponent_max=24
    )
    bits = values.view(np.uint16)
    sign = bits >> 15
    exponent = (bits >> 10) & 0x1F
    mantissa = bits & 0x3FF
    assert (int(sign.min()), int(sign.max())) == (0, 1)
    assert (int(exponent.min()), int(exponent.max())) == (0, 24)
    assert (int(mantissa.min()), int(mantissa.max())) == (0, 1023)
    assert np.isfinite(values).all()


def test_raw_bf16_sampler_uses_all_requested_finite_fields():
    values = sample_finite_bf16_fields(
        np.random.default_rng(1), (200_000,), exponent_max=136
    )
    bits = values.view(torch.uint16).numpy()
    sign = bits >> 15
    exponent = (bits >> 7) & 0xFF
    mantissa = bits & 0x7F
    assert (int(sign.min()), int(sign.max())) == (0, 1)
    assert (int(exponent.min()), int(exponent.max())) == (0, 136)
    assert (int(mantissa.min()), int(mantissa.max())) == (0, 127)
    assert bool(torch.isfinite(values).all())


def test_bf16_random_qcol_experiment_smoke_cpu():
    args = argparse.Namespace(
        bits=4,
        group_size=128,
        mxu_rows=128,
        extra_bits=19,
        reduce_extra_bits=10,
        activation_format="bf16",
        m=4,
        n=4,
        k_values=(128,),
        exponent_max_by_k={128: 136},
        finite_target=1.0,
        trials=1,
        base_seed=17,
        atol=0.0,
        rtol=0.0,
        device="cpu",
    )
    result = run(args)
    assert result["status"] == "pass"
    assert result["config"]["activation_format"] == "bf16"
    assert result["config"]["operation"] == "raw_bf16_times_signed_int"
    assert result["summary"]["overall"]["qcol_correctness"] == {
        "torch_allclose_cases": 1,
        "cuda_cases": 0,
        "cuda_allclose_cases": 0,
    }


def test_raw_fpxint_case_uses_full_int4_and_identity_qparams():
    config = FpIntConfig(4, group_size=128, mxu_rows=128)
    activation, weight, scale, zero = make_case(
        m=32, k=32768, n=32, config=config, seed=11, exponent_max=20
    )
    assert activation.shape == (32, 32768)
    assert weight.shape == (32, 32768)
    assert (int(weight.min()), int(weight.max())) == (-8, 7)
    assert scale.shape == (32, 256)
    assert np.all(scale == 1)
    assert np.all(zero == 0)
    stats = activation_field_stats(activation)
    assert stats["positive_sign_bits"] > 0
    assert stats["negative_sign_bits"] > 0


@pytest.mark.parametrize("activation_format", ["fp16", "bf16"])
@pytest.mark.parametrize("scale_format", ["fp16", "bf16"])
def test_sampled_scale_matches_all_backends(activation_format, scale_format):
    config = FpIntConfig(4, group_size=128, mxu_rows=128, activation_format=activation_format)
    kwargs = dict(m=4, k=257, n=5, config=config, seed=72,
                  exponent_max=config.exponent_bias, scale_mode="log-uniform",
                  scale_format=scale_format)
    case = make_case(**kwargs)
    repeated = make_case(**kwargs)
    tensors = tuple(value if isinstance(value, torch.Tensor) else torch.from_numpy(value) for value in case)
    scale = tensors[2]
    assert scale.dtype == (torch.float16 if scale_format == "fp16" else torch.bfloat16)
    assert bool(((scale >= 2**-4) & (scale <= 1)).all())
    assert scale.unique().numel() > 1
    assert not tensors[3].any()
    repeated_scale = repeated[2] if isinstance(repeated[2], torch.Tensor) else torch.from_numpy(repeated[2])
    assert torch.equal(scale, repeated_scale)
    expected = torch.from_numpy(qcol_real_2scomp_reference(*case, config)).float()
    backends = [("fpint_torch", "cpu")]
    if torch.cuda.is_available():
        backends += [("fpint_torch", "cuda"), ("fpint_cuda", "cuda")]
    for backend, device in backends:
        actual = fpint_linear(*(value.to(device) for value in tensors), config, backend=backend, has_zero=False)
        torch.testing.assert_close(actual.cpu().float(), expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("group_size", [2, -1])
def test_scaled_reference_and_baseline_use_group_scales(dtype, group_size):
    activation = torch.tensor([[1, 2, 3]], dtype=dtype)
    weight = torch.tensor([[3, -1, 2]], dtype=torch.int8)
    scale = torch.tensor([[0.3, 0.7]] if group_size == 2 else [[0.3]], dtype=dtype)
    expanded = [float(scale[0, 0]), float(scale[0, 0]), float(scale[0, -1])]
    expected = sum(a * w * s for a, w, s in zip([1, 2, 3], [3, -1, 2], expanded))
    assert fp64_reference_linear(activation, weight, scale, group_size).item() == expected
    rounded_weight = torch.tensor([[w * s for w, s in zip([3, -1, 2], expanded)]], dtype=dtype)
    expected_baseline = torch.nn.functional.linear(activation, rounded_weight)
    for mode in (True, False):
        assert torch.equal(conventional_linear(activation, weight, scale, group_size, mode), expected_baseline)


@pytest.mark.parametrize("dtype,flag", [
    (torch.float16, "allow_fp16_reduced_precision_reduction"),
    (torch.bfloat16, "allow_bf16_reduced_precision_reduction"),
])
def test_baseline_scopes_reduction_flag_even_on_error(monkeypatch, dtype, flag):
    original = getattr(torch.backends.cuda.matmul, flag)
    observed = []
    def fail(*args):
        observed.append(getattr(torch.backends.cuda.matmul, flag))
        raise RuntimeError("injected failure")
    monkeypatch.setattr(torch.nn.functional, "linear", fail)
    for mode in (True, False):
        with pytest.raises(RuntimeError, match="injected failure"):
            conventional_linear(torch.ones(1, 2, dtype=dtype), torch.ones(1, 2, dtype=torch.int8), reduced_precision=mode)
        assert getattr(torch.backends.cuda.matmul, flag) == original
    assert observed == [True, False]


def test_three_candidates_share_mask_including_false_baseline():
    result = paired_error_metrics(
        torch.tensor([1., 2., float("inf"), 4.]),
        torch.tensor([1., 2., 3., float("nan")]),
        torch.tensor([1., 2., 3., 4.], dtype=torch.float64),
        conventional_full_precision=torch.tensor([1., float("inf"), 3., 4.]),
    )
    assert result["coverage"]["common_finite_elements"] == 1
    assert result["coverage"]["conventional_full_precision_nonfinite"] == 1
    for key in ("conventional_err", "conventional_full_precision_err", "fp_int_err"):
        assert result[key]["elements"] == 1
        assert result[key]["rmse"] == 0


def test_sampled_scale_run_and_report_cpu():
    args = argparse.Namespace(
        bits=4, group_size=128, mxu_rows=128, extra_bits=19, reduce_extra_bits=10,
        activation_format="bf16", m=4, n=4, k_values=(128,),
        exponent_max_by_k={128: 127}, finite_target=1.0, trials=2,
        base_seed=17, atol=0., rtol=0., device="cpu", scale_mode="log-uniform",
        scale_format="activation", scale_log2_min=-4., scale_log2_max=0.,
    )
    result = run(args)
    assert result["status"] == "pass"
    assert result["config"]["scale_sampling"]["format"] == "bf16"
    assert result["config"]["baseline_reduced_precision_modes"] == [True, False]
    assert result["summary"]["overall"]["conventional_full_precision_err"]["elements"] == 32
    report = render_report(result)
    assert "GPU True" in report and "GPU False" in report
    assert "Uniform(-4.0, 0.0)" in report
    assert "2개 trial" in report


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_scaled_cancellation_preserves_exact_power_of_two_restoration():
    # CUDA torch.ldexp(1., -19) used to perturb FP32 tie rounding, producing
    # -462 instead of -464 at [14, 28] after scale and K-tile accumulation.
    config = FpIntConfig(4, group_size=128, mxu_rows=128, activation_format="bf16")
    case = make_case(m=32, k=4096, n=32, config=config, seed=20261154,
                     exponent_max=134, scale_mode="log-uniform")
    expected = torch.from_numpy(qcol_real_2scomp_reference(*case, config)).float()
    assert expected[14, 28].item() == -464
    tensors = tuple((value if isinstance(value, torch.Tensor) else torch.from_numpy(value)).cuda() for value in case)
    for backend in ("fpint_torch", "fpint_cuda"):
        actual = fpint_linear(*tensors, config, backend=backend, has_zero=False)
        torch.testing.assert_close(actual.cpu().float(), expected, atol=0, rtol=0)


def test_exponent_map_parser():
    assert parse_exponent_max_map("128:25,256:24") == {128: 25, 256: 24}
    with pytest.raises(argparse.ArgumentTypeError):
        parse_exponent_max_map("128:255")


def test_cli_exponent_map_uses_internal_field_name(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "measure_fpint_qcol_accuracy.py",
            "--k-values",
            "128",
            "--exp-max-by-k",
            "128:25",
            "--output",
            str(tmp_path / "result.json"),
        ],
    )
    args = parse_args()
    assert args.exponent_max_by_k == {128: 25}


def test_paired_errors_use_one_common_finite_mask():
    reference = torch.tensor([1.0003, 2.0, 3.0, 4.0], dtype=torch.float64)
    conventional = torch.tensor([1.0, float("nan"), 3.0, float("inf")])
    fp_int = torch.tensor([1.0, 2.0, float("inf"), 4.0])
    result = paired_error_metrics(conventional, fp_int, reference)
    coverage = result["coverage"]
    assert coverage["common_finite_elements"] == 1
    assert coverage["conventional_nonfinite"] == 2
    assert coverage["fp_int_nonfinite"] == 1
    assert result["conventional_err"]["elements"] == 1
    assert result["fp_int_err"]["elements"] == 1
    assert result["conventional_err"]["rmse"] == pytest.approx(0.0003)


def test_paired_errors_exclude_fp64_values_that_overflow_rounded_fp16():
    reference = torch.tensor([70_000.0], dtype=torch.float64)
    finite_candidate = torch.tensor([64_000.0], dtype=torch.float16)
    result = paired_error_metrics(finite_candidate, finite_candidate, reference)
    assert result["coverage"]["fp16_rounded_reference_nonfinite"] == 1
    assert result["coverage"]["common_finite_elements"] == 0
    assert result["conventional_err"]["rmse"] is None


def test_fp16_ulp_distance_handles_zero_subnormal_and_sign_crossing():
    zero = torch.tensor([0x0000], dtype=torch.int16).view(torch.float16)
    negative_zero = torch.tensor([-32768], dtype=torch.int16).view(torch.float16)
    min_subnormal = torch.tensor([0x0001], dtype=torch.int16).view(torch.float16)
    negative_min_subnormal = torch.tensor([-32767], dtype=torch.int16).view(
        torch.float16
    )
    assert fp16_ulp_distance(zero, negative_zero).item() == 0
    assert fp16_ulp_distance(zero, min_subnormal).item() == 1
    assert fp16_ulp_distance(negative_min_subnormal, min_subnormal).item() == 2


def test_fp64_and_conventional_paths_do_not_accept_mxu_configuration():
    activation = torch.tensor([[1.0, 2.0]], dtype=torch.float16)
    weight = torch.tensor([[3, -1]], dtype=torch.int8)
    assert conventional_linear(activation, weight).item() == 1.0
    assert fp64_reference_linear(activation, weight).item() == 1.0


def test_k_summary_reports_element_and_trial_standard_deviations():
    reference = torch.zeros(2, dtype=torch.float64)
    records = []
    for trial, (conv_value, fp_int_value) in enumerate(((1.0, 2.0), (3.0, 4.0))):
        numerical = paired_error_metrics(
            torch.tensor([conv_value, -conv_value], dtype=torch.float16),
            torch.tensor([fp_int_value, -fp_int_value], dtype=torch.float16),
            reference,
        )
        records.append(
            {
                "trial": trial,
                "k": 128,
                "input_stats": {"positive_sign_bits": 1, "negative_sign_bits": 1},
                "coverage": numerical["coverage"],
                "comparisons": {
                    "fpint_torch_vs_qcol_reference": {"allclose": True},
                    "conventional_err": numerical["conventional_err"],
                    "fp_int_err": numerical["fp_int_err"],
                },
            }
        )
    row = summarize_records(records, finite_target=1.0)["by_k"]["128"]
    assert row["conventional_err"]["trial_rmse_mean"] == pytest.approx(2.0)
    assert row["conventional_err"]["trial_rmse_sample_std"] == pytest.approx(
        2**0.5
    )
    assert row["conventional_err"]["global_signed_error_std"] == pytest.approx(
        5**0.5
    )
    assert row["comparison"]["trial_rmse_mean"] == {
        "delta_fp_int_minus_conventional": pytest.approx(1.0),
        "ratio_fp_int_over_conventional": pytest.approx(1.5),
    }


def test_qcol_allclose_metrics_handles_tolerance_and_infinities():
    expected = torch.tensor([float("inf"), -float("inf"), 1.0], dtype=torch.float16)
    actual = torch.tensor([float("inf"), -float("inf"), 1.0005])
    result = qcol_allclose_metrics(actual, expected, 1e-3, 1e-3)
    assert result["allclose"]
    assert result["reference_nonfinite"] == 2


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


def test_mxu128_runner_invalidates_stale_fpint_kernel_results():
    script = Path(__file__).parents[1] / "scripts" / "run_fpint_mxu128_llama31_8b.sh"
    source = script.read_text(encoding="utf-8")
    assert 'backend != "fpint_cuda"' in source
    assert 'metadata.get("fpint_cuda_kernel_sha256") == fpint_kernel_sha256' in source


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


def test_full_result_aggregation_rejects_task_batch_mismatch():
    shards = [
        ["wikitext"],
        ["hellaswag"],
        ["arc_easy", "openbookqa"],
        ["arc_challenge", "winogrande"],
    ]
    standard = [_full_payload("standard", tasks) for tasks in shards]
    fpint = [_full_payload("fpint_cuda", tasks) for tasks in shards]
    fpint[1]["spinquant_quantization"]["lm_eval_batch_size"] = "64"
    with pytest.raises(ValueError, match="hellaswag uses batch 64, expected 32"):
        aggregate_full_results(standard, fpint)


def test_report_contains_only_random_fpxint_results():
    error = {
        "elements": 1024,
        "global_rmse": 0.5,
        "global_signed_error_mean": 0.1,
        "global_signed_error_std": 0.49,
        "global_mean_ulp": 1.5,
        "global_std_ulp": 0.75,
        "global_max_ulp": 4,
        "trials_with_metrics": 1,
        "trial_rmse_mean": 0.5,
        "trial_rmse_sample_std": 0.0,
        "trial_mean_ulp_mean": 1.5,
        "trial_mean_ulp_sample_std": 0.0,
        "trial_p50_ulp_mean": 1.0,
        "trial_p95_ulp_mean": 3.0,
    }
    summary = {
        "cases": 1,
        "coverage": {
            "output_elements": 1024,
            "fp64_reference_nonfinite": 0,
            "fp16_rounded_reference_nonfinite": 1,
            "conventional_nonfinite": 1,
            "fp_int_nonfinite": 0,
            "common_finite_elements": 1023,
            "common_finite_fraction": 1023 / 1024,
        },
        "finite_target_met": True,
        "activation_sign_bits": {"positive": 2048, "negative": 2048},
        "qcol_correctness": {
            "torch_allclose_cases": 1,
            "cuda_cases": 1,
            "cuda_allclose_cases": 1,
        },
        "conventional_err": error,
        "fp_int_err": error,
        "comparison": {
            "trial_rmse_mean": {
                "delta_fp_int_minus_conventional": 0.0,
                "ratio_fp_int_over_conventional": 1.0,
            },
            "trial_mean_ulp_mean": {
                "delta_fp_int_minus_conventional": 0.0,
                "ratio_fp_int_over_conventional": 1.0,
            },
        },
    }
    random = {
        "status": "pass",
        "environment": {
            "fpint_cuda_kernel_sha256": "abc",
            "allow_fp16_reduced_precision_reduction": True,
        },
        "config": {
            "m": 32,
            "n": 32,
            "k_values": [128],
            "trials": 1,
            "finite_target": 0.999,
            "weight_bits": 4,
            "mxu_rows": 128,
            "group_size": 128,
            "integer_range": [-8, 7],
            "fp16_fields": {
                "sign": [0, 1],
                "exponent_min": 0,
                "exponent_max_by_k": {"128": 24},
                "mantissa": [0, 1023],
            },
        },
        "summary": {"overall": summary, "by_k": {"128": summary}},
    }
    report = render_report(random)
    assert "FP64-reference FP×INT" in report
    assert "Conventional RMSE" in report
    assert "FPINT mean ULP" in report
    for forbidden in ("Llama", "Smoke", "WikiText", "HellaSwag", "perplexity"):
        assert forbidden not in report
