#!/usr/bin/env python3
"""Validate unquantized evaluation artifacts and render a model-level report."""

import argparse
import json
import os
from pathlib import Path

from summarize_fpint_mxu128 import TASK_METRICS


EXPECTED_SAMPLES = {
    "hellaswag": 10042, "arc_easy": 2376, "arc_challenge": 1172,
    "winogrande": 1267, "openbookqa": 500,
}
MODELS = (("llama2-7b", "Llama-2 7B", "fp16"), ("llama31-8b", "Llama-3.1 8B", "bf16"))


def collect(root):
    models = {}
    for name, label, dtype in MODELS:
        tasks, metadata, sources = {}, [], []
        for shard in ("accuracy", "wikitext"):
            path = root / f"{name}-{shard}.json"
            data = json.loads(path.read_text())
            meta = data["spinquant_quantization"]
            expected = {
                "weight_bits": 16, "activation_bits": 16, "query_bits": 16,
                "probability_bits": 16, "key_bits": 16, "value_bits": 16,
                "rotate": False, "linear_backend": "standard", "compute_dtype": dtype,
                "attention_backend": "eager", "model_max_length": 2048,
                "num_fewshot": 0, "eval_limit": None, "seed": 0,
                "quantized_checkpoint": None, "rotation_checkpoint": None,
                "lm_eval_batch_size": "1" if shard == "wikitext" else "32",
            }
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise ValueError(f"{path}: {key}={meta.get(key)!r}, expected {value!r}")
            expected_tasks = {"wikitext"} if shard == "wikitext" else set(EXPECTED_SAMPLES)
            if set(data["results"]) != expected_tasks:
                raise ValueError(f"{path}: unexpected task set")
            for task in expected_tasks:
                count = data["n-samples"][task]
                if count["original"] != count["effective"]:
                    raise ValueError(f"{path}: incomplete {task}")
                expected_count = 62 if task == "wikitext" else EXPECTED_SAMPLES[task]
                if count["effective"] != expected_count:
                    raise ValueError(f"{path}: changed sample count for {task}")
                metric, kind = TASK_METRICS[task]
                value = data["results"][task][metric]
                stderr_key = metric.replace(",none", "_stderr,none")
                tasks[task] = {
                    "metric": metric, "kind": kind, "value": value,
                    "stderr": data["results"][task].get(stderr_key),
                    "samples": count["effective"],
                }
            metadata.append(meta)
            sources.append(str(path))
        if metadata[0]["model"] != metadata[1]["model"]:
            raise ValueError("model snapshots differ across shards")
        total = sum(tasks[t]["samples"] for t in EXPECTED_SAMPLES)
        micro = sum(tasks[t]["value"] * tasks[t]["samples"] for t in EXPECTED_SAMPLES) / total
        models[name] = {
            "label": label, "dtype": dtype, "model": metadata[0]["model"],
            "tasks": tasks, "micro_accuracy": micro, "accuracy_samples": total,
            "evaluation_seconds_sum": sum(m["evaluation_seconds"] for m in metadata),
            "sources": sources,
            "environment": {
                "transformers": data["transformers_version"],
                "torch": next(line for line in data["pretty_env_info"].splitlines()
                              if line.startswith("PyTorch version:")),
                "lm_eval_git_hash": data["git_hash"],
            },
        }
    return models


def render(models, output):
    lines = [
        "# Llama-2 / Llama-3.1 비양자화 baseline",
        "",
        "[문서 목차](README.md)",
        "",
        "원본 pretrained base model을 사용해 weight·activation·Q/K/V·attention probability 양자화와 rotation을 모두 끄고 평가했다.",
        "비양자화는 원본 모델의 FP16/BF16 연산을 뜻하며 FP32 평가를 뜻하지 않는다.",
        "",
        "## 설정",
        "",
        "- Llama-2 7B: FP16; Llama-3.1 8B: BF16",
        "- W/A/Q/P/K/V bits=16 (bypass), rotation=False, standard Linear; quantized checkpoint 미사용",
        "- 기존 모델 평가와 동일한 ptq.py 및 모델 구현·tokenizer 로딩 경로 사용",
        "- Eager attention, max length=2048, zero-shot, seed=0, eval limit 없음",
        "- Accuracy batch size=32; WikiText batch size=1",
        "- HellaSwag/ARC Easy/ARC Challenge/OpenBookQA: acc_norm; WinoGrande: acc",
        "- WikiText: lm-eval wikitext task의 word_perplexity",
        "- Micro average: 5개 accuracy task의 sample 수로 가중 평균 (총 15,357개)",
        "",
        "## 정확도",
        "",
        "| Task | Samples | Llama-2 7B FP16 | Llama-3.1 8B BF16 |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for task, count in EXPECTED_SAMPLES.items():
        cells = [f"{100 * models[name]['tasks'][task]['value']:.6f}%" for name, _, _ in MODELS]
        lines.append(f"| {task} | {count} | " + " | ".join(cells) + " |")
    lines.append(
        "| Micro average | 15357 | "
        + " | ".join(f"{100 * models[name]['micro_accuracy']:.6f}%" for name, _, _ in MODELS) + " |"
    )
    lines += [
        "", "## WikiText perplexity", "",
        "| Model | Word perplexity |", "| :--- | ---: |",
    ]
    for name, label, _ in MODELS:
        lines.append(f"| {label} | {models[name]['tasks']['wikitext']['value']:.8f} |")
    lines += ["", "## 원본 및 재현", ""]
    for name, label, dtype in MODELS:
        model = models[name]
        lines += [
            f"- {label} ({dtype}): {model['model']}",
            f"- Shard 평가 시간 합계: {model['evaluation_seconds_sum']:.2f} s",
            f"- {model['environment']['torch']}; Transformers {model['environment']['transformers']}; lm-eval git {model['environment']['lm_eval_git_hash']}",
        ]
        for source in model["sources"]:
            path = Path(source)
            lines.append(f"- [{path.name}]({os.path.relpath(path, output.parent)})")
    lines += [
        "", chr(96) * 3 + "bash",
        "conda activate spinquant",
        "bash scripts/run_unquantized_llama_baselines.sh",
        "python summarize_unquantized_llama_baselines.py --input-root results/unquantized-llama-baselines --output docs/FP-INT-hw-acc/unquantized_llama_baselines.md",
        chr(96) * 3, "",
        "Runner는 로컬 Hugging Face cache에서 weight shard가 모두 존재하는 snapshot을 찾는다.",
        "LLAMA2_MODEL/LLAMA3_MODEL로 경로를 지정할 수 있으며, CUDA_DEVICES는 네 GPU 번호다.",
        "소량 점검은 EVAL_LIMIT과 별도 OUTPUT_ROOT를 지정한다. 이 보고서는 limit 없는 전체 결과만 허용한다.",
        "",
        "JSON에는 task config/version, sample 수, stderr 및 양자화 bypass 설정이 기록되어 있다.",
        "기존 ptq.py metadata의 weight_quantizer/scale_dtype는 공통 기본값이며, 여기서는 weight_bits=16이므로 실제 weight quantizer나 scale은 사용하지 않는다.",
        "이전 W4 결과와 비교할 때 task별 metric, dtype, tokenizer와 dataset/model revision 차이를 함께 확인한다.",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = collect(args.input_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(models, args.output))
    (args.input_root / "summary.json").write_text(json.dumps(models, indent=2) + "\n")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
