"""Pinned input adapters for matrix-output numerical-accuracy workloads."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class MatrixWorkloadExample:
    """One exact text request measured by the FP16/AQP forwards."""

    workload: str
    family: str
    example_id: str
    text: str
    kind: str
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class MatrixWorkload:
    name: str
    family: str
    config: dict[str, Any]
    examples: list[MatrixWorkloadExample]


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _take_nonempty(dataset: Any, count: int, text_key: str) -> list[tuple[int, dict[str, Any]]]:
    selected = []
    for index, row in enumerate(dataset):
        if str(row.get(text_key, "")).strip():
            selected.append((index, dict(row)))
            if len(selected) == count:
                break
    if len(selected) != count:
        raise ValueError(f"requested {count} non-empty examples but found {len(selected)}")
    return selected


def _load_dataset(
    name: str, config: str | None, split: str, streaming: bool, revision: str | None
) -> Any:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency is runtime-only
        raise RuntimeError("matrix workload adapters require `datasets`") from exc
    return load_dataset(name, config, split=split, streaming=streaming, revision=revision)


def _rolling_text_workload(name: str, config: dict[str, Any]) -> MatrixWorkload:
    dataset = _load_dataset(
        config["dataset"], config.get("dataset_config"), config["split"], True,
        config.get("revision"),
    )
    examples = []
    for index, row in _take_nonempty(dataset, int(config["source_documents"]), config["text_key"]):
        text = str(row[config["text_key"]]).strip()
        examples.append(
            MatrixWorkloadExample(
                workload=name,
                family=config["family"],
                example_id=f"{name}:{index}",
                text=text,
                kind="rolling_lm",
                source_metadata={
                    "dataset": config["dataset"],
                    "dataset_config": config.get("dataset_config"),
                    "split": config["split"],
                    "revision": config.get("revision"),
                    "source_index": index,
                    "text_sha256": sha256_text(text),
                },
            )
        )
    return MatrixWorkload(name, config["family"], config, examples)


def _arc_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    choices = row["choices"]
    labels, texts = choices["label"], choices["text"]
    rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
    return f"Question: {row['question']}\nChoices:\n{rendered}\nAnswer:", list(texts)


def _hellaswag_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    context = f"{row['ctx_a']} {row['ctx_b']}".replace("[title]", "").strip()
    return context, list(row["endings"])


def _winogrande_prompt(row: dict[str, Any]) -> tuple[str, list[str]]:
    before, after = row["sentence"].split("_", 1)
    return before, [f"{row['option1']}{after}", f"{row['option2']}{after}"]


MCQ_PROMPTS = {
    "arc_challenge": _arc_prompt,
    "hellaswag": _hellaswag_prompt,
    "winogrande": _winogrande_prompt,
}


def _mcq_workload(name: str, config: dict[str, Any]) -> MatrixWorkload:
    if name not in MCQ_PROMPTS:
        raise ValueError(f"unsupported MCQ adapter: {name}")
    dataset = _load_dataset(
        config["dataset"], config.get("dataset_config"), config["split"], True,
        config.get("revision"),
    )
    examples = []
    for index, row in _take_nonempty(dataset, int(config["source_documents"]), config["id_key"]):
        prompt, choices = MCQ_PROMPTS[name](row)
        source_id = str(row.get(config["id_key"], index))
        for choice_index, continuation in enumerate(choices):
            text = f"{prompt} {continuation}"
            examples.append(
                MatrixWorkloadExample(
                    workload=name,
                    family=config["family"],
                    example_id=f"{name}:{source_id}:choice-{choice_index}",
                    text=text,
                    kind="choice",
                    source_metadata={
                        "dataset": config["dataset"],
                        "dataset_config": config.get("dataset_config"),
                        "split": config["split"],
                        "revision": config.get("revision"),
                        "source_id": source_id,
                        "choice_index": choice_index,
                        "prompt_sha256": sha256_text(prompt),
                        "continuation_sha256": sha256_text(continuation),
                    },
                )
            )
    return MatrixWorkload(name, config["family"], config, examples)


def _ruler_niah_workload(name: str, config: dict[str, Any], root: Path) -> MatrixWorkload:
    path = root / config["jsonl"]
    if not path.is_file():
        raise FileNotFoundError(path)
    examples = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            text = str(row["input"])
            examples.append(
                MatrixWorkloadExample(
                    workload=name,
                    family=config["family"],
                    example_id=f"{name}:{index}",
                    text=text,
                    kind="prompt_only",
                    source_metadata={
                        "ruler_task": config["ruler_task"],
                        "ruler_repo": config["ruler_repo"],
                        "ruler_revision": config["ruler_revision"],
                        "seed": config["seed"],
                        "context_length": config["context_length"],
                        "input_sha256": sha256_text(text),
                        "generator_record": {
                            key: row[key]
                            for key in ("length", "token_position_answer", "outputs")
                            if key in row
                        },
                    },
                )
            )
    if len(examples) != int(config["examples"]):
        raise ValueError(f"{name}: expected {config['examples']} RULER examples, got {len(examples)}")
    return MatrixWorkload(name, config["family"], config, examples)


def legacy_wikitext_workload(name: str, fp_results_path: str | Path) -> MatrixWorkload:
    payload = json.loads(Path(fp_results_path).read_text(encoding="utf-8"))
    examples = []
    for sample in payload["samples"]["wikitext"]:
        text = str(sample["target"])
        examples.append(
            MatrixWorkloadExample(
                workload=name,
                family="ppl",
                example_id=f"{name}:{sample['doc_id']}",
                text=text,
                kind="rolling_lm",
                source_metadata={
                    "legacy_fp_results": str(fp_results_path),
                    "doc_id": sample["doc_id"],
                    "doc_hash": sample["doc_hash"],
                    "text_sha256": sha256_text(text),
                },
            )
        )
    return MatrixWorkload(name, "ppl", {"adapter": "legacy_wikitext"}, examples)


def load_workloads(
    manifest_path: str | Path,
    selected: list[str],
    *,
    legacy_fp_results_path: str | Path | None = None,
) -> tuple[dict[str, Any], list[MatrixWorkload]]:
    manifest_file = Path(manifest_path)
    manifest = yaml.safe_load(manifest_file.read_text(encoding="utf-8"))
    configured = manifest.get("workloads", {})
    missing = [name for name in selected if name not in configured]
    if missing:
        raise ValueError(f"workloads absent from manifest: {missing}")
    loaded = []
    for name in selected:
        config = dict(configured[name])
        adapter = config["adapter"]
        if adapter == "legacy_wikitext":
            if legacy_fp_results_path is None:
                raise ValueError("wikitext requires --fp-results-path")
            loaded.append(legacy_wikitext_workload(name, legacy_fp_results_path))
        elif adapter == "rolling_text":
            loaded.append(_rolling_text_workload(name, config))
        elif adapter == "mcq":
            loaded.append(_mcq_workload(name, config))
        elif adapter == "ruler_niah":
            loaded.append(_ruler_niah_workload(name, config, manifest_file.parent.parent))
        else:
            raise ValueError(f"{name}: unsupported adapter {adapter!r}")
    return manifest, loaded
