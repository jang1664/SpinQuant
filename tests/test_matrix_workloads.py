from pathlib import Path

from utils.matrix_workloads import (
    _arc_prompt,
    _hellaswag_prompt,
    _winogrande_prompt,
    load_workloads,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "workloads" / "matrix_accuracy_v1.yaml"


def test_mcq_adapters_render_one_request_per_choice():
    prompt, choices = _arc_prompt(
        {
            "question": "Which is correct?",
            "choices": {"label": ["A", "B"], "text": ["first", "second"]},
        }
    )
    assert "A. first" in prompt
    assert choices == ["first", "second"]

    prompt, choices = _hellaswag_prompt(
        {"ctx_a": "A person", "ctx_b": "walks", "endings": ["home", "away"]}
    )
    assert prompt == "A person walks"
    assert choices == ["home", "away"]

    prompt, choices = _winogrande_prompt(
        {"sentence": "Alex thanked _ warmly.", "option1": "Sam", "option2": "Lee"}
    )
    assert prompt == "Alex thanked "
    assert choices == ["Sam warmly.", "Lee warmly."]


def test_ruler_manifest_is_pinned_and_loadable():
    manifest, workloads = load_workloads(
        MANIFEST, ["ruler_niah_single_1_2k"]
    )
    workload = workloads[0]
    assert manifest["version"] == 1
    assert workload.family == "long_context_stress"
    assert workload.config["ruler_task"] == "niah_single_1"
    assert len(workload.examples) == 4
    assert all(example.kind == "prompt_only" for example in workload.examples)
    assert all("special magic number" in example.text for example in workload.examples)
