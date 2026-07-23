import json

import pytest

from summarize_hadamard_comparison import build_rows, load_metrics


def write_result(path, offset):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {
            "hellaswag": {"acc_norm,none": 0.70 + offset},
            "arc_easy": {"acc_norm,none": 0.60 + offset},
            "arc_challenge": {"acc_norm,none": 0.40 + offset},
            "winogrande": {"acc,none": 0.65 + offset},
            "openbookqa": {"acc_norm,none": 0.45 + offset},
            "wikitext": {"word_perplexity,none": 9.0 + offset},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_metrics_selects_expected_fields(tmp_path):
    result = tmp_path / "model.json"
    write_result(result, 0.0)
    assert load_metrics(result) == {
        "HellaSwag": 0.70,
        "ARC Easy": 0.60,
        "ARC Challenge": 0.40,
        "Winogrande": 0.65,
        "OpenBookQA": 0.45,
        "WikiText PPL": 9.0,
    }


def test_build_rows_computes_zero_minus_factorized(tmp_path):
    factorized = tmp_path / "factorized"
    zero_padding = tmp_path / "zero_padding"
    write_result(factorized / "llama2-7b.json", 0.0)
    write_result(zero_padding / "llama2-7b.json", 0.02)
    rows = build_rows(
        factorized,
        zero_padding,
        ["llama2-7b"],
    )
    assert rows[0]["model"] == "llama2-7b"
    assert rows[0]["metric"] == "HellaSwag"
    assert rows[0]["delta"] == pytest.approx(0.02)


def test_load_metrics_reports_missing_metric(tmp_path):
    result = tmp_path / "broken.json"
    result.write_text('{"results": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="hellaswag"):
        load_metrics(result)
