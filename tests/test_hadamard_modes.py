import sys

import pytest
import torch

from utils import hadamard_utils
from utils.process_args import parser_gen


def parse_spinquant_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["ptq.py", *args])
    parsed, unknown = parser_gen()
    assert unknown == []
    return parsed


def test_online_hadamard_mode_defaults_to_factorized(monkeypatch):
    args = parse_spinquant_args(monkeypatch)
    assert args.online_had_mode == "factorized"


def test_online_hadamard_mode_accepts_zero_padding(monkeypatch):
    args = parse_spinquant_args(
        monkeypatch, "--online_had_mode", "zero_padding"
    )
    assert args.online_had_mode == "zero_padding"


def test_online_hadamard_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["ptq.py", "--online_had_mode", "unknown"],
    )
    with pytest.raises(SystemExit):
        parser_gen()


def test_factorized_dispatch_matches_legacy_call():
    x = torch.randn(2, 12)
    expected = hadamard_utils.matmul_hadU(x)
    actual = hadamard_utils.matmul_hadU_dispatch(x, mode="factorized")
    torch.testing.assert_close(actual, expected)


def test_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported online Hadamard mode"):
        hadamard_utils.matmul_hadU_dispatch(
            torch.randn(2, 12), mode="unknown"
        )
