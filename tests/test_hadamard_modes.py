import sys

import pytest
import torch

from utils import hadamard_utils
from utils.process_args import parser_gen


CUDA_READY = torch.cuda.is_available()


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


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
@pytest.mark.parametrize("dim", [137, 11008, 14336])
def test_zero_padding_matches_direct_fast_kernel(dim):
    from fast_hadamard_transform import hadamard_transform

    torch.manual_seed(0)
    x = torch.randn(2, dim, device="cuda", dtype=torch.float32)
    expected = hadamard_transform(x.contiguous()) / torch.tensor(
        dim, device=x.device, dtype=torch.float32
    ).sqrt()
    actual = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
@pytest.mark.parametrize("dim", [128, 8192])
def test_modes_match_for_power_of_two_dimensions(dim):
    torch.manual_seed(0)
    x = torch.randn(2, dim, device="cuda", dtype=torch.float32)
    factorized = hadamard_utils.matmul_hadU_cuda(
        x, mode="factorized"
    )
    zero_padding = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    torch.testing.assert_close(zero_padding, factorized)


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
def test_non_power_of_two_modes_are_distinct():
    torch.manual_seed(0)
    x = torch.randn(2, 11008, device="cuda", dtype=torch.float32)
    factorized = hadamard_utils.matmul_hadU_cuda(
        x, mode="factorized"
    )
    zero_padding = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    assert not torch.allclose(zero_padding, factorized)
