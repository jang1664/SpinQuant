import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

from eval_utils.main import add_probability_quantization, ptq_model
from eval_utils.modeling_llama import LlamaAttention, LlamaForCausalLM
from eval_utils.rotation_utils import QKRotationWrapper
from utils.quant_utils import ActQuantizer
from utils.process_args import parser_gen
from utils.utils import HadamardTransform


def parse_spinquant_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["ptq.py", *args])
    parsed, unknown = parser_gen()
    assert unknown == []
    return parsed


def test_aqp_argument_defaults_and_q_groupsize_fallback(monkeypatch):
    args = parse_spinquant_args(monkeypatch, "--k_groupsize", "128")
    assert (args.q_bits, args.p_bits) == (16, 16)
    assert args.q_groupsize == 128
    assert args.p_groupsize == -1
    assert args.p_asym is True
    assert args.attention_backend == "auto"


def test_quantizer_repr_exposes_configured_bit_width():
    quantizer = ActQuantizer()
    quantizer.configure(bits=8, groupsize=-1, sym=False, clip_ratio=1.0)
    assert "bits=8" in repr(quantizer)
    assert "asymmetric" in repr(quantizer)


def test_probability_groupsize_rejects_ambiguous_grouping(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["ptq.py", "--p_groupsize", "16"]
    )
    with pytest.raises(SystemExit):
        parser_gen()


def qk_config():
    return SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
    )


def qk_kwargs(q_bits=4, k_bits=16):
    return {
        "q_bits": q_bits,
        "q_groupsize": 8,
        "q_sym": True,
        "q_clip_ratio": 1.0,
        "k_bits": k_bits,
        "k_groupsize": 8,
        "k_sym": True,
        "k_clip_ratio": 1.0,
    }


def test_q_quantization_is_gqa_safe_and_k_can_bypass():
    torch.manual_seed(0)
    q = torch.randn(2, 4, 3, 8)
    k = torch.randn(2, 2, 3, 8)
    wrapper = QKRotationWrapper(
        lambda: (q, k), qk_config(), **qk_kwargs(q_bits=4, k_bits=16)
    )

    quantized_q, bypassed_k = wrapper()
    expected_q = HadamardTransform.apply(q.float()) / q.shape[-1] ** 0.5
    expected_k = HadamardTransform.apply(k.float()) / k.shape[-1] ** 0.5

    assert quantized_q.shape == q.shape
    assert bypassed_k.shape == k.shape
    assert not torch.equal(quantized_q, expected_q)
    torch.testing.assert_close(bypassed_k, expected_k)


def test_q_can_bypass_while_k_is_quantized():
    torch.manual_seed(1)
    q = torch.randn(1, 4, 3, 8)
    k = torch.randn(1, 2, 3, 8)
    wrapper = QKRotationWrapper(
        lambda: (q, k), qk_config(), **qk_kwargs(q_bits=16, k_bits=4)
    )

    bypassed_q, quantized_k = wrapper()
    expected_q = HadamardTransform.apply(q.float()) / q.shape[-1] ** 0.5
    expected_k = HadamardTransform.apply(k.float()) / k.shape[-1] ** 0.5

    torch.testing.assert_close(bypassed_q, expected_q)
    assert not torch.equal(quantized_k, expected_k)


class RecordingQuantizer(ActQuantizer):
    def find_params(self, x):
        self.input_to_qdq = x.detach().clone()
        super().find_params(x)

    def forward(self, x):
        output = super().forward(x)
        self.output_from_qdq = output.detach().clone()
        return output


def tiny_llama_config():
    return LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attention_dropout=0.0,
    )


def ptq_args(aqp_bits):
    return SimpleNamespace(
        seed=0,
        rotate=False,
        load_qmodel_path=None,
        save_qmodel_path=None,
        export_to_et=False,
        w_bits=16,
        w_rtn=False,
        a_bits=aqp_bits,
        a_groupsize=-1,
        a_asym=False,
        a_clip_ratio=1.0,
        v_bits=16,
        v_asym=False,
        v_clip_ratio=1.0,
        int8_down_proj=False,
        q_bits=aqp_bits,
        q_groupsize=8,
        q_asym=False,
        q_clip_ratio=1.0,
        k_bits=16,
        k_groupsize=8,
        k_asym=False,
        k_clip_ratio=1.0,
        k_pre_rope=False,
        p_bits=aqp_bits,
        p_groupsize=-1,
        p_asym=True,
        p_clip_ratio=1.0,
    )


def test_p_is_quantized_after_softmax_before_pv():
    torch.manual_seed(0)
    attention = LlamaAttention(tiny_llama_config(), layer_idx=0).eval()
    quantizer = RecordingQuantizer()
    quantizer.configure(bits=4, groupsize=-1, sym=False, clip_ratio=1.0)
    attention.p_quantizer = quantizer

    output, probabilities, _ = attention(
        torch.randn(2, 5, 32),
        position_ids=torch.arange(5).unsqueeze(0).expand(2, -1),
        output_attentions=True,
    )

    assert output.shape == (2, 5, 32)
    assert probabilities.shape == (2, 4, 5, 5)
    assert quantizer.input_to_qdq.shape == (2 * 4 * 5, 5)
    assert torch.isfinite(quantizer.input_to_qdq).all()
    assert (quantizer.input_to_qdq >= 0).all()
    torch.testing.assert_close(
        quantizer.input_to_qdq.sum(-1),
        torch.ones(2 * 4 * 5),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.isfinite(quantizer.output_from_qdq).all()
    assert not torch.equal(
        quantizer.input_to_qdq, quantizer.output_from_qdq
    )
    torch.testing.assert_close(
        probabilities.reshape(-1, 5), quantizer.output_from_qdq
    )


def test_p_quantization_requires_eager_attention():
    model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        model=SimpleNamespace(layers=[]),
    )
    args = SimpleNamespace(p_bits=4)
    with pytest.raises(ValueError, match="requires eager attention"):
        add_probability_quantization(model, args)


def test_p16_is_a_strict_configuration_bypass():
    sentinel = object()
    attention = SimpleNamespace(p_quantizer=sentinel)
    model = SimpleNamespace(
        config=SimpleNamespace(_attn_implementation="sdpa"),
        model=SimpleNamespace(
            layers=[SimpleNamespace(self_attn=attention)]
        ),
    )
    add_probability_quantization(model, SimpleNamespace(p_bits=16))
    assert attention.p_quantizer is sentinel


@pytest.mark.parametrize("num_key_value_heads", [4, 2])
def test_full_model_aqp16_vs_aqp4_smoke(num_key_value_heads):
    torch.manual_seed(0)
    config = tiny_llama_config()
    config.num_key_value_heads = num_key_value_heads
    config.vocab_size = 64
    config._attn_implementation = "eager"
    reference = LlamaForCausalLM(config).eval()
    quantized = LlamaForCausalLM(config).eval()
    quantized.load_state_dict(reference.state_dict())

    reference = ptq_model(ptq_args(16), reference)
    quantized = ptq_model(ptq_args(4), quantized)
    input_ids = torch.randint(0, config.vocab_size, (1, 6))

    with torch.inference_mode():
        reference_logits = reference(input_ids, use_cache=False).logits
        quantized_logits = quantized(input_ids, use_cache=False).logits

    assert reference_logits.shape == quantized_logits.shape == (1, 6, 64)
    assert torch.isfinite(reference_logits).all()
    assert torch.isfinite(quantized_logits).all()
    assert not torch.equal(reference_logits, quantized_logits)
