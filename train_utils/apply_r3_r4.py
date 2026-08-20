# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import math

import torch
import tqdm

from utils import quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
)
from utils.utils import HadamardTransform


def R4_rotate_down_proj_weights(layer, online_had_mode):
    # Rotate the MLP output weights and bias.
    W = layer.mlp.down_proj
    apply_exact_had_to_linear(
        W,
        had_dim=-1,
        output=False,
        online_had_mode=online_had_mode,
    )  # apply exact (inverse) hadamard on the weights of mlp output


@torch.inference_mode()
def rotate_model(model, args):
    config = model.config
    num_heads = config.num_attention_heads
    model_dim = config.hidden_size
    head_dim = model_dim // num_heads

    utils.cleanup_memory()
    layers = [layer for layer in model.model.layers]
    for idx, layer in enumerate(
        tqdm.tqdm(layers, unit="layer", desc="Applying R4 rotation to W_down")
    ):
        R4_rotate_down_proj_weights(layers[idx], args.online_had_mode)


class QKRotationWrapper(torch.nn.Module):
    def __init__(self, func, config, *args, **kwargs):
        super().__init__()
        self.config = config
        num_heads = config.num_attention_heads
        model_dim = config.hidden_size
        head_dim = model_dim // num_heads
        assert is_pow2(
            head_dim
        ), f"Only power of 2 head_dim is supported for K-cache Quantization!"
        self.func = func
        self.q_quantizer = quant_utils.ActQuantizer()
        self.k_quantizer = quant_utils.ActQuantizer()
        self.q_bits = kwargs["q_bits"]
        self.k_bits = kwargs["k_bits"]
        self.q_groupsize = kwargs["q_groupsize"]
        self.k_groupsize = kwargs["k_groupsize"]
        for tensor_name, groupsize in (
            ("Q", self.q_groupsize),
            ("K", self.k_groupsize),
        ):
            assert groupsize in [
                -1,
                head_dim,
            ], f"Only token-wise/{head_dim}g quantization is supported for {tensor_name}"
        self.q_quantizer.configure(
            bits=self.q_bits,
            groupsize=-1,
            sym=kwargs["q_sym"],
            clip_ratio=kwargs["q_clip_ratio"],
        )
        self.k_quantizer.configure(
            bits=self.k_bits,
            groupsize=-1,
            sym=kwargs["k_sym"],
            clip_ratio=kwargs["k_clip_ratio"],
        )

    def extra_repr(self):
        return (
            f"q_bits={self.q_bits}, k_bits={self.k_bits}, "
            f"q_groupsize={self.q_groupsize}, k_groupsize={self.k_groupsize}"
        )

    @staticmethod
    def _quantize_states(states, quantizer, groupsize):
        bsz, num_heads, seq_len, head_dim = states.shape
        if groupsize == -1:
            token_wise = states.transpose(1, 2).reshape(
                -1, num_heads * head_dim
            )
            quantizer.find_params(token_wise)
            return (
                quantizer(token_wise)
                .reshape(bsz, seq_len, num_heads, head_dim)
                .transpose(1, 2)
                .to(states)
            )

        per_head = states.reshape(-1, head_dim)
        quantizer.find_params(per_head)
        return quantizer(per_head).reshape(states.shape).to(states)

    def forward(self, *args, **kwargs):
        q, k = self.func(*args, **kwargs)
        dtype = q.dtype
        q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
        k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)

        if self.q_bits < 16:
            q = self._quantize_states(q, self.q_quantizer, self.q_groupsize)
            self.q_quantizer.free()
        if self.k_bits < 16:
            k = self._quantize_states(k, self.k_quantizer, self.k_groupsize)
            self.k_quantizer.free()

        return q, k


def add_qk_rotation_wrapper_after_function_call_in_forward(
    module,
    function_name,
    *args,
    **kwargs,
):
    """
    This function adds a rotation wrapper after the output of a function call in forward.
    Only calls directly in the forward function are affected. calls by other functions called in forward are not affected.
    """
    import functools

    from utils import monkeypatch

    attr_name = f"{function_name}_qk_rotation_wrapper"
    assert not hasattr(module, attr_name)
    wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
        module,
        "forward",
        function_name,
        functools.partial(QKRotationWrapper, *args, **kwargs),
    )
    setattr(module, attr_name, wrapper)
