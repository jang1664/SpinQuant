# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import functools
import math

import torch
import tqdm

from utils import monkeypatch, quant_utils, utils
from utils.hadamard_utils import (
    apply_exact_had_to_linear,
    is_pow2,
    random_hadamard_matrix,
)
from utils.utils import HadamardTransform
from utils.profile import measure, profile


def random_orthogonal_matrix(size, device):
  """
  Generate a random orthogonal matrix of the specified size.
  First, we generate a random matrix with entries from a standard distribution.
  Then, we use QR decomposition to obtain an orthogonal matrix.
  Finally, we multiply by a diagonal matrix with diag r to adjust the signs.

  Args:
  size (int): The size of the matrix (size x size).

  Returns:
  torch.Tensor: An orthogonal matrix of the specified size.
  """
  torch.cuda.empty_cache()
  random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
  q, r = torch.linalg.qr(random_matrix)
  q *= torch.sign(torch.diag(r)).unsqueeze(0)
  return q


def get_orthogonal_matrix(size, mode, device="cuda"):
  if mode == "random":
    return random_orthogonal_matrix(size, device)
  elif mode == "hadamard":
    return random_hadamard_matrix(size, device)
  else:
    raise ValueError(f"Unknown mode {mode}")


def rotate_embeddings(model, R1: torch.Tensor) -> None:
  # Rotate the embeddings.
  for W in [model.model.embed_tokens]:
    dtype = W.weight.data.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_inputs(layer, R1) -> None:
  # Rotate the WQ, WK and WV matrices of the self-attention layer.
  for W in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]:
    dtype = W.weight.dtype
    W_ = W.weight.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_attention_output(layer, R1) -> None:
  # Rotate output matrix of the self-attention layer.
  W = layer.self_attn.o_proj

  dtype = W.weight.data.dtype
  W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
  W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
  if W.bias is not None:
    b = W.bias.data.to(device="cuda", dtype=torch.float64)
    W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_mlp_input(layer, R1):
  # Rotate the MLP input weights.
  mlp_inputs = [layer.mlp.up_proj, layer.mlp.gate_proj]
  for W in mlp_inputs:
    dtype = W.weight.dtype
    W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
    W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_mlp_output(layer, R1, online_had_mode):
  # Rotate the MLP output weights and bias.
  W = layer.mlp.down_proj
  dtype = W.weight.data.dtype
  W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
  W.weight.data = torch.matmul(R1.T, W_).to(device="cpu", dtype=dtype)
  apply_exact_had_to_linear(
      W,
      had_dim=-1,
      output=False,
      online_had_mode=online_had_mode,
  )  # apply exact (inverse) hadamard on the weights of mlp output
  if W.bias is not None:
    b = W.bias.data.to(device="cuda", dtype=torch.float64)
    W.bias.data = torch.matmul(R1.T, b).to(device="cpu", dtype=dtype)


def rotate_head(model, R1: torch.Tensor) -> None:
  # Rotate the head.
  W = model.lm_head
  dtype = W.weight.data.dtype
  W_ = W.weight.data.to(device="cuda", dtype=torch.float64)
  W.weight.data = torch.matmul(W_, R1).to(device="cpu", dtype=dtype)


def rotate_ov_proj(layer, head_num, head_dim, R2=None):
  v_proj = layer.self_attn.v_proj
  o_proj = layer.self_attn.o_proj

  apply_exact_had_to_linear(v_proj, had_dim=head_dim, output=True, R2=R2)
  apply_exact_had_to_linear(o_proj, had_dim=head_dim, output=False, R2=R2)


@torch.inference_mode()
def rotate_model(model, args):
  R1 = get_orthogonal_matrix(model.config.hidden_size, args.rotate_mode)
  if args.optimized_rotation_path is not None:
    R_cpk = args.optimized_rotation_path
    R1 = torch.load(R_cpk)["R1"].cuda().to(torch.float64)
  config = model.config
  num_heads = config.num_attention_heads
  model_dim = config.hidden_size
  head_dim = model_dim // num_heads

  rotate_embeddings(model, R1)
  rotate_head(model, R1)
  utils.cleanup_memory()
  layers = [layer for layer in model.model.layers]
  for idx, layer in enumerate(tqdm.tqdm(layers, unit="layer", desc="Rotating")):
    if args.optimized_rotation_path is not None:
      key = f"model.layers.{idx}.self_attn.R2"
      R2 = torch.load(R_cpk)[key].cuda().to(torch.float64)
    else:
      R2 = get_orthogonal_matrix(head_dim, args.rotate_mode)
    rotate_attention_inputs(layers[idx], R1)
    rotate_attention_output(layers[idx], R1)
    rotate_mlp_input(layers[idx], R1)
    rotate_mlp_output(layers[idx], R1, args.online_had_mode)
    rotate_ov_proj(layers[idx], num_heads, head_dim, R2=R2)


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
    with measure("qroate"):
      q = (HadamardTransform.apply(q.float()) / math.sqrt(q.shape[-1])).to(dtype)
    with measure("krotate"):
      k = (HadamardTransform.apply(k.float()) / math.sqrt(k.shape[-1])).to(dtype)

    if self.q_bits < 16:
      with measure("q_quantize"):
        q = self._quantize_states(q, self.q_quantizer, self.q_groupsize)
      self.q_quantizer.free()
    if self.k_bits < 16:
      with measure("k_quantize"):
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

  attr_name = f"{function_name}_qk_rotation_wrapper"
  assert not hasattr(module, attr_name)
  wrapper = monkeypatch.add_wrapper_after_function_call_in_method(
      module,
      "forward",
      function_name,
      functools.partial(QKRotationWrapper, *args, **kwargs),
  )
  setattr(module, attr_name, wrapper)
