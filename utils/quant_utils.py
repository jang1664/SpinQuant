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
import transformers

from fpint_emul import FpIntConfig, fpint_linear
from train_utils.quant_linear import QuantizeLinear
from utils import hadamard_utils
from utils.utils import HadamardTransform
from utils.profile import measure, profile
import time
from global_params import ZP_INT8, SIGNED_KV, ZP_CLAMP, SCALE_NO_UPCAST
print(f"Importing quant utils. ZP_INT8: {ZP_INT8}, SIGNED_KV: {SIGNED_KV}, ZP_CLAMP: {ZP_CLAMP}, SCALE_NO_UPCAST: {SCALE_NO_UPCAST}")

def get_minq_maxq(bits, sym):
  if sym:
    maxq = torch.tensor(2 ** (bits - 1) - 1)
    minq = -maxq - 1
  else:
    maxq = torch.tensor(2**bits - 1)
    minq = 0

  return minq, maxq


def asym_quant(x, scale, zero, maxq):
  scale = scale.to(x.device)
  zero = zero.to(x.device)
  q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
  return q, scale, zero


def asym_dequant(q, scale, zero):
  return scale * (q - zero)


def asym_quant_dequant(x, scale, zero, maxq):
  return asym_dequant(*asym_quant(x, scale, zero, maxq))


def sym_quant(x, scale, maxq):
  scale = scale.to(x.device)
  q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
  return q, scale


def sym_dequant(q, scale):
  return scale * q


def sym_quant_dequant(x, scale, maxq):
  return sym_dequant(*sym_quant(x, scale, maxq))


class STEQuantize(torch.autograd.Function):
  @staticmethod
  def forward(ctx, x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq + 1), maxq)
    return scale * q

  @staticmethod
  def backward(ctx, grad_output):
    # Straight-through estimator: just pass the gradient through
    return grad_output, None, None


class AsymSTEQuantize(torch.autograd.Function):
  @staticmethod
  @profile("asmy_ste_quantize")
  def forward(ctx, x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    if not SIGNED_KV:
      q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    else:
      q = torch.clamp(torch.round(x / scale) + zero, -(maxq+1)//2, (maxq-1)//2)
    return scale * (q - zero)

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output, None, None, None


class ActQuantizer(torch.nn.Module):
  """
  A class for quantizing the activations. We only support (both sym. and asym.) per-token quantization
  for the activations.
  """

  def __init__(self) -> None:
    super(ActQuantizer, self).__init__()
    self.register_buffer("maxq", torch.tensor(0))
    self.register_buffer("scale", torch.zeros(1))
    self.register_buffer("zero", torch.zeros(1))
    self.bits = 16

  def free(self) -> None:
    self.zero = None
    self.scale = None

  def extra_repr(self) -> str:
    if self.bits == 16:
      return "bits=16 (bypass)"
    scheme = "symmetric" if getattr(self, "sym", False) else "asymmetric"
    groupsize = getattr(self, "groupsize", -1)
    return f"bits={self.bits}, {scheme}, groupsize={groupsize}"

  def forward(self, x):
    x_dtype = x.dtype
    if self.bits == 16:
      return x
    elif self.sym:
      return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
    return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(x_dtype)

  # Different from `forward`, this method returns quantized integers, scales (and zeros if asymmetric).
  def quantize(self, x):
    if self.sym:
      return sym_quant(x, self.scale, self.maxq)
    else:
      return asym_quant(x, self.scale, self.zero, self.maxq)

  def configure(
      self, bits: int, groupsize: int = -1, sym: bool = False, clip_ratio: float = 1.0,
  ) -> None:
    _, self.maxq = get_minq_maxq(bits, sym)
    self.bits = bits
    self.groupsize = groupsize
    self.sym = sym
    self.clip_ratio = clip_ratio
    assert (
        self.clip_ratio <= 1 and self.clip_ratio > 0
    ), "Clip ratio should be in (0, 1]"

  @profile("find_params_per_token_groupwise")
  def find_params_per_token_groupwise(self, x) -> None:
    init_shape = x.shape
    reshaped_x = x.reshape(
        -1, x.shape[-2], x.shape[-1] // self.groupsize, self.groupsize
    )

    xmax = torch.amax(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
    xmin = torch.amin(reshaped_x, dim=3, keepdim=True) * self.clip_ratio
    if self.sym:
      xmax = torch.maximum(torch.abs(xmin), xmax)
      tmp = xmax == 0
      self.scale = xmax / self.maxq
      self.scale[tmp] = 1
      self.zero = torch.zeros_like(self.scale)
    else:
      if ZP_CLAMP:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
      self.scale = (xmax - xmin) / self.maxq
      if SCALE_NO_UPCAST:
        self.scale = self.scale.to(x.dtype)
      self.zero = torch.round(-xmin / self.scale)
      if ZP_INT8:
        self.zero = self.zero.to(torch.int8)
      if SIGNED_KV:
        self.zero = self.zero - 8

    self.scale = self.scale.repeat(1, 1, 1, self.groupsize).reshape(init_shape)
    self.zero = self.zero.repeat(1, 1, 1, self.groupsize).reshape(init_shape)

  @profile("find_params")
  def find_params(self, x) -> None:
    if self.bits == 16:
      return

    dev = x.device
    self.maxq = self.maxq.to(dev)

    init_shape = x.shape

    if self.groupsize > 0:
      # group-wise per-token quantization
      self.find_params_per_token_groupwise(x)
      # utils.cleanup_memory(verbos=False)
      return

    reshaped_x = x.reshape((-1, x.shape[-1]))

    if ZP_CLAMP:
      tmp = torch.zeros(reshaped_x.shape[0], device=dev)
      xmin = torch.minimum(reshaped_x.min(1)[0], tmp) * self.clip_ratio
      xmax = torch.maximum(reshaped_x.max(1)[0], tmp) * self.clip_ratio
    else:
      xmax = torch.amax(reshaped_x, dim=1) * self.clip_ratio
      xmin = torch.amin(reshaped_x, dim=1) * self.clip_ratio
    if self.sym:
      xmax = torch.maximum(torch.abs(xmin), xmax)
      tmp = xmax == 0
      self.scale = (xmax / self.maxq).unsqueeze(1).repeat(1, reshaped_x.shape[-1])
      self.scale[tmp] = 1
      self.scale = self.scale.reshape(init_shape)
      self.zero = torch.zeros_like(self.scale)
    else:
      if ZP_CLAMP:
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1
      self.scale = (xmax - xmin) / self.maxq
      if SCALE_NO_UPCAST:
        self.scale = self.scale.to(x.dtype)
      self.zero = torch.round(-xmin / self.scale)
      if ZP_INT8:
        self.zero = self.zero.to(torch.int8)
      if SIGNED_KV:
        self.zero = self.zero - 8

      self.scale = (
          self.scale.unsqueeze(1)
          .repeat(1, reshaped_x.shape[-1])
          .reshape(init_shape)
      )
      self.zero = (
          self.zero.unsqueeze(1)
          .repeat(1, reshaped_x.shape[-1])
          .reshape(init_shape)
      )


class ActQuantWrapper(torch.nn.Module):
  """
  This class is a wrapper for the activation quantization.
  We extract the FP features in the forward pass and quantize the rest using
  the self.quantizer object.
  If a rotation Q is provided, the weight matrix will be rotated,
  a pre-forward hook will be registered to rotate the activation before quantization.
  """

  def __init__(self, module: torch.nn.Linear) -> None:
    super(ActQuantWrapper, self).__init__()
    # assert isinstance(module, torch.nn.Linear)
    self.module = module
    self.weight = module.weight
    self.bias = module.bias
    self.quantizer = ActQuantizer()
    self.out_quantizer = ActQuantizer()
    self.register_buffer("had_K", torch.tensor(0))
    self._buffers["had_K"] = None
    self.K = 1
    self.online_full_had = False
    self.online_partial_had = False
    self.online_had_mode = hadamard_utils.ONLINE_HAD_MODE_FACTORIZED
    self.had_dim = 0
    self.fp32_had = False
    self.linear_backend = "standard"
    self.fpint_fallback_reason = "FPINT backend is disabled"
    self._fpint_config = None
    self._fpint_has_zero = False
    self.register_buffer("fpint_weight", torch.empty(0, dtype=torch.int8))
    self.register_buffer("fpint_scale", torch.empty(0, dtype=torch.float16))
    self.register_buffer("fpint_zero", torch.empty(0, dtype=torch.int32))
    self.register_buffer("fpint_group_index", torch.empty(0, dtype=torch.int32))
    # [format_version, bits, group_size, mxu_rows, extra_bits, reduce_extra_bits]
    self.register_buffer("fpint_meta", torch.empty(0, dtype=torch.int32))

  def _load_from_state_dict(
      self,
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
  ):
    # Quantized buffer shapes are model-dependent. Resize before the normal
    # loader copies them, and inject empty defaults for legacy checkpoints so
    # standard inference remains backward compatible.
    for name in (
        "fpint_weight",
        "fpint_scale",
        "fpint_zero",
        "fpint_group_index",
        "fpint_meta",
    ):
      key = prefix + name
      if key in state_dict:
        value = state_dict[key]
        self._buffers[name] = torch.empty(
            value.shape, dtype=value.dtype, device=self._buffers[name].device
        )
      else:
        state_dict[key] = self._buffers[name]
    super()._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )

  @property
  def has_fpint_metadata(self) -> bool:
    return self.fpint_meta.numel() == 6 and self.fpint_weight.numel() > 0

  def set_fpint_metadata(
      self,
      *,
      weight,
      scale,
      zero,
      group_index,
      bits,
      group_size,
      mxu_rows=32,
      extra_bits=19,
      reduce_extra_bits=10,
  ) -> None:
    device = self.module.weight.device
    self.fpint_weight = weight.detach().to(device=device, dtype=torch.int8)
    self.fpint_scale = scale.detach().to(device=device, dtype=torch.float16)
    self.fpint_zero = zero.detach().to(device=device, dtype=torch.int32)
    self.fpint_group_index = group_index.detach().to(
        device=device, dtype=torch.int32
    )
    self.fpint_meta = torch.tensor(
        [1, bits, group_size, mxu_rows, extra_bits, reduce_extra_bits],
        dtype=torch.int32,
        device=device,
    )

  def configure_linear_backend(
      self,
      backend="standard",
      *,
      mxu_rows=32,
      extra_bits=19,
      reduce_extra_bits=10,
  ) -> None:
    if backend not in ("standard", "fpint_torch", "fpint_cuda"):
      raise ValueError(f"Unknown Linear backend: {backend!r}")
    if backend == "standard":
      self.linear_backend = backend
      self.fpint_fallback_reason = "FPINT backend is disabled"
      self._fpint_config = None
      return
    if not self.has_fpint_metadata:
      raise ValueError(
          "FPINT backend requested for a quantized Linear without integer "
          "weight metadata; regenerate the quantized checkpoint"
      )
    version, bits, group_size, _, _, _ = self.fpint_meta.tolist()
    if version != 1:
      raise ValueError(f"Unsupported FPINT metadata version: {version}")
    config = FpIntConfig(
        weight_bits=bits,
        group_size=group_size,
        mxu_rows=mxu_rows,
        extra_bits=extra_bits,
        reduce_extra_bits=reduce_extra_bits,
        activation_format=(
            "bf16" if self.module.weight.dtype == torch.bfloat16 else "fp16"
        ),
    )
    k = self.fpint_weight.shape[1]
    expected_group_index = torch.arange(
        k, device=self.fpint_group_index.device, dtype=torch.int32
    )
    if group_size != -1:
      expected_group_index = expected_group_index // group_size
    else:
      expected_group_index.zero_()
    if not torch.equal(self.fpint_group_index, expected_group_index):
      raise ValueError(
          "FPINT QCOL does not support grouped GPTQ act-order because its "
          "scale groups are non-contiguous in the original K order"
      )
    if tuple(self.fpint_weight.shape) != tuple(self.module.weight.shape):
      raise ValueError("FPINT integer weight shape does not match Linear weight")
    if tuple(self.fpint_scale.shape) != (
        self.fpint_weight.shape[0],
        config.group_count(k),
    ):
      raise ValueError("FPINT compact scale shape is inconsistent with metadata")
    # Full value validation belongs here, once per installed weight, rather
    # than in every inference call.
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    if bool(((self.fpint_weight < qmin) | (self.fpint_weight > qmax)).any()):
      raise ValueError(f"FPINT weight is outside the INT{bits} signed range")
    if not bool(torch.isfinite(self.fpint_scale).all()) or bool(
        (self.fpint_scale <= 0).any()
    ):
      raise ValueError("FPINT scale must contain finite positive values")
    maximum_zero = (
        int(self.fpint_zero.to(torch.int64).abs().max())
        if self.fpint_zero.numel()
        else 0
    )
    config.validate_zero_bound(maximum_zero)
    self.fpint_meta[3:] = torch.tensor(
        [mxu_rows, extra_bits, reduce_extra_bits],
        device=self.fpint_meta.device,
        dtype=torch.int32,
    )
    self.linear_backend = backend
    self.fpint_fallback_reason = ""
    self._fpint_config = config
    self._fpint_has_zero = bool(self.fpint_zero.any())

  def _fpint_forward(self, x):
    if self._fpint_config is None:
      raise RuntimeError("FPINT Linear backend was not configured")
    return fpint_linear(
        x,
        self.fpint_weight,
        self.fpint_scale,
        self.fpint_zero,
        self._fpint_config,
        bias=self.bias,
        backend=self.linear_backend,
        validate_values=False,
        has_zero=self._fpint_has_zero,
    )

  def extra_repr(self) -> str:
    str_ = f"Input Quantizer Bits: {self.quantizer.bits}"
    if self.quantizer.bits < 16:
      str_ += (
          f" (Asymmetric Per-Token)"
          if not self.quantizer.sym
          else f" (Symmetric Per-Token)"
      )

    str_ += f"\nOutput Quantizer Bits: {self.out_quantizer.bits}"
    if self.out_quantizer.bits < 16:
      str_ += (
          f" (Asymmetric Per-Token)"
          if not self.out_quantizer.sym
          else f" (Symmetric Per-Token)"
      )

    return str_

  def forward(self, x, R1=None, R2=None, transpose=False):
    x_dtype = x.dtype

    # Rotate, if needed
    if self.online_full_had:
      if self.fp32_had:  # Full Hadamard in FP32
        x = hadamard_utils.matmul_hadU_cuda(
            x.float(),
            self.had_K,
            self.K,
            mode=self.online_had_mode,
        ).to(x_dtype)
      else:  # Full Hadamard in FP16
        x = hadamard_utils.matmul_hadU_cuda(
            x,
            self.had_K,
            self.K,
            mode=self.online_had_mode,
        )

    elif self.online_partial_had:
      # todo: implement this in QAttention to avoid reshaping!

      if self.fp32_had:
        x = x.float()

      init_shape = x.shape
      if self.K == 1:
        x = (
            HadamardTransform.apply(
                x.reshape(
                    -1, init_shape[-1] // self.had_dim, self.had_dim
                ).transpose(1, 2)
            )
            / math.sqrt(init_shape[-1] // self.had_dim)
        ).transpose(1, 2)
      else:
        x = (
            self.had_K.to(x.dtype)
            @ x.reshape(-1, init_shape[-1] // self.had_dim, self.had_dim)
        ) / math.sqrt(init_shape[-1] // self.had_dim)

      if self.fp32_had:
        x = x.to(x_dtype)
      x = x.reshape(init_shape)

    if self.quantizer.bits < 16:  # Quantize, if needed
      self.quantizer.find_params(x)
      x = self.quantizer(x).to(x_dtype)
      self.quantizer.free()
    if R1 is not None:
      if self.linear_backend != "standard":
        raise ValueError(
            "FPINT inference backend cannot execute the training-time R1/R2 "
            "Linear interface"
        )
      x = self.module(x, R1, R2, transpose).to(x_dtype)
    else:
      with measure("linear"):
        if self.linear_backend == "standard":
          x = self.module(x).to(x_dtype)
        else:
          x = self._fpint_forward(x).to(x_dtype)

    if self.out_quantizer.bits < 16:  # Quantize the output, if needed
      self.out_quantizer.find_params(x)
      x = self.out_quantizer(x).to(x_dtype)
      self.out_quantizer.free()

    return x


class WeightQuantizer(torch.nn.Module):
  """From GPTQ Repo"""

  def __init__(self, shape: int = 1) -> None:
    super(WeightQuantizer, self).__init__()
    self.register_buffer("maxq", torch.tensor(0))
    self.register_buffer("scale", torch.zeros(shape))
    self.register_buffer("zero", torch.zeros(shape))

  def configure(
      self,
      bits,
      perchannel: bool = False,
      sym: bool = True,
      mse: bool = False,
      norm: float = 2.4,
      grid: int = 100,
      maxshrink: float = 0.8,
      weight_groupsize: int = -1,
  ) -> None:
    self.bits = bits
    self.perchannel = perchannel
    self.sym = sym
    self.mse = mse
    self.norm = norm
    self.grid = grid
    self.maxshrink = maxshrink
    self.weight_groupsize = weight_groupsize
    if sym:
      self.maxq = torch.tensor(2 ** (bits - 1) - 1)
    else:
      self.maxq = torch.tensor(2**bits - 1)

  def find_params_weight_groupwise(self, x) -> None:
    scales = []
    zeros = []
    for start in range(0, x.shape[-1], self.weight_groupsize):
      group = x[:, start : start + self.weight_groupsize]
      xmin = torch.amin(group, dim=-1, keepdim=True)
      xmax = torch.amax(group, dim=-1, keepdim=True)
      if self.sym:
        xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
        best_scale = xmax / self.maxq
        best_zero = torch.zeros_like(best_scale)
      else:
        all_zero = (xmin == 0) & (xmax == 0)
        xmin = torch.where(all_zero, -torch.ones_like(xmin), xmin)
        xmax = torch.where(all_zero, torch.ones_like(xmax), xmax)
        best_scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
        best_zero = torch.round(-xmin / best_scale)
      if self.mse:
        best_error = torch.full(
            (group.shape[0],), float("inf"), device=x.device, dtype=torch.float32
        )
        for i in range(int(self.maxshrink * self.grid)):
          p = 1 - i / self.grid
          xmin1, xmax1 = p * xmin, p * xmax
          if self.sym:
            scale1 = xmax1 / self.maxq
            zero1 = torch.zeros_like(scale1)
            candidate = sym_quant_dequant(group, scale1, self.maxq)
          else:
            scale1 = (xmax1 - xmin1).clamp(min=1e-5) / self.maxq
            zero1 = torch.round(-xmin1 / scale1)
            candidate = asym_quant_dequant(group, scale1, zero1, self.maxq)
          error = (
              (candidate - group)
              .abs()
              .float()
              .pow(self.norm)
              .sum(dim=-1)
          )
          better = error < best_error
          best_error[better] = error[better]
          best_scale[better] = scale1[better]
          best_zero[better] = zero1[better]
      scales.append(best_scale.expand(-1, group.shape[-1]))
      zeros.append(best_zero.expand(-1, group.shape[-1]))
    self.scale = torch.cat(scales, dim=-1)
    self.zero = torch.cat(zeros, dim=-1)

  def find_params(self, x) -> None:
    if self.bits == 16:
      return
    dev = x.device
    self.maxq = self.maxq.to(dev)

    shape = x.shape

    if self.weight_groupsize > 0:
      # group-wise per-token quantization
      self.find_params_weight_groupwise(x)
      # utils.cleanup_memory(verbos=False)
      return
    elif self.perchannel:
      x = x.flatten(1)
    else:
      x = x.flatten().unsqueeze(0)

    tmp = torch.zeros(x.shape[0], device=dev)
    xmin = torch.minimum(x.min(1)[0], tmp)
    xmax = torch.maximum(x.max(1)[0], tmp)

    if self.sym:
      xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5)
      self.scale = xmax / self.maxq
      self.zero = torch.zeros_like(self.scale)
    else:
      tmp = (xmin == 0) & (xmax == 0)
      xmin[tmp] = -1
      xmax[tmp] = +1
      self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
      self.zero = torch.round(-xmin / self.scale)

    if self.mse:
      best = torch.full([x.shape[0]], float("inf"), device=dev)
      for i in range(int(self.maxshrink * self.grid)):
        p = 1 - i / self.grid
        xmin1 = p * xmin
        xmax1 = p * xmax

        if self.sym:
          scale1 = xmax1 / self.maxq
          zero1 = torch.zeros_like(scale1)
          q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
        else:
          scale1 = (xmax1 - xmin1) / self.maxq
          zero1 = torch.round(-xmin1 / scale1)
          q = asym_quant_dequant(
              x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq
          )

        q -= x
        q.abs_()
        q.pow_(self.norm)
        err = torch.sum(q, 1)
        tmp = err < best
        if torch.any(tmp):
          best[tmp] = err[tmp]
          self.scale[tmp] = scale1[tmp]
          self.zero[tmp] = zero1[tmp]
    if not self.perchannel:
      tmp = shape[0]
      self.scale = self.scale.repeat(tmp)
      self.zero = self.zero.repeat(tmp)

    shape = [-1] + [1] * (len(shape) - 1)
    self.scale = self.scale.reshape(shape)
    self.zero = self.zero.reshape(shape)
    return

  # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
  def quantize(self, x):
    x_dtype = x.dtype
    if self.ready() and self.bits < 16:
      if self.sym:
        return STEQuantize.apply(x, self.scale, self.maxq).to(x_dtype)
      return AsymSTEQuantize.apply(x, self.scale, self.zero, self.maxq).to(
          x_dtype
      )
    return x

  # Return int value and scale in addtional to fake quantized weight
  def fake_quantize(self, x):
    quantized, integer, scale, _ = self.fake_quantize_with_metadata(x)
    return quantized, integer, scale

  def fake_quantize_with_metadata(self, x):
    x_dtype = x.dtype
    if self.ready() and self.bits < 16:
      scale = self.scale.to(x.device)
      if self.sym:
        q = torch.clamp(torch.round(x / scale), -(self.maxq + 1), self.maxq)
        zero = torch.zeros_like(q, dtype=torch.int32)
      else:
        unsigned = torch.clamp(
            torch.round(x / scale) + self.zero.to(x.device), 0, self.maxq
        )
        signed_offset = 1 << (self.bits - 1)
        q = unsigned - signed_offset
        zero = torch.round(self.zero.to(x.device)).to(torch.int32) - signed_offset
      q = q.to(torch.int8)
      dequantized = scale * (q.to(scale.dtype) - zero.to(scale.dtype))
      return dequantized.to(x_dtype), q, scale, zero
    return None, None, None, None

  def enabled(self):
    return self.maxq > 0

  def ready(self):
    return torch.all(self.scale != 0)


def stash_fpint_metadata(
    module,
    *,
    integer_weight,
    expanded_scale,
    expanded_zero,
    bits,
    group_size,
    group_index=None,
) -> None:
  """Store compact CPU metadata until the owning wrapper is available."""
  if integer_weight.ndim != 2:
    raise ValueError("FPINT integer weight must have shape [N, K]")
  n, k = integer_weight.shape
  if expanded_scale.ndim != 2 or expanded_scale.shape[0] != n:
    raise ValueError("FPINT scale must have shape [N, K] or [N, 1]")
  if expanded_zero.ndim != 2 or expanded_zero.shape[0] != n:
    raise ValueError("FPINT zero must have shape [N, K] or [N, 1]")
  if expanded_scale.shape[1] == 1:
    expanded_scale = expanded_scale.expand(n, k)
  if expanded_zero.shape[1] == 1:
    expanded_zero = expanded_zero.expand(n, k)
  if expanded_scale.shape[1] != k or expanded_zero.shape[1] != k:
    raise ValueError("FPINT expanded scale/zero K does not match integer weight")
  if group_index is None:
    if group_size == -1:
      group_index = torch.zeros(k, device=integer_weight.device, dtype=torch.long)
    else:
      group_index = torch.arange(k, device=integer_weight.device) // group_size
  group_index = group_index.to(device=expanded_scale.device, dtype=torch.long)
  group_count = int(group_index.max().item()) + 1 if k else 0
  compact_scale = torch.empty(
      (n, group_count), device=expanded_scale.device, dtype=expanded_scale.dtype
  )
  compact_zero = torch.empty(
      (n, group_count), device=expanded_zero.device, dtype=torch.int32
  )
  for group in range(group_count):
    positions = torch.nonzero(group_index == group, as_tuple=False).flatten()
    if positions.numel() == 0:
      raise ValueError(f"FPINT quantization group {group} is empty")
    first = int(positions[0].item())
    group_scales = expanded_scale[:, positions]
    group_zeros = expanded_zero[:, positions]
    expected_scale = expanded_scale[:, first : first + 1]
    expected_zero = expanded_zero[:, first : first + 1]
    if not torch.equal(group_scales, expected_scale.expand_as(group_scales)):
      raise ValueError("scale is not constant inside its FPINT quantization group")
    if not torch.equal(
        group_zeros.to(torch.int32), expected_zero.to(torch.int32).expand_as(group_zeros)
    ):
      raise ValueError("zero-point is not constant inside its FPINT quantization group")
    compact_scale[:, group] = expanded_scale[:, first]
    compact_zero[:, group] = expanded_zero[:, first].to(torch.int32)
  module._fpint_pending_metadata = {
      "weight": integer_weight.detach().to(device="cpu", dtype=torch.int8),
      "scale": compact_scale.detach().to(device="cpu", dtype=torch.float16),
      "zero": compact_zero.detach().to(device="cpu", dtype=torch.int32),
      "group_index": group_index.detach().to(device="cpu", dtype=torch.int32),
      "bits": int(bits),
      "group_size": int(group_size),
  }


def configure_fpint_linears(model, args):
  """Install pending metadata, select a backend and return layer coverage."""
  backend = getattr(args, "linear_backend", "standard")
  mxu_rows = getattr(args, "fpint_mxu_rows", 32)
  extra_bits = getattr(args, "fpint_extra_bits", 19)
  reduce_extra_bits = getattr(args, "fpint_reduce_extra_bits", 10)
  wrappers = find_qlayers(model, layers=[ActQuantWrapper])
  coverage = []
  for name, wrapper in wrappers.items():
    pending = getattr(wrapper.module, "_fpint_pending_metadata", None)
    if pending is not None:
      wrapper.set_fpint_metadata(
          **pending,
          mxu_rows=mxu_rows,
          extra_bits=extra_bits,
          reduce_extra_bits=reduce_extra_bits,
      )
      del wrapper.module._fpint_pending_metadata

    if backend == "standard":
      wrapper.configure_linear_backend("standard")
      selected = "standard"
      reason = "FPINT backend is disabled"
    elif wrapper.has_fpint_metadata:
      wrapper.configure_linear_backend(
          backend,
          mxu_rows=mxu_rows,
          extra_bits=extra_bits,
          reduce_extra_bits=reduce_extra_bits,
      )
      selected = backend
      reason = ""
    elif "lm_head" in name:
      wrapper.configure_linear_backend("standard")
      selected = "standard"
      reason = "lm_head remains in floating point"
      wrapper.fpint_fallback_reason = reason
    else:
      raise ValueError(
          f"FPINT backend requested but {name!r} has no integer weight metadata. "
          "Legacy dequantized checkpoints must be regenerated or re-quantized."
      )
    coverage.append(
        {
            "name": name,
            "backend": selected,
            "reason": reason,
            "bits": int(wrapper.fpint_meta[1]) if wrapper.has_fpint_metadata else None,
        }
    )
  if backend != "standard":
    fpint_count = sum(item["backend"] == backend for item in coverage)
    standard_count = len(coverage) - fpint_count
    print(
        f"Linear backend coverage: {fpint_count} {backend}, "
        f"{standard_count} standard"
    )
  model.fpint_linear_coverage = coverage
  return coverage


def add_actquant(
    module: ActQuantWrapper,
    name: str = "",
    layers=[
        torch.nn.Linear,
        QuantizeLinear,
        ActQuantWrapper,
        transformers.models.falcon.modeling_falcon.FalconLinear,
    ],
) -> None:
  if isinstance(module, ActQuantWrapper):
    return
  for attr in dir(module):
    tmp = getattr(module, attr)
    if type(tmp) in layers:
      setattr(module, attr, ActQuantWrapper(tmp))
    if type(tmp) is torch.nn.Sequential:
      replaced = []
      for i, child in enumerate(tmp.children()):
        if type(child) in layers:
          replaced.append(ActQuantWrapper(child))
        else:
          replaced.append(child)
      setattr(module, attr, torch.nn.Sequential(*replaced))
    if type(tmp) is torch.nn.ModuleList:
      replaced = []
      for i, child in enumerate(tmp.children()):
        if type(child) in layers:
          replaced.append(ActQuantWrapper(child))
        else:
          replaced.append(child)
      setattr(module, attr, torch.nn.ModuleList(replaced))
  for name1, child in module.named_children():
    add_actquant(child, name + "." + name1 if name != "" else name1, layers)


def find_qlayers(
    module,
    layers=[torch.nn.Linear, ActQuantWrapper, QuantizeLinear],
    name: str = "",
):
  # fix for llama embedding layer
  if type(module) in [torch.nn.Embedding] and type(module) in layers:
    return {"embed_tokens": module}
  if type(module) in layers:
    return {name: module}
  res = {}
  for name1, child in module.named_children():
    res.update(
        find_qlayers(
            child, layers=layers, name=name + "." + name1 if name != "" else name1
        )
    )
  return res
