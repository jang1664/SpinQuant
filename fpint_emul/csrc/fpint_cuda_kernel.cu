#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBlockN = 32;
constexpr int kBlockM = 4;
constexpr int kReductionLanes = 4;
constexpr int kThreadsPerBlock = kBlockN * kBlockM * kReductionLanes;
template <typename scalar_t>
__device__ __forceinline__ unsigned scalar_bits(const scalar_t value);

template <>
__device__ __forceinline__ unsigned scalar_bits<__half>(const __half value) {
  return __half_as_ushort(value);
}

template <>
__device__ __forceinline__ unsigned scalar_bits<__nv_bfloat16>(
    const __nv_bfloat16 value) {
  return __bfloat16_as_ushort(value);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t scalar_zero();

template <>
__device__ __forceinline__ __half scalar_zero<__half>() {
  return __float2half(0.0f);
}

template <>
__device__ __forceinline__ __nv_bfloat16 scalar_zero<__nv_bfloat16>() {
  return __float2bfloat16_rn(0.0f);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t scalar_from_float(const float value);

template <>
__device__ __forceinline__ __half scalar_from_float<__half>(const float value) {
  return __float2half_rn(value);
}

template <>
__device__ __forceinline__ __nv_bfloat16 scalar_from_float<__nv_bfloat16>(
    const float value) {
  return __float2bfloat16_rn(value);
}

template <typename scalar_t, int kMantissaBits, int kExponentBits>
__device__ __forceinline__ int scalar_exponent(const scalar_t value) {
  return (scalar_bits(value) >> kMantissaBits) & ((1 << kExponentBits) - 1);
}

template <typename scalar_t, int kMantissaBits, int kExponentBits>
__device__ __forceinline__ int64_t align_scalar(
    const scalar_t value, const int maximum_exponent, const int extra_bits) {
  const unsigned bits = scalar_bits(value);
  const bool negative = (bits >> 15) != 0;
  const int exponent =
      (bits >> kMantissaBits) & ((1 << kExponentBits) - 1);
  const int exponent_for_align = exponent == 0 ? 1 : exponent;
  const int64_t hidden =
      (static_cast<int64_t>(exponent != 0) << kMantissaBits) |
      static_cast<int64_t>(bits & ((1 << kMantissaBits) - 1));
  const int shift = maximum_exponent - exponent_for_align;
  const int64_t shifted = hidden << extra_bits;
  const int64_t aligned = shift >= 63 ? 0 : shifted >> shift;
  return negative ? -aligned : aligned;
}

__device__ __forceinline__ float restore_scaled_integer(
    const int64_t value, const int binary_exponent) {
  if (value == 0) {
    return 0.0f;
  }
  const uint64_t magnitude = value < 0
      ? static_cast<uint64_t>(-(value + 1)) + 1
      : static_cast<uint64_t>(value);
  const int value_exponent = 63 - __clzll(magnitude) + binary_exponent;
  if (value_exponent >= -126) {
    // Scaling a rounded integer by a power of two is exact while the result is
    // normal.  This avoids slow FP64 arithmetic on the common model path.
    return ldexpf(__ll2float_rn(value), binary_exponent);
  }
  // Construct a subnormal in units of 2^-149 and round the complete integer
  // expression once.  This avoids both FP32 double rounding and slow FP64.
  const int subnormal_shift = binary_exponent + 149;
  uint64_t fraction;
  if (subnormal_shift >= 0) {
    fraction = magnitude << subnormal_shift;
  } else {
    const int right_shift = -subnormal_shift;
    if (right_shift > 64) {
      fraction = 0;
    } else if (right_shift == 64) {
      fraction = magnitude > (uint64_t{1} << 63) ? 1 : 0;
    } else {
      fraction = magnitude >> right_shift;
      const uint64_t remainder =
          magnitude & ((uint64_t{1} << right_shift) - 1);
      const uint64_t halfway = uint64_t{1} << (right_shift - 1);
      fraction += remainder > halfway ||
          (remainder == halfway && (fraction & 1));
    }
  }
  const uint32_t sign = value < 0 ? 0x80000000u : 0u;
  return __uint_as_float(sign | static_cast<uint32_t>(fraction));
}

template <typename scalar_t, int kMantissaBits, int kExponentBits,
          int kExponentBias>
__global__ void fpint_qcol_kernel(
    const scalar_t* __restrict__ activation,
    const int8_t* __restrict__ weight,
    const void* __restrict__ scale,
    const int32_t* __restrict__ zero,
    scalar_t* __restrict__ output,
    const int rows,
    const int n_columns,
    const int k_columns,
    const int group_size,
    const int group_count,
    const int mxu_rows,
    const int extra_bits,
    const int reduce_extra_bits,
    const bool has_zero,
    const bool scale_is_bf16) {
  extern __shared__ unsigned char shared_bytes[];
  int64_t* shared_main = reinterpret_cast<int64_t*>(shared_bytes);
  int64_t* shared_reduce = shared_main + kBlockM * mxu_rows;
  int8_t* shared_weight = reinterpret_cast<int8_t*>(
      shared_reduce + (has_zero ? kBlockM * mxu_rows : 0));
  __shared__ int maximum_exponents[kBlockM];

  const int thread = threadIdx.x;
  const int output_index = thread / kReductionLanes;
  const int reduction_lane = thread % kReductionLanes;
  const int lane = output_index % kBlockN;
  const int local_row = output_index / kBlockN;
  const int row = blockIdx.y * kBlockM + local_row;
  const int column = blockIdx.x * kBlockN + lane;
  float accumulator = 0.0f;

  const int tile_count = (k_columns + mxu_rows - 1) / mxu_rows;
  for (int tile = 0; tile < tile_count; ++tile) {
    const int tile_start = tile * mxu_rows;
    if (thread < kBlockM * kBlockN) {
      const int exponent_row = thread / kBlockN;
      const int exponent_lane = thread % kBlockN;
      const int global_row = blockIdx.y * kBlockM + exponent_row;
      int local_maximum = 1;
      if (global_row < rows) {
        for (int offset = exponent_lane; offset < mxu_rows;
             offset += kBlockN) {
          const int k = tile_start + offset;
          if (k < k_columns) {
            int exponent =
                scalar_exponent<scalar_t, kMantissaBits, kExponentBits>(
                    activation[global_row * k_columns + k]);
            exponent = exponent == 0 ? 1 : exponent;
            local_maximum = max(local_maximum, exponent);
          }
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_maximum = max(
            local_maximum,
            __shfl_down_sync(0xffffffff, local_maximum, offset));
      }
      if (exponent_lane == 0) {
        maximum_exponents[exponent_row] = local_maximum;
      }
    }
    __syncthreads();

    for (int index = thread; index < kBlockM * mxu_rows;
         index += kThreadsPerBlock) {
      const int activation_row = index / mxu_rows;
      const int offset = index - activation_row * mxu_rows;
      const int global_row = blockIdx.y * kBlockM + activation_row;
      const int k = tile_start + offset;
      const scalar_t value =
          (global_row < rows && k < k_columns)
          ? activation[global_row * k_columns + k]
          : scalar_zero<scalar_t>();
      shared_main[index] =
          align_scalar<scalar_t, kMantissaBits, kExponentBits>(
              value, maximum_exponents[activation_row], extra_bits);
      if (has_zero) {
        shared_reduce[index] =
            align_scalar<scalar_t, kMantissaBits, kExponentBits>(
                value,
                maximum_exponents[activation_row],
                reduce_extra_bits);
      }
    }
    for (int index = thread; index < kBlockN * mxu_rows;
         index += kThreadsPerBlock) {
      const int local_column = index / mxu_rows;
      const int offset = index - local_column * mxu_rows;
      const int global_column = blockIdx.x * kBlockN + local_column;
      const int k = tile_start + offset;
      shared_weight[index] =
          (global_column < n_columns && k < k_columns)
          ? weight[global_column * k_columns + k]
          : 0;
    }
    __syncthreads();

    if (row < rows && column < n_columns) {
      int64_t inner = 0;
      int64_t reduction = 0;
      for (int offset = reduction_lane; offset < mxu_rows;
           offset += kReductionLanes) {
        const int k = tile_start + offset;
        if (k < k_columns) {
          inner += shared_main[local_row * mxu_rows + offset] *
              static_cast<int64_t>(
                  shared_weight[lane * mxu_rows + offset]);
          if (has_zero) {
            reduction += shared_reduce[local_row * mxu_rows + offset];
          }
        }
      }
      const unsigned subgroup_mask =
          ((1u << kReductionLanes) - 1u) <<
          (((thread & 31) / kReductionLanes) * kReductionLanes);
      for (int offset = kReductionLanes / 2; offset > 0; offset >>= 1) {
        inner += __shfl_down_sync(
            subgroup_mask, inner, offset, kReductionLanes);
        reduction += __shfl_down_sync(
            subgroup_mask, reduction, offset, kReductionLanes);
      }
      if (reduction_lane == 0) {
        const int group = group_size == -1 ? 0 : tile_start / group_size;
        int64_t post = inner;
        if (has_zero) {
          const int64_t zero_value = zero[column * group_count + group];
          post -= zero_value * reduction *
              (int64_t{1} << (extra_bits - reduce_extra_bits));
        }
        // Preserve the original FP32 scale and K-tile accumulation order.
        const int binary_exponent = maximum_exponents[local_row] -
            kExponentBias - kMantissaBits - extra_bits;
        // Match the RTL int-to-FP conversion as one rounded P * 2^q operation.
        float contribution = restore_scaled_integer(post, binary_exponent);
        const int scale_index = column * group_count + group;
        const float scale_value = scale_is_bf16
            ? __bfloat162float(static_cast<const __nv_bfloat16*>(scale)[scale_index])
            : __half2float(static_cast<const __half*>(scale)[scale_index]);
        contribution = __fmul_rn(contribution, scale_value);
        accumulator = __fadd_rn(accumulator, contribution);
      }
    }
    __syncthreads();
  }

  if (reduction_lane == 0 && row < rows && column < n_columns) {
    output[row * n_columns + column] = scalar_from_float<scalar_t>(accumulator);
  }
}

}  // namespace

torch::Tensor fpint_qcol_cuda(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor zero,
    int64_t group_size,
    int64_t mxu_rows,
    int64_t extra_bits,
    int64_t reduce_extra_bits,
    bool has_zero) {
  TORCH_CHECK(activation.is_cuda(), "activation must be CUDA");
  TORCH_CHECK(weight.is_cuda() && scale.is_cuda() && zero.is_cuda(),
              "all FPINT tensors must be CUDA");
  TORCH_CHECK(
      activation.scalar_type() == at::kHalf ||
          activation.scalar_type() == at::kBFloat16,
      "activation must be FP16 or BF16");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(scale.scalar_type() == at::kHalf || scale.scalar_type() == at::kBFloat16,
              "scale must be FP16 or BF16");
  TORCH_CHECK(zero.scalar_type() == at::kInt, "zero must be int32");
  TORCH_CHECK(activation.is_contiguous() && weight.is_contiguous() &&
                  scale.is_contiguous() && zero.is_contiguous(),
              "FPINT CUDA tensors must be contiguous");
  TORCH_CHECK(activation.dim() == 2 && weight.dim() == 2 &&
                  scale.dim() == 2 && zero.dim() == 2,
              "FPINT CUDA expects rank-2 tensors");
  TORCH_CHECK(mxu_rows > 0 && mxu_rows <= 256,
              "fpint_cuda supports mxu_rows in [1, 256]");
  TORCH_CHECK(extra_bits >= reduce_extra_bits && extra_bits < 52,
              "invalid FPINT extra bit widths");

  const auto rows = activation.size(0);
  const auto k = activation.size(1);
  const auto n = weight.size(0);
  TORCH_CHECK(weight.size(1) == k, "weight K mismatch");
  TORCH_CHECK(scale.size(0) == n && zero.sizes() == scale.sizes(),
              "scale/zero shape mismatch");
  TORCH_CHECK(rows <= 65535LL * kBlockM,
              "flattened FPINT row count exceeds CUDA grid limit");

  const c10::cuda::CUDAGuard device_guard(activation.device());
  auto output = torch::empty({rows, n}, activation.options());
  if (rows == 0 || n == 0) {
    return output;
  }
  const dim3 block(kThreadsPerBlock);
  const dim3 grid((n + kBlockN - 1) / kBlockN,
                  (rows + kBlockM - 1) / kBlockM);
  const size_t aligned_arrays = has_zero ? 2 : 1;
  const size_t shared_memory =
      aligned_arrays * kBlockM * mxu_rows * sizeof(int64_t) +
      kBlockN * mxu_rows * sizeof(int8_t);
  auto stream = at::cuda::getCurrentCUDAStream(activation.device().index());
  if (activation.scalar_type() == at::kHalf) {
    fpint_qcol_kernel<__half, 10, 5, 15>
        <<<grid, block, shared_memory, stream>>>(
            reinterpret_cast<const __half*>(activation.data_ptr<at::Half>()),
            weight.data_ptr<int8_t>(),
            scale.data_ptr(),
            zero.data_ptr<int32_t>(),
            reinterpret_cast<__half*>(output.data_ptr<at::Half>()),
            static_cast<int>(rows),
            static_cast<int>(n),
            static_cast<int>(k),
            static_cast<int>(group_size),
            static_cast<int>(scale.size(1)),
            static_cast<int>(mxu_rows),
            static_cast<int>(extra_bits),
            static_cast<int>(reduce_extra_bits),
            has_zero, scale.scalar_type() == at::kBFloat16);
  } else {
    fpint_qcol_kernel<__nv_bfloat16, 7, 8, 127>
        <<<grid, block, shared_memory, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(
                activation.data_ptr<at::BFloat16>()),
            weight.data_ptr<int8_t>(),
            scale.data_ptr(),
            zero.data_ptr<int32_t>(),
            reinterpret_cast<__nv_bfloat16*>(
                output.data_ptr<at::BFloat16>()),
            static_cast<int>(rows),
            static_cast<int>(n),
            static_cast<int>(k),
            static_cast<int>(group_size),
            static_cast<int>(scale.size(1)),
            static_cast<int>(mxu_rows),
            static_cast<int>(extra_bits),
            static_cast<int>(reduce_extra_bits),
            has_zero, scale.scalar_type() == at::kBFloat16);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
