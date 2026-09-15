#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <torch/extension.h>

#include <cstdint>

namespace {

constexpr int kBlockN = 32;
constexpr int kBlockM = 4;
constexpr int kMantissaBits = 10;
constexpr int kExponentBias = 15;

__device__ __forceinline__ int fp16_exponent(const __half value) {
  return (__half_as_ushort(value) >> 10) & 0x1f;
}

__device__ __forceinline__ int64_t align_fp16(
    const __half value, const int maximum_exponent, const int extra_bits) {
  const unsigned bits = __half_as_ushort(value);
  const bool negative = (bits >> 15) != 0;
  const int exponent = (bits >> 10) & 0x1f;
  const int exponent_for_align = exponent == 0 ? 1 : exponent;
  const int64_t hidden =
      (static_cast<int64_t>(exponent != 0) << kMantissaBits) |
      static_cast<int64_t>(bits & 0x3ff);
  const int shift = maximum_exponent - exponent_for_align;
  const int64_t aligned = (hidden << extra_bits) >> shift;
  return negative ? -aligned : aligned;
}

__global__ void fpint_qcol_kernel(
    const __half* __restrict__ activation,
    const int8_t* __restrict__ weight,
    const __half* __restrict__ scale,
    const int32_t* __restrict__ zero,
    __half* __restrict__ output,
    const int rows,
    const int n_columns,
    const int k_columns,
    const int group_size,
    const int group_count,
    const int mxu_rows,
    const int extra_bits,
    const int reduce_extra_bits,
    const bool has_zero) {
  extern __shared__ unsigned char shared_bytes[];
  int64_t* shared_main = reinterpret_cast<int64_t*>(shared_bytes);
  int64_t* shared_reduce = shared_main + kBlockM * mxu_rows;
  int8_t* shared_weight = reinterpret_cast<int8_t*>(
      shared_reduce + (has_zero ? kBlockM * mxu_rows : 0));
  __shared__ int maximum_exponents[kBlockM];

  const int lane = threadIdx.x;
  const int local_row = threadIdx.y;
  const int row = blockIdx.y * kBlockM + local_row;
  const int column = blockIdx.x * kBlockN + lane;
  const int linear_thread = local_row * kBlockN + lane;
  float accumulator = 0.0f;

  const int tile_count = (k_columns + mxu_rows - 1) / mxu_rows;
  for (int tile = 0; tile < tile_count; ++tile) {
    const int tile_start = tile * mxu_rows;
    int local_maximum = 1;
    if (row < rows) {
      for (int offset = lane; offset < mxu_rows; offset += kBlockN) {
        const int k = tile_start + offset;
        if (k < k_columns) {
          int exponent = fp16_exponent(activation[row * k_columns + k]);
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
    if (lane == 0) {
      maximum_exponents[local_row] = local_maximum;
    }
    __syncthreads();

    if (row < rows) {
      for (int offset = lane; offset < mxu_rows; offset += kBlockN) {
        const int k = tile_start + offset;
        const __half value =
            k < k_columns ? activation[row * k_columns + k]
                          : __float2half(0.0f);
        shared_main[local_row * mxu_rows + offset] =
            align_fp16(value, maximum_exponents[local_row], extra_bits);
        if (has_zero) {
          shared_reduce[local_row * mxu_rows + offset] = align_fp16(
              value, maximum_exponents[local_row], reduce_extra_bits);
        }
      }
    }
    for (int index = linear_thread; index < kBlockN * mxu_rows;
         index += kBlockN * kBlockM) {
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
      for (int offset = 0; offset < mxu_rows; ++offset) {
        inner += shared_main[local_row * mxu_rows + offset] *
            static_cast<int64_t>(shared_weight[lane * mxu_rows + offset]);
        if (has_zero) {
          reduction += shared_reduce[local_row * mxu_rows + offset];
        }
      }
      const int group = group_size == -1 ? 0 : tile_start / group_size;
      int64_t post = inner;
      if (has_zero) {
        const int64_t zero_value = zero[column * group_count + group];
        post -= zero_value * reduction *
            (int64_t{1} << (extra_bits - reduce_extra_bits));
      }
      // Intrinsics make the two specified FP32 operations explicit and avoid
      // silently fusing the scale multiply with the K-ordered accumulation.
      const int binary_exponent = maximum_exponents[local_row] -
          kExponentBias - kMantissaBits - extra_bits;
      float contribution = __fmul_rn(
          static_cast<float>(post), ldexpf(1.0f, binary_exponent));
      contribution = __fmul_rn(
          contribution, __half2float(scale[column * group_count + group]));
      accumulator = __fadd_rn(accumulator, contribution);
    }
    __syncthreads();
  }

  if (row < rows && column < n_columns) {
    output[row * n_columns + column] = __float2half_rn(accumulator);
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
  TORCH_CHECK(activation.scalar_type() == at::kHalf, "activation must be FP16");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(scale.scalar_type() == at::kHalf, "scale must be FP16");
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
  const dim3 block(kBlockN, kBlockM);
  const dim3 grid((n + kBlockN - 1) / kBlockN,
                  (rows + kBlockM - 1) / kBlockM);
  const size_t aligned_arrays = has_zero ? 2 : 1;
  const size_t shared_memory =
      aligned_arrays * kBlockM * mxu_rows * sizeof(int64_t) +
      kBlockN * mxu_rows * sizeof(int8_t);
  auto stream = at::cuda::getCurrentCUDAStream(activation.device().index());
  fpint_qcol_kernel<<<grid, block, shared_memory, stream>>>(
      reinterpret_cast<const __half*>(activation.data_ptr<at::Half>()),
      weight.data_ptr<int8_t>(),
      reinterpret_cast<const __half*>(scale.data_ptr<at::Half>()),
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
      has_zero);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
