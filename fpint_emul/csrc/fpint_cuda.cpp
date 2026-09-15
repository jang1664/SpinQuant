#include <torch/extension.h>

torch::Tensor fpint_qcol_cuda(
    torch::Tensor activation,
    torch::Tensor weight,
    torch::Tensor scale,
    torch::Tensor zero,
    int64_t group_size,
    int64_t mxu_rows,
    int64_t extra_bits,
    int64_t reduce_extra_bits,
    bool has_zero);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "qcol_real_2scomp",
      &fpint_qcol_cuda,
      "QCOL_REAL_2SCOMP FP16 x INT CUDA forward");
}
