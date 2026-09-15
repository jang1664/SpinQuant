"""Run inside the transferred source directory with the spinquant environment."""
import importlib
import importlib.util
import pathlib
import tempfile
import time
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline, CUDA_HOME
from fast_hadamard_transform import hadamard_transform

torch.manual_seed(0)
print('ENV', torch.__version__, torch.version.cuda, CUDA_HOME, flush=True)
print('ARCH', torch.cuda.get_arch_list(), flush=True)
for name in ('transformers', 'accelerate', 'datasets', 'eval_utils.main', 'result_analysis.load_model'):
    mod = importlib.import_module(name)
    print('IMPORT_OK', name, getattr(mod, '__version__', ''), flush=True)
for device in range(torch.cuda.device_count()):
    x = torch.randn(32, 128, device=f'cuda:{device}', dtype=torch.float16)
    assert torch.isfinite(x @ x.T).all()
    y = hadamard_transform(x, scale=128**-0.5)
    z = hadamard_transform(y, scale=128**-0.5)
    torch.testing.assert_close(z, x, atol=0.004, rtol=0.004)
    print('GPU_AND_HADAMARD_OK', device, torch.cuda.get_device_name(device),
          torch.cuda.get_device_capability(device), flush=True)

build = tempfile.mkdtemp(prefix='spinquant-cuda-smoke-')
module = load_inline(name='spinquant_blackwell_smoke',
    cpp_sources='torch::Tensor smoke_cuda(torch::Tensor x);',
    cuda_sources=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
__global__ void kernel(const float* x, float* y) {
    int i = threadIdx.x;
    y[i] = x[i] + 1.0f;
}
torch::Tensor smoke_cuda(torch::Tensor x) {
    auto y = torch::empty_like(x);
    kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<float>(),y.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
''', functions=['smoke_cuda'], build_directory=build, verbose=True)
stream = torch.cuda.Stream()
with torch.cuda.stream(stream):
    x = torch.arange(32, device='cuda', dtype=torch.float32)
    y = module.smoke_cuda(x)
stream.synchronize()
assert torch.equal(y, x + 1)
print('CUSTOM_CUDA_BUILD_STREAM_RUN_OK', flush=True)

spec = importlib.util.spec_from_file_location('hwref', pathlib.Path('fpint_emul/py/fpint_emul.py'))
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
import figna_source as gpu
# Align the imported emulator's MXU tile with the supplied hardware reference.
gpu.MXU_K = ref.MXU_K
M, K, N = 4, 128, 32
x = torch.randn(M, K, device='cuda', dtype=torch.float16)
for bits in (4, 8):
    w = torch.randint(-(2**(bits-1)), 2**(bits-1), (K, N), device='cuda', dtype=torch.int8)
    scale = (torch.rand(K//32, N, device='cuda') * 0.005 + 0.001).half()
    for asymmetric in (False, True):
        zero = torch.randint(-4, 5, scale.shape, device='cuda', dtype=torch.int16) if asymmetric else torch.zeros_like(scale, dtype=torch.int16)
        expected = ref.fpint_gemm_qcol_real_2scomp(
            x.cpu().numpy().view(np.uint16), w.cpu().numpy(),
            scale.cpu().numpy().view(np.uint16), zero.cpu().numpy(), M, N, K).view(np.float16)
        actual = gpu.fpint_gemm_qcol_real_2scomp_torch(x, w,
            scale.repeat_interleave(32, 0), zero.repeat_interleave(32, 0), 32)
        torch.testing.assert_close(actual.cpu(), torch.from_numpy(expected), atol=1e-3, rtol=1e-3)
        print('QCOL_REFERENCE_ALLCLOSE_OK', bits, asymmetric,
              float((actual.cpu()-torch.from_numpy(expected)).abs().max()), flush=True)

# A timing probe, not a performance acceptance target or a full model benchmark.
x = torch.randn(32, 4096, device='cuda', dtype=torch.float16)
w = torch.randint(-8, 8, (4096, 4096), device='cuda', dtype=torch.int8)
scale = torch.full(w.shape, 0.01, device='cuda', dtype=torch.float16)
zero = torch.zeros_like(w, dtype=torch.int16)
dequantized_weight = w.half() * scale
for label, fn in (
    ('PREDEQUANTIZED_MATMUL', lambda: x @ dequantized_weight),
    ('DEQUANTIZE_PLUS_MATMUL', lambda: x @ (w.half()*scale)),
    ('SOURCE_FPINT_QCOL', lambda: gpu.fpint_gemm_qcol_real_2scomp_torch(x,w,scale,zero,32)),
):
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        out = fn()
    torch.cuda.synchronize()
    print('TIMING_MS', label, (time.perf_counter()-start)*100, 'finite', bool(torch.isfinite(out).all()), flush=True)
print('ALL_SMOKE_CHECKS_PASSED', flush=True)
