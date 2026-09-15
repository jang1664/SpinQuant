# FP×INT Linear 구현 결과

## 구현 요약

`QCOL_REAL_2SCOMP`의 연산 순서를 현재 SpinQuant의 inference Linear에 연결했다. 기본 backend는 기존 `standard`이며, `--linear_backend fpint_torch` 또는 `fpint_cuda`를 선택한 경우에만 FP×INT 경로가 활성화된다.

- `fpint_emul/config.py`: weight bit, K축 quant group, MXU row, main/reduce extra bit의 독립 설정과 overflow/shape 계약.
- `fpint_emul/reference.py`: 기존 하드웨어 NumPy 코드와 독립된 일반화 CPU reference. quant group과 MXU tile index를 분리하고 K/N tail 및 여러 batch 차원을 지원한다.
- `fpint_emul/torch_backend.py`: MXU tile 순서대로 FP32 누적하는 Torch backend. CUDA와 CPU tensor에서 같은 API를 사용한다.
- `fpint_emul/csrc/`: FP16 분해/prealignment, INT64 MAC, zero-point reduce, scale, 순차 FP32 누적을 한 CUDA kernel로 처리한다. Activation과 weight를 shared memory에서 M/N tile 내 재사용한다.
- `fpint_emul/cuda_backend.py`: backend를 처음 사용할 때 현재 PyTorch/CUDA 환경의 GPU arch로 extension을 JIT build한다. PyTorch current stream과 tensor device를 사용한다.
- `utils/quant_utils.py`: `ActQuantWrapper` backend dispatch, persistent integer metadata, layer coverage, legacy checkpoint 처리. Hadamard와 activation/output QDQ의 앞뒤 순서는 유지했다.
- `eval_utils/gptq_utils.py`: GPTQ/RTN 모두 signed INT4/INT8 weight, FP16 compact scale, signed zero-point와 K group mapping을 저장한다. asymmetric unsigned 표현은 weight와 zero에 같은 offset을 적용해 signed 표현으로 변환한다.
- `utils/process_args.py`, `eval_utils/main.py`, `result_analysis/load_model.py`: CLI, PTQ, checkpoint save/load 및 분석 loader 연결.
- `measure_fpint_linear.py`: accuracy, latency, peak allocation, 실제 projection shape와 tiny Llama 실행을 재현한다.

정수 weight는 unpacked `int8 [N,K]`, scale은 `float16 [N,G]`, zero는 `int32 [N,G]`다. Bias는 FP×INT 결과를 FP16으로 만든 뒤 FP16 add한다. `lm_head`처럼 정수 metadata가 없는 floating-point layer만 이유를 기록하고 standard에 남긴다. 다른 대상 layer의 metadata 누락은 오류다.

## 실행 방법

기존 실행 명령에 backend와 하드웨어 설정을 추가한다. 최초 `fpint_cuda` forward에서 현재 GPU arch용 extension을 build하며 이후 PyTorch extension cache를 사용한다.

```bash
export CUDA_HOME=/usr/local/cuda
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6  # berlin1은 12.0
export MAX_JOBS=2

# 기존 동작
--linear_backend standard

# PyTorch emulation 또는 custom CUDA
--w_bits 4 --w_groupsize 32 --w_rtn \
--linear_backend fpint_torch \
--fpint_mxu_rows 32 --fpint_extra_bits 19 --fpint_reduce_extra_bits 10

--w_bits 4 --w_groupsize 32 --w_rtn \
--linear_backend fpint_cuda \
--fpint_mxu_rows 32 --fpint_extra_bits 19 --fpint_reduce_extra_bits 10
```

GPTQ도 같은 옵션을 사용하되 `--w_rtn`을 생략한다. 새 checkpoint는 format version과 integer metadata를 저장한다. 기존 dequantized checkpoint에 metadata가 없으면 FPINT backend는 재양자화가 필요하다는 오류를 내고, `standard` backend는 이전처럼 load된다.

## 수치 검증

합격 기준은 `atol=1e-3, rtol=1e-3`이다. `tests/test_fpint_emul.py`는 다음 범위를 포함한다.

- INT4/INT8 × symmetric/asymmetric
- `group_size == mxu_rows`, 서로 다른 group/MXU 크기, per-channel
- K/N tail, 1D/2D/여러 batch 차원, empty batch, bias, 비연속 입력
- zero/subnormal/넓은 exponent/cancellation, 큰 K, weight 경계, 음수·큰 zero
- int64 overflow 거부
- RTN/GPTQ metadata, grouped act-order mapping, state-dict round trip, legacy metadata 오류
- activation QDQ와 online Hadamard 이후 FPINT dispatch
- CUDA current stream 및 multi-GPU device guard

기본 `group_size=mxu_rows=32`는 원래 `fpint_emul/py/fpint_emul.py::fpint_gemm_qcol_real_2scomp`와 FP16 결과가 일치한다. benchmark의 독립 reference 비교에서 Torch/CUDA 모두 max/mean/RMS absolute error와 tolerance 초과 비율이 0이었다. CUDA 테스트는 INT4/INT8, symmetric/asymmetric 네 조합을 모두 포함한다.

전체 현재 repo 테스트 결과:

```text
78 passed, 3 warnings in 6.48s
```

실행 명령:

```bash
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate spinquant
export CUDA_HOME=/usr/local/cuda
export PATH="$CONDA_PREFIX/bin:$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
pytest -q tests
```

## 성능 결과

FP16 activation, signed INT4 weight, `group_size=mxu_rows=32`, warmup 3회와 측정 20회 결과다. `standard_predequantized`는 FP16 weight를 미리 만든 Tensor Core baseline이고, `dequantize_plus_linear`는 매 호출 weight dequantization을 포함한다. FPINT CUDA는 dequantized weight를 만들지 않는다.

| GPU / zero | Shape (M,K,N) | predequantized | dequantize + Linear | FPINT CUDA |
| --- | ---: | ---: | ---: | ---: |
| RTX A6000 / zero=0 | 1,3072,3072 | 0.039 ms | 0.922 ms | 0.317 ms |
| RTX A6000 / zero=0 | 32,3072,3072 | 0.061 ms | 0.935 ms | 0.968 ms |
| RTX A6000 / nonzero | 1,3072,3072 | 0.041 ms | 0.936 ms | 0.348 ms |
| RTX A6000 / nonzero | 32,3072,3072 | 0.040 ms | 0.952 ms | 1.204 ms |
| RTX PRO 6000 Blackwell / zero=0 | 1,3072,3072 | 0.014 ms | 0.201 ms | 0.234 ms |
| RTX PRO 6000 Blackwell / zero=0 | 32,3072,3072 | 0.015 ms | 0.205 ms | 0.370 ms |

순차 Torch FPINT는 A6000의 `(1,3072,3072)`에서 164.28 ms라서 실제 모델 경로로 부족했다. 이 결과 때문에 CUDA kernel을 구현했다. FPINT CUDA의 측정 중 추가 peak allocation은 출력 크기와 같았다: M=1은 6,144 bytes, M=32는 196,608 bytes. 반면 호출마다 dequantize하는 경로는 113,270,784 bytes가 추가됐다.

Llama-3.2-3B의 MLP projection 크기를 반영한 추가 A6000 측정에서 FPINT CUDA는 decode형 `(1,3072,8192)` 0.503 ms, `(1,8192,3072)` 0.876 ms였다. M=32에서는 각각 2.667/3.219 ms였다. 이때 매 호출 dequantization은 약 302 MB를 추가로 사용했지만 FPINT CUDA는 출력 크기만 할당했다. 원본은 [A6000 Llama projection shapes](a6000-fpint-llama32-shapes.json)에 있다.

tiny Llama는 hidden 64, intermediate 128, 1 layer, sequence 32의 repo `LlamaForCausalLM`을 RTN INT4로 만들어 실행했다. 7개 attention/MLP projection은 FPINT CUDA, `lm_head`는 standard였고 모든 출력은 finite였다.

| GPU | symmetric | latency | coverage |
| --- | --- | ---: | --- |
| RTX A6000 | yes | 1.803 ms | 7 FPINT CUDA, 1 standard |
| RTX A6000 | no | 1.834 ms | 7 FPINT CUDA, 1 standard |
| RTX PRO 6000 Blackwell | yes | 1.281 ms | 7 FPINT CUDA, 1 standard |

원본 결과:

- [A6000 symmetric](a6000-fpint-symmetric.json)
- [A6000 asymmetric](a6000-fpint-asymmetric.json)
- [A6000 Torch backend](a6000-fpint-torch.json)
- [A6000 Llama projection shapes](a6000-fpint-llama32-shapes.json)
- [Blackwell symmetric](blackwell-fpint-symmetric.json)

재현 명령:

```bash
python measure_fpint_linear.py \
  --warmup 3 --iterations 20 \
  --output-json agent-tasks/fp-int-emul/a6000-fpint-symmetric.json

python measure_fpint_linear.py \
  --asymmetric --warmup 3 --iterations 20 \
  --output-json agent-tasks/fp-int-emul/a6000-fpint-asymmetric.json
```

## Blackwell 검증

기존 berlin1의 `/home/jaeyong.jang/.conda/envs/spinquant`에서 PyTorch 2.7.0+cu128, CUDA toolkit 12.8과 `TORCH_CUDA_ARCH_LIST=12.0`으로 새 kernel을 직접 build했다. RTX PRO 6000 Blackwell Server Edition에서 위 benchmark와 tiny model이 통과했다. 동일 테스트 파일의 43개 테스트도 통과해 INT4/INT8 × symmetric/asymmetric, per-channel/큰-K/subnormal/큰 zero, PTQ checkpoint round-trip, current stream, GPU 1 device guard를 확인했다.

```bash
ssh berlin1
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate spinquant
cd /home/jaeyong.jang/spinquant-setup.fP5oR2/source
PYTHONPATH=. pytest -q tests/test_fpint_emul.py
python measure_fpint_linear.py --warmup 3 --iterations 20
```

## 알려진 제한

- 학습 backward, attention QK/PV, KV cache는 범위 밖이다.
- INT4 packing은 하지 않으므로 integer checkpoint weight는 element당 1 byte다.
- positive `group_size`는 MXU row의 배수여야 한다. Scale boundary가 MXU tile을 가르는 설정은 오류다.
- grouped GPTQ와 act-order를 함께 사용하면 원래 K 순서의 scale group이 불연속일 수 있다. Metadata/group mapping은 보존하지만 QCOL backend 선택 시 설명 있는 오류를 낸다. Per-channel act-order는 지원한다.
- FP16 activation의 NaN/Inf는 public API 검증에서 거부한다. Wrapper steady-state는 전체 tensor scan 비용을 피하기 위해 metadata를 한 번 검증하고 매 forward에는 shape/device 계약만 확인하므로 입력은 finite여야 한다.
- predequantized Tensor Core baseline은 특히 prefill에서 여전히 훨씬 빠르다. FPINT CUDA의 목적은 목표 하드웨어 연산 의미를 유지하면서 Torch emulation의 병목과 full dequantized-weight 임시 allocation을 없애는 것이다.
- 실제 3B/7B checkpoint 전체 평가와 downstream task accuracy는 checkpoint가 생성되는 실험과 별도이며, 여기서는 실제 projection shape와 작은 end-to-end model 실행까지 검증했다.
- 검증 명령은 repo의 maintained suite인 `pytest -q tests`다. 아무 경로 제한 없이 root에서 `pytest`를 실행하면 vendored `fast-hadamard-transform/tests`의 미설치 `einops`와 기존 `fpint_emul/py` 검증 코드의 미선언 `vsc`/누락 visualization helper 때문에 별도 collection 오류가 난다. 새 FPINT package의 legacy symbol re-export는 유지했으며 이 두 기존 test dependency 문제는 runtime backend와 분리했다.
