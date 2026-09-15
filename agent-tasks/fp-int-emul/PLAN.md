# FP×INT Linear 하드웨어 에뮬레이션 이식 계획

## 목표와 확정 사항

- 현재 SpinQuant의 양자화된 Linear를 하드웨어 방식의 FP×INT 에뮬레이션으로 실행한다.
- 기준 연산은 `QCOL_REAL_2SCOMP`다. Weight 양자화 그룹은 GEMM reduction 축 K를 따라 형성된다.
- FP16 activation, signed INT4/INT8 weight, 대칭·비대칭 양자화를 지원한다.
- 검증 합격 기준은 `allclose`다. Bit-exact 일치를 요구하지 않는다. 다만 하드웨어의 alignment 및 zero-point 보정 의미를 유지한다.
- 성능 목표는 미리 고정하지 않는다. 실제 Linear 및 모델 실행 시간·메모리를 측정하고 사용자가 체감할 수 있는 실행 경로를 제공한다. 속도가 부족하면 custom CUDA kernel을 구현한다.
- Git merge 대신 필요한 코드를 복사하고 현재 repo에 맞게 수정한다.
- 이번 범위는 inference용 Linear다. QKᵀ/PV, attention backend, KV cache 에뮬레이션, 학습용 backward는 제외한다.

## 기준 소스와 이식 범위

### 연산 의미의 기준

- `fpint_emul/py/fpint_emul.py::prealign`
- `fpint_emul/py/fpint_emul.py::fpint_gemm_qcol_real_2scomp`

이 파일은 기존 하드웨어 연산 순서를 이해하는 기준이다. 하드코딩과 인덱스 문제까지 정답으로 고정하지 않는다. 원본은 보존하고 별도 일반화 reference를 만든다.

### 재사용할 구현

`/home/jaeyongjang/project.local/SpinQuant_fpint`에서 다음 부분만 선별한다.

- `utils/figna_utils.py`: FP16 bit 분해·prealignment, QCOL 정수 MAC 및 zero-point 보정, Linear 진입점.
- `utils/quant_utils.py`: `ActQuantWrapper`의 custom Linear 분기.
- `eval_utils/gptq_utils.py`: 양자화 시 정수 weight와 quantization parameter를 보존하는 구조.
- `eval_utils/main.py`, `utils/process_args.py`: 활성화 및 옵션 연결 방식.
- `test_figna_opt.py`: 테스트 입력 구성과 timing 방식 참고. 현재 구현을 capture한 뒤 자기 자신과 비교하는 방식은 독립 reference 검증으로 대체한다.

기존 FP32 collapsed fast path와 FP64 block reduction은 기준 연산으로 채택하지 않는다. 최적화 후보로서만 검토하고 reference 대비 allclose를 통과해야 사용한다.

## 환경설정

### 사용할 환경과 확인 결과 (2026-09-15)

`spinquant` conda 환경을 사용한다. 아래 항목은 설치 여부만 확인한 것이 아니라 주요 import, GPU 연산, 작은 CUDA extension 빌드·실행까지 확인했다. 현 시점에서 PyTorch backend 개발과 custom CUDA extension 개발을 막는 환경 문제는 발견되지 않았다.

| 항목 | 확인된 값 |
| --- | --- |
| Python | `/home/jaeyongjang/.conda/envs/spinquant/bin/python`, 3.10.16 |
| PyTorch | `2.7.0+cu126`, CUDA runtime 12.6 |
| CUDA toolkit | `/usr/local/cuda`, nvcc 12.6.85 |
| Host C++ compiler | GCC 12.4.0 |
| GPU | NVIDIA RTX A6000 × 4, 각 49140 MiB, compute capability 8.6 |
| NVIDIA driver | 575.57.08 (`nvidia-smi`의 CUDA 표시는 12.9; 실제 빌드 toolkit은 12.6) |
| NumPy / pytest / ninja | 2.1.2 / 9.1.1 / 1.11.1.4 |
| transformers / accelerate / datasets | 4.44.2 / 0.34.2 / 4.8.5 |
| fast_hadamard_transform | 1.0.4.post1 |

### 실행 및 빌드 설정

```bash
conda activate spinquant
export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=2
```

현재 활성화되지 않은 셸에서는 `ninja` 명령을 찾지 못했다. `/home/jaeyongjang/.conda/envs/spinquant/bin/ninja`는 설치되어 있고 해당 bin을 PATH에 넣은 빌드는 성공했다. 절대 경로 Python만 호출해도 외부 빌드 도구의 PATH가 자동 설정되는 것은 아니므로 conda 활성화를 우선한다. 환경을 활성화하기 어려운 실행 도구에서는 아래처럼 환경 bin을 명시한다.

```bash
PATH=/home/jaeyongjang/.conda/envs/spinquant/bin:$PATH \
TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2 \
/home/jaeyongjang/.conda/envs/spinquant/bin/python <script.py>
```

`nvcc`는 conda 환경 내부가 아닌 시스템 toolkit을 사용한다. 실제 실험 시에는 사용할 GPU를 `CUDA_VISIBLE_DEVICES`로 선택하고, 측정 직전 다른 작업의 사용량을 확인한다. 설치 패키지 변경이나 재설치는 이번 점검에서 수행하지 않았다.

### 통과한 점검과 검증 범위

- `python -m pip check`: `No broken requirements found`.
- 주요 패키지 및 `eval_utils.main`, `result_analysis.load_model` import 성공.
- GPU 0에서 FP16 행렬곱 실행 및 finite 출력 확인.
- 설치된 `fast_hadamard_transform`을 FP16 CUDA tensor에 실행하여 finite 출력 확인.
- PyTorch `load_inline`으로 C++/CUDA extension을 SM 8.6 대상으로 실제 컴파일·링크·로드했다. Current CUDA stream에서 실행하는 간단한 커널의 출력이 기대값과 일치했다.
- 재현용 임시 스크립트: `/tmp/spinquant-env-check.0wknN1/check.py`. 같은 폴더에 빌드 산출물이 있다. `/tmp` 파일은 장기 보관용이 아니며 최종 backend 구현 시 정식 smoke test로 대체한다.
- GPU 4개는 모두 인식했다. 초기 환경 점검 후 목표 FP×INT kernel을 구현해 GPU 0 수치·성능, non-default stream, GPU 1 device guard까지 검증했다. 전체 3B/7B checkpoint 평가는 수행하지 않았다.

### Blackwell / berlin1 이식 상태 (2026-09-15)

- 현재 환경을 [spinquant-source-environment.yml](spinquant-source-environment.yml)로 `conda env export --no-builds` 했다. 이는 현재 머신의 원본 snapshot이며 원격 설치용 manifest로 그대로 사용하지 않는다.
- 원본 PyTorch는 `2.7.0+cu126`이며 확인된 binary arch 목록에는 Blackwell이 없다. [PyTorch 2.7 공식 릴리스](https://pytorch.org/blog/pytorch-2-7/)에 따라 같은 2.7.0의 CUDA 12.8 wheel로 설치했다.
- Export에는 CUDA 11/12 관련 pip package가 함께 있고 머신 고유 `prefix`도 있다. 원격용 환경은 이 항목을 정리하고 PyTorch CUDA 12.8 wheel의 의존성으로 CUDA runtime package를 해결한다. GPU 빌드 계열을 바꾸면서 기존 NVIDIA package pin을 그대로 강제하지 않는다.
- SSH 인증 복구 후 `/home/jaeyong.jang/.conda/envs/spinquant`를 새로 생성했다. 기존 `CIM` 환경과 시스템 CUDA 13.0은 변경하지 않았다.
- 서버: x86_64, RTX PRO 6000 Blackwell Server Edition × 4 (각 97887 MiB), compute capability 12.0, driver 580.126.20.
- 설치: Python 3.10.16, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, torchaudio 2.7.0+cu128, conda CUDA toolkit 12.8 (nvcc 12.8.93). 나머지 Python 패키지는 [berlin1-requirements.txt](berlin1-requirements.txt)의 원본 export 버전을 사용했다.
- `fast_hadamard_transform` 로컬 소스를 전송해 [berlin1-hadamard-setup.py](berlin1-hadamard-setup.py)의 `compute_120/sm_120` 대상으로 재빌드했다. 버전은 1.0.4.post1이며 Blackwell 4개 GPU 모두에서 실행 및 Hadamard 왕복 allclose 검증을 통과했다.
- conda 환경에 `CUDA_HOME`, toolkit header용 `CPATH`, `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=2`를 설정했다. [berlin1-activate.sh](berlin1-activate.sh)를 activation hook으로 설치해 host compiler를 시스템 GCC/G++ 13.3으로 통일했다.
- `pip check`, 주요 모델 모듈 import, GPU 행렬곱과 Hadamard가 통과했다. 이후 실제 FP×INT CUDA kernel을 SM120으로 build했고 최종 FP×INT test 40개와 실제 projection/tiny-model benchmark를 통과했다.
- 실행 위치: `/home/jaeyong.jang/spinquant-setup.fP5oR2/source`. [BERLIN1_RESULTS.md](BERLIN1_RESULTS.md)에 재실행 명령, 측정 수치, 검증 한계를 기록했다. 설치 결과 snapshot은 [spinquant-berlin1-environment.yml](spinquant-berlin1-environment.yml), 로그는 [berlin1-smoke.log](berlin1-smoke.log)다.

## Reference

### 연산 기준 및 PyTorch 이식 소스

| 파일 | 참고할 내용 |
| --- | --- |
| [fpint_emul.py](/home/jaeyongjang/project.local/SpinQuant/fpint_emul/py/fpint_emul.py) | `prealign`, `fpint_gemm_qcol_real_2scomp`: 목표 연산 순서. FP64 reference는 별도 수치 오차 비교용. |
| [FPINT_EMUL_EQUATIONS.md](/home/jaeyongjang/project.local/SpinQuant/fpint_emul/FPINT_EMUL_EQUATIONS.md) | 연산 수식 설명. 고정 크기 설명은 실제 config와 대조한다. |
| [figna_utils.py](/home/jaeyongjang/project.local/SpinQuant_fpint/utils/figna_utils.py) | GPU PyTorch prealignment 및 QCOL 구현. Fast path는 reference 검증 후 최적화 후보로만 사용. |
| [quant_utils.py](/home/jaeyongjang/project.local/SpinQuant_fpint/utils/quant_utils.py) | `ActQuantWrapper`의 custom Linear 분기. |
| [gptq_utils.py](/home/jaeyongjang/project.local/SpinQuant_fpint/eval_utils/gptq_utils.py) | 정수 weight/scale 보존. Zero-point, RTN, act-order 보완 필요. |
| [main.py](/home/jaeyongjang/project.local/SpinQuant_fpint/eval_utils/main.py), [process_args.py](/home/jaeyongjang/project.local/SpinQuant_fpint/utils/process_args.py) | Backend 활성화와 옵션 연결. |
| [test_figna_opt.py](/home/jaeyongjang/project.local/SpinQuant_fpint/test_figna_opt.py) | QCOL 입력 구성, allclose 및 timing 측정 예시. |

### prealign 저장소: CUDA 구현 참고

2026-09-15 소스 탐색 결과다. 아래 코드는 구조 재사용 후보이며 목표 QCOL_REAL_2SCOMP와 수치적으로 동일하거나 성능이 충분하다고 검증한 결과는 아니다. 이 탐색에서 빌드·GPU 실행은 하지 않았다.

| 우선순위 | 파일 | 참고할 내용 / 목표와 차이 |
| --- | --- | --- |
| 높음 | [fp16_prealign_linear_cuda_kernel.cu](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/fp16_prealign_linear/cuda/fp16_prealign_linear_cuda_kernel.cu) | 가장 가까운 커널. FP16 bit 분해, extra-bit alignment, shared memory의 activation/weight 재사용, int64 MAC, `num_systolic_row`마다 FP32 누적, FP16 출력. 그룹 scale과 별도의 zero-point reduce 경로가 없다. Weight는 int8이 아니라 FP16에 담긴 정수를 커널 안에서 int로 변환한다. |
| 높음 | [fp16_prealign_linear_cuda.cpp](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/fp16_prealign_linear/cuda/fp16_prealign_linear_cuda.cpp) | PyBind forward, CUDA/contiguous 검사, device guard. Shape/dtype/device 일치 검사를 보강할 출발점. |
| 높음 | [setup.py](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/fp16_prealign_linear/cuda/setup.py) | `CUDAExtension`/`BuildExtension` 구성. SM 70/75/80/86 고정 목록은 현재 GPU에 맞게 변경한다. |
| 보조 | [decompose_fp16_cuda_kernel.cu](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/decompose_fp16/cuda/decompose_fp16_cuda_kernel.cu) | FP16 sign/exponent/fraction 추출을 독립 커널로 구현한 예. 중간값 진단에 참고하되 dtype dispatch 및 bit reinterpretation을 보완한다. |
| 보조 | [debug_script.py](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/fp16_prealign_linear/cuda/debug_script.py) | Extension 호출과 `F.linear` 비교 예시. Ones weight 등 제한된 입력 및 상대오차 출력만 있으므로 새로운 allclose 테스트로 대체한다. |
| 보조 | [prealign_mm.py](/home/jaeyongjang/project.local/prealign/prealign_mm/prealign_mm.py) | Python의 분해→alignment→정수 MAC→normalization 구조. FP32/binary weight 중심이며 `chunk` 분할·반복 메모리 정리 방식을 그대로 이식하지 않는다. |
| 낮음 | [prealign_linear_cuda_kernel.cu](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/prealign_linear/cuda/prealign_linear_cuda_kernel.cu) | FP32 형식의 유사 tiled integer MAC 커널. FP16 버전과 공통 구조 비교용. |
| 낮음 | [bf16_prealign_linear_cuda_kernel.cu](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/bf16_prealign_linear/cuda/bf16_prealign_linear_cuda_kernel.cu) | BF16 변형의 alignment/MAC 구조 참고. BF16 지원은 이번 범위에 추가하지 않는다. |
| 낮음 | [hgemm_cuda_kernel.cu](/home/jaeyongjang/project.local/prealign/prealign_mm/extension/hgemm/cuda/hgemm_cuda_kernel.cu) | Shared-memory GEMM 구조 비교용. `__hfma`와 half accumulator를 사용하므로 목표 FP32 누적 reference로 쓰지 않는다. |
| 낮음 | [quant_cuda_kernel.cu](/home/jaeyongjang/project.local/prealign/gptq/cuda_extension/quant_cuda_kernel.cu) | Packed 3-bit weight unpacking, scale/zero 적용, half2 연산 예. Prealignment가 없는 dequantization 기반 GEMV이며 목표 커널과 다르다. Packing을 후속 검토할 때만 참고한다. |

### CUDA 이식 시 수정해야 할 사항

- 핵심 재사용 대상은 FP16 커널의 tiling/정수 MAC 구조다. 목표의 main/reduce 이중 alignment, QCOL scale/zero indexing, signed int8 weight 입력을 추가한다.
- 기존 `normalization`은 FP32 bit pattern을 수동 구성하며 round-to-nearest-even 경로에 미검증 주석이 있다. 이를 정답으로 사용하지 않고 목표 reference의 FP32 변환 의미에 맞춰 재구현·검증한다. 가변 shift의 폭과 정수 overflow도 점검한다.
- Host에서 `abs + max_pool1d`로 MXU 그룹 최대값을 구한다. 이 전처리 비용을 따로 측정하고 필요하면 exponent reduction을 커널에 융합한다. 모든 shape에서 pooling 차원과 tail이 올바른지도 검사한다.
- 커널은 `BLOCK_SIZE=16`인 16×16 thread/shared-memory tile과 독립적인 `num_systolic_row`를 사용한다. 두 크기를 분리하는 설계는 참고하되 고정 block 크기가 성능상 최선이라고 가정하지 않는다.
- `<<<grid, blocks>>>`에는 PyTorch current CUDA stream이 지정되어 있지 않다. 새 binding은 current stream으로 실행하고 launch error를 검사하며 multi-GPU device guard를 유지한다.
- 선언된 dtype dispatch 범위와 실제 `at::Half` pointer 사용을 일치시킨다. `.contiguous()` 결과를 만든 뒤 원본 pointer를 넘기는 부분, `MIN`으로 경계 주소를 clamp하는 부분은 명시적인 검증/masking으로 정리한다.
- `*_bak*`, `*.failed_speedup`, `*.with_fp32` 등의 대안 파일도 있으나 검증된 최적화로 간주하지 않는다. 우선 현재 FP16 `.cu`를 출발점으로 삼는다.

## 1. 연산 계약과 일반화 reference

### QCOL 처리 순서

1. MXU ROW 크기 T의 K축 tile마다 FP16 exponent 최댓값을 구한다.
2. 가수 절댓값을 오른쪽 shift한 뒤 부호를 적용한다. Main/reduce 경로는 각각 별도의 extra-bit 설정을 사용한다. Subnormal 처리는 기존 prealign을 따른다.
3. Tile 내 정수 MAC `I = sum(A_main * W_int)`와 reduction `R = sum(A_reduce)`를 계산한다.
4. `P = I - Z * R * 2^(extra_main - extra_reduce)`로 zero-point를 보정한다.
5. Tile exponent로 실수 값을 복원해 FP32로 변환하고 FP16 scale을 FP32로 올려 곱한다.
6. K축 순서로 tile 기여분을 FP32 누적하고 최종 FP16 출력을 만든다.
7. Bias가 있으면 기존 Linear 인터페이스에 맞춰 더한다. Bias의 dtype과 더하는 시점은 reference와 GPU 구현에서 동일하게 명시한다.

`Z=0`은 동일한 수학적 의미를 유지하며 reduce 계산을 생략할 수 있다. Zero-point가 0이 아닌 경우 main alignment 결과로 reduce 경로를 대체하지 않는다.

### 독립 설정

- `weight_bits`: 4 또는 8.
- `group_size`: K축 quantization block 크기. `-1`은 출력 채널마다 K 전체가 한 그룹.
- `mxu_rows`: prealignment 및 정수 reduction tile 크기. 기본값은 제공된 기준 코드의 32.
- `extra_bits`: 기본 19, `reduce_extra_bits`: 기본 10.
- GPU 처리 tile 크기는 위 하드웨어 수치 설정과 별개로 둔다. CUDA block 크기를 바꾸어 하드웨어 alignment 범위가 변하지 않게 한다.

K축이 양자화 축이라는 사실과 `group_size == mxu_rows`는 구분한다. 우선 한 MXU tile 안에서 scale/zero가 일정한 조합을 지원한다. 양의 group size는 MXU ROW의 배수로 검증하고, per-channel 그룹과 마지막 K tail은 명시적으로 처리한다. Scale 경계가 MXU tile 내부를 가르는 조합은 임의 재정렬하지 않고 설명 있는 오류를 낸다. 추후 필요하면 실제 하드웨어의 분할 규칙을 먼저 정의한다.

### Shape와 저장 형식

- Linear 입력은 `[..., K]`, weight는 `[N, K]`, 출력은 `[..., N]`으로 일반화한다. 2D 입력과 여러 batch 차원을 지원한다.
- 정수 weight는 unpacked `int8`로 보관한다. INT4 packing은 필수 범위에 넣지 않는다.
- Scale/zero는 `[N, ceil(K / group_size)]`의 compact 형식으로 보존한다. Per-channel은 `[N, 1]`이다.
- K/N tail은 mask 또는 neutral padding으로 처리한다. Padding이 유효 입력의 alignment 및 그룹 의미를 바꾸지 않는지 검증한다.
- 비연속 입력·device 이동·빈 batch 등은 API에서 명확히 처리한다.
- FP16 외 activation dtype, NaN/Inf 입력, integer overflow 가능 설정은 정책을 명시하고 검증한다. 암묵적 BF16 reinterpretation이나 int8 wraparound는 허용하지 않는다.

### 원본에서 바로 고칠 의미상의 문제

- QCOL의 `aligned_exp_data[m, kg]`는 quantization group 인덱스를 사용한다. 일반화 reference는 MXU tile 인덱스와 quantization group 인덱스를 분리한다.
- `QBLOCK // MXU_K`에 의존한 반복문을 일반화하여 tail과 독립 설정을 처리한다.
- 관련 수식 문서의 16 고정 설명과 현재 코드의 32 설정이 다르므로 구현 계약에는 실제 설정값을 기록한다.

## 2. GPU PyTorch backend

예정 파일 구조:

- `fpint_emul/config.py`: 설정 및 입력 계약 검증.
- `fpint_emul/reference.py`: CPU NumPy 일반화 QCOL reference.
- `fpint_emul/torch_backend.py`: CUDA tensor 기반 PyTorch 에뮬레이션.
- `fpint_emul/linear.py`: backend dispatch와 Linear tensor 인터페이스.
- 필요한 package 초기화 파일. 기존 `fpint_emul/py` 검증 코드는 보존한다.

Prealignment는 vectorize하고, M/N 방향으로 작업을 나누어 큰 중간 tensor를 제한한다. Full-size scale/zero 복제와 전체 `[K/T, M, N]` materialization을 피한다. 정수 contraction을 FP64 연산으로 대체한다면 값 범위와 정확한 정수 표현 가능 범위를 확인한다.

먼저 reference에 가까운 누적 순서로 구현한다. 재결합·누적 순서 변경은 별도 최적화로 적용하고 allclose 및 오차 분포로 판단한다. 전역 TF32 설정을 영구 변경하지 않으며 필요 시 예외 발생에도 원복한다.

## 3. 현재 양자화·Linear 흐름에 연결

### Weight 추출

- `eval_utils/gptq_utils.py`의 GPTQ/RTN 모두에서 실제 양자화 당시의 정수 weight, scale, zero, bits, group mapping을 보존한다. Export 옵션에 종속시키지 않는다.
- 현재 `WeightQuantizer.fake_quantize`는 대칭식과 3개 반환값을 사용한다. 비대칭식과 metadata 추출을 함께 정리하되 모든 호출자를 확인해 기존 경로를 보존한다.
- 기존 unsigned 비대칭 정수는 signed 표현으로 변환할 때 weight와 zero-point에 같은 offset을 적용하여 `W-Z`를 유지한다. Zero는 int16/int32에 저장하고 weight 범위로 clamp하지 않는다.
- Scale을 FP16으로 만드는 시점과 그에 따른 오차를 명시한다. 기존 FP32 scale이 FP16과 같다고 가정하지 않는다. 수치 비교에는 동일한 FP16 scale을 쓰는 기준 연산도 제공한다.
- `act_order` 사용 시 정수 weight·scale·zero/group mapping에도 올바른 순열을 적용한다. 원래 K 순서에서 QCOL 그룹이 연속적이지 않게 되는 경우 compact 그룹으로 억지 압축하지 않고, 올바른 mapping 지원 또는 명시적인 미지원 오류를 선택한다.

### Linear 분기

- `ActQuantWrapper.forward`에서 기존 online Hadamard와 activation QDQ 이후 Linear 호출만 교체한다.
- 현재 `online_had_mode`, profiling, activation quantizer, output quantizer 흐름을 보존한다. A8 등은 기존 QDQ를 수행한 FP16 activation으로 FP×INT에 진입한다.
- 이미 회전한 weight를 사용하고, runtime `R1/R2`를 요구하는 학습 경로는 custom inference로 조용히 우회하지 않는다.
- FP weight로 남는 `lm_head` 등은 기존 Linear를 유지한다. 어떤 layer가 emulation/기존 경로를 쓰는지 이름과 이유를 기록한다. 대상 quantized layer의 metadata 부족은 silent fallback하지 않는다.

### 옵션 및 checkpoint

- `utils/process_args.py`, `eval_utils/main.py`, `result_analysis/load_model.py`에 backend(`standard`, `fpint_torch`, 후속 `fpint_cuda`) 및 수치 설정을 전달한다. 기본은 기존 standard 경로다.
- 새 checkpoint에는 정수 metadata와 format version을 저장하고 load 전에 buffer를 준비한다. `.to(device)` 및 state dict round-trip을 검증한다.
- 기존 checkpoint는 먼저 보유 metadata를 검사한다. Dequantized weight만으로 원래 GPTQ 정수값/그룹 scale을 정확히 복구할 수 있다고 가정하지 않는다. 부족하면 FP×INT용 재양자화/checkpoint 재생성 경로를 안내한다.
- `ptq.py`의 평가/결과 저장 기능을 유지한다. 상대 repo의 weight 삭제 코드를 그대로 이식하지 않고 wrapper의 alias와 device 이동을 포함해 메모리 사용을 검토한다.

## 4. 수치 검증

### 합격 기준

- 동일 정수 weight·FP16 scale·zero·하드웨어 설정에서 GPU backend와 일반화 CPU reference를 비교한다.
- `abs(actual-reference) <= atol + rtol*abs(reference)`를 사용한다. 초기 후보는 `atol=1e-3, rtol=1e-3`이며 FP16 출력 및 실제 값 범위를 확인해 구현 최적화 전에 기준을 고정한다. 실패를 숨기기 위해 실행마다 tolerance를 늘리지 않는다.
- 최대 절대오차, 평균/RMS 오차, tolerance 초과 비율을 함께 보고한다. Bit-exact 여부는 합격 요건이 아니다.
- FP64 수학적 GEMM 및 기존 QDQ Linear 대비 오차는 하드웨어 근사 오차 지표로 별도 기록한다. 이 비교가 하드웨어 reference 일치 검증을 대체하지 않는다.

### 테스트 범위

- 기본 `group_size=mxu_rows=32`에서 기존 QCOL_REAL_2SCOMP와 reference 비교.
- `group_size != mxu_rows`, per-channel, MXU 크기 변경, K/N tail, 여러 batch 차원, bias, 비연속 입력.
- INT4/INT8, 대칭/비대칭, nonzero/음수/큰 zero-point, signed 변환 동등성.
- Zero, subnormal, 넓은 exponent 차이, cancellation, 큰 K, weight 최솟값/최댓값.
- GPTQ와 RTN 생성 경로, checkpoint round-trip, 기존 checkpoint 오류 처리, act-order 조합.
- FP×INT 비활성화 시 기존 모델 동작, Hadamard/activation QDQ 보존, quantized layer dispatch coverage.

작은 CPU reference 테스트와 GPU kernel 테스트를 분리한다. 전체 모델 검증은 repo의 로컬 모델/실제 Linear shape를 확인한 뒤 작은 입력으로 시작한다. CPU triple-loop reference를 전체 모델 실행에 사용하지 않는다.

## 5. 성능 실측과 CUDA 구현 단계

### 먼저 제공할 실행 결과

- 재현 가능한 benchmark 스크립트: seed, model/layer shape, bits/group/MXU 설정, GPU·PyTorch 정보 기록.
- 기존 QDQ Linear, PyTorch FP×INT, CUDA 구현 후 CUDA FP×INT를 동일 입력으로 비교.
- Warmup 및 CUDA 동기화 후 반복 latency와 peak GPU memory 측정. Weight 준비 시간과 steady-state forward 시간을 분리한다.
- 작은 M의 decode 형태와 큰 M의 prefill 형태, attention projection/MLP의 대표 K/N을 포함한다. 여기서 decode 형태는 Linear shape 측정을 뜻하며 KV cache 변경은 포함하지 않는다.
- 짧은 모델 실행 시간과 layer별 병목도 제공하여 사용자가 체감 성능을 판단할 수 있게 한다. 수치 검증 이후에 성능을 비교한다.

### 속도가 부족하면 구현할 custom CUDA kernel

- `fpint_emul/csrc/` 및 Python binding을 추가하고 extension은 backend 선택 시 로드한다.
- Prealignment와 integer MAC/zero correction을 GPU kernel로 구현한다. Activation alignment를 여러 output column에서 재사용하고 compact scale/zero를 직접 읽는다.
- M/N tile 병렬화, K tile 순서, partial accumulation, int64 범위를 설계한다. FMA/누적 재배치로 생기는 차이도 allclose 검증에 포함한다.
- Symmetric zero=0 경로와 비대칭 경로를 지원하며 지원하지 않는 dtype/shape/settings를 명시한다.
- PyTorch/CPU reference와 같은 테스트를 실행하고 실제 모델 shape에서 latency·메모리를 재측정한다. 검증을 통과하지 않는 CUDA 구현을 기본 경로로 승격하지 않는다.
- 하드웨어 근사 의미를 없애는 단순 dequantize+GEMM은 custom FP×INT backend로 대체하지 않는다.

## 완료 조건 및 산출물

1. 독립 설정을 받는 QCOL_REAL_2SCOMP reference와 PyTorch backend가 구현되어 있다.
2. 대칭/비대칭 INT4/INT8 quantized Linear를 현재 repo에서 실행하고, 설정·layer coverage를 확인할 수 있다.
3. Reference allclose, quantization metadata 및 checkpoint 검증을 통과한다.
4. 기존 standard 실행과 Q/P/K/V 양자화·Hadamard 동작이 유지된다.
5. 실제 shape 및 작은 모델 실행의 수치·latency·메모리 보고서와 재현 명령이 있다.
6. 사용자가 성능을 판단할 수 있는 PyTorch 실행 경로를 먼저 전달한다. 부족하다고 판단되면 이 계획의 CUDA 단계를 이어 수행하고 같은 검증/보고서를 제공한다.

## 수행 결과 (2026-09-15)

위 완료 조건에 해당하는 reference, Torch backend, custom CUDA backend, quantization/checkpoint 연결, 테스트와 benchmark를 구현했다. 순차 Torch backend의 실제 shape 성능이 부족해 CUDA 단계까지 수행했다. 구현 파일, 수치 범위, A6000/Blackwell 성능, 재현 명령 및 알려진 제한은 [IMPLEMENTATION_RESULTS.md](IMPLEMENTATION_RESULTS.md)에 기록했다.

현재 문서는 계획만 작성한 상태다. Backend 구현, 파일 이식, benchmark 실행은 후속 작업이다.
