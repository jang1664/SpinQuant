# berlin1 Blackwell 환경 설치 및 smoke test

2026-09-15 실행. 환경 설치와 최초 smoke test 후, 이식한 FP×INT Linear CUDA kernel의 SM120 build, 수치 테스트, 실제 projection shape 및 tiny-model benchmark까지 완료했다. 전체 3B/7B checkpoint 평가는 수행하지 않았다.

## 실행 방법

```bash
ssh berlin1
source /home/tools/anaconda3/etc/profile.d/conda.sh
conda activate spinquant
cd /home/jaeyong.jang/spinquant-setup.fP5oR2/source
python berlin1-smoke.py
```

환경 경로는 `/home/jaeyong.jang/.conda/envs/spinquant`다. `spinquant` 이름으로 활성화할 수 있다. 서버에 전달한 소스는 환경 검증용 snapshot이며, 모델 checkpoint는 복사하지 않았다. 다른 모델 실행에는 해당 소스/모델 경로를 별도로 준비해야 한다.

## 설치 내역

- Python 3.10.16, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, torchaudio 2.7.0+cu128.
- CUDA toolkit 12.8, nvcc 12.8.93을 새 conda 환경에 설치했다. 서버의 CUDA 13.0은 변경하지 않았다.
- Host compiler는 `/usr/bin/gcc`, `/usr/bin/g++` 13.3.0을 사용한다.
- CUDA/PyTorch 및 로컬 Hadamard를 제외한 Python 패키지는 원본 환경 export의 버전을 유지했다. `pip check` 통과.
- 설치된 PyTorch arch 목록: `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120 compute_120`.
- RTX PRO 6000 Blackwell Server Edition GPU 4개, 각 97887 MiB, compute capability 12.0. Driver 580.126.20.
- `CUDA_HOME`은 conda 환경 root, `CPATH`는 그 아래 `targets/x86_64-linux/include`로 설정했다. 환경 활성화 시 기존 시스템 CUDA_HOME을 덮어쓴다는 경고는 의도된 환경 선택이다.
- `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=2`를 환경에 저장했다. Conda compiler activation 이후 시스템 compiler를 사용하도록 `zz-spinquant-compiler.sh` hook을 추가했다.

## Hadamard 재빌드

현재 repo의 `fast-hadamard-transform` 소스를 검증용 원격 폴더에 복사했다. 원본 setup의 고정 GPU 목록을 `compute_120,code=sm_120`으로 교체하고 `FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE`로 실제 source build를 수행했다.

```bash
cd /home/jaeyong.jang/spinquant-setup.fP5oR2/source/fast-hadamard-transform
FAST_HADAMARD_TRANSFORM_FORCE_BUILD=TRUE python -m pip install --no-build-isolation --no-deps .
```

설치 버전: 1.0.4.post1. Wheel SHA256: `6d39058824cd41865d9aabe91110b5758d0aedadceadc9f97286d5ba8a6502b7`.
빌드 로그: `/home/jaeyong.jang/spinquant-setup.fP5oR2/hadamard-build.log`.
원래 로컬 repo의 Hadamard setup.py는 변경하지 않았다.

## 검증 결과

| 검사 | 결과 |
| --- | --- |
| Python 패키지 의존성 | `No broken requirements found` |
| transformers / accelerate / datasets | 4.44.2 / 0.34.2 / 4.8.5 import 성공 |
| `eval_utils.main`, `result_analysis.load_model` | import 성공 |
| GPU 0–3 FP16 matmul | 모두 finite 출력 |
| GPU 0–3 Hadamard transform | 모두 실행 및 두 번 적용한 결과의 입력 대비 allclose 통과 (`atol=rtol=0.004`) |
| Custom CUDA extension | SM120 대상으로 컴파일·링크·로드 성공, non-default current stream에서 결과 일치 |
| INT4 symmetric QCOL | reference allclose 통과, 최대 절대차 0 |
| INT4 asymmetric QCOL | reference allclose 통과, 최대 절대차 0 |
| INT8 symmetric QCOL | reference allclose 통과, 최대 절대차 0 |
| INT8 asymmetric QCOL | reference allclose 통과, 최대 절대차 0 |

QCOL은 전달한 `SpinQuant_fpint/utils/figna_utils.py`를 `figna_source.py`로 import했다. 비교 전 모듈의 `MXU_K`를 기준 코드에 맞춰 32로 설정했다. CPU reference는 제공된 `fpint_emul/py/fpint_emul.py::fpint_gemm_qcol_real_2scomp`다. Shape `(M,K,N)=(4,128,32)`, group size 32, seed 0, `atol=rtol=1e-3`이다. 이 4개 입력에서의 일치는 모든 shape·값 범위에 대한 일반적인 정확성 보장은 아니다.

최초 Custom CUDA smoke는 간단한 덧셈 커널로 extension 빌드 환경과 stream 연동을 검사했다. 이후 실제 `fpint_emul/csrc` kernel도 SM120으로 build했다. 빌드 중 CC/NVCC_CCBIN의 compiler-bindir 중복 경고가 있었으나 실제 GCC 13.3 빌드와 실행은 통과했다.

## 간단한 Linear 형태 timing

GPU 0, FP16 입력 `(32,4096)`, INT4 weight `(4096,4096)`, zero=0, scale=0.01, MXU/group size=32. 한 번 warmup 후 10회 실행의 host wall time을 CUDA synchronize로 감싸 평균했다.

| 실행 경로 | 평균 ms |
| --- | ---: |
| 미리 dequantize한 weight의 FP16 matmul | 0.02435 |
| 매번 dequantize + FP16 matmul | 0.06341 |
| 기존 PyTorch FP×INT QCOL | 0.56337 |

모든 출력은 finite였다. 세 번째 경로는 기존 소스의 zero=0 FP32 collapsed fast path를 사용한다. 일반화 reference의 순차 FP32 누적 경로 성능을 뜻하지 않는다. 이 표는 환경 준비 단계의 초기 측정이며 아래 최종 kernel benchmark와 구분한다.

## 이식한 FP×INT CUDA 최종 검증

`fpint_emul/csrc/fpint_cuda_kernel.cu`를 conda CUDA 12.8의 `sm_120`으로 JIT build했다. 최종 `PYTHONPATH=. pytest -q tests/test_fpint_emul.py` 결과는 `43 passed, 1 warning in 5.52s`였다. 이 실행은 INT4/INT8 × symmetric/asymmetric, K/N tail, 독립 group/MXU 크기, per-channel/큰-K, subnormal/큰 zero, PTQ checkpoint round-trip, current stream과 GPU 1 device guard를 포함한다.

새 kernel의 `(M,K,N)=(1,3072,3072)` latency는 0.234 ms, `(32,3072,3072)`는 0.370 ms였다. 추가 peak allocation은 각각 출력 크기인 6,144/196,608 bytes였다. 1-layer tiny Llama의 7개 projection을 FPINT CUDA로 실행한 latency는 1.281 ms였고 출력은 finite였다. 상세 수치와 원본 JSON은 [IMPLEMENTATION_RESULTS.md](IMPLEMENTATION_RESULTS.md), [blackwell-fpint-symmetric.json](blackwell-fpint-symmetric.json)에 있다.

## 보관 파일

- [원본 환경](spinquant-source-environment.yml), [berlin1 설치 결과](spinquant-berlin1-environment.yml): 환경 snapshot. GPU 특화 build 설정과 로컬 Hadamard 재빌드까지 포함하는 자동 설치 파일은 아니다.
- [Python 패키지 pin](berlin1-requirements.txt), [Hadamard setup](berlin1-hadamard-setup.py), [compiler activation hook](berlin1-activate.sh).
- [실행 스크립트](berlin1-smoke.py), [실행 로그](berlin1-smoke.log).
- 서버 작업 폴더: `/home/jaeyong.jang/spinquant-setup.fP5oR2`.
