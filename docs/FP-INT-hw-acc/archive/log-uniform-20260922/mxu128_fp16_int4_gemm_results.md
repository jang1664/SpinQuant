# MXU ROW 128 FP16×INT4 GEMM accuracy

실행 명령, raw 파일 위치, 이전 결과와의 차이는 [실험 설명](mxu128_scaled_gemm_experiment.md)을 참조한다.

## 설정

- 상태: `pass`
- Shape: M=32, N=32; K=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
- K당 30 trials; base seed=20260915
- Activation: sign/exponent/mantissa raw fields independently uniform; exponent min=0
- Signed INT weight: [-8, 7] uniform
- Scale: log2(S) ~ Uniform(-4.0, 0.0), S = round_fp16(2^log2(S)); output channel × K-group별 독립 sample
- Activation/weight/scale RNG는 case seed의 독립 SeedSequence child stream
- Zero-point=0, bias 없음
- MXU rows/group size=128/128; main/reduction extra bits=19/10
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch 2.7.0+cu128; CUDA 12.8

## 비교 정의

- FP64 reference: A.double() @ (INT.double() × expanded_scale.double()).T
- GPU baseline: A @ cast_FP16(INT.float() × expanded_scale.float()).T
- 동일 A/INT/scale에서 reduced-precision reduction=True와 False를 각각 명시적으로 설정
- 각 baseline 호출 후 원래 reduction 설정 복원; True는 reduced reduction을 허용한다는 뜻이며 실제 사용은 GPU/kernel에 의존
- FPINT: QCOL_REAL_2SCOMP; MXU tile의 정수 누적을 FP32로 복원하고 scale을 FP32로 곱한 뒤 K 순서로 FP32 누적
- FP64 reference, rounded reference, baseline True/False, FPINT가 모두 finite인 하나의 공통 mask 사용
- RMSE: unrounded FP64 reference 기준; ULP: dtype으로 반올림한 FP64 reference와의 representable-value distance
- Baseline 오차에는 dequantized weight의 dtype 반올림 오차도 포함
- 아래 mean ± std는 30개 trial metric의 평균과 sample std(ddof=1)

## Finite coverage

| K | EXP max | Common / total | Fraction | FP64 nonfinite | Rounded ref nonfinite | GPU True nonfinite | GPU False nonfinite | FPINT nonfinite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 24 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 256 | 24 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 512 | 24 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1024 | 23 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2048 | 23 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 22 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8192 | 21 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 16384 | 21 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 32768 | 20 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |

## RMSE

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.070233 ± 0.15118842 | 1.070233 ± 0.15118842 | 0.8011113 ± 0.098023094 | 0.74853914 | 0.74853914 |
| 256 | 1.560086 ± 0.15707707 | 1.560086 ± 0.15707707 | 1.1704314 ± 0.11721277 | 0.75023519 | 0.75023519 |
| 512 | 2.2913563 ± 0.20251964 | 2.2913563 ± 0.20251964 | 1.6945718 ± 0.13042928 | 0.73954965 | 0.73954965 |
| 1024 | 1.5941373 ± 0.081303835 | 1.5941373 ± 0.081303835 | 1.2068774 ± 0.062346276 | 0.75707245 | 0.75707245 |
| 2048 | 2.8625277 ± 0.12884049 | 2.306018 ± 0.10778129 | 1.7476804 ± 0.10116913 | 0.61053747 | 0.75787805 |
| 4096 | 2.0646668 ± 0.07415168 | 1.6470743 ± 0.066304038 | 1.2441134 ± 0.056470929 | 0.60257343 | 0.75534745 |
| 8192 | 1.5003517 ± 0.045127974 | 1.18763 ± 0.041423308 | 0.89185615 ± 0.035450248 | 0.59443139 | 0.75095458 |
| 16384 | 2.0823972 ± 0.034505089 | 1.6613249 ± 0.033578966 | 1.2391756 ± 0.032556481 | 0.59507167 | 0.74589598 |
| 32768 | 1.5010416 ± 0.041959648 | 1.2079891 ± 0.044476406 | 0.90851284 ± 0.039106298 | 0.60525493 | 0.75208692 |

## Mean ULP

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.4842122 ± 6.9204691 | 3.4842122 ± 6.9204691 | 6.5104167e-05 ± 0.00024776185 | 1.8685477e-05 | 1.8685477e-05 |
| 256 | 2.5217122 ± 5.8620167 | 2.5217122 ± 5.8620167 | 0.00016276042 ± 0.00045032523 | 6.4543612e-05 | 6.4543612e-05 |
| 512 | 4.3710286 ± 9.0410941 | 4.3710286 ± 9.0410941 | 0.00045572917 ± 0.00066547401 | 0.00010426131 | 0.00010426131 |
| 1024 | 2.3064779 ± 5.5759963 | 2.3064779 ± 5.5759963 | 0.00026041667 ± 0.00043923481 | 0.00011290664 | 0.00011290664 |
| 2048 | 3.4105469 ± 4.5881806 | 2.6802409 ± 5.0566449 | 0.00052083333 ± 0.00061407249 | 0.00015271256 | 0.00019432333 |
| 4096 | 3.3332357 ± 4.9684789 | 2.5805013 ± 4.9227241 | 0.00055338542 ± 0.00071087113 | 0.00016602049 | 0.0002144488 |
| 8192 | 7.6044271 ± 12.339938 | 5.2161133 ± 10.817571 | 0.0029622396 ± 0.0098710744 | 0.00038954145 | 0.0005679017 |
| 16384 | 4.9503906 ± 8.406268 | 3.4473307 ± 7.0653225 | 0.001171875 ± 0.0013905599 | 0.00023672374 | 0.00033993692 |
| 32768 | 4.5285807 ± 7.3708101 | 3.4987956 ± 7.1005626 | 0.0020182292 ± 0.0015795252 | 0.00044566483 | 0.00057683541 |

## 전체 오차

| Candidate | Relative L2 | Max abs | Mean ULP | Max ULP |
| :--- | ---: | ---: | ---: | ---: |
| GPU True | 0.00031685795 | 17.655831 | 4.0567347 | 32092 |
| GPU False | 0.00027747756 | 17.655831 | 3.345157 | 32092 |
| FPINT | 0.00020821677 | 15.909605 | 0.00090784144 | 56 |

## Correctness / 재현

- Torch vs QCOL reference all-close: 270/270
- CUDA vs QCOL reference all-close: 270/270
- Tolerance: atol=0.001, rtol=0.001
- K별 common finite fraction target: 0.999
- CUDA kernel SHA256: `5e587e94a14bddf37b4050aecb47650696a4c0532a1bfefde80876aed865302d`
- Measurement script SHA256: `3ba30003cfdb374393c7d16e479cf1495f22b8cae2ed4bfc7f5ec468121b22de`
- 완료 시각 (UTC): 2026-09-22T12:24:18.014652+00:00
- JSON/CSV: case별 seed, 세 candidate 오차, finite coverage; JSON에 scale 범위와 raw-bit SHA256 추가 기록

## 해석 범위

- FP16/BF16은 activation exponent의 실제 하한이 다르므로 포맷 간 동일 실수 입력 비교가 아니다.
- K별 exponent 상한이 달라 K 증가와 입력 분포 변화의 영향을 함께 포함한다.
- Scale RNG의 원본 샘플은 포맷 간 같지만 FP16/BF16 반올림 후 값은 다르다.
- Zero-point=0이므로 zero-point 보정 경로와 reduction extra-bit 효과는 이 실험에서 검증하지 않는다.
- 수치 정확도 실험이며 latency/throughput 결과는 포함하지 않는다.
