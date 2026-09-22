# MXU ROW 128 BF16×INT4 GEMM accuracy

실행 명령, raw 파일 위치, 이전 결과와의 차이는 [실험 설명](mxu128_scaled_gemm_experiment.md)을 참조한다.

## 설정

- 상태: `pass`
- Shape: M=32, N=32; K=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
- K당 30 trials; base seed=20260915
- Activation: sign/exponent/mantissa raw fields independently uniform; exponent min=0
- Signed INT weight: [-8, 7] uniform
- Scale: BF16 independent uniform raw fields: sign [0, 1], exponent [0, 241], mantissa [0, 127]; output channel × K-group별 독립 sample
- Activation/weight/scale RNG는 case seed의 독립 SeedSequence child stream
- Zero-point=0, bias 없음
- MXU rows/group size=128/128; main/reduction extra bits=19/10
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch 2.7.0+cu128; CUDA 12.8

## 비교 정의

- FP64 reference: A.double() @ (INT.double() × expanded_scale.double()).T
- GPU baseline: A @ cast_BF16(INT.float() × expanded_scale.float()).T
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
| 128 | 136 | 30714 / 30720 | 0.99980469 | 0 | 6 | 6 | 6 | 6 |
| 256 | 136 | 30715 / 30720 | 0.99983724 | 0 | 5 | 5 | 5 | 5 |
| 512 | 136 | 30714 / 30720 | 0.99980469 | 0 | 5 | 6 | 6 | 5 |
| 1024 | 135 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2048 | 135 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 134 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8192 | 133 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 16384 | 133 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 32768 | 132 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |

## RMSE

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.1552535e+34 ± 1.6654381e+34 | 1.1552535e+34 ± 1.6654381e+34 | 8.6430061e+33 ± 1.2928261e+34 | 0.748148 | 0.748148 |
| 256 | 1.7934524e+34 ± 2.0224785e+34 | 1.7934524e+34 ± 2.0224785e+34 | 1.2012217e+34 ± 1.2476166e+34 | 0.66978177 | 0.66978177 |
| 512 | 2.8489471e+34 ± 2.2707759e+34 | 2.8489471e+34 ± 2.2707759e+34 | 1.9909004e+34 ± 1.5103567e+34 | 0.69881973 | 0.69881973 |
| 1024 | 2.7222693e+34 ± 1.2920379e+34 | 2.7222693e+34 ± 1.2920379e+34 | 2.0619846e+34 ± 9.4213376e+33 | 0.75745061 | 0.75745061 |
| 2048 | 4.8899145e+34 ± 1.7108826e+34 | 3.9739133e+34 ± 1.2512372e+34 | 2.9628886e+34 ± 8.6821549e+33 | 0.60591829 | 0.74558463 |
| 4096 | 3.9235397e+34 ± 9.8126527e+33 | 3.0541903e+34 ± 7.1461281e+33 | 2.2895394e+34 ± 5.4106939e+33 | 0.58353925 | 0.74963874 |
| 8192 | 2.4536516e+34 ± 4.231395e+33 | 2.019827e+34 ± 3.3531153e+33 | 1.5051588e+34 ± 2.4132246e+33 | 0.61343621 | 0.74519193 |
| 16384 | 3.6110399e+34 ± 4.9096225e+33 | 2.8993425e+34 ± 3.6484779e+33 | 2.2086924e+34 ± 2.8340695e+33 | 0.61164997 | 0.7617908 |
| 32768 | 2.6938458e+34 ± 2.4474289e+33 | 2.1564038e+34 ± 1.8878556e+33 | 1.622925e+34 ± 1.6384262e+33 | 0.60245652 | 0.75260717 |

## Mean ULP

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 10.62239 ± 22.894687 | 10.62239 ± 22.894687 | 0 ± 0 | 0 | 0 |
| 256 | 7.415401 ± 13.496365 | 7.415401 ± 13.496365 | 0 ± 0 | 0 | 0 |
| 512 | 3.667024 ± 11.481829 | 3.667024 ± 11.481829 | 0 ± 0 | 0 | 0 |
| 1024 | 16.501953 ± 31.879924 | 16.501953 ± 31.879924 | 0 ± 0 | 0 | 0 |
| 2048 | 23.881738 ± 34.99299 | 9.7626302 ± 20.52863 | 0 ± 0 | 0 | 0 |
| 4096 | 22.67194 ± 32.649239 | 22.129525 ± 35.727327 | 3.2552083e-05 ± 0.0001782951 | 1.4357873e-06 | 1.4709798e-06 |
| 8192 | 32.359408 ± 43.171028 | 16.54834 ± 30.771831 | 3.2552083e-05 ± 0.0001782951 | 1.0059542e-06 | 1.9670906e-06 |
| 16384 | 13.053809 ± 28.89691 | 6.795931 ± 18.086314 | 0 ± 0 | 0 | 0 |
| 32768 | 29.172982 ± 53.650939 | 20.798437 ± 32.47147 | 3.2552083e-05 ± 0.0001782951 | 1.1158298e-06 | 1.5651216e-06 |

## 전체 오차

| Candidate | Global RMSE | Relative L2 | Max abs | Mean ULP | Max ULP |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GPU True | 3.3853942e+34 | 0.0025910052 | 1.257838e+36 | 17.706232 | 62211 |
| GPU False | 2.9408015e+34 | 0.0022507369 | 1.2492397e+36 | 12.694256 | 61833 |
| FPINT | 2.14875e+34 | 0.0016445418 | 6.5842369e+35 | 1.0851362e-05 | 1 |

## Correctness / 재현

- Torch vs QCOL reference all-close: 270/270
- CUDA vs QCOL reference all-close: 270/270
- Tolerance: atol=0.001, rtol=0.001
- K별 common finite fraction target: 0.999
- CUDA kernel SHA256: `5e587e94a14bddf37b4050aecb47650696a4c0532a1bfefde80876aed865302d`
- Measurement script SHA256: `2bbfd810995f1b0fa7f431464caab1ab659409680aee3ca6629da3ed9c6e494e`
- 완료 시각 (UTC): 2026-09-22T13:03:05.510813+00:00
- JSON/CSV: case별 seed, 세 candidate 오차, finite coverage; JSON에 scale 범위와 raw-bit SHA256 추가 기록

## 해석 범위

- FP16/BF16은 activation exponent의 실제 하한이 다르므로 포맷 간 동일 실수 입력 비교가 아니다.
- K별 exponent 상한이 달라 K 증가와 입력 분포 변화의 영향을 함께 포함한다.
- Scale 분포와 dtype은 위 설정을 따른다. 포맷 간 동일 실수 scale 비교가 아니다.
- Zero-point=0이므로 zero-point 보정 경로와 reduction extra-bit 효과는 이 실험에서 검증하지 않는다.
- 수치 정확도 실험이며 latency/throughput 결과는 포함하지 않는다.
