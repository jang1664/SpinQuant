# MXU ROW 128 BF16×INT4 GEMM accuracy

실행 명령, raw 파일 위치, 이전 결과와의 차이는 [실험 설명](mxu128_scaled_gemm_experiment.md)을 참조한다.

## 설정

- 상태: `pass`
- Shape: M=32, N=32; K=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
- K당 30 trials; base seed=20260915
- Activation: sign/exponent/mantissa raw fields independently uniform; exponent min=0
- Signed INT weight: [-8, 7] uniform
- Scale: log2(S) ~ Uniform(-4.0, 0.0), S = round_bf16(2^log2(S)); output channel × K-group별 독립 sample
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
| 128 | 136 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 256 | 136 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 512 | 136 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 1024 | 135 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2048 | 135 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 134 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8192 | 133 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 16384 | 133 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 32768 | 132 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |

## RMSE

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 3.6035977 ± 0.41530383 | 3.6035977 ± 0.41530383 | 2.7229753 ± 0.31853724 | 0.75562689 | 0.75562689 |
| 256 | 5.3112855 ± 0.56763191 | 5.3112855 ± 0.56763191 | 3.9943093 ± 0.45999662 | 0.75204191 | 0.75204191 |
| 512 | 7.5961767 ± 0.66188469 | 7.5961767 ± 0.66188469 | 5.7427919 ± 0.51434168 | 0.75601084 | 0.75601084 |
| 1024 | 5.3482983 ± 0.30813946 | 5.3482983 ± 0.30813946 | 4.0254088 ± 0.21846208 | 0.75265225 | 0.75265225 |
| 2048 | 9.529109 ± 0.46500182 | 7.5539393 ± 0.33532076 | 5.7166075 ± 0.32037131 | 0.59990996 | 0.7567717 |
| 4096 | 6.7419387 ± 0.24131615 | 5.363079 ± 0.21777561 | 4.0510748 ± 0.16731761 | 0.60087684 | 0.75536363 |
| 8192 | 4.8312425 ± 0.14420585 | 3.8412435 ± 0.1567113 | 2.9111376 ± 0.11787151 | 0.602565 | 0.75786333 |
| 16384 | 6.7574966 ± 0.16911271 | 5.385962 ± 0.15524876 | 4.037545 ± 0.15289829 | 0.59749124 | 0.74964231 |
| 32768 | 4.7541724 ± 0.11350004 | 3.8127499 ± 0.11211917 | 2.8578225 ± 0.11737636 | 0.60111882 | 0.74954366 |

## Mean ULP

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5.8778646 ± 11.851533 | 5.8778646 ± 11.851533 | 6.5104167e-05 ± 0.00024776185 | 1.107616e-05 | 1.107616e-05 |
| 256 | 14.610091 ± 24.575795 | 14.610091 ± 24.575795 | 3.2552083e-05 ± 0.0001782951 | 2.2280548e-06 | 2.2280548e-06 |
| 512 | 17.083203 ± 23.538372 | 17.083203 ± 23.538372 | 9.765625e-05 ± 0.00029797713 | 5.716507e-06 | 5.716507e-06 |
| 1024 | 8.6271484 ± 13.70836 | 8.6271484 ± 13.70836 | 3.2552083e-05 ± 0.0001782951 | 3.7732147e-06 | 3.7732147e-06 |
| 2048 | 21.999349 ± 27.247959 | 10.765788 ± 17.167377 | 3.2552083e-05 ± 0.0001782951 | 1.4796839e-06 | 3.0236601e-06 |
| 4096 | 19.975977 ± 26.221261 | 19.305176 ± 24.537313 | 3.2552083e-05 ± 0.0001782951 | 1.6295616e-06 | 1.6861842e-06 |
| 8192 | 20.927995 ± 27.258073 | 9.7732422 ± 16.577856 | 6.5104167e-05 ± 0.00024776185 | 3.110865e-06 | 6.6614707e-06 |
| 16384 | 23.731348 ± 24.51091 | 20.3625 ± 24.347707 | 6.5104167e-05 ± 0.00024776185 | 2.7433826e-06 | 3.197258e-06 |
| 32768 | 20.84043 ± 24.676163 | 14.965788 ± 21.67966 | 0.00016276042 ± 0.00037016506 | 7.8098398e-06 | 1.0875499e-05 |

## 전체 오차

| Candidate | Relative L2 | Max abs | Mean ULP | Max ULP |
| :--- | ---: | ---: | ---: | ---: |
| GPU True | 0.0025094513 | 97.112913 | 17.074823 | 33257 |
| GPU False | 0.0021947537 | 97.112913 | 13.485645 | 33188 |
| FPINT | 0.0016562119 | 62.049892 | 6.5104167e-05 | 1 |

## Correctness / 재현

- Torch vs QCOL reference all-close: 270/270
- CUDA vs QCOL reference all-close: 270/270
- Tolerance: atol=0.001, rtol=0.001
- K별 common finite fraction target: 0.999
- CUDA kernel SHA256: `5e587e94a14bddf37b4050aecb47650696a4c0532a1bfefde80876aed865302d`
- Measurement script SHA256: `3ba30003cfdb374393c7d16e479cf1495f22b8cae2ed4bfc7f5ec468121b22de`
- 완료 시각 (UTC): 2026-09-22T12:24:29.768587+00:00
- JSON/CSV: case별 seed, 세 candidate 오차, finite coverage; JSON에 scale 범위와 raw-bit SHA256 추가 기록

## 해석 범위

- FP16/BF16은 activation exponent의 실제 하한이 다르므로 포맷 간 동일 실수 입력 비교가 아니다.
- K별 exponent 상한이 달라 K 증가와 입력 분포 변화의 영향을 함께 포함한다.
- Scale RNG의 원본 샘플은 포맷 간 같지만 FP16/BF16 반올림 후 값은 다르다.
- Zero-point=0이므로 zero-point 보정 경로와 reduction extra-bit 효과는 이 실험에서 검증하지 않는다.
- 수치 정확도 실험이며 latency/throughput 결과는 포함하지 않는다.
