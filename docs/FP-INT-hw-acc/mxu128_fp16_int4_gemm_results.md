# MXU ROW 128 FP16×INT4 GEMM accuracy

[문서 목차](README.md)

실행 명령, raw 파일 위치, 이전 결과와의 차이는 [실험 설명](mxu128_scaled_gemm_experiment.md)을 참조한다.

## 설정

- 상태: `pass`
- Shape: M=32, N=32; K=[128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
- K당 30 trials; base seed=20260915
- Activation: sign/exponent/mantissa raw fields independently uniform; exponent min=0
- Signed INT weight: [-8, 7] uniform
- Scale: FP16 independent uniform raw fields: sign [0, 1], exponent [0, 15], mantissa [0, 1023]; output channel × K-group별 독립 sample
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
| 256 | 24 | 30717 / 30720 | 0.99990234 | 0 | 3 | 3 | 3 | 3 |
| 512 | 24 | 30715 / 30720 | 0.99983724 | 0 | 5 | 5 | 5 | 5 |
| 1024 | 23 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 2048 | 23 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4096 | 22 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 8192 | 21 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 16384 | 21 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |
| 32768 | 20 | 30720 / 30720 | 1 | 0 | 0 | 0 | 0 | 0 |

## RMSE

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.1716057 ± 0.36827608 | 1.1716057 ± 0.36827608 | 0.87280702 ± 0.27441921 | 0.74496654 | 0.74496654 |
| 256 | 1.540678 ± 0.44968238 | 1.540678 ± 0.44968238 | 1.139538 ± 0.321087 | 0.73963412 | 0.73963412 |
| 512 | 2.3080585 ± 0.39651016 | 2.3080585 ± 0.39651016 | 1.7570956 ± 0.27434062 | 0.76128729 | 0.76128729 |
| 1024 | 1.6319049 ± 0.19771009 | 1.6319049 ± 0.19771009 | 1.2347228 ± 0.1425207 | 0.7566144 | 0.7566144 |
| 2048 | 2.9143197 ± 0.29280443 | 2.3365455 ± 0.24923665 | 1.7652446 ± 0.20465213 | 0.60571411 | 0.75549334 |
| 4096 | 2.1280672 ± 0.10703771 | 1.6994375 ± 0.10569579 | 1.2754267 ± 0.082421294 | 0.59933575 | 0.75049936 |
| 8192 | 1.5346315 ± 0.075664942 | 1.2309297 ± 0.062129166 | 0.91261165 ± 0.047869309 | 0.59467802 | 0.74140031 |
| 16384 | 2.1558013 ± 0.065597488 | 1.7177075 ± 0.061235513 | 1.2980452 ± 0.053817437 | 0.60211725 | 0.75568465 |
| 32768 | 1.5838991 ± 0.045018354 | 1.2656758 ± 0.037454889 | 0.95298463 ± 0.032817442 | 0.60167005 | 0.75294529 |

## Mean ULP

| K | GPU True mean ± std | GPU False mean ± std | FPINT mean ± std | FPINT / True | FPINT / False |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.9386719 ± 3.1915019 | 1.9386719 ± 3.1915019 | 3.2552083e-05 ± 0.0001782951 | 1.6790919e-05 | 1.6790919e-05 |
| 256 | 2.5557474 ± 5.484092 | 2.5557474 ± 5.484092 | 6.5104167e-05 ± 0.00024776185 | 2.5473631e-05 | 2.5473631e-05 |
| 512 | 4.2752345 ± 8.2251447 | 4.2752345 ± 8.2251447 | 0.0001953125 ± 0.00047288496 | 4.5684628e-05 | 4.5684628e-05 |
| 1024 | 6.7368815 ± 10.627351 | 6.7368815 ± 10.627351 | 0.00035807292 ± 0.00060053506 | 5.3151138e-05 | 5.3151138e-05 |
| 2048 | 3.6734049 ± 6.0957845 | 2.1383789 ± 4.0906993 | 0.00032552083 ± 0.00046822588 | 8.8615559e-05 | 0.00015222785 |
| 4096 | 2.4226237 ± 1.5835564 | 1.6394206 ± 0.83751524 | 0.00061848958 ± 0.00060053506 | 0.00025529742 | 0.00037726108 |
| 8192 | 4.0988932 ± 7.4392396 | 2.5045247 ± 4.7406901 | 0.00068359375 ± 0.00068575513 | 0.0001667752 | 0.0002729435 |
| 16384 | 4.0776693 ± 7.5103579 | 4.1457682 ± 8.1337292 | 0.0018554687 ± 0.0042632988 | 0.00045503169 | 0.00044755728 |
| 32768 | 4.2759115 ± 7.0410006 | 3.4089518 ± 6.7289237 | 0.0015950521 ± 0.0015490424 | 0.00037303207 | 0.00046790103 |

## 전체 오차

| Candidate | Global RMSE | Relative L2 | Max abs | Mean ULP | Max ULP |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GPU True | 1.9689629 | 0.00031483496 | 28.508569 | 3.78379 | 33388 |
| GPU False | 1.7239164 | 0.0002756523 | 28.508569 | 3.2602795 | 32080 |
| FPINT | 1.2978034 | 0.0002075173 | 15.963376 | 0.00063659249 | 23 |

## Correctness / 재현

- Torch vs QCOL reference all-close: 270/270
- CUDA vs QCOL reference all-close: 270/270
- Tolerance: atol=0.001, rtol=0.001
- K별 common finite fraction target: 0.999
- CUDA kernel SHA256: `5e587e94a14bddf37b4050aecb47650696a4c0532a1bfefde80876aed865302d`
- Measurement script SHA256: `2bbfd810995f1b0fa7f431464caab1ab659409680aee3ca6629da3ed9c6e494e`
- 완료 시각 (UTC): 2026-09-22T13:02:53.330381+00:00
- JSON/CSV: case별 seed, 세 candidate 오차, finite coverage; JSON에 scale 범위와 raw-bit SHA256 추가 기록

## 해석 범위

- FP16/BF16은 activation exponent의 실제 하한이 다르므로 포맷 간 동일 실수 입력 비교가 아니다.
- K별 exponent 상한이 달라 K 증가와 입력 분포 변화의 영향을 함께 포함한다.
- Scale 분포와 dtype은 위 설정을 따른다. 포맷 간 동일 실수 scale 비교가 아니다.
- Zero-point=0이므로 zero-point 보정 경로와 reduction extra-bit 효과는 이 실험에서 검증하지 않는다.
- 수치 정확도 실험이며 latency/throughput 결과는 포함하지 않는다.
