# FP16 × INT4 GEMM — 최종 설정 결과

[공통 실험 설정](mxu128_scaled_gemm_experiment.md)

## 설정

- Activation / scale dtype: FP16 / FP16
- Scale exponent field: discrete uniform [0, 15]
- Scale sign: uniform {0,1}; mantissa: uniform [0, 1023]
- Scale은 output channel × K-group별 샘플링; 음수·양수·0·subnormal 포함
- Activation: 기존 K별 exponent 상한 고정; sign/exponent/mantissa independent uniform
- Signed INT4 weight [-8,7] uniform; zero-point=0; bias 없음
- M=N=32; MXU rows=128; group size=128; extra bits=19/10
- K당 30 trials; base seed=20260915
- GPU baseline reduced-precision reduction=True/False 각각 측정

## Finite 비율

전체 출력 수를 분모로 집계했다. Common은 FP64, rounded reference, GPU True/False, FPINT의 finite 교집합이다.

| K | Activation EXP field | Rounded ref finite | GPU True finite | GPU False finite | FPINT finite | Common finite |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: |
| 128 | [0, 24] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 256 | [0, 24] | 99.9902% | 99.9902% | 99.9902% | 99.9902% | 99.9902% |
| 512 | [0, 24] | 99.9837% | 99.9837% | 99.9837% | 99.9837% | 99.9837% |
| 1024 | [0, 23] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 2048 | [0, 23] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 4096 | [0, 22] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 8192 | [0, 21] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 16384 | [0, 21] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |
| 32768 | [0, 20] | 100.0000% | 100.0000% | 100.0000% | 100.0000% | 100.0000% |

모든 K에서 common finite ≥99.9%를 만족했다. 최악 K는 99.9837%다.

## 검증 및 원본

- FP64 reference는 모든 원소가 finite였다.
- 이 설정의 CUDA/QCOL reference 검사: 9/9 cases (각 K의 첫 trial).
- Finite 값은 exact equality, NaN 위치와 Inf 부호는 일치 여부를 검사했다.
- 이 설정에서는 finite coverage를 측정했다. RMSE/ULP는 아직 측정하지 않았다.
- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition; PyTorch 2.7.0+cu128; CUDA 12.8
- 완료 UTC: 2026-09-22T12:40:53.684582+00:00
- [Raw JSON](../../results/fpint-scale-exponent-sweep/fp16.json), [CSV](../../results/fpint-scale-exponent-sweep/fp16.csv): scale_exp_max=15인 records를 사용했다.
- 이 결과는 지정된 seed/shape/분포의 관측값이며 모든 입력에 대한 finite 보장은 아니다.
