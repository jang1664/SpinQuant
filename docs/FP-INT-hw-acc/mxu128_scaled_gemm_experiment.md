# FP16/BF16 × INT4 GEMM — 최종 실험 설정

2026-09-22 마지막 scale exponent 실험에서 모든 K의 common finite 비율이 99.9% 이상인 설정을 사용한다.
Activation exponent 상한은 고정하며, 아래 범위는 모두 IEEE 저장 exponent field 값이다.

## Scale

| Dtype | Sign field | Exponent field | Mantissa field |
| :--- | :--- | :--- | :--- |
| FP16 | uniform {0,1} | uniform integer [0,15] | uniform integer [0,1023] |
| BF16 | uniform {0,1} | uniform integer [0,241] | uniform integer [0,127] |

- Sign/exponent/mantissa는 독립적으로 샘플링한다.
- Activation과 scale의 dtype은 같고, output channel × K-group별로 scale을 생성한다.
- Scale shape: [N, ceil(K/128)]. 음수·양수·0·subnormal을 포함한다.
- 정상수의 실제 지수는 field−bias이며 FP16 bias=15, BF16 bias=127이다. Field 0은 zero/subnormal이다.
- FP16 scale의 절댓값은 2 미만, BF16은 2^115 미만이다.

## Activation

Sign은 uniform {0,1}, mantissa는 FP16 [0,1023] / BF16 [0,127]의 integer uniform이다.
Exponent도 아래 범위의 integer uniform으로 독립 샘플링한다.

| K | FP16 EXP field | BF16 EXP field |
| ---: | :--- | :--- |
| 128 | [0,24] | [0,136] |
| 256 | [0,24] | [0,136] |
| 512 | [0,24] | [0,136] |
| 1024 | [0,23] | [0,135] |
| 2048 | [0,23] | [0,135] |
| 4096 | [0,22] | [0,134] |
| 8192 | [0,21] | [0,133] |
| 16384 | [0,21] | [0,133] |
| 32768 | [0,20] | [0,132] |

## GEMM 및 비교 조건

- M=N=32; K당 30 trials; base seed=20260915
- Signed INT4 weight: uniform integer [-8,7]
- Zero-point=0; bias 없음
- MXU rows=128; group size=128; main/reduction extra bits=19/10
- FP64 reference: A.double() @ (Q.double() × expanded_scale.double()).T
- GPU baseline: A @ cast_dtype(Q.float() × expanded_scale.float()).T
- GPU reduced-precision reduction=True/False를 동일 입력에서 각각 측정하고 원래 flag를 복원한다.
- FPINT: QCOL_REAL_2SCOMP CUDA, tile별 정수 누적 후 FP32 복원·scale 곱·K 순서 누적, 최종 dtype cast
- Finite 비율의 분모는 전체 출력 수다. Common finite는 FP64 reference, rounded reference, 두 baseline, FPINT의 교집합이다.
- 이 설정에서 측정한 지표는 finite/Inf/NaN 비율이다. RMSE/ULP 결과는 포함하지 않는다.

## 결과

| Dtype | Scale EXP field | 전체 common finite | 최악 K의 common finite |
| :--- | :--- | ---: | ---: |
| FP16 | [0,15] | 99.9971% | 99.9837% |
| BF16 | [0,241] | 99.9939% | 99.9805% |

[FP16 상세 결과](mxu128_fp16_int4_gemm_results.md) · [BF16 상세 결과](mxu128_bf16_int4_gemm_results.md)

## 재현

저장소 루트에서 spinquant 환경을 사용한다. 각 명령은 해당 최종 scale 범위만 측정한다.

```bash
conda activate spinquant
python measure_fpint_scale_sweep.py \
  --activation-format fp16 --scale-exp-min 0 --scale-exp-max-values 15 \
  --trials 30 --base-seed 20260915 --device cuda:0 \
  --output results/fpint-final-setting/fp16.json

python measure_fpint_scale_sweep.py \
  --activation-format bf16 --scale-exp-min 0 --scale-exp-max-values 241 \
  --trials 30 --base-seed 20260915 --device cuda:0 \
  --output results/fpint-final-setting/bf16.json
```

Scale의 exponent는 동일 U~Uniform[0,1)에서 floor(U×(Emax+1))로 생성한다.
같은 K/trial의 activation·weight와 scale sign/mantissa는 exponent 범위에 관계없이 동일하다.
위 명령은 마지막 sweep의 해당 설정을 재현한다. K 목록·순서는 유지해야 한다.
