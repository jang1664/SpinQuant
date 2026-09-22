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
- RMSE는 공통 finite mask에서 unrounded FP64 reference 대비 계산한다. ULP는 FP64 reference를 출력 dtype으로 반올림한 값과의 representable-value distance다.
- Baseline 오차에는 INT×scale을 weight dtype으로 반올림하는 오차도 포함한다.

## 결과

| Dtype | Scale EXP field | 전체 common finite | 최악 K의 common finite |
| :--- | :--- | ---: | ---: |
| FP16 | [0,15] | 99.9971% | 99.9837% |
| BF16 | [0,241] | 99.9939% | 99.9805% |

[FP16 상세 결과](../../mxu128_fp16_int4_gemm_results.md) · [BF16 상세 결과](mxu128_bf16_int4_gemm_results.md)

아래 Global RMSE는 모든 K/trial의 공통 finite 원소를 합쳐 `sqrt(sum(error²)/count)`로 계산했다.
K별 trial RMSE의 mean ± sample std와 ULP는 상세 결과에 기록했다.

| Dtype | GPU True RMSE | GPU False RMSE | FPINT RMSE |
| :--- | ---: | ---: | ---: |
| FP16 | 1.9689629 | 1.7239164 | 1.2978034 |
| BF16 | 3.3853942e34 | 2.9408015e34 | 2.1487500e34 |

| Dtype | GPU True relative L2 | GPU False relative L2 | FPINT relative L2 |
| :--- | ---: | ---: | ---: |
| FP16 | 0.00031483496 | 0.00027565230 | 0.00020751730 |
| BF16 | 0.0025910052 | 0.0022507369 | 0.0016445418 |

BF16은 scale EXP field 상한 241을 사용하므로 절대 출력 크기와 RMSE가 크다.
FP16과 동일 실수 scale 분포가 아니며 절대 RMSE를 포맷 간 직접 비교하지 않는다.

- 각 dtype 270 cases 모두 측정했다. Torch/CUDA 각각 독립 QCOL reference와 270/270 cases에서 정확히 일치했다.
- 540 cases의 activation/weight 및 scale SHA256, common finite 개수가 마지막 sweep의 해당 설정과 일치함을 확인했다.
- 관련 테스트: 97 passed, 1 skipped. GPU 하나만 노출하여 multi-GPU 테스트는 skip됐다.
- [FP16 raw JSON](../../../../results/fpint-final-setting-accuracy/fp16-int4-scaled.json), [CSV](../../../../results/fpint-final-setting-accuracy/fp16-int4-scaled.csv)
- [BF16 raw JSON](../../../../results/fpint-final-setting-accuracy/archive-exp241/bf16-int4-scaled.json), [CSV](../../../../results/fpint-final-setting-accuracy/archive-exp241/bf16-int4-scaled.csv)

## 재현

저장소 루트에서 spinquant 환경을 사용한다. 두 포맷의 정확도 측정과 결과 문서 생성을 함께 실행한다.

```bash
conda activate spinquant
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash scripts/run_fpint_scaled_gemm.sh
```

개별 포맷의 최종 scale 범위만 측정하려면:

```bash
conda activate spinquant
python measure_fpint_qcol_accuracy.py \
  --activation-format fp16 --scale-mode raw-fields --scale-exp-min 0 --scale-exp-max 15 \
  --trials 30 --base-seed 20260915 --device cuda:0 \
  --output results/fpint-final-setting-accuracy/fp16-int4-scaled.json

python measure_fpint_qcol_accuracy.py \
  --activation-format bf16 --scale-mode raw-fields --scale-exp-min 0 --scale-exp-max 241 \
  --trials 30 --base-seed 20260915 --device cuda:0 \
  --output results/fpint-final-setting-accuracy/bf16-int4-scaled.json
```

Scale의 exponent는 동일 U~Uniform[0,1)에서 floor(U×(Emax+1))로 생성한다.
같은 K/trial의 activation·weight와 scale sign/mantissa는 exponent 범위에 관계없이 동일하다.
위 명령은 마지막 sweep의 해당 설정을 재현한다. K 목록·순서는 유지해야 한다.
