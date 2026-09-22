# FP16/BF16 × INT4 GEMM: sampled scale, reduction True/False

2026-09-22, `spinquant` conda 환경에서 실행했다. FP16 activation + FP16 scale,
BF16 activation + BF16 scale을 각각 측정했다. 두 실험 모두 **PASS**다.

후속 [scale exponent sweep](../../mxu128_scale_exponent_sweep.md)은 activation 상한을
고정하고 scale의 sign/exponent/mantissa를 uniform raw field로 샘플링한다.
이 문서의 양수 log-uniform 설정과 구분하며, 후속 실험에서는 음수와 0 scale도 포함한다.

## 설정

| 항목 | 값 |
| :--- | :--- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, physical GPU 1 |
| Software | PyTorch 2.7.0+cu128, CUDA 12.8, NVIDIA driver 580.126.20 |
| M / N | 32 / 32 |
| K | 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 |
| Trials | 각 K당 30; dtype당 270 cases, 276,480 output elements |
| Activation | 기존 raw sign/exponent/mantissa independent uniform sampler 유지 |
| Weight | signed INT4 [-8, 7] uniform |
| Scale | 양수; log2(S) ~ Uniform(-4, 0), 즉 [2^-4, 1] log-uniform |
| Scale dtype | FP16 activation이면 FP16, BF16 activation이면 BF16 |
| Scale granularity | output channel × K-group, shape [N, ceil(K/128)] |
| Scale 생성 | FP64에서 원본 샘플 생성 후 scale dtype으로 반올림 |
| Zero-point / bias | 0 / 없음 |
| MXU rows / group size | 128 / 128 |
| Main / reduction extra bits | 19 / 10 |
| Base seed | 20260915 |
| Finite target | K별 공통 finite 비율 ≥0.999; 실제 양쪽 실험 모두 1.0 |

FP16 exponent field 상한은 K 순서대로 `24,24,24,23,23,22,21,21,20`,
BF16은 `136,136,136,135,135,134,133,133,132`다. 하한은 모두 field 0이다.
Mantissa field는 FP16 `0..1023`, BF16 `0..127`이다.

Activation, weight, scale은 `SeedSequence(case_seed)`의 첫째/둘째/셋째 child
RNG를 사용한다. 두 dtype에서 scale의 원본 샘플은 동일하지만 dtype별 반올림
후 값은 다르다. 기존 `base_seed + case_index` 방식이므로 K 목록이나 순서를
바꾸면 같은 K/trial의 seed도 달라진다. 재현하려면 K 목록과 순서를 유지한다.

## 계산과 측정

동일한 activation A, integer weight Q, scale S에 대해 다음을 계산한다.

1. **FP64 reference:** `A.double() @ (Q.double() * S_expanded.double()).T`.
2. **GPU True:** `A @ cast_dtype(Q.float() * S_expanded.float()).T`, 해당 dtype의
   `allow_*_reduced_precision_reduction=True`.
3. **GPU False:** GPU True와 같은 입력·dequantization,
   `allow_*_reduced_precision_reduction=False`.
4. **FPINT:** MXU tile별 정수 누적 → FP32 복원 → FP32 scale 곱 → K 순서로 FP32 누적
   → activation dtype 출력.

Baseline 호출마다 reduction flag를 명시적으로 설정하고, 호출 후 원래 값을
복원한다. True는 reduced reduction을 **허용**하는 설정이다. 실제 사용 여부는
GPU와 선택된 GEMM kernel에 의존한다. False에서도 출력 dtype은 FP16/BF16이다.

RMSE는 반올림 전 FP64 reference, ULP는 FP64 reference를 출력 dtype으로
반올림한 값 기준이다. FP64 reference, rounded reference, GPU True, GPU False,
FPINT가 모두 finite인 하나의 mask를 세 candidate에 동일하게 적용한다.
이번 실행은 제외된 원소가 없다. 표의 trial std는 `ddof=1`이다.

이 baseline은 기존 `fpint_linear(..., backend="standard")`의 dequantization
순서와 같다. 따라서 **baseline 오차에는 Q×S를 FP16/BF16 weight로 반올림하는
오차도 포함**한다. FPINT와의 차이를 accumulation 정밀도 차이만으로 해석하지 않는다.

## 결과

아래 relative L2는 해당 dtype의 모든 K/trial 원소를 합쳐 계산했다.

| Activation / scale | GPU True relative L2 | GPU False relative L2 | FPINT relative L2 |
| :--- | ---: | ---: | ---: |
| FP16 / FP16 | 0.00031685795 | 0.00027747756 | 0.00020821677 |
| BF16 / BF16 | 0.0025094513 | 0.0021947537 | 0.0016562119 |

- K≤1024에서는 GPU True/False의 RMSE가 같고, K≥2048에서는 False가 더 작았다.
- FP16의 K별 trial-mean RMSE 비율 `FPINT / GPU False`는 0.7395–0.7579다.
- BF16의 같은 비율은 0.7495–0.7579다.
- 두 dtype 각각 Torch/reference와 CUDA/reference가 270/270 cases에서 정확히
  일치했다(`max_abs=0`). 합계 540 cases, 552,960 output elements다.
- FPINT 정확도 순위는 diagnostic이며, pass 조건은 reference 일치와 finite coverage다.

상세 RMSE, ULP, nonfinite counts, 소스 hash:

- [FP16 결과](mxu128_fp16_int4_gemm_results.md)
- [BF16 결과](mxu128_bf16_int4_gemm_results.md)

FP16/BF16 activation의 실제 지수 하한과 mantissa 분포가 다르므로 이 표를
동일 실수 입력에 대한 dtype 간 정확도 비교로 해석하지 않는다. K별 exponent
상한도 변하므로 절대 RMSE 추이에는 입력 크기 변화가 포함된다.

## 실행과 산출물

저장소 루트에서 이번 실행을 재현한다. 각 case 안에서 True/False를 모두 측정한다.

```bash
conda activate spinquant
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  bash scripts/run_fpint_scaled_gemm.sh
```

Runner는 FP16/BF16 측정 후 각각 Markdown report를 생성한다.
`TRIALS`, `BASE_SEED`, `DEVICE`, `SCALE_FORMAT`, `OUTPUT_ROOT`, `REPORT_ROOT`를
환경 변수로 지정할 수 있다. 기본 `SCALE_FORMAT=activation`이며 scale dtype을
activation과 독립적으로 고정하려면 `SCALE_FORMAT=fp16` 또는 `bf16`을 사용한다.

한 포맷만 실행하거나 다른 scale 범위를 시험하려면:

```bash
python measure_fpint_qcol_accuracy.py \
  --activation-format bf16 --scale-format bf16 \
  --scale-mode log-uniform --scale-log2-min -4 --scale-log2-max 0 \
  --bits 4 --group-size 128 --mxu-rows 128 \
  --extra-bits 19 --reduce-extra-bits 10 \
  --m 32 --n 32 --trials 30 --base-seed 20260915 --device cuda:0 \
  --output results/fpint-mxu128-scaled-gemm/bf16-int4-scaled.json

python summarize_fpint_mxu128.py \
  --random results/fpint-mxu128-scaled-gemm/bf16-int4-scaled.json \
  --output docs/FP-INT-hw-acc/mxu128_bf16_int4_gemm_results.md
```

`--scale-mode identity`는 scale=1을 사용한다. `--k-values`를 바꾸면
`--exp-max-by-k`에도 정확히 같은 K 목록을 지정해야 한다.

로컬 raw 산출물(`results/`는 gitignore 대상):

- [FP16 JSON](../../../../results/fpint-mxu128-scaled-gemm/fp16-int4-scaled.json), [CSV](../../../../results/fpint-mxu128-scaled-gemm/fp16-int4-scaled.csv)
- [BF16 JSON](../../../../results/fpint-mxu128-scaled-gemm/bf16-int4-scaled.json), [CSV](../../../../results/fpint-mxu128-scaled-gemm/bf16-int4-scaled.csv)

JSON의 `conventional_err`는 True, `conventional_full_precision_err`는 False,
`fp_int_err`는 FPINT다. 이름의 full_precision은 reduction flag=False를 뜻하며
FP64 baseline이라는 뜻이 아니다. Scale 샘플 전체 대신 seed, 분포, dtype,
관측 min/max, scale bit-pattern SHA256을 저장한다.

## 구현 검증과 수정

- `reference.py`, `torch_backend.py`, CUDA kernel에 BF16 scale 지원을 추가했다.
  기존 FP16 scale도 계속 지원한다. Scale 곱과 누적 연산은 FP32를 유지한다.
- 첫 BF16 실행에서 Torch GPU 경로만 seed=20261154, K=4096의 한 원소가
  reference와 달랐다. 이 환경에서 `torch.ldexp(1., -19)`가 정확한 값보다
  FP64 1 ULP 작았고, 최종 BF16 출력이 -464 대신 -462가 됐다.
- Torch의 power-of-two factor를 FP64 exponent bit로 정확히 구성하도록 수정하고
  회귀 테스트를 추가했다. CUDA kernel의 정수 복원 계산은 이 수정의 대상이 아니다.
- 수정 후 두 dtype의 270-case 실험을 모두 재실행했다. 위 결과와 raw 파일은 재실행 결과다.
- 관련 테스트: `89 passed, 1 skipped` (`tests/test_fpint_mxu128_experiment.py`,
  `tests/test_fpint_emul.py`). 실행 시 GPU 하나만 노출하여 multi-GPU 테스트는 skip됐다.
- 이번 범위는 GEMM numerical accuracy다. 모델 평가와 RTL 재검증은 실행하지 않았다.

기존 scale=1 결과는 [FP16 unit-level 문서](../../mxu128_numerical_acc_results_unit_level.md)와
[BF16 archive](../identity-scale-20260922/mxu128_bf16_int4_gemm_results.md)에 보존했다.
