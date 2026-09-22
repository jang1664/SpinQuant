# FP×INT 실험 문서

## 현재 GEMM-level 실험

**[공통 설정과 결과 요약](mxu128_scaled_gemm_experiment.md)**부터 읽고 dtype별 상세 결과를 확인한다.

| 문서 | 역할 | 상태 |
| :--- | :--- | :--- |
| [공통 설정·결과 요약](mxu128_scaled_gemm_experiment.md) | Activation/scale sampling, 비교 정의, 실행 명령, Global RMSE 요약 | 현재 설정 |
| [FP16 × INT4 결과](mxu128_fp16_int4_gemm_results.md) | K별 RMSE·ULP, finite coverage, reference 검증 | Scale EXP field [0,15] |
| [BF16 × INT4 결과](mxu128_bf16_int4_gemm_results.md) | K별 RMSE·ULP, finite coverage, reference 검증 | Scale EXP field [0,127] |
| [Scale exponent sweep](mxu128_scale_exponent_sweep.md) | Scale 범위를 넓혔을 때 경로별 finite/Inf/NaN 비율과 heatmap | 범위 탐색 자료 |

현재 scale은 activation과 같은 dtype이며 sign/exponent/mantissa를 독립 uniform raw field로 샘플링한다.
음수·양수·0·subnormal을 포함하고, 두 dtype 모두 scale 절댓값은 2 미만이다.
Zero-point=0, M=N=32, K=128…32768, K당 30 trials, MXU rows/group size=128이다.
GPU baseline의 reduced-precision reduction True/False와 FPINT를 동일 입력에서 비교한다.

Sweep에서 발견한 BF16 finite 경계 EXP=241은 **현재 정확도 실험의 상한이 아니다**.
현재는 절대 출력 크기를 제한하기 위해 EXP=127을 사용한다.

실험과 문서 생성:

```bash
conda activate spinquant
bash scripts/run_fpint_scaled_gemm.sh
```

원본 JSON/CSV는 저장소의 `results/fpint-final-setting-accuracy/`에 있다.
`results/`는 gitignore 대상이며 Markdown에는 결과·조건·소스 hash를 기록한다.

## 관련 검증과 모델 평가

| 문서 | 범위 |
| :--- | :--- |
| [RTL vs CUDA](mxu128_rtl_gpu_comparison.md) | 기록된 RTL/CUDA 버전의 bit-exact 검증. 현재 scale 실험의 재검증 결과는 아님 |
| [Llama 2 FP16](mxu128_llama2_fp16_model_results.md) | 모델 task accuracy·perplexity·실행 시간 |
| [Llama 3.1 BF16](mxu128_llama31_bf16_model_results.md) | 모델 task accuracy·perplexity·실행 시간 |
| [Llama 3.1 FP16 진단](mxu128_numerical_acc_results_model_level.md) | 모델 Linear/logit 오차와 downstream 평가 |

모델 보고서에 포함된 scale=1 GEMM 표는 해당 모델 실행 당시의 사전 검증 기록이다.
현재 GEMM 결과는 위 FP16/BF16 전용 문서를 기준으로 읽는다.

## 이전 실험

[Archive 목차](archive/README.md)에 scale=1, 양수 log-uniform, BF16 EXP=241 및 bugfix 이전 결과를 보존한다.
이전 `mxu128_numerical_acc_results_unit_level.md` 경로는 archive 안내 문서로 유지한다.
