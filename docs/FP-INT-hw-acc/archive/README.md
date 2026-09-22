# 이전 FP×INT 실험

현재 실험은 [문서 목차](../README.md)와 [공통 설정](../mxu128_scaled_gemm_experiment.md)을 참조한다.
아래 결과는 각 실행 당시의 설정·코드에 대한 기록이다.

| 보관 위치 | 설정 / 내용 |
| :--- | :--- |
| [Scale=1 FP16](identity-scale-20260922/mxu128_numerical_acc_results_unit_level.md) | 기존 FP16 unit-level GEMM 결과 |
| [Scale=1 BF16](identity-scale-20260922/mxu128_bf16_int4_gemm_results.md) | 기존 BF16 GEMM 결과 |
| [양수 log-uniform 설정](log-uniform-20260922/mxu128_scaled_gemm_experiment.md) | Scale [2^-4,1], FP16/BF16 RMSE·ULP |
| [양수 log-uniform FP16](log-uniform-20260922/mxu128_fp16_int4_gemm_results.md) | 위 설정의 FP16 상세 결과 |
| [양수 log-uniform BF16](log-uniform-20260922/mxu128_bf16_int4_gemm_results.md) | 위 설정의 BF16 상세 결과 |
| [BF16 EXP=241 설정](bf16-scale-exp241-20260922/mxu128_scaled_gemm_experiment.md) | 넓은 raw-field scale 범위의 정확도 실험 |
| [BF16 EXP=241 결과](bf16-scale-exp241-20260922/mxu128_bf16_int4_gemm_results.md) | BF16 절대 RMSE가 큰 이전 결과 |
| [Bugfix 이전 BF16 GEMM](pre-bugfix-20260920/mxu128_bf16_int4_gemm_results.md) | 수정 이전 코드 결과 |
| [Bugfix 이전 BF16 모델](pre-bugfix-20260920/mxu128_llama31_bf16_model_results.md) | 수정 이전 모델 평가 |
| [Bugfix 이전 RTL 비교](pre-bugfix-20260920/mxu128_rtl_gpu_comparison.md) | 수정 이전 RTL/CUDA 비교 |

Archive에 적힌 재현 명령은 당시 설정이다. 현재 실행 script의 기본값은 이후 변경되었으므로
과거 결과를 재현할 때는 문서에 기록된 sampling·dtype·소스 hash를 함께 확인한다.
