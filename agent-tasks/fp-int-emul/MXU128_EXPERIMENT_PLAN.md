# MXU ROW 128 QCOL / Llama 3.1 8B 검증 계획

## 확정 설정

- 연산: `QCOL_REAL_2SCOMP`, INT4, `group_size=mxu_rows=128`.
- Random 검증: symmetric와 asymmetric weight parameter를 모두 사용한다.
- 모델: Llama 3.1 8B, optimized rotation, symmetric W4 GPTQ/group 128/clipping.
- 모델의 A/Q/K/V/P는 FP16으로 유지하고 Linear backend만 `standard`와 `fpint_cuda`로 바꾼다.
- Random QCOL reference 일치는 correctness gate로 사용한다.
- 모델 smoke의 all-close/L2/cosine은 diagnostic으로만 기록한다. Full 평가는
  backend coverage, 반복 실행, finite output을 확인하는 `sanity_v1` gate 이후 실행한다.

## 실행 단계

1. Gaussian 및 FP16 sign/exponent/mantissa 독립 sampling으로 K `127,128,129,255,256,257,1024,4096,14336`을 30 seeds 실행한다.
2. NumPy QCOL reference 대비 Torch/CUDA all-close를 확인하고 standard QDQ 차이는 별도 기록한다.
3. `linear_backend=standard`로 W4 GPTQ checkpoint를 한 번 생성한다. Format version, group mapping과 checkpoint SHA256을 기록한다.
4. WikiText 4 documents × 256 tokens에서 standard repeatability, 224개 Linear 출력과 logits를 paired 비교한다. All-close는 gate로 사용하지 않고 coverage와 finite 여부만 검사한다.
5. Sanity smoke 통과 후 4개 GPU에 WikiText와 5개 accuracy task를 나눠 두 backend로 실행한다.
6. WikiText word perplexity와 task별 accuracy/stderr, delta, weighted micro accuracy를 보고한다. Full 결과에는 자동 품질 pass/fail을 두지 않는다.

Full 평가에서 WikiText는 긴 sequence 때문에 lm-eval batch 1을 유지한다. Multiple-choice
task는 Berlin1 사전 측정에서 batch 32가 batch 1보다 약 6–7배 빨랐고 batch 128보다
빨랐으므로 batch 32로 실행한다. 같은 shard의 standard와 FPINT는 같은 batch를 사용한다.

재현 진입점은 `scripts/run_fpint_mxu128_llama31_8b.sh`다. Raw 결과는 `results/fpint-mxu128-llama31-8b`, 로그는 `logs/fpint-mxu128-llama31-8b`, 요약은 `MXU128_EXPERIMENT_RESULTS.md`에 저장한다.

## Berlin1 환경

- Conda: `/home/jaeyong.jang/.conda/envs/spinquant`, PyTorch 2.7.0+cu128, CUDA 12.8, `TORCH_CUDA_ARCH_LIST=12.0`.
- Model: `/data/hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b`.
- Rotation/checkpoint는 `/mnt/nfs-vlsi/jaeyongjang`의 절대 경로를 사용한다. Remote source의 끊어진 repo-relative symlink에는 의존하지 않는다.
- Conda 환경의 `bin`을 PATH에 포함해 JIT extension이 `ninja`를 찾도록 한다.

## Reference

| 파일 | 용도 |
| --- | --- |
| `fpint_emul/py/fpint_emul.py` | 원본 `prealign` 및 `fpint_gemm_qcol_real_2scomp` 연산 순서 |
| `fpint_emul/py/fan_in_sweep.py` | FP16 component sampling, fan-in/seed 통계 구조 |
| `measure_fpint_linear.py` | 일반화된 row/group accuracy 및 latency 측정 |
| `tests/test_fpint_emul.py` | 독립 reference, CUDA, metadata/checkpoint 회귀 검증 |
| `measure_logit_divergence.py` | WikiText paired logits와 task choice 비교 로직 |
| `measure_matrix_output_accuracy.py` | paired module output observer 구조 |
| `scripts/run_hard_workload_comparison.sh` | quantized checkpoint 재사용과 lm-eval orchestration |

## 완료 조건

- Random reference와 row-128 regression이 모두 통과한다.
- 새 checkpoint가 정확히 224개 FPINT projection metadata를 포함하고 `lm_head`만 standard로 남는다.
- Paired smoke가 224개 projection coverage, exact standard repeatability 및 finite output을 확인한다.
- Full WikiText/5-task accuracy와 PPL 결과 및 재현 명령이 결과 문서에 기록된다.
