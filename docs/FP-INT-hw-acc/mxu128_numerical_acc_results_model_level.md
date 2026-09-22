# MXU ROW 128 FPxINT model-level numerical accuracy

[문서 목차](README.md) · 현재 GEMM 실험: [공통 설정·결과](mxu128_scaled_gemm_experiment.md)

## 요약

Llama 3.1 8B의 224개 quantized Linear를 `QCOL_REAL_2SCOMP` FPINT CUDA
backend로 실행하고, 동일한 W4 checkpoint를 일반 GPU QDQ Linear로 실행한 결과와
비교했다. Full workload에서는 element-wise all-close를 품질 기준으로 사용하지 않고
최종 WikiText perplexity와 downstream task accuracy를 비교했다.

- WikiText PPL: `8.4200217` → `8.4200611` (`+0.0000394`, `+0.000468%`)
- 5개 accuracy task의 micro average: `74.5133%` → `74.4937%`
  (`-0.0195 pp`)
- HellaSwag, ARC Easy, OpenBookQA 점수는 동일했다.
- ARC Challenge에서 1개, WinoGrande에서 2개 결과가 달라져 전체 15,357개
  sample 중 FPINT의 정답이 3개 적었다.
- 관측된 task 차이는 각 task의 standard error보다 훨씬 작다.

따라서 이 조건에서는 Linear/logit의 엄격한 element-wise all-close 실패가 최종
PPL 또는 task accuracy의 의미 있는 저하로 이어지지 않았다. 이 실험에는 자동
품질 pass/fail threshold를 두지 않았으며, 위 수치는 두 backend의 관측 결과다.

## 실험 조건

| 항목 | 설정 |
| --- | --- |
| Model | Meta Llama 3.1 8B |
| Weight quantization | symmetric W4 GPTQ, clipping, group size 128 |
| Rotation | SpinQuant optimized rotation |
| Linear reference | standard GPU QDQ Linear |
| Linear candidate | FPINT CUDA `QCOL_REAL_2SCOMP` |
| Activation / Q / K / V / P | FP16 |
| MXU reduction row | 128 |
| FPINT extra bits | main 19, reduction 10 |
| FPINT 적용 범위 | decoder projection 224개 |
| Standard 유지 범위 | `lm_head` |
| Attention | eager |
| Evaluation | zero-shot lm-eval |
| Batch size | WikiText 1, multiple-choice task 32 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| PyTorch / CUDA | PyTorch 2.7.0+cu128 / CUDA 12.8 |

Standard와 FPINT는 checkpoint, task, sample, batch size를 동일하게 유지했다.
Checkpoint SHA256은
`5c54a66a6465547d1cb60e0c2217cc0884d033862628404ef53ab94abe6e8083`이다.

## 실행 전 sanity 검증

Full workload 실행 여부는 `sanity_v1`으로 결정했다. 기존 `atol=rtol=0.001`
all-close는 diagnostic으로만 기록하고 gate에서는 제외했다.

| 검사 | 결과 |
| --- | --- |
| Standard / FPINT backend coverage | 통과: 각각 224개, fallback 없음 |
| `lm_head` backend | 통과: standard 유지 |
| Workload non-empty | 통과 |
| Standard exact repeatability | 통과: `atol=rtol=0`, 차이 0 |
| Logit finite | 통과: non-finite 0 |
| Linear output complete / finite | 통과: 224개, non-finite 0 |

Unit-level prerequisite도 통과했다. Gaussian/componentwise FP16 input,
symmetric/asymmetric zero, 9개 K 크기와 30개 seed를 조합한 1,080 case에서
Torch 및 CUDA를 독립 NumPy reference와 비교한 2,160회가 모두 정확히 일치했다.
세부 결과는 [unit-level 결과](mxu128_numerical_acc_results_unit_level.md)를 참고한다.

## Paired model smoke

WikiText 4개 document에서 각 256 token, 총 1,024 token을 동일 입력으로 실행했다.

### Linear output diagnostic

| Metric | 결과 |
| --- | ---: |
| 관측 Linear | 224 |
| `atol=rtol=0.001` all-close Linear | 8 / 224 |
| All-close 밖 element | 251,102,601 / 1,409,286,144 (17.8177%) |
| Maximum absolute error | 0.1015625 |
| MAE | 0.00080231 |
| RMSE | 0.00128907 |
| Relative L2 error | 0.00198254 |
| Cosine similarity | 0.99999803 |
| Non-finite | 0 |

### Final logit diagnostic

| Metric | 결과 |
| --- | ---: |
| Logit all-close (`atol=rtol=0.001`) | False |
| Maximum absolute error | 0.11328125 |
| RMSE | 0.00984093 |
| Relative L2 error | 0.00310814 |
| Cosine similarity | 0.99999521 |
| Symmetric KL | 0.00003005 nats |
| JS divergence | 0.00000751 nats |
| Top-1 agreement | 99.9023% (1,023 / 1,024 token) |
| Standard / FPINT smoke PPL | 9.3094218 / 9.3066113 |
| Non-finite | 0 |

이 결과는 작은 오차가 layer를 거치며 누적되어 엄격한 element-wise all-close는
통과하지 않지만, logit 방향과 확률 분포 및 top-1 prediction은 거의 유지됨을
보여준다. Full 평가는 이 diagnostic 수치에 threshold를 적용하지 않았다.

## Full workload 결과

Accuracy에는 task의 canonical metric을 사용했다. HellaSwag, ARC Easy,
ARC Challenge와 OpenBookQA는 normalized accuracy, WinoGrande는 accuracy다.

| Task | Samples | Standard | FPINT CUDA | Delta (pp) | Standard stderr | FPINT stderr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| HellaSwag | 10,042 | 77.8231% | 77.8231% | 0.0000 | 0.4146 pp | 0.4146 pp |
| ARC Easy | 2,376 | 78.9141% | 78.9141% | 0.0000 | 0.8370 pp | 0.8370 pp |
| ARC Challenge | 1,172 | 50.7679% | 50.6826% | -0.0853 | 1.4610 pp | 1.4610 pp |
| WinoGrande | 1,267 | 74.0331% | 73.8753% | -0.1579 | 1.2323 pp | 1.2347 pp |
| OpenBookQA | 500 | 44.0000% | 44.0000% | 0.0000 | 2.2221 pp | 2.2221 pp |
| **Micro average** | **15,357** | **74.5133%** | **74.4937%** | **-0.0195** | - | - |

WikiText 62개 sample의 word perplexity는 다음과 같다.

| Metric | Standard | FPINT CUDA | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| Word perplexity | 8.4200217 | 8.4200611 | +0.0000394 | +0.000468% |

## 성능

현재 kernel은 correctness-oriented hardware emulation이며 일반 Tensor Core Linear를
대체할 수준으로 최적화된 kernel은 아니다. K reduction을 output당 4개 thread로
분할한 뒤 smoke 처리량은 기존 44.73에서 78.57 token/s로 1.76배 개선됐지만,
standard의 1,759.88 token/s보다는 22.40배 느렸다.

Full lm-eval에서 기록한 backend별 evaluation time은 다음과 같다. Shard는 GPU
4장에서 병렬 실행했으므로 합계는 wall-clock 시간이 아니라 각 GPU의 평가 시간
합이다.

| Shard | Standard | FPINT CUDA | FPINT / Standard |
| --- | ---: | ---: | ---: |
| WikiText | 45.64 s | 2,295.80 s | 50.30× |
| HellaSwag | 221.35 s | 22,590.34 s | 102.06× |
| ARC Easy + OpenBookQA | 46.70 s | 2,044.88 s | 43.78× |
| ARC Challenge + WinoGrande | 40.10 s | 1,553.73 s | 38.74× |
| **합계** | **353.80 s** | **28,484.75 s** | **80.51×** |

가장 긴 HellaSwag FPINT shard의 실제 evaluation 구간은 약 6시간 16분이었다.
수치 정확도는 유지되지만, 반복적인 full evaluation에 사용하려면 추가 kernel
최적화가 필요하다.

## 해석 및 결론

1. 독립 unit reference와 CUDA가 정확히 일치하므로 CUDA kernel이 의도한
   `QCOL_REAL_2SCOMP` 연산 순서를 구현한다는 근거가 확보됐다.
2. Standard QDQ와 FPINT hardware emulation은 같은 연산이 아니므로 중간 Linear와
   final logit의 엄격한 all-close 실패는 예상 가능한 현상이다.
3. 그럼에도 full WikiText PPL 변화는 `+0.000468%`, 5-task micro accuracy 변화는
   `-0.0195 pp`였고, 세 task는 완전히 동일했다.
4. 이번 workload에서는 FPINT numerical difference가 model-level 품질에 미치는
   영향이 매우 작다. 다만 이는 Llama 3.1 8B, 해당 W4 checkpoint와 평가 task에
   대한 관측 결과이며 다른 model, quantization 설정 또는 generation workload로
   일반화하려면 별도 평가가 필요하다.
5. 다음 병목은 numerical accuracy가 아니라 emulation 속도다.

## 재현 및 결과 파일

- 실행 진입점: `scripts/run_fpint_mxu128_llama31_8b.sh`
- 결과 집계: `summarize_fpint_mxu128.py`
- Paired smoke: `measure_fpint_backend_accuracy.py`
- Full metric:
  `/mnt/nfs-vlsi/jaeyongjang/results/fpint-mxu128-llama31-8b/full-metrics.json`
- Raw shard:
  `/mnt/nfs-vlsi/jaeyongjang/results/fpint-mxu128-llama31-8b/full-shards/`
- Log:
  `/mnt/nfs-vlsi/jaeyongjang/logs/fpint-mxu128-llama31-8b/`

실험 완료일은 2026-09-16이다. 사용한 CUDA kernel SHA256은
`e2a6e927927e5f5ae806b5e2cd7ebaaeaf80e2274d3d8f9547c132d5211b64c6`이다.
