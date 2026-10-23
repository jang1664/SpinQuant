# Llama-2 / Llama-3.1 비양자화 baseline

[문서 목차](README.md)

원본 pretrained base model을 사용해 weight·activation·Q/K/V·attention probability 양자화와 rotation을 모두 끄고 평가했다.
비양자화는 원본 모델의 FP16/BF16 연산을 뜻하며 FP32 평가를 뜻하지 않는다.

## 설정

- Llama-2 7B: FP16; Llama-3.1 8B: BF16
- W/A/Q/P/K/V bits=16 (bypass), rotation=False, standard Linear; quantized checkpoint 미사용
- 기존 모델 평가와 동일한 ptq.py 및 모델 구현·tokenizer 로딩 경로 사용
- Eager attention, max length=2048, zero-shot, seed=0, eval limit 없음
- Accuracy batch size=32; WikiText batch size=1
- HellaSwag/ARC Easy/ARC Challenge/OpenBookQA: acc_norm; WinoGrande: acc
- WikiText: lm-eval wikitext task의 word_perplexity
- Micro average: 5개 accuracy task의 sample 수로 가중 평균 (총 15,357개)

## 정확도

| Task | Samples | Llama-2 7B FP16 | Llama-3.1 8B BF16 |
| :--- | ---: | ---: | ---: |
| hellaswag | 10042 | 75.960964% | 79.038040% |
| arc_easy | 2376 | 74.579125% | 80.976431% |
| arc_challenge | 1172 | 46.245734% | 53.412969% |
| winogrande | 1267 | 68.981847% | 73.954223% |
| openbookqa | 500 | 44.200000% | 45.000000% |
| Micro average | 15357 | 71.869506% | 75.854659% |

## WikiText perplexity

| Model | Word perplexity |
| :--- | ---: |
| Llama-2 7B | 9.15459692 |
| Llama-3.1 8B | 7.93639451 |

## 원본 및 재현

- Llama-2 7B (fp16): /data/hf_cache/hub/models--meta-llama--Llama-2-7b-hf/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9
- Shard 평가 시간 합계: 301.36 s
- PyTorch version: 2.7.0+cu128; Transformers 4.44.2; lm-eval git 42b2d88
- [llama2-7b-accuracy.json](../../results/unquantized-llama-baselines/llama2-7b-accuracy.json)
- [llama2-7b-wikitext.json](../../results/unquantized-llama-baselines/llama2-7b-wikitext.json)
- Llama-3.1 8B (bf16): /data/hf_cache/hub/models--meta-llama--Meta-Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b
- Shard 평가 시간 합계: 286.51 s
- PyTorch version: 2.7.0+cu128; Transformers 4.44.2; lm-eval git 42b2d88
- [llama31-8b-accuracy.json](../../results/unquantized-llama-baselines/llama31-8b-accuracy.json)
- [llama31-8b-wikitext.json](../../results/unquantized-llama-baselines/llama31-8b-wikitext.json)

```bash
conda activate spinquant
bash scripts/run_unquantized_llama_baselines.sh
python summarize_unquantized_llama_baselines.py --input-root results/unquantized-llama-baselines --output docs/FP-INT-hw-acc/unquantized_llama_baselines.md
```

Runner는 로컬 Hugging Face cache에서 weight shard가 모두 존재하는 snapshot을 찾는다.
LLAMA2_MODEL/LLAMA3_MODEL로 경로를 지정할 수 있으며, CUDA_DEVICES는 네 GPU 번호다.
소량 점검은 EVAL_LIMIT과 별도 OUTPUT_ROOT를 지정한다. 이 보고서는 limit 없는 전체 결과만 허용한다.

JSON에는 task config/version, sample 수, stderr 및 양자화 bypass 설정이 기록되어 있다.
기존 ptq.py metadata의 weight_quantizer/scale_dtype는 공통 기본값이며, 여기서는 weight_bits=16이므로 실제 weight quantizer나 scale은 사용하지 않는다.
이전 W4 결과와 비교할 때 task별 metric, dtype, tokenizer와 dataset/model revision 차이를 함께 확인한다.
