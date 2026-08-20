# Hard Workload Evaluation Plan

## 목적

Llama 모델에서 다음 세 조건을 동일한 평가 설정으로 비교한다.

1. **FP base**: 원본 full-precision 모델. Weight, A, Q, P 모두 quantization하지 않는다.
2. **AQP no-quant**: SpinQuant의 고정 W4/K4/V4 checkpoint와 동일 rotation을 사용하되, A/Q/P는 16-bit bypass한다.
3. **AQP 8-bit**: 위와 동일한 W4/K4/V4 checkpoint와 rotation을 사용하고 A/Q/P를 각각 8-bit quantize한다.

이렇게 하면 FP base 대비 weight-only/rotation 비용과 A/Q/P 8-bit activation 비용을 분리해서 볼 수 있다.

## 평가 workload

다음 네 태스크를 사용한다.

| 이름 | lm-eval task | 측정 성격 | 기본 metric |
|---|---|---|---|
| MMLU | `mmlu` | 광범위한 지식과 기본 추론 | `acc` 또는 `acc_norm` |
| GSM8K | `gsm8k_cot` | 다단계 수학 추론 | `exact_match` |
| BBH | `bbh_cot_zeroshot` | 논리·조합·패턴 추론 | `exact_match` |
| GPQA Diamond | `gpqa_diamond_zeroshot` | 고난도 과학 추론 | `acc_norm` |

`lm-evaluation-harness`가 제공하는 task 이름을 실행 전에 `lm_eval --tasks list`로 검증한다. GSM8K/BBH는 generative task이므로 각 조건에서 동일한 prompt, few-shot 수, generation 설정을 사용해야 한다.

## 코드 변경 계획

### 1. `ptq.py`의 task 하드코딩 제거

- `task_names` 전역 상수를 삭제한다.
- `process_args_ptq()`에서 `--eval_tasks` 문자열 인자를 추가한다.
- 쉼표 구분 문자열을 list로 변환하고 공백을 제거한다.
- 기본값은 현재 regression을 보존하기 위해 기존 task 목록으로 둔다.
- 평가 직전에 선택된 task 목록을 log와 결과 metadata에 기록한다.

예시:

```bash
--eval_tasks mmlu,gsm8k_cot,bbh_cot_zeroshot,gpqa_diamond
```

### 2. FP base 실행 경로 추가

기존 PTQ 경로는 W4 checkpoint를 로드하므로 FP base를 별도 condition으로 실행한다.

- `--w_bits 16 --k_bits 16 --v_bits 16`
- `--a_bits 16 --q_bits 16 --p_bits 16`
- rotation 및 GPTQ checkpoint를 사용하지 않음
- 가능하면 `attention_backend=sdpa` 또는 기본 backend를 사용
- 메모리 부족 시 동일 eager backend로 fallback하되, backend를 metadata에 기록

FP base는 모델 원본 자체의 reference이며, AQP no-quant와 동일한 모델 weight가 아니다. 따라서 결과 해석 시 FP base→AQP no-quant 차이는 주로 W4/K4/V4 및 rotation의 비용으로 본다.

### 3. AQP no-quant와 AQP 8-bit 실행 통합

기존 `run_ptq_w4kv4_aqp_comparison.sh`를 세 condition 실행 스크립트로 확장한다.

- 공통: 모델, rotation checkpoint, W4/K4/V4 checkpoint, seed, max length, batch size
- `aqp16`: `a_bits=q_bits=p_bits=16`
- `aqp8`: `a_bits=q_bits=p_bits=8`
- 결과 파일과 log 파일을 condition별로 분리
- 결과가 이미 있으면 skip하되, task 목록/실험 설정이 다른 경우에는 output directory를 새로 사용

권장 디렉터리 구조:

```text
results/hard-workloads/<model>/
  fp_base.json
  aqp16.json
  aqp8.json
  logs/fp_base.log
  logs/aqp16.log
  logs/aqp8.log
  SUMMARY.md
```

### 4. Quantization 실제 적용 검증

각 실행 log와 JSON metadata에서 다음을 확인한다.

- FP base: weight/activation/query/probability bits가 모두 16 또는 bypass
- AQP no-quant: W/K/V=4, A/Q/P=16
- AQP 8-bit: W/K/V=4, A/Q/P=8
- `attention_backend=eager`가 P quantization에서 사용됨
- QK wrapper repr에 `q_bits=8`이 표시됨
- P quantizer repr에 `bits=8`이 표시됨
- 결과 JSON의 task set이 네 workload와 정확히 일치함

가능하면 unit test에서 AQP8 실행 시 Q/P quantizer의 QDQ 출력이 입력과 다름을 확인한다.

## 공통 평가 설정

- model: 현재 사용 중인 Llama 3.2 3B
- dtype: 기존 PTQ 평가와 동일한 `float16`
- seed: `0`
- `num_fewshot`: `0`으로 고정
- batch size: GPU 메모리에 맞추되 세 condition 동일
- `model_max_length`: 일반 workload는 `2048`
- generation: GSM8K/BBH에서 동일한 `max_gen_toks`, stop 조건, sampling 비활성화
- task별 score와 전체 raw JSON을 모두 보존

RULER는 이번 네 태스크에는 포함하지 않는다. 이후 Q/P가 긴 context에서 어떻게 영향을 주는지 별도로 분석할 때 `max_length=4096/8192` 실험으로 추가한다.

## 결과 요약 및 분석

요약 스크립트를 추가하거나 확장해 각 task에 대해 다음을 출력한다.

```text
task | fp_base | aqp16 | aqp8 | aqp8-fp(pp) | aqp8-aqp16(pp)
```

- MMLU/BBH/GPQA: accuracy 계열은 percentage point 차이로 표시
- GSM8K: exact-match percentage point 차이로 표시
- task aggregate는 단순 평균하지 않고 task별로 우선 해석
- 낮은 절대 정확도라도 AQP16 대비 AQP8의 degradation을 핵심 지표로 삼음

## 검증 순서

1. task 이름과 metric을 lm-eval에서 확인한다.
2. 작은 subset 또는 `limit` 옵션으로 세 condition의 end-to-end 실행을 확인한다.
3. full 네 task를 FP base, AQP16, AQP8 순서로 실행한다.
4. JSON metadata, log의 quantizer repr, task 목록을 자동 검사한다.
5. 기존 AQP quantization unit test와 shell syntax check를 실행한다.
6. `SUMMARY.md`에 결과와 각 condition의 정확한 command/configuration을 기록한다.

## 예상 해석

- FP base→AQP16: W4/K4/V4와 rotation으로 인한 손실
- AQP16→AQP8: A/Q/P 8-bit quantization 자체의 추가 손실
- 특히 GSM8K/BBH/GPQA에서 AQP8의 reasoning degradation 여부를 중점 확인
- GPQA의 낮은 정확도는 quantization 차이를 가릴 수 있으므로 절대값보다 paired difference를 우선한다.
