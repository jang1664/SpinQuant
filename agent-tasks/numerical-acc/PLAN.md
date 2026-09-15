# AQP8 W4/KV4 행렬곱 출력 정확도 비교 계획

## 목표

기존 W4A8KV4 vs W4A16KV4 분포(logit) 비교를 대체하는 것이 아니라,
**A/Q/P가 모두 8-bit인 AQP8 W4/KV4**와 **A/Q/P가 모두 16-bit인
AQP16 W4/KV4**의 수치 오차를 각각 FP16 참조 모델에 대해 측정한다.

- `error_A`: `FP16 vs AQP8 W4/KV4`의 행렬곱 출력 오차
- `error_B`: `FP16 vs AQP16 W4/KV4`의 행렬곱 출력 오차

핵심 비교 대상은 `error_A`와 `error_B`다. 즉, A/Q/P를 8-bit로 낮췄을 때
각 그룹의 FP16 대비 오차가 A/Q/P=16 대비 얼마나 커지는지 측정한다.

비교 대상은 모델 안의 행렬곱 출력으로 한정한다. 보고서는 아래 세 그룹만
생성한다.

1. `Linear`: 모든 `nn.Linear`/SpinQuant linear projection의 출력
2. `QK`: attention의 `Q @ K^T` 출력
3. `PV`: attention의 `P @ V` 출력 (`P`는 softmax 뒤의 attention probability)

logit, softmax 출력 자체, RMSNorm/SiLU/잔차 더하기, RoPE, KV cache tensor,
perplexity, task accuracy는 측정하거나 보고하지 않는다.

## 고정 실험 조건

| 항목 | FP16 참조 | AQP16 | AQP8 |
| --- | --- | --- | --- |
| 모델 | 동일 모델/동일 입력 | 동일 | 동일 |
| weights | FP16 | 공통 SpinQuant W4 checkpoint | AQP16과 같은 checkpoint |
| weight / KV cache | FP16 | W4, K4, V4 | W4, K4, V4 |
| A | FP16 | 16-bit (bypass) | 8-bit |
| Q (QK의 Q 입력) | FP16 | 16-bit (bypass) | 8-bit |
| P (PV의 P 입력) | FP16 | 16-bit (bypass) | 8-bit |
| attention backend | eager | eager | eager |
| rotation, sequence length, seed, prompts | 같은 optimized rotation, 고정 | 고정 | 고정 |

`AQP8`/`AQP16` 설정은
`scripts/run_hard_workload_comparison.sh`의 설정을 기준으로 한다.
특히 AQP8에는 `--a_bits 8 --q_bits 8 --q_groupsize 128 --p_bits 8
--p_groupsize -1`을, AQP16에는 대응하는 16-bit bypass 설정을 사용한다.
기존 hard-workload와 같이 A와 P는 asymmetric, Q는 symmetric (`q_asym=False`)
설정을 유지한다.
두 조건은 같은 rotation과 같은 W4 checkpoint를 로드해야 한다.

## 구현 계획

### 1. 전용 실행 스크립트와 결과 경로를 추가한다

- `scripts/run_matrix_output_accuracy.sh`를 추가한다.
- 기본 대상은 기존 divergence 실험과 같은 `llama2-7b`, `llama3.1-8b`,
  `llama3.2-3b`이며, 모델 경로 인자로 재정의할 수 있게 한다.
- 결과는 기존 `w4a8kv4...`/`w4a16kv4...` 폴더와 섞지 않고,
  `results/aqp-matrix-output-accuracy/<model>/matrix-output-accuracy.json`에 쓴다.
  한 JSON 안에 동일 FP16 reference 기준의 `error_A`와 `error_B`를 함께 넣어
  입력·checkpoint·rotation의 일치를 명시한다.
- 빠른 smoke test용 문서/토큰 제한 옵션과, 전체 WikiText 62문서 실행 옵션을
  분리한다. 최종 보고에는 전체 실행만 사용한다.

### 2. AQP 조건을 hard-workload 방식으로 모델에 적용한다

- `ptq.py` 또는 `result_analysis.load_model.load_model` 호출 경로가
  `a_bits`, `q_bits`, `p_bits`를 모두 받도록 확인하고, 전용 계측 러너에서는
  세 값을 명시적으로 전달한다. Q/P를 생략하여 기본값 16이 되는 일이 없어야 한다.
- P quantization이 실제 attention 구현에서 수행되도록 `eager` backend를
  강제한다. `p_bits < 16`이면 Flash/SDPA를 쓰지 않는다.
- 시작 시 각 attention layer의 QK quantizer bit-width와 P quantizer bit-width,
  그리고 linear activation bit-width를 검증해 JSON `configuration`에 기록한다.
  한 layer라도 기대값과 다르면 실행을 실패시킨다.

### 3. 행렬곱 출력 계측 지점을 명시적으로 삽입한다

단순 forward hook만으로 QK/PV는 얻을 수 없으므로, eager
`eval_utils/modeling_llama.py::LlamaAttention.forward`에 read-only observer
callback(또는 동등한 collector)을 둔다. collector는 detach한 tensor만 받고
모델 계산값을 수정하지 않는다.

| 그룹 | 수집 tensor | 수집 시점 | 제외 대상 |
| --- | --- | --- | --- |
| `Linear` | 각 linear projection의 forward output | projection 직후 | 입력 activation, bias 단독값, 이후 nonlinearity/residual |
| `QK` | `torch.matmul(query_states, key_states.transpose(2, 3))`의 raw output | scale, mask, softmax 이전 | scale/mask/softmax tensor |
| `PV` | `torch.matmul(attn_weights, value_states)`의 raw output | P QDQ 후, output projection 이전 | P tensor 자체, O projection output(이는 Linear로 집계) |

- `Linear`은 attention의 Q/K/V/O projection과 MLP projection 및 `lm_head`를
  포함한 모든 실제 linear 행렬곱을 이름으로 식별한다. 동일한 모듈 이름이 FP16과
  quantized 모델에서 1:1로 대응하는지 실행 전 검증한다.
- QK/PV record key는 `model.layers.<N>.self_attn.qk` 및 `.pv`로 고정한다.
  cache를 쓰지 않는 teacher-forced WikiText forward로 실행해, 두 모델의
  shape와 causal sequence가 동일하도록 한다.
- collector는 tensor 전체를 디스크에 저장하지 않는다. 한 번의 FP16 forward와
  한 번의 quantized forward에서 대응 tensor를 즉시 비교·누적해 GPU/host 메모리
  사용량을 제한한다.

### 4. 수치 metric을 FP16 대비로 누적한다

각 대응 출력 원소에 대해 아래 두 오차를 독립적으로 계산한다.

```text
error_A = AQP8_output  - FP16_output
error_B = AQP16_output - FP16_output
```

각 error의 모든 layer, head, token, matrix element를 그룹별로 합쳐 아래
metric을 계산한다.

| metric | 정의 |
| --- | --- |
| `elements` | 비교한 출력 원소 수 |
| `mae` | `mean(abs(error_A))` 또는 `mean(abs(error_B))` |
| `rmse` | `sqrt(mean(error_A^2))` 또는 `sqrt(mean(error_B^2))` |
| `relative_l2_error` | `||error_A||_2 / max(||fp16||_2, eps)` 또는 `error_B` 대응값 |
| `max_abs_error` | `max(abs(error_A))` 또는 `max(abs(error_B))` |
| `cosine_similarity` | flatten한 FP16/AQP8 또는 FP16/AQP16 출력의 전역 cosine similarity |

- 평균 계열 metric은 batch/layer 평균이 아니라 **원소 수 가중 전역 집계**로
  계산한다.
- `relative_l2_error`와 cosine도 각 record의 평균이 아니라 그룹 전체의 합산
  dot product 및 squared norm으로 계산한다.
- dtype은 누적 시 FP64로 올려 반올림 오차를 줄인다.
- AQP8 대 AQP16 tensor 차이를 별도의 error로 계산하지 않는다. 두 FP16 기준
  error를 비교하기 위해, 각 error metric에 아래 파생 비교값을 계산한다.

| 파생값 | 정의 | 해석 |
| --- | --- | --- |
| `A_minus_B` | `metric(error_A) - metric(error_B)` | 양수면 AQP8의 오차가 더 큼 |
| `A_over_B` | `metric(error_A) / max(metric(error_B), eps)` | 1보다 크면 AQP8의 오차가 더 큼 |

`mae`, `rmse`, `relative_l2_error`, `max_abs_error`에 대해 위 파생값을
보고한다. cosine은 `A_minus_B = cosine_A - cosine_B`만 보고하며, 음수면
AQP8의 FP16 일치도가 더 낮음을 뜻한다.

### 5. 결과 형식과 보고서를 제한한다

모델별 JSON은 재현 정보, 세 group의 `error_A`/`error_B`, 그리고 두 error의
비교값만 가진다.

```json
{
  "conditions": {
    "AQP8": {"w_bits": 4, "k_bits": 4, "v_bits": 4,
              "a_bits": 8, "q_bits": 8, "p_bits": 8},
    "AQP16": {"w_bits": 4, "k_bits": 4, "v_bits": 4,
               "a_bits": 16, "q_bits": 16, "p_bits": 16},
    "attention_backend": "eager"
  },
  "scope": {"dataset": "WikiText lm-eval samples", "documents": 62,
            "sequence_length": 2048},
  "groups": {
    "Linear": {
      "error_A_fp16_vs_aqp8": {"elements": 0, "mae": 0.0, "rmse": 0.0,
        "relative_l2_error": 0.0, "max_abs_error": 0.0, "cosine_similarity": 1.0},
      "error_B_fp16_vs_aqp16": {"elements": 0, "mae": 0.0, "rmse": 0.0,
        "relative_l2_error": 0.0, "max_abs_error": 0.0, "cosine_similarity": 1.0},
      "A_vs_B": {
        "mae": {"difference": 0.0, "ratio": 1.0},
        "rmse": {"difference": 0.0, "ratio": 1.0},
        "relative_l2_error": {"difference": 0.0, "ratio": 1.0},
        "max_abs_error": {"difference": 0.0, "ratio": 1.0},
        "cosine_similarity": {"difference": 0.0}
      }
    },
    "QK": {},
    "PV": {}
  }
}
```

- 별도 summarizer는 모델별로 정확히 `Linear`, `QK`, `PV` 세 행만 출력한다.
  각 행에는 `error_A`, `error_B`, `A_minus_B`, `A_over_B`를 병기하며,
  `A_minus_B`가 양수이거나 `A_over_B`가 1보다 큰 metric을 AQP8의 추가
  오차로 명확히 표시한다.
- per-layer/per-head/per-token 결과, logits, 분포 KL/JS/EAR, task accuracy 및
  perplexity는 JSON과 Markdown 보고서에서 모두 제외한다.

## 검증 및 실행 순서

1. unit test: FP16 vs FP16 observer에서 세 그룹 모두 MAE/RMSE/max error=0,
   relative L2=0, cosine=1인지 확인한다.
2. observer test: 작은 eager Llama 입력에서 각 attention layer마다 QK와 PV가
   정확히 한 record씩, 모든 projection이 Linear record로 수집되는지 확인한다.
3. configuration test: AQP8에서 QK wrapper `q_bits=8`, attention P quantizer
   `bits=8`이 확인되고, AQP16에서는 모두 16-bit bypass인지 확인한다.
4. smoke run: 한 문서와 짧은 token limit으로 FP16/AQP16/AQP8의 record-key,
   shape, element count가 동일한지 확인한다.
5. full run: 세 모델 전체 WikiText 62문서를 실행하고 `error_A`, `error_B`,
   `A_minus_B`, `A_over_B` 및 JSON checksum/configuration, shared W4 checkpoint
   경로, rotation SHA를 보관한다.
6. final review: 보고서가 세 group뿐인지, AQP8 JSON에 `a_bits=q_bits=p_bits=8`이
   명시됐는지, 그리고 AQP16과 AQP8이 동일 checkpoint/입력 범위를 썼는지 확인한다.

## 완료 기준

- 세 모델 각각에 AQP16 및 AQP8 결과가 있고, `error_A`와 `error_B`가 동일
  FP16 출력·동일 입력 범위에서 수집된다. AQP8의 A/Q/P bit-width는 모두 8로
  검증된다.
- 모든 비교가 FP16 대응 행렬곱 출력에 대해서만 수행된다.
- 최종 Markdown/JSON 요약은 `Linear`, `QK`, `PV` 세 group만 포함하며, 각
  group에서 error_A가 error_B보다 얼마나 큰지 수치로 제시한다.
