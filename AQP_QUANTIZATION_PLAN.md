# SpinQuant A/Q/P activation quantization 계획

## 목표

기존 SpinQuant의 linear-input activation quantization(A)에 더해, attention의 두 BMM 입력인
Query(Q)와 softmax probability(P)를 fake-quantize 한다. 첫 실험의 목표는 세 요소의 개별 효과를
분리하는 것이 아니라, **동일한 rotation 조건에서 A/Q/P를 모두 quantize했을 때의 전체 영향**을
빠르게 확인하는 것이다.

표기와 삽입 위치는 다음과 같다.

```text
hidden ── Linear(q_proj) ── RoPE + R2 ── [Q quant] ── QK^T ── softmax ── [P quant] ── PV ── o_proj
   └──────────────────────────── [A quant: 각 Linear 입력] ─────────────────────────────────────┘
```

- **A**: 기존 `ActQuantWrapper`가 모든 `Linear`의 입력을 quantize한다.
- **Q**: RoPE 및 online R2 Hadamard transform 뒤, `QK^T` 직전에 quantize한다.
- **P**: softmax (및 training 시 dropout) 뒤, `P @ V` 직전에 quantize한다.

여기서 quantization은 dequantization까지 포함한 fake quantization(QDQ)이다. 첫 구현의 목적은
정확도 분석이며, 정수 BMM kernel 도입은 범위에서 제외한다.

## 회전(rotation) 원칙

Q/P ablation 중 rotation이 또 다른 변수가 되면 결과를 해석할 수 없다. 따라서 아래를 고정한다.

1. `--rotate`와 같은 R1/R2 checkpoint를 두 A/Q/P 조건에서 동일하게 사용한다.
2. 기존 SpinQuant의 R2 경로를 유지한다. 즉 V/O에는 fuse된 R2를 적용하고, Q/K에는 RoPE 뒤
   online Hadamard를 적용한다.
3. Q quantizer는 **R2 뒤**에 둔다. R2로 완화된 분포를 quantize하는 것이며, QK dot product의
   등가성도 보존한다.
4. P에는 rotation을 적용하지 않는다. P는 확률 simplex(음수 불가, 행 합 1)이므로 feature-space
   orthogonal rotation의 대상이 아니다.

CEVA-AI-Labs fork의 Q 구현은 참고하되 그대로 가져오지 않는다. 그 fork는 Q를 K와 같은
`k_bits`로 quantize하지만 Q/K Hadamard 부분을 주석 처리한다. 이 상태를 쓰면 Q quantization과
R2 제거의 효과가 섞인다.

## 인터페이스

기존 옵션의 의미를 바꾸지 않고 아래 옵션을 추가한다.

| 옵션 | 기본값 | 의미 |
| --- | ---: | --- |
| `--a_bits` | 기존값 | Linear 입력 A의 bit-width |
| `--q_bits` | `16` | QK BMM의 Q 입력 bit-width. `16`은 bypass |
| `--p_bits` | `16` | PV BMM의 P 입력 bit-width. `16`은 bypass |
| `--q_groupsize` | `None` → `k_groupsize` | Q의 token-wise(`-1`) 또는 head-wise(`head_dim`) 설정 |
| `--q_asym` | `False` | Q는 signed symmetric quantization을 기본으로 사용 |
| `--p_groupsize` | `-1` | attention head·query별 한 probability row 전체를 한 group으로 사용 |
| `--p_asym` | `True` | P는 non-negative 분포이므로 asymmetric quantization을 기본으로 사용 |
| `--p_clip_ratio` | `1.0` | P clipping 비율. 초기 ablation에서는 clipping을 하지 않음 |

`16`은 현재 SpinQuant의 full-precision bypass 관례를 그대로 따른다. 첫 비교에서는 중간 조합을
실행하지 않고 `(A, Q, P) = (16, 16, 16)`과 `(4, 4, 4)`만 실행한다.

## 구현 단계

### 1. 인자와 공통 config 추가

대상: `utils/process_args.py`

- `q_bits`, `q_groupsize`, `q_asym`, `q_clip_ratio`를 추가한다.
- `p_bits`, `p_groupsize`, `p_asym`, `p_clip_ratio`를 추가한다.
- `q_groupsize`가 지정되지 않으면 기존 `k_groupsize`를 사용한다.
- P는 `p_groupsize == -1`만 우선 지원한다. 다른 값은 명시적으로 오류를 내어 모호한
  broadcasting을 막는다.

### 2. Q quantizer를 K quantizer와 분리

대상: `eval_utils/main.py`, `eval_utils/rotation_utils.py`

- Q/K wrapper 설치 조건을 `k_bits < 16`에서
  `q_bits < 16 or k_bits < 16`으로 변경한다.
- wrapper config에 Q와 K의 bit-width/group-size/대칭성/clip ratio를 각각 전달한다.
- `QKRotationWrapper`에 `q_quantizer`와 `k_quantizer`를 별도로 configure한다.
- Q branch는 Q shape를 사용한다. GQA에서는 `num_attention_heads != num_key_value_heads`이므로
  CEVA의 수정처럼 `num_heads_q`, `head_dim_q`로 reshape한다.
- Q 또는 K가 16-bit이면 그 텐서만 bypass한다. 다른 쪽의 quantization과 R2는 계속 적용한다.
- 기존 Q/K Hadamard 코드는 유지하고 주석 처리하지 않는다.

동일한 변경을 rotation optimization 경로(`train_utils/apply_r3_r4.py` 및 해당 config 전달부)에도
반영한다. 그래야 나중에 Q가 포함된 loss로 rotation을 재최적화할 수 있다.

### 3. P quantizer 추가

대상: `eval_utils/modeling_llama.py`, `eval_utils/main.py`

- 각 eager `LlamaAttention` 인스턴스에 `p_quantizer`와 `p_bits`를 attach하는 작은 helper를 만든다.
  `p_bits == 16`이면 quantizer를 만들지 않는다.
- attention forward에서 아래 위치에 삽입한다.

  ```python
  attn_weights = softmax(scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
  attn_weights = dropout(attn_weights, ...)
  attn_weights = quantize_p_rows(attn_weights)  # p_bits < 16일 때만
  attn_output = torch.matmul(attn_weights, value_states)
  ```

- `quantize_p_rows`는 `[batch, head, query, kv]`를 `[-1, kv]`로 reshape한다. 따라서 각
  `(batch, head, query)` probability row가 독립적인 dynamic quantization group이 된다.
- `ActQuantizer`를 `groupsize=-1`, `sym=False`, `clip_ratio=1.0`으로 configure하고, 매 forward마다
  `find_params` 후 QDQ를 수행한다. output은 원래 dtype으로 복원한다.
- NaN/Inf가 없고 0 이상이라는 조건을 debug assertion 또는 단위 테스트로 검사한다. QDQ 뒤의
  행 합은 정확히 1일 필요는 없으며, 별도의 renormalization은 첫 실험에서는 하지 않는다.
  renormalization은 quantization 자체 외의 새 연산이므로 별도 실험 변수로 남긴다.

P가 명시적으로 보이는 eager attention만 첫 지원 대상이다. SDPA/FlashAttention은 softmax tensor를
노출하지 않으므로 `p_bits < 16`일 때 eager implementation을 강제하고, 실행 로그에 이를 남긴다.
학습용 custom Llama attention(`train_utils/modeling_llama_quant.py`)에도 같은 위치를 추가한다.

### 4. 첫 실험에서는 rotation을 재최적화하지 않음

현재 A4/KV4용으로 학습된 같은 R1/R2 checkpoint를 두 조건 모두에 적용한다. 첫 결과를 확인하기
전에는 Q/P가 포함된 loss로 rotation을 다시 학습하지 않는다. 그래야 측정된 차이가 rotation 변경이
아니라 A/Q/P QDQ 삽입에서 발생했음을 해석할 수 있다.

all-quant 결과가 유의미하면 다음 단계에서만 `(A, Q, P) = (4, 4, 4)`용 R1/R2를 재최적화하고,
그 결과는 고정-rotation 결과와 분리해 보고한다.

## 검증 계획

### 단위/구조 검증

1. `q_bits=p_bits=16`에서 원본과 동일한 seed, 입력으로 logits가 허용 오차 내 일치하는지 확인한다.
2. `q_bits=4`에서 QK BMM 입력 Q가 실제로 QDQ되며 K가 16-bit면 변하지 않는지 hook으로 확인한다.
3. GQA 모델(Llama-3 8B 또는 Qwen2.5)에서 Q/K head 수가 달라도 reshape 오류 없이 실행되는지 확인한다.
4. `p_bits=4`에서 softmax 직후와 PV 직전 사이에 P가 QDQ되는지 확인하고, shape·dtype·finite 여부를
   검사한다.
5. `p_bits < 16`에서 Flash/SDPA가 선택되지 않고 eager attention이 쓰이는지 확인한다.

### 실험 조건

초기에는 W4, KV4, 같은 calibration set, 같은 seed, 같은 R1/R2 checkpoint를 고정한다.

| ID | 조건 | A | Q | P |
| --- | --- | ---: | ---: | ---: |
| 0 | A/Q/P no-quant | FP16 | FP16 | FP16 |
| 1 | A/Q/P all-quant | INT4 | INT4 | INT4 |

각 설정에 대해 WikiText-2 perplexity와 기존 zero-shot task를 측정한다. 결과에는 모델, weight/KV
bit-width, rotation checkpoint hash, A/Q/P bit-width, Q/P quantization scheme, attention backend를
함께 기록한다.

## 완료 기준

- `(A, Q, P) = (16, 16, 16)`과 `(4, 4, 4)`를 동일한 실행 경로에서 선택할 수 있다.
- Q는 RoPE+R2 뒤/QK BMM 앞, P는 softmax 뒤/PV BMM 앞에 정확히 한 번만 삽입된다.
- R1/R2와 W/KV 설정은 두 조건에서 변경되지 않는다.
- Llama GQA smoke test와 기존 Llama-2 smoke test가 통과한다.
- 두 고정-rotation 결과가 같은 평가 설정과 metadata로 재현 가능하게 저장된다.

## 구현 상태

- CLI, Q/K 독립 quantizer, eager P quantizer, 학습 경로를 구현했다.
- `scripts/run_ptq_w4kv4_aqp_comparison.sh`가 하나의 W4 checkpoint와 rotation checkpoint를
  공유하고 eager attention을 고정하여 `A/Q/P=(16,16,16)`과 `(4,4,4)`를 순서대로 실행한다.
- 결과 JSON에는 A/Q/P/W/K/V 설정, attention backend, seed, rotation checkpoint 경로와 SHA-256이
  저장된다.
- `tests/test_aqp_quantization.py`에서 MHA(Llama-2형)와 GQA(Llama-3형) 전체-model smoke test,
  Q/K 독립 bypass, P 삽입 위치 및 eager 강제를 검증한다.
- `models/llama*`, `rotation_llama-*`, `saved_models` 아래의 기존 모델·rotation·W4 checkpoint를
  확인했다. 현재 GPU들이 다른 작업에 사용 중이어서 실제 WikiText-2 및 zero-shot 평가는 아직
  실행하지 않았다. 아래처럼 기존 W4 checkpoint를 지정하면 GPTQ를 반복하지 않고 두 조건을
  실행할 수 있다.

  ```bash
  WEIGHT_CHECKPOINT=saved_models/llama3.2-3b/a16w4kv4-vasym.pt \
      scripts/run_ptq_w4kv4_aqp_comparison.sh \
      models/llama3.2-3b \
      rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin \
      results/aqp-comparison/llama3.2-3b
  ```
