# MXU ROW 128 QCOL / Llama 3.1 8B 실험 결과

## 설정

- QCOL_REAL_2SCOMP, INT4, weight group size 128, MXU ROW 128
- Llama 3.1 8B W4 GPTQ symmetric, A/Q/K/V/P FP16
- Reference backend: standard QDQ Linear
- Candidate backend: FPINT CUDA

## Random QCOL

- 상태: `pass`
- Reference 비교 수: 2160
- 실패 수: 0
- 최대 절대 오차: 0
- Standard QDQ all-close: 540/1080 (최대 절대 오차 4)

## 실제 Linear smoke

- 상태: `pass` (`sanity_v1`)
- All-close는 diagnostic이며 smoke gate에 사용하지 않음
- 관측 Linear: 224
- 통과 / 실패 Linear: 8 / 216
- All-close: `False` (`atol=0.001`, `rtol=0.001`)
- Max abs / MAE / RMSE: 0.1015625 / 0.00080230778 / 0.001289073
- Relative L2 / cosine: 0.001982542 / 0.99999803
- 허용 오차 밖 element: 251102601/1409286144 (0.17817716)
- Relative L2가 큰 Linear:
  - `model.layers.27.self_attn.o_proj`: 0.0068469725
  - `model.layers.24.self_attn.o_proj`: 0.006769082
  - `model.layers.23.self_attn.o_proj`: 0.0065286773
  - `model.layers.25.self_attn.o_proj`: 0.0064645545
  - `model.layers.19.self_attn.o_proj`: 0.0062991635

## Smoke logits

- 문서 / token: 4 / 1024
- Logit all-close: `False`
- Logit max abs / RMSE: 0.11328125 / 0.0098409318
- Symmetric KL: 3.0053532e-05
- JS divergence: 7.5138948e-06
- Top-1 agreement: 0.99902344
- Standard / FPINT perplexity: 9.3094218 / 9.3066113

## 성능 참고

- Standard tokens/s: 1759.8805
- FPINT CUDA tokens/s: 78.569549

## Full WikiText

- Standard / FPINT PPL: 8.4200217 / 8.4200611
- Absolute / relative delta: 3.9395417e-05 / 4.6787786e-06

## Downstream tasks

| Task | Samples | Standard | FPINT | Delta (pp) | Standard stderr | FPINT stderr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hellaswag | 10042 | 0.77823143 | 0.77823143 | 0 | 0.0041458721 | 0.0041458721 |
| arc_easy | 2376 | 0.78914141 | 0.78914141 | 0 | 0.0083703045 | 0.0083703045 |
| arc_challenge | 1172 | 0.50767918 | 0.50682594 | -0.085324232 | 0.014609667 | 0.014610029 |
| winogrande | 1267 | 0.74033149 | 0.73875296 | -0.1578532 | 0.012322701 | 0.012346915 |
| openbookqa | 500 | 0.44 | 0.44 | 0 | 0.022221332 | 0.022221332 |

- Micro accuracy standard / FPINT: 0.74513251 / 0.74493716
- Micro accuracy delta: -0.019535065 pp

## 전체 workload 성능

- Standard lm-eval seconds 합: 353.8036
- FPINT CUDA lm-eval seconds 합: 28484.753
- 자동 품질 pass/fail 기준 없음

Checkpoint SHA256: `5c54a66a6465547d1cb60e0c2217cc0884d033862628404ef53ab94abe6e8083`

원본 JSON/CSV에 환경, coverage, layer별 오차와 checkpoint 경로가 기록되어 있다.
