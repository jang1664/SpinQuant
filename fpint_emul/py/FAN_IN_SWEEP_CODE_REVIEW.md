# FP16-INT4 Fan-in Sweep Code Review

## 1. 문서 범위

이 문서는 `fan_in_sweep.py`를 중심으로 다음 실험 경로를 설명한다.

```text
독립 성분 기반 FP16/INT4 입력 생성
        ↓
REF / NVIDIA Tensor Core / FINISH 계산
        ↓
FP16 bit-pattern 기준 ULP 계산
        ↓
seed별 raw CSV export
        ↓
CSV를 다시 읽어 95% CI 계산 및 plot 생성
```

관련 파일은 다음과 같다.

- `fan_in_sweep.py`: 실험 데이터 생성, 세 계산 경로, CSV 및 plot CLI
- `fpint_emul.py`: FINISH scalar emulator와 하드웨어 상수
- `visualize.py`: exact FP16 ULP 거리
- `test_fan_in_sweep.py`: vectorized/scalar 일치 및 plotting 회귀 테스트

이 문서의 line number는 현재 코드 기준이다.

## 2. 현재 실험 설정

### 하드웨어 관련 상수

`fpint_emul.py:22-57`

- `QBLOCK = 32`: quantization parameter가 공유되는 block 크기다.
- `MXU_K = 32`: 한 번에 prealign되는 K 방향 tile 크기다.
- `MXU_N = 16`: Tensor Core 실행을 위해 M/N을 16의 배수로 제한할 때 사용한다.
- `EXTRA_BIT = 19`: main aligned significand의 추가 fractional precision이다.
- `EXTRA_BIT_FOR_REDUCE = 10`: QCOL zero-point correction용 reduce precision이다.
- `EXTRA_BIT_FOR_REDUCE_QROW = 10`: QROW zero-point correction용 reduce precision이다.

`fan_in_sweep.py:34-52`

- K sweep: `32, 64, ..., 32768`
- 입력 exponent: 정수 `[-24, 15]` uniform
- mantissa field: `Normal(512, 256^2)`를 반올림하고 `[0, 1023]`으로 clip
- scale: `[2^-14, 2^-11]` uniform
- 비교 method:
  - `gpu_tensor_core`: 실제 NVIDIA GPU 실행
  - `gpu_model`: FP16 product와 순차 FP32 accumulation을 사용하는 보조 model
  - `ours`: FINISH real two's-complement emulator

scale 범위를 작게 둔 이유는 넓은 입력 exponent 범위에서 FP16 output overflow를 방지하기 위해서다.

## 3. 입력 sampling

### FP16 bit 변환

`fan_in_sweep.py:55-64`

`bits_to_fp16()`은 `uint16`을 FP16으로 bit-cast한다. `fp16_to_bits()`는 입력을 FP16으로 반올림한 뒤 raw `uint16` bit pattern을 돌려준다. 이후 계산과 ULP 비교가 decimal value가 아니라 FP16 representation을 기준으로 동작하게 만드는 기본 함수다.

### sign, exponent, mantissa 독립 sampling

`fan_in_sweep.py:77-93`

1. sign bit를 `{0, 1}`에서 uniform sampling한다.
2. exponent를 inclusive range `[-24, 15]`에서 discrete-uniform sampling한다.
3. mantissa code를 정규분포에서 뽑아 반올림하고 `[0, 1023]`으로 제한한다.
4. 다음 식으로 실수를 구성한다.

```text
x = (-1)^sign × (1 + mantissa / 1024) × 2^exponent
```

5. 마지막에 FP16으로 반올림한다.

주의할 점은 sampled exponent가 `-14`보다 작으면 최종 FP16에서는 subnormal encoding이 된다는 것이다. 따라서 `[-24, -15]`의 값은 FP16 exponent field에 그대로 저장되는 것이 아니라 subnormal significand로 표현된다.

### trial 전체 operand 생성

`fan_in_sweep.py:96-131`

- `SeedSequence.spawn(4)`를 사용해 input, weight, QCOL parameter, QROW parameter RNG를 분리한다.
- weight는 signed INT4 범위 `[-8, 7]`에서 uniform sampling한다.
- zero point는 `[-4, 3]`에서 uniform sampling한다.
- QCOL scale/zero shape은 `(K / QBLOCK, N)`이다.
- QROW scale/zero shape은 `(K, ceil(N / QBLOCK))`이다.

RNG stream을 분리했기 때문에 한 operand의 sampling 구현을 바꾸더라도 다른 operand sequence가 불필요하게 변하는 것을 줄일 수 있다.

## 4. Quantization parameter 확장

`fan_in_sweep.py:134-151`

`_expand_qparams()`는 compressed scale/zero tensor를 GEMM의 `(K, N)` shape으로 확장한다.

- QCOL: K축으로 각 parameter를 `QBLOCK`번 반복한다.
- QROW: N축으로 각 parameter를 `QBLOCK`번 반복한다.

이 함수는 REF, GPU model, 실제 Tensor Core 경로가 동일한 logical scale/zero mapping을 사용하도록 한다.

## 5. 비교 대상 계산 경로

### REF

`fan_in_sweep.py:154-173`

REF는 다음 식을 FP64로 계산하고, 각 K checkpoint에서 최종 결과만 FP16으로 반올림한다.

```text
y_ref(K) = RN_FP16(Σ[k=0..K-1] x[k] × (w[k] - z[k]) × scale[k])
```

FP16 operand의 곱과 최대 K=32768 누적을 FP64로 수행하기 때문에 이 실험에서 accurate-operation reference 역할을 한다. `np.cumsum()`을 한 번 수행하고 필요한 K prefix만 꺼내므로 K마다 전체 REF GEMM을 다시 계산하지 않는다.

### 보조 GPU model

`fan_in_sweep.py:180-204`

이 경로는 다음 순서를 명시적으로 modeling한다.

1. `(weight - zero) × scale`을 FP16으로 반올림
2. input과 dequantized weight의 product를 FP16으로 반올림
3. product를 순차 FP32 누적
4. 각 K checkpoint output을 FP16으로 반올림

이 값은 raw CSV에는 저장되지만 현재 figure에는 표시하지 않는다.

### 공통 prealign

`fan_in_sweep.py:207-223`

`_prealign_grouped()`는 각 `MXU_K=32` group 안에서 다음을 수행한다.

- FP16 sign/exponent/mantissa 분해
- denormal에 대해 effective exponent를 1로 설정
- group maximum exponent 계산
- significand에 extra fractional bit 추가
- maximum exponent에 맞춰 right shift
- sign을 적용해 signed `int64` aligned value 생성

QCOL과 QROW vectorized emulator가 같은 prealign 구현을 공유한다.

### FINISH QCOL

`fan_in_sweep.py:226-257`

QCOL의 핵심 계산은 다음과 같다.

1. input을 `(M, K/QBLOCK, QBLOCK)`으로 묶는다 (`242-245`).
2. main dot product를 `einsum`으로 계산한다 (`247`).
3. aligned input reduce sum과 zero point를 곱해 correction을 만든다 (`248-250`).
4. tile exponent와 FP16 scale을 적용해 floating-point tile value로 복원한다 (`251-255`).
5. K group 방향으로 FP64 prefix accumulation하고 각 K에서 FP16으로 반올림한다 (`256-257`).

즉, QCOL에서는 scale이 integer dot/correction 이후 tile output에 적용된다.

### FINISH QROW

`fan_in_sweep.py:260-301`

QROW는 QCOL과 operation ordering이 다르다.

1. input과 expanded scale을 먼저 곱하고 FP16으로 반올림한다 (`275-282`).
2. `(M, N, K/QBLOCK, QBLOCK)` 순서로 재배치한다 (`282-285`).
3. weight dot과 per-element zero correction을 각각 계산한다 (`287-295`).
4. exponent factor로 tile value를 복원한다 (`296-299`).
5. K group 방향으로 FP64 prefix accumulation한다 (`300-301`).

QROW error가 QCOL보다 큰 주된 구조적 이유는 scale multiplication과 FP16 rounding이 prealign 이전에 들어가기 때문이다.

### 실제 NVIDIA Tensor Core

`fan_in_sweep.py:304-347`

- `TORCH_DEVICE_BACKEND_AUTOLOAD=0`으로 현재 환경의 누락된 `torch_vortex` backend 자동 로드를 막는다.
- deterministic algorithm을 활성화한다.
- `(weight - zero) × scale`을 FP16으로 만든다.
- PyTorch FP16 `matmul`을 각 K prefix에 대해 실제 CUDA device에서 수행한다.
- output을 CPU로 가져와 FP16 bit pattern으로 저장한다.

`fan_in_sweep.py:350-374`

작은 K와 큰 K를 profiler로 실행하고 kernel name에 `wmma` 또는 `tensorop`이 존재하는지 확인한다. 이 검사는 실행이 단순 CUDA core GEMM으로 빠지지 않았는지 확인하기 위한 guard다.

## 6. ULP metric과 통계

### Exact FP16 ULP distance

`visualize.py:137-162`

FP16 bit pattern을 수직선 위에서 단조 증가하는 정수로 mapping한 뒤 두 정수의 차이를 계산한다.

- `+0`과 `-0`은 같은 점으로 취급한다.
- sign을 가로지르는 거리도 representable FP16 value 개수 기준으로 계산한다.
- NaN이 포함되면 ULP가 정의되지 않으므로 예외를 발생시킨다.

### Seed별 raw metric

`fan_in_sweep.py:377-403`

REF와 평가 결과에 Inf/NaN이 없는지 먼저 검사한다. 이후 각 `(seed, qdir, K, method)`에 대해 다음을 CSV row로 만든다.

- `mean_ulp`
- `max_ulp`
- `p50_ulp`
- `p95_ulp`
- `sample_count = M × N`

### 100 seeds 요약과 95% CI

`fan_in_sweep.py:406-429`

plot에서 사용하는 bar height는 seed별 `mean_ulp`의 평균이다. Error bar는 seed 평균들에 대한 Student-t 95% confidence interval이다.

```text
half_width = t(0.975, n-1) × sample_std / sqrt(n)
```

하한은 ULP가 음수가 될 수 없으므로 0으로 clip한다.

## 7. CSV export 경로

`fan_in_sweep.py:577-692`

`export_data()`가 data CLI의 main orchestration 함수다.

1. config와 Tensor Core kernel을 검증한다 (`581-595`).
2. sampling 설정과 시스템 정보를 `metadata.json`에 기록한다 (`596-631`).
3. seed별로 QCOL/QROW의 REF, GPU model, GPU, FINISH를 계산한다 (`633-678`).
4. 새 row를 정렬한 뒤 temporary CSV에 먼저 쓰고 rename한다 (`680-685`).
5. 완료 상태와 row count를 metadata에 기록한다 (`688-692`).

resume은 `(seed, qdir, K, method)` tuple을 key로 사용한다 (`590-593`, `633-639`). 따라서 이미 완료된 seed는 다시 계산하지 않는다.

현재 100-seed 설정에서는 row 수가 다음과 같다.

```text
100 seeds × 2 qdirs × 11 K values × 3 methods = 6600 rows
```

## 8. CSV-only plotting

### CSV reader

`fan_in_sweep.py:528-543`

`plot_csv()`는 raw CSV path 하나만 받는다. 필수 column을 검증하고, seed summary를 `summary.csv`로 저장한 뒤 plot 함수를 호출한다. 실험 sampling이나 GPU가 plotting 단계에 필요하지 않다.

### Figure 구성

`fan_in_sweep.py:432-525`

- 전체 크기: `3.5 × 1.4 inch`
- subplot: QCOL/QROW를 `1 × 2`로 배치
- y scale: log, floor `1e-5`
- 표시 method: GPU와 FINISH
- y tick/label: 왼쪽 QCOL panel에만 표시
- legend: 오른쪽 QROW panel의 lower-right에만 표시
- font: 기본 4.0 pt, title 4.2 pt
- 출력: PNG 200 dpi, SVG, PDF

실제 평균이 0이면 log axis에 직접 표시할 수 없으므로 `2 × 10^-5` 위치에 얇은 bar를 그리고 `0` annotation을 추가한다 (`476-503`). 이는 시각화용 처리이며 CSV 값은 0으로 유지된다.

## 9. CLI

`fan_in_sweep.py:723-762`

### Data 생성

```bash
conda activate vortex
python fan_in_sweep.py data \
  --k-values 32,64,128,256,512,1024,2048,4096,8192,16384,32768 \
  --m 16 --n 16 --qblock 32 --trials 100 \
  --device cuda:0 --verify-tensor-core \
  --output <result-directory>
```

### 기존 run 이어서 생성

```bash
python fan_in_sweep.py data \
  --trials 100 \
  --output <result-directory> \
  --resume
```

### CSV에서 plot 생성

```bash
python fan_in_sweep.py plot \
  <result-directory>/raw_seed_metrics.csv
```

## 10. 테스트 coverage

`test_fan_in_sweep.py:35-54`

- 모든 65536개 FP16 bit pattern의 bit-cast round trip
- zero, subnormal, sign crossing, 인접 normal 값의 exact ULP 검증
- NaN 거부 검증

`test_fan_in_sweep.py:57-73`

- seed reproducibility
- finite/nonzero input
- 양/음 sign 존재
- INT4 weight와 zero-point 범위

`test_fan_in_sweep.py:76-147`

- vectorized REF/GPU model과 scalar implementation의 bit-exact 일치
- vectorized QCOL/QROW와 scalar real two's-complement implementation의 bit-exact 일치

`test_fan_in_sweep.py:150-175`

- 실제 CUDA output shape, 반복 실행 bit equality, finite output

`test_fan_in_sweep.py:178-207`

- CSV-only plot 생성
- PNG/SVG/PDF 존재
- PNG가 `700 × 280 px`, 즉 `3.5 × 1.4 inch @ 200 dpi`인지 검증

현재 focused suite는 9개 test가 통과한다.

## 11. Review findings와 해석상 주의점

### 11.1 Medium: 긴 K에 대한 scalar/vectorized bit-exact test가 없다

vectorized FINISH와 scalar FINISH의 일치 검증은 현재 K=128까지만 수행한다 (`test_fan_in_sweep.py:119-147`). 실제 실험은 K=32768까지 간다. group prefix accumulation이나 큰 정수 중간값에서 발생할 수 있는 문제를 더 직접 검증하려면 K=1024 이상의 일부 case를 테스트에 추가하는 것이 좋다.

### 11.2 Medium: Tensor Core 검증은 양 끝 K만 확인한다

profiler 검증은 최소/최대 K만 검사한다 (`fan_in_sweep.py:350-374`). 중간 K에서 다른 kernel이 선택될 가능성까지 엄밀하게 배제하지는 않는다. 모든 K를 profile하면 더 강한 검증이 되지만 실험 시작 시간이 증가한다.

또한 kernel name에 `wmma` 또는 `tensorop` 문자열이 있는지 보는 heuristic이므로 PyTorch/CUTLASS naming이 바뀌면 실제 Tensor Core를 사용해도 false failure가 날 수 있다.

### 11.3 Medium: ULP 분포가 heavy-tailed일 때 Student-t CI가 불안정할 수 있다

현재 confidence interval은 seed별 mean ULP가 정규분포에 가까워진다는 가정을 사용한다 (`fan_in_sweep.py:406-429`). ULP는 cancellation 근처에서 큰 outlier가 발생할 수 있어 분포가 비대칭일 수 있다. 현재 100 seeds는 30 seeds보다 안정적이지만, 논문용 통계에는 bootstrap CI 또는 median/p95의 병행 표시를 고려할 수 있다.

### 11.4 Low: GPU model은 CSV에 있지만 plot에는 없다

`gpu_model`은 보조 분석을 위해 생성되고 CSV에 저장된다 (`fan_in_sweep.py:180-204`, `670`). plot은 의도적으로 실제 GPU와 FINISH만 선택한다 (`442-445`). CSV column의 method를 보고 GPU model과 실제 GPU를 혼동하지 않아야 한다.

### 11.5 Low: QROW와 QCOL의 rounding point가 다르다

QROW는 scale multiplication을 FP16으로 반올림한 후 prealign한다 (`277-285`). QCOL은 integer dot/correction 후 scale을 적용한다 (`247-256`). 두 plot의 ULP 크기를 비교할 때 단순히 quantization 방향만 바뀐 동일 operation ordering이라고 해석하면 안 된다.

### 11.6 Low: plotting은 CSV 옆 파일을 덮어쓴다

`plot_csv()`는 `summary.csv`와 세 plot 파일을 입력 CSV directory에 고정된 이름으로 저장한다 (`541-543`, `518-525`). 여러 plot style을 보존하려면 실행 전에 directory를 분리하거나 향후 output prefix option을 추가해야 한다.

## 12. Review 결론

현재 코드는 다음 경계를 명확히 분리하고 있다.

- sampling 및 실험 실행: `export_data()`
- raw data artifact: `raw_seed_metrics.csv`
- 통계 집계: `summarize_results()`
- CSV-only 시각화: `plot_csv()`와 `plot_summary()`

핵심 vectorized 계산은 scalar emulator와 bit-exact 비교되고, 실제 NVIDIA path는 CUDA 반복성 및 profiler kernel name으로 검증된다. 현재 결과를 해석할 때 가장 중요한 사항은 QCOL/QROW의 scale rounding 위치가 다르다는 점과 mean ULP의 Student-t CI가 outlier에 민감하다는 점이다.
