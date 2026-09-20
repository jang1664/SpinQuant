# BF16 RTL/GPU bugfix 재검증 진행 상황

- 2026-09-20: 기존 RTL↔GPU 비교에서 BF16 normal packer 1 ULP 오차와
  BF16 subnormal exponent/restore 불일치를 분리해 확인했다.
- 2026-09-20: hardware 저장소에서 `feat/bugfix` branch를 만들었다.
- 2026-09-20: RTL BF16 RNE bias와 subnormal effective exponent를 수정하고,
  testbench golden helper 및 RTL 문서를 함께 갱신했다.
- 2026-09-20: CUDA int-to-FP restore를 우선 FP64 `ldexp` 후 단일 FP32 RNE
  변환으로 고쳐 원인을 검증하고 BF16 subnormal CUDA 회귀 테스트를 추가했다.
  이후 normal은 `__ll2float_rn` + `ldexpf`, subnormal은 정수 RNE bit 구성으로
  최적화해 FP64 없이 동일한 단일 반올림 의미를 구현했다.
- 2026-09-20: 로컬 Verilator 비교에서 BF16/FP16 2,048개 출력이 모두
  bit-exact였고, 당시 SpinQuant 전체 테스트 111개가 통과했다.
- 2026-09-20: full-evaluation 결과 metadata에 CUDA kernel hash를 기록하고,
  변경된 kernel로 생성되지 않은 FPINT shard를 resume 시 재실행하도록 보강했다.
- 2026-09-20: berlin1 전체 테스트 111개가 통과했다. 새 kernel로 BF16×INT4
  270-case를 재측정해 Torch/CUDA reference 270/270 및 finite coverage 100%를
  확인했고, paired Llama 3.1 smoke sanity도 통과했다.
- 2026-09-20: 4개 GPU에서 Llama 3.1 full FPINT shard를 시작했다. 기존
  standard shard는 checkpoint/rotation 조건 검증 후 재사용하고 FPINT shard만
  kernel hash 불일치로 재실행 중이다.
- 2026-09-20: 최종 kernel로 ARC Challenge + WinoGrande full shard가 완료됐다.
  결과 JSON의 kernel SHA를 검증했으며 accuracy는 각각 51.962457%와
  72.296764%다. 나머지 세 shard는 계속 실행 중이다.
- 2026-09-20: ARC Easy + OpenBookQA full shard도 최종 kernel SHA로 완료됐다.
  normalized accuracy는 각각 79.587542%와 43.400000%다. WikiText와
  HellaSwag shard는 계속 실행 중이다.
- 2026-09-20: GPU 1에서 타 사용자 workload를 확인해 해당 GPU의 HellaSwag
  timing 오염을 방지했다. 내 experiment process group만 종료한 뒤 GPU 순서를
  `1,0,2,3`으로 바꿔 HellaSwag를 비점유 GPU 0에 재시작했다.
- 2026-09-20: 정상 FP32 결과는 `__ll2float_rn` 뒤 power-of-two scale을 쓰고,
  subnormal은 정수 quotient/remainder와 tie-even으로 직접 bit를 구성하도록
  정리했다. FP64와 double-rounding을 모두 제거했으며 로컬 111 tests, RTL
  bit-exact 비교, berlin1 53 tests와 paired smoke가 통과했다. 최종 kernel
  SHA는 `267945ac381abc66e7075f50e79fe5a28f00188b6673ed921755f05cb424acc4`다.
- 2026-09-20: WikiText full shard가 최종 kernel SHA로 완료됐다. 62개 문서,
  word perplexity `8.42695620331156`, evaluation time `2524.1763460610528 s`이며
  BF16 compute/activation/output과 checkpoint/rotation provenance를 확인했다.
- 2026-09-20: berlin1 유휴 GPU에서 HellaSwag batch 64/96/128/256을 짧게
  측정했다. limit-128에서는 batch 64가 빨랐지만 full workload의 긴 request
  구간에서는 `1.29 req/s`로 기존 batch 32의 `1.40 req/s`보다 느렸다. 또한
  batch가 수치 결과에 영향을 주므로 최종 실험은 원래 조건인 batch 32로
  복귀했다. 집계기의 task별 batch 일치 검증과 보고서 batch 표시는 유지했다.
- 2026-09-20: lm-eval native request sharding을 preloaded model에 연결한 2-GPU
  시험은 request 계산 뒤 NCCL watchdog illegal-memory 오류가 발생해 결과를
  생성하지 못했다. 이 경로는 채택하지 않고 관련 코드를 제거했으며, 검증된
  single-GPU batch-32 경로로 복귀했다.
- 2026-09-20: berlin1에서 HellaSwag batch-32 full shard가 오류 없이 완료됐다.
  10,042개 sample의 normalized accuracy는 Standard GPU QDQ `77.703645%`,
  FPINT CUDA `77.663812%`로 delta는 `-0.039833%p`다. 전체 8개 shard에서
  BF16 compute/activation/output, checkpoint/rotation/kernel SHA, task별 batch,
  sample 수 및 224개 FPINT projection + floating-point lm_head coverage를 검증했다.
- 2026-09-20: model 결과를 재집계해 accuracy micro average는 Standard
  `74.428599%`, FPINT `74.441623%`(`+0.013023%p`), WikiText word perplexity는
  Standard `8.42729060`, FPINT `8.42695620`으로 문서를 갱신했다.
- 2026-09-20: 최종 로컬 회귀에서 SpinQuant `112 passed`, Verilator RTL↔GPU
  BF16/FP16 2,048/2,048 bit-exact, `git diff --check` 통과를 재확인했다.
