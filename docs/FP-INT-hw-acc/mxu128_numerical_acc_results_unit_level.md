# MXU ROW 128 QCOL unit-level numerical accuracy

## 설정

- 연산: `QCOL_REAL_2SCOMP`
- Activation: FP16
- Weight: signed INT4
- Weight group size: 128
- MXU reduction row: 128
- Main / reduction extra bits: 19 / 10
- Output: FP16
- 비교 기준: 독립 NumPy reference
- 실행 환경: NVIDIA RTX PRO 6000 Blackwell Server Edition,
  PyTorch 2.7.0+cu128, CUDA 12.8

## Random sweep

다음 축의 Cartesian product를 30개 seed로 실행했다.

- Input distribution: Gaussian, FP16 sign/exponent/mantissa componentwise sampling
- Weight zero: symmetric, asymmetric
- K: `127, 128, 129, 255, 256, 257, 1024, 4096, 14336`
- M / N: 4 / 37
- Seed: 20260915부터 case마다 1씩 증가
- All-close diagnostic: `atol=rtol=0.001`

총 1,080개 input case이며, 각 case에서 Torch backend와 CUDA backend를
독립 reference에 각각 비교했다.

## 결과

| 비교 | 통과 / 전체 | Maximum absolute error |
| --- | ---: | ---: |
| FPINT Torch vs NumPy reference | 1,080 / 1,080 | 0 |
| FPINT CUDA vs NumPy reference | 1,080 / 1,080 | 0 |
| FPINT CUDA vs standard QDQ | 540 / 1,080 | 4 |

Reference 비교 2,160회는 FP16 output이 모두 정확히 일치했다. CUDA kernel은
K가 MXU row 또는 quantization group 경계에 맞지 않는 tail, 큰 K, symmetric와
asymmetric zero, 넓은 FP16 exponent 분포를 모두 처리했다.

Standard QDQ와의 차이는 오류가 아니다. Standard는 FP16 dequantized weight로
일반 GPU Linear를 실행하는 반면 FPINT는 activation exponent alignment, integer
dot product, tile별 scale 및 FP32 순차 accumulation을 수행한다. 따라서 두 연산의
FP16 rounding 경로가 다르다.

## 결론

- Torch와 CUDA 구현 모두 의도한 `QCOL_REAL_2SCOMP` reference와 일치한다.
- MXU row 128과 weight group 128이 정렬된 주 실험 조건뿐 아니라 K tail도 통과한다.
- Standard QDQ와의 all-close는 FPINT correctness gate로 사용할 수 없다.
- 실제 model-level 영향은
  [model-level 결과](mxu128_numerical_acc_results_model_level.md)에서 PPL과
  downstream accuracy로 평가한다.

## 결과 파일

- 측정 script: `measure_fpint_qcol_accuracy.py`
- Raw JSON:
  `/mnt/nfs-vlsi/jaeyongjang/results/fpint-mxu128-llama31-8b/random-qcol.json`
- Raw CSV:
  `/mnt/nfs-vlsi/jaeyongjang/results/fpint-mxu128-llama31-8b/random-qcol.csv`
