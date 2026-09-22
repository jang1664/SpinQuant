# MXU ROW 128 FP×INT unit-level numerical accuracy

이 문서는 기존 scale=1 실험 결과를 보존한다. 2026-09-22에 추가한
FP16/BF16 scale sampling 및 GPU reduced-precision reduction True/False 비교는
[새 실험 설정](../../mxu128_scaled_gemm_experiment.md),
[FP16 결과](../../mxu128_fp16_int4_gemm_results.md),
[BF16 결과](../../mxu128_bf16_int4_gemm_results.md)를 참조한다.

## 실험 정의

동일한 FP16 activation과 signed INT4 weight로 세 경로를 계산했다.

1. `FP64_ref`: activation과 weight를 FP64로 승격한 GPU Linear
2. `Conventional`: INT4 weight만 FP16으로 cast한 실제 GPU FP16 Linear
3. `FPINT`: `QCOL_REAL_2SCOMP` CUDA hardware emulation

`Conventional_err = Conventional - FP64_ref`,
`FP_INT_err = FPINT - FP64_ref`로 정의했다. MXU row 128과 main/reduction
extra bits 19/10은 FPINT 경로에만 적용된다. Conventional은 Berlin1
PyTorch의 default Tensor Core 설정
`allow_fp16_reduced_precision_reduction=True`를 사용했다.

## Input

- M=N=32, K=`128..32768` power-of-two, K당 30 trials
- Activation sign/exponent/mantissa field를 독립 uniform sampling
- Sign bit `0..1`, mantissa `0..1023`
- Weight는 signed INT4 `-8..7` uniform sampling
- Scale=1, zero-point=0

각 K에서 common finite output을 99.9% 이상 확보하도록 exponent field
상한을 다음과 같이 조정했다.

| K | 128 | 256 | 512 | 1024 | 2048 | 4096 | 8192 | 16384 | 32768 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EXP max | 24 | 24 | 24 | 23 | 23 | 22 | 21 | 21 | 20 |

Activation sign bit는 양수 31,396,120개, 음수 31,395,560개였다.

## Metric

- RMSE는 candidate FP16 output을 FP64로 올려 unrounded FP64 reference와 비교했다.
- ULP는 FP64 reference를 correctly-rounded FP16으로 변환한 값과 candidate
  FP16 bit pattern 사이의 exact representable-value distance다.
- FP64 reference, FP16-rounded reference, Conventional, FPINT가 모두 finite인
  common mask를 두 error에 동일하게 적용했다.
- 표의 `mean ± std`는 30개 trial metric의 mean과 sample std(`ddof=1`)다.
- 원소별 signed error와 ULP의 population std도 raw JSON과
  [상세 결과](../../../../agent-tasks/fp-int-emul/MXU128_EXPERIMENT_RESULTS.md)에 기록했다.

## 결과

| K | Common finite | Conv RMSE mean ± std | FPINT RMSE mean ± std | FPINT/Conv RMSE | Conv mean ULP ± std | FPINT mean ULP ± std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 1.9790936 ± 0.0947782 | 1.9790927 ± 0.0947783 | 0.9999995 | 0.00149740 ± 0.00238841 | 0.00006510 ± 0.00024776 |
| 256 | 1 | 2.7551951 ± 0.1409866 | 2.7551887 ± 0.1409841 | 0.9999977 | 0.00257161 ± 0.00284984 | 0.00032552 ± 0.00059226 |
| 512 | 0.9994141 | 3.8980988 ± 0.1140638 | 3.8980837 ± 0.1140655 | 0.9999961 | 0.00332241 ± 0.00193177 | 0.00026054 ± 0.00056990 |
| 1024 | 1 | 2.8276444 ± 0.1081050 | 2.8276063 ± 0.1081131 | 0.9999865 | 0.00712891 ± 0.00479206 | 0.00039063 ± 0.00070701 |
| 2048 | 0.9992839 | 5.6687153 ± 0.1591597 | 3.9984284 ± 0.1579884 | 0.7053500 | 4.064229 ± 8.051941 | 0.00048854 ± 0.00135121 |
| 4096 | 1 | 4.0961378 ± 0.1120170 | 2.8974354 ± 0.1077941 | 0.7073579 | 2.525716 ± 5.232312 | 0.00048828 ± 0.00061496 |
| 8192 | 1 | 2.9696318 ± 0.0793433 | 2.0692133 ± 0.0691083 | 0.6967912 | 3.297656 ± 6.375443 | 0.00097656 ± 0.00214568 |
| 16384 | 1 | 4.1807329 ± 0.1238503 | 2.9814538 ± 0.1108139 | 0.7131414 | 5.772949 ± 10.092003 | 0.00231120 ± 0.00367624 |
| 32768 | 1 | 3.0172404 ± 0.0796185 | 2.1298918 ± 0.0537904 | 0.7059072 | 2.934701 ± 5.552041 | 0.00240885 ± 0.00253535 |

K≤1024에서는 두 경로의 RMSE가 거의 같았다. K≥2048에서 FPINT RMSE는
Conventional의 약 69.7–71.3%였다. FPINT mean ULP는 모든 K에서 Conventional보다
작았다. 이 비교는 Conventional의 Blackwell default reduced-precision reduction을
포함하므로 다른 GPU accumulation 설정으로 일반화하면 안 된다.

FPINT Torch/CUDA는 독립 QCOL reference all-close를 각각 270/270 case에서
통과했다.

## 결과 파일

- 측정 script: `measure_fpint_qcol_accuracy.py`
- Raw JSON:
  `/mnt/nfs-vlsi/jaeyongjang/results/fpint-mxu128-llama31-8b/random-qcol-fp64-errors.json`
- Raw CSV:
  `/mnt/nfs-vlsi/jaeyongjang/results/fpint-mxu128-llama31-8b/random-qcol-fp64-errors.csv`
