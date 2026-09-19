# Model-level GPU QDQ vs FPINT

- Compute / activation / output dtype: `FP16`
- Weight: symmetric GPTQ INT4, group size 128
- Scale / accumulator dtype: `FP16` / `FP32`
- Quantized checkpoint SHA256: `ddd707a14a4fddab498c224d746dbbaebb515b5633b2c22ad7469fc6133546a9`
- Rotation optimization dtype: `fp16`
- Rotation SHA256: `44498f11e4746f8e8a683209e6268d91428e90cf40c807eb73f784f81e7c1883`
- Quality gate: 없음(관측값 보고)

## Accuracy

| Task | Samples | Standard GPU QDQ | FPINT CUDA | Delta (pp) |
| --- | ---: | ---: | ---: | ---: |
| hellaswag | 10042 | 75.323641% | 75.323641% | +0.000000 |
| arc_easy | 2376 | 74.116162% | 74.116162% | +0.000000 |
| arc_challenge | 1172 | 45.392491% | 45.392491% | +0.000000 |
| winogrande | 1267 | 68.429361% | 68.350434% | -0.078927 |
| openbookqa | 500 | 43.600000% | 43.600000% | +0.000000 |
| **Micro average** | **15357** | **71.250895%** | **71.244384%** | **-0.006512** |

## WikiText perplexity

| Standard GPU QDQ | FPINT CUDA | Absolute delta | Relative delta |
| ---: | ---: | ---: | ---: |
| 9.43339305 | 9.43326214 | -0.00013091 | -0.001388% |

## Evaluation time

- Standard shard time sum: 399.19 s
- FPINT shard time sum: 31282.80 s


# MXU ROW 128 FP64-reference FP×INT 오차 비교

## 설정

- 상태: `pass`
- Shape: M=32, N=32
- K: 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768
- Trials: 30
- Finite target per K: 0.999
- FP16 raw field uniform sampling: sign [0, 1], exponent min 0, mantissa [0, 1023]
- Signed INT4 uniform sampling: [-8, 7]
- Scale=1, zero-point=0 (raw FP×INT)
- MXU row / group size: 128 / 128
- FP64 reference: FP64 activation × FP64-cast integer weight on GPU
- Conventional: FP16 activation × FP16-cast integer weight on GPU
- FPINT: QCOL_REAL_2SCOMP CUDA emulation
- RMSE는 unrounded FP64 reference, ULP는 correctly-rounded FP16 reference 기준
- 세 output이 모두 finite인 common mask에서 두 error를 paired 비교

## Finite coverage

| K | EXP max | Common finite | Fraction | FP16-ref non-finite | Conventional non-finite | FPINT non-finite | Target |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 128 | 24 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |
| 256 | 24 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |
| 512 | 24 | 30702 / 30720 | 0.99941406 | 18 | 18 | 18 | pass |
| 1024 | 23 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |
| 2048 | 23 | 30698 / 30720 | 0.99928385 | 22 | 22 | 22 | pass |
| 4096 | 22 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |
| 8192 | 21 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |
| 16384 | 21 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |
| 32768 | 20 | 30720 / 30720 | 1 | 0 | 0 | 0 | pass |

## RMSE / signed error

전체 Conventional relative L2 / max abs: n/a / n/a
전체 FPINT relative L2 / max abs: n/a / n/a

아래 `mean ± std`는 30개 trial metric의 평균과 sample std다.

| K | Conventional RMSE mean ± std | FPINT RMSE mean ± std | FPINT-Conv | FPINT/Conv | Conventional signed-error std | FPINT signed-error std |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1.9790936 ± 0.094778211 | 1.9790927 ± 0.09477826 | -9.3475154e-07 | 0.99999953 | 1.9812521 | 1.9812514 |
| 256 | 2.7551951 ± 0.14098659 | 2.7551887 ± 0.14098408 | -6.3571174e-06 | 0.99999769 | 2.7586177 | 2.7586138 |
| 512 | 3.8980988 ± 0.11406382 | 3.8980837 ± 0.11406549 | -1.512992e-05 | 0.99999612 | 3.8994955 | 3.8994506 |
| 1024 | 2.8276444 ± 0.10810497 | 2.8276063 ± 0.10811314 | -3.8141231e-05 | 0.99998651 | 2.8295515 | 2.8295197 |
| 2048 | 5.6687153 ± 0.15915973 | 3.9984284 ± 0.15798835 | -1.6702869 | 0.70535002 | 5.6708479 | 4.0013499 |
| 4096 | 4.0961378 ± 0.112017 | 2.8974354 ± 0.10779414 | -1.1987025 | 0.70735788 | 4.0975687 | 2.8992474 |
| 8192 | 2.9696318 ± 0.079343347 | 2.0692133 ± 0.069108283 | -0.90041847 | 0.69679121 | 2.9706145 | 2.0703022 |
| 16384 | 4.1807329 ± 0.12385031 | 2.9814538 ± 0.11081394 | -1.1992792 | 0.71314141 | 4.1823382 | 2.9834261 |
| 32768 | 3.0172404 ± 0.079618486 | 2.1298918 ± 0.053790351 | -0.88734861 | 0.70590722 | 3.0182454 | 2.1305314 |

## FP16 ULP error

| K | Conventional mean ULP ± std | FPINT mean ULP ± std | FPINT-Conv | FPINT/Conv | Conventional ULP std | FPINT ULP std | Conventional / FPINT p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.0014973958 ± 0.002388411 | 6.5104167e-05 ± 0.00024776185 | -0.0014322917 | 0.043478261 | 0.074811203 | 0.0080684526 | 0 / 0 |
| 256 | 0.0025716146 ± 0.0028498383 | 0.00032552083 ± 0.0005922641 | -0.0022460937 | 0.12658228 | 0.086677091 | 0.019761554 | 0 / 0 |
| 512 | 0.0033224133 ± 0.0019317713 | 0.00026054395 ± 0.00056989986 | -0.0030618694 | 0.078420089 | 0.069818606 | 0.016140057 | 0 / 0 |
| 1024 | 0.0071289062 ± 0.004792058 | 0.000390625 ± 0.00070700558 | -0.0067382812 | 0.054794521 | 0.16823211 | 0.02134424 | 0 / 0 |
| 2048 | 4.064229 ± 8.0519412 | 0.00048853587 ± 0.0013512138 | -4.0637404 | 0.00012020383 | 262.81663 | 0.038283871 | 3.8816667 / 0 |
| 4096 | 2.5257161 ± 5.2323124 | 0.00048828125 ± 0.00061496438 | -2.5252279 | 0.00019332388 | 170.14565 | 0.022091691 | 3.8333333 / 0 |
| 8192 | 3.2976563 ± 6.3754426 | 0.0009765625 ± 0.0021456805 | -3.2966797 | 0.00029613836 | 203.54244 | 0.067500652 | 3.7233333 / 0 |
| 16384 | 5.7729492 ± 10.092003 | 0.0023111979 ± 0.0036762441 | -5.770638 | 0.0004003496 | 340.4303 | 0.12458733 | 3.89 / 0 |
| 32768 | 2.9347005 ± 5.5520413 | 0.0024088542 ± 0.0025353469 | -2.9322917 | 0.00082081771 | 180.61575 | 0.083037576 | 3.8333333 / 0 |

## Correctness / reproducibility

- Activation sign-bit positive / negative: 31396120 / 31395560
- FPINT Torch vs QCOL reference all-close: 270/270
- FPINT CUDA vs QCOL reference all-close: 270/270
- Conventional reduced-precision reduction: `True`

CUDA kernel SHA256: `9df9a6766bc476cca91a6d9a615844e6e5cb104073e03e67871ccf9b6a5add71`

원본 JSON/CSV에는 case별 seed, field 분포, common finite mask, Conventional_err와 FP_INT_err가 기록되어 있다.
