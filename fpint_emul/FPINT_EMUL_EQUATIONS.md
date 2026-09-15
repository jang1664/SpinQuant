# FPINT GEMM Emulator 수식 정리

이 문서는 `py/fpint_emul.py`의 계산을 발표 자료에서 설명할 수 있도록 수식으로 정리한 것이다.
수식은 코드의 현재 동작을 기준으로 하며, 모든 FP16 입출력은 bit pattern으로 저장된다.

## 1. 표기법과 상수

| 기호 | 코드 | 의미 |
|---|---|---|
| $X_{m,k}$ | `input_data[m, k]` | FP16 activation |
| $W_{k,n}$ | `weight_data[k, n]` | signed INT4 weight 또는 encoded weight |
| $S$ | `scale_data` | FP16 scale |
| $Z$ | `zero_data` | signed zero point |
| $Y_{m,n}$ | `output_data[m, n]` | FP16 GEMM 결과 |
| $B$ | `QBLOCK` | quantization block 크기, $B=16$ |
| $T$ | `MXU_K` | 내적 tile 크기, $T=16$ |
| $P$ | `EXTRA_BIT` | main alignment 여유 비트, $P=19$ |
| $R$ | `EXTRA_BIT_FOR_REDUCE` | reduction alignment 여유 비트, $R=10$ |
| $b$ | `IN_EXP_BIAS` | FP16 exponent bias, $b=15$ |
| $F$ | `IN_MAN_WIDTH` | FP16 fraction 비트 수, $F=10$ |

두 alignment 결과의 fractional-bit 차이는 다음과 같다.

$$
\Delta = P-R = 19-10 = 9
$$

QCOL의 quantization group과 QROW의 quantization group은 각각 다음과 같다.

$$
g(k)=\left\lfloor\frac{k}{B}\right\rfloor,
\qquad
h(n)=\min\left(\left\lfloor\frac{n}{B}\right\rfloor,\;N_G-1\right)
$$

여기서 $N_G$는 실제 `scale_data` 또는 `zero_data`가 가진 group 수이다.

## 2. 고정소수점 변환

`FixedPointArray`에서 fractional bit 수가 $n_f$일 때 실수를 raw integer로 바꾸는 식은 다음과 같다.

$$
q=\operatorname{round}\left(x\,2^{n_f}\right)
$$

raw integer를 다시 실수로 바꾸는 식은 다음과 같다.

$$
\hat{x}=q\,2^{-n_f}=\frac{q}{2^{n_f}}
$$

## 3. FP16 분해와 값 복원

16-bit FP16 bit pattern을 $u$라고 하면 sign, exponent, fraction은 다음과 같이 추출된다.

$$
s=(u\gg15)\mathbin{\&}1
$$

$$
e=(u\gg10)\mathbin{\&}31
$$

$$
f=u\mathbin{\&}1023
$$

hidden bit와 보정 exponent를 다음과 같이 정의한다.

$$
h=\begin{cases}
0, & e=0 \quad\text{(subnormal 또는 zero)}\\
1, & e\ne0 \quad\text{(normal)}
\end{cases}
$$

$$
\bar{e}=\max(e,1)
$$

11-bit significand integer는 다음과 같다.

$$
H=h\,2^{10}+f
$$

따라서 finite FP16 값은 다음 식으로 표현할 수 있다.

$$
x=(-1)^s H\,2^{\bar{e}-15-10}
$$

이 표현은 normal과 subnormal을 하나의 식으로 다룬다.

## 4. Pre-alignment

각 $K$-tile $\mathcal{T}_t=\{16t,\ldots,16t+15\}$ 안에서 가장 큰 보정 exponent를 공통 exponent로 선택한다.

$$
E_{m,t}=\max_{k\in\mathcal{T}_t}\bar{e}_{m,k}
$$

extra bit 수가 $p$일 때 각 significand를 공통 exponent에 맞춘 signed integer는 다음과 같다.

$$
A^{(p)}_{m,k}
=(-1)^{s_{m,k}}
\left\lfloor
\frac{H_{m,k}\,2^p}
{2^{E_{m,t}-\bar{e}_{m,k}}}
\right\rfloor,
\qquad k\in\mathcal{T}_t
$$

코드는 양의 significand를 먼저 logical right shift한 뒤 sign을 적용한다. 따라서 위 식의 floor는
절댓값에 적용되며, shift에서 버려지는 하위 비트는 반올림되지 않는다.

정렬된 integer를 실수 영역으로 복원하는 scale factor는 다음과 같다.

$$
C(E,p)=2^{E-15}\,2^{-(10+p)}
=2^{E-15-10-p}
$$

따라서 원래 activation의 근삿값은 다음과 같다.

$$
x_{m,k}\approx A^{(p)}_{m,k}\,C(E_{m,t},p)
$$

main path와 reduction path에서는 각각 다음 integer를 사용한다.

$$
A_{m,k}=A^{(P)}_{m,k},
\qquad
R_{m,k}=A^{(R)}_{m,k}
$$

reduction path를 main path의 binary point로 맞추는 연산은 다음과 같다.

$$
R_{m,k}\ll\Delta=R_{m,k}\,2^{P-R}
$$

## 5. 공통 GEMM 기준식

양자화 weight의 dequantization은 다음과 같다.

$$
W^{\mathrm{deq}}=S\,(W-Z)
$$

따라서 GEMM의 공통 기준식은 다음과 같다.

$$
Y_{m,n}=\sum_{k=0}^{K-1}X_{m,k}\,S\,(W_{k,n}-Z)
$$

### QCOL

QCOL에서는 $K$ 방향의 16개 weight가 scale과 zero point를 공유한다.

$$
Y^{\mathrm{QCOL}}_{m,n}
=\sum_{k=0}^{K-1}
X_{m,k}\,S_{g(k),n}
\left(W_{k,n}-Z_{g(k),n}\right)
$$

### QROW

QROW에서는 $N$ 방향의 16개 weight가 scale과 zero point를 공유한다.

$$
Y^{\mathrm{QROW}}_{m,n}
=\sum_{k=0}^{K-1}
X_{m,k}\,S_{k,h(n)}
\left(W_{k,n}-Z_{k,h(n)}\right)
$$

## 6. Reference 구현의 정밀도

$\mathcal{R}_{16}(x)$와 $\mathcal{R}_{32}(x)$를 각각 FP16, FP32 반올림이라고 정의한다.

### `fpint_gemm_gpu`

각 항은 FP16 곱셈 두 번을 거친 후 FP32로 누적된다.

$$
d_{k,n}=\mathcal{R}_{16}(W_{k,n}-Z)
$$

$$
q_{k,n}=\mathcal{R}_{16}(S\,d_{k,n})
$$

$$
p_{m,k,n}=\mathcal{R}_{16}(X_{m,k}\,q_{k,n})
$$

$$
a_{m,n}^{(k+1)}
=\mathcal{R}_{32}\left(a_{m,n}^{(k)}+p_{m,k,n}\right),
\qquad a_{m,n}^{(0)}=0
$$

$$
Y_{m,n}=\mathcal{R}_{16}\left(a_{m,n}^{(K)}\right)
$$

### `fpint_gemm_ref`

입력 FP16 값을 FP64로 변환한 후 전체 식을 FP64로 계산하고 마지막에 한 번만 FP16으로 변환한다.

$$
Y_{m,n}
=\mathcal{R}_{16}\left(
\sum_{k=0}^{K-1}
\operatorname{FP64}(X_{m,k})
\operatorname{FP64}(S)
\operatorname{FP64}(W_{k,n}-Z)
\right)
$$

## 7. FPINT 공통 tile 계산

이하 수식은 하나의 16-element tile $\mathcal{T}_t$에 대한 계산이다. 최종 출력은 모든 tile의
실수 contribution $Q_t$를 누적한 뒤 FP16으로 변환한다.

$$
Y_{m,n}=\mathcal{R}_{16}\left(\sum_t Q_{m,n,t}\right)
$$

integer post-processing 결과를 $J_{m,n,t}$라고 하면 실수 복원값은 다음과 같다.

$$
F_{m,n,t}=J_{m,n,t}\,C(E_{m,t},P)
$$

즉, 코드의 `post_inner_product_fp`는 다음 식이다.

$$
F_{m,n,t}
=J_{m,n,t}
\,2^{E_{m,t}-15}
\,2^{-(10+19)}
$$

## 8. QCOL 구현

QCOL에서는 scale을 integer 내적 이후에 곱한다. 따라서 pre-alignment 입력은 원래 activation이다.

$$
U_{m,k}=X_{m,k}
$$

### 8.1 QCOL 2's-complement encoding

tile 내의 세 가지 정수 합을 정의한다.

$$
I=\sum_{k\in\mathcal{T}_t}A_{m,k}W_{k,n}
$$

$$
A_\Sigma=\sum_{k\in\mathcal{T}_t}A_{m,k}
$$

$$
R_\Sigma=\sum_{k\in\mathcal{T}_t}R_{m,k}
$$

코드의 post-processing은 다음과 같다.

$$
J
=2I+A_\Sigma
+\left(-1-2Z_{g,n}\right)
\left(R_\Sigma\,2^\Delta\right)
$$

복원 후 scale과 $1/2$을 적용한다.

$$
Q_{m,n,t}=\frac{S_{g,n}}{2}\,F_{m,n,t}
$$

alignment 오차를 제외하면 위 두 식은 다음 weight 보정 형태를 구현한다.

$$
Q_{m,n,t}\approx
S_{g,n}\sum_{k\in\mathcal{T}_t}
X_{m,k}\left(W_{k,n}-Z_{g,n}\right)
$$

### 8.2 QCOL zero-less encoding

저장된 odd weight $W^{\mathrm{odd}}$를 내부 signed weight $Q$로 변환한다.

$$
Q_{k,n}=\left\lfloor\frac{W^{\mathrm{odd}}_{k,n}-1}{2}\right\rfloor
$$

odd weight는 다음과 같이 복원된다.

$$
W^{\mathrm{odd}}_{k,n}=2Q_{k,n}+1
$$

정수 합과 post-processing은 다음과 같다.

$$
I=\sum_{k\in\mathcal{T}_t}A_{m,k}Q_{k,n},
\qquad
A_\Sigma=\sum_{k\in\mathcal{T}_t}A_{m,k},
\qquad
R_\Sigma=\sum_{k\in\mathcal{T}_t}R_{m,k}
$$

$$
J=2I+A_\Sigma-Z_{g,n}\left(R_\Sigma\,2^\Delta\right)
$$

$$
Q_{m,n,t}=S_{g,n}\,F_{m,n,t}
$$

따라서 alignment 오차를 제외하면 다음 식과 같다.

$$
Q_{m,n,t}\approx
S_{g,n}\sum_{k\in\mathcal{T}_t}
X_{m,k}\left(W^{\mathrm{odd}}_{k,n}-Z_{g,n}\right)
$$

### 8.3 QCOL real 2's-complement encoding

signed weight를 변환 없이 직접 사용한다.

$$
I=\sum_{k\in\mathcal{T}_t}A_{m,k}W_{k,n},
\qquad
R_\Sigma=\sum_{k\in\mathcal{T}_t}R_{m,k}
$$

$$
J=I-Z_{g,n}\left(R_\Sigma\,2^\Delta\right)
$$

$$
Q_{m,n,t}=S_{g,n}\,F_{m,n,t}
$$

## 9. QROW 구현

QROW에서는 $k$마다 scale이 다르므로 scale을 activation에 먼저 곱하고 FP16으로 반올림한다.

$$
U_{m,k,n}
=\mathcal{R}_{16}\left(X_{m,k}S_{k,h(n)}\right)
$$

이후 $U_{m,k,n}$를 main path와 reduction path로 각각 pre-align한다.

$$
A_{m,k,n}=\operatorname{Prealign}_{P}(U_{m,k,n})
$$

$$
R_{m,k,n}=\operatorname{Prealign}_{R}(U_{m,k,n})
$$

따라서 QROW의 post-processing 이후에는 별도의 scale 곱셈이 없다.

### 9.1 QROW 2's-complement encoding

$$
I=\sum_{k\in\mathcal{T}_t}A_{m,k,n}W_{k,n}
$$

$$
A_\Sigma=\sum_{k\in\mathcal{T}_t}A_{m,k,n}
$$

zero correction을 포함한 reduction 합은 다음과 같다.

$$
R_Z
=\sum_{k\in\mathcal{T}_t}
R_{m,k,n}\left(1+2Z_{k,h(n)}\right)
$$

$$
J=2I+A_\Sigma-R_Z\,2^\Delta
$$

$$
Q_{m,n,t}=\frac{1}{2}F_{m,n,t}
$$

alignment 오차를 제외하면 다음 식을 구현한다.

$$
Q_{m,n,t}\approx
\sum_{k\in\mathcal{T}_t}
X_{m,k}S_{k,h(n)}
\left(W_{k,n}-Z_{k,h(n)}\right)
$$

### 9.2 QROW zero-less encoding

$$
Q_{k,n}=\left\lfloor\frac{W^{\mathrm{odd}}_{k,n}-1}{2}\right\rfloor
$$

$$
I=\sum_{k\in\mathcal{T}_t}A_{m,k,n}Q_{k,n}
$$

$$
A_\Sigma=\sum_{k\in\mathcal{T}_t}A_{m,k,n}
$$

$$
R_Z
=\sum_{k\in\mathcal{T}_t}
R_{m,k,n}Z_{k,h(n)}
$$

$$
J=2I+A_\Sigma-R_Z\,2^\Delta
$$

$$
Q_{m,n,t}=F_{m,n,t}
$$

alignment 오차를 제외하면 다음 식을 구현한다.

$$
Q_{m,n,t}\approx
\sum_{k\in\mathcal{T}_t}
X_{m,k}S_{k,h(n)}
\left(W^{\mathrm{odd}}_{k,n}-Z_{k,h(n)}\right)
$$

### 9.3 QROW real 2's-complement encoding

$$
I=\sum_{k\in\mathcal{T}_t}A_{m,k,n}W_{k,n}
$$

$$
R_Z
=\sum_{k\in\mathcal{T}_t}
R_{m,k,n}Z_{k,h(n)}
$$

$$
J=I-R_Z\,2^\Delta
$$

$$
Q_{m,n,t}=F_{m,n,t}
$$

## 10. Formula flow

$$
Y_{m,n}
=\sum_k X_{m,k}\,S\left(W_{k,n}-Z\right)
$$

$$
Y_{m,n}
=\sum_k S\left(X_{m,k}W_{k,n}-X_{m,k}Z\right)
$$

$$
x=(-1)^s\left(h2^F+f\right)2^{\bar e-b-F},
\qquad
\bar e=\max(e,1)
$$

$$
E_t=\max_{k\in\mathcal T_t}\bar e_k
$$

$$
A_k^{(p)}
=(-1)^{s_k}
\left\lfloor
\frac{(h_k2^F+f_k)2^p}
{2^{E_t-\bar e_k}}
\right\rfloor
$$

$$
C_t=C(E_t,P)=2^{E_t-b-F-P}
$$

$$
X_k\approx A_k^{(P)}C_t
\approx\left(A_k^{(R)}2^{P-R}\right)C_t
$$

$$
A_k=A_k^{(P)},
\qquad
R_k=A_k^{(R)},
\qquad
\Delta=P-R
$$

$$
\boxed{
Y_{m,n}=\mathcal R_{16}\left(\sum_t Q_{m,n,t}\right)
}
$$

$$
\boxed{\mathrm{QCOL}}
$$

$$
Y_{m,n}
=\sum_t S_{t,n}
\sum_{k\in\mathcal T_t}
X_{m,k}\left(W_{k,n}-Z_{t,n}\right)
$$

$$
Y_{m,n}
\approx
\sum_t S_{t,n}C_t
\left[
\sum_{k\in\mathcal T_t}A_{m,k}W_{k,n}
-Z_{t,n}2^\Delta
\sum_{k\in\mathcal T_t}R_{m,k}
\right]
$$

$$
I_W=\sum_{k\in\mathcal T_t}A_{m,k}W_{k,n},
\qquad
A_\Sigma=\sum_{k\in\mathcal T_t}A_{m,k},
\qquad
R_\Sigma=\sum_{k\in\mathcal T_t}R_{m,k}
$$

$$
\begin{aligned}
J_{\mathrm{real}}
&=I_W-Z_{t,n}2^\Delta R_\Sigma,\\
Q_{m,n,t}^{\mathrm{real}}
&=S_{t,n}C_tJ_{\mathrm{real}}
\end{aligned}
$$

$$
Q_{k,n}^{\mathrm{odd}}
=\left\lfloor\frac{W_{k,n}^{\mathrm{odd}}-1}{2}\right\rfloor,
\qquad
W_{k,n}^{\mathrm{odd}}=2Q_{k,n}^{\mathrm{odd}}+1
$$

$$
I_Q=\sum_{k\in\mathcal T_t}A_{m,k}Q_{k,n}^{\mathrm{odd}}
$$

$$
\begin{aligned}
J_{\mathrm{odd}}
&=2I_Q+A_\Sigma-Z_{t,n}2^\Delta R_\Sigma,\\
Q_{m,n,t}^{\mathrm{odd}}
&=S_{t,n}C_tJ_{\mathrm{odd}}
\end{aligned}
$$

$$
\begin{aligned}
J_{\mathrm{2s}}
&=2I_W+A_\Sigma
+\left(-1-2Z_{t,n}\right)2^\Delta R_\Sigma,\\
Q_{m,n,t}^{\mathrm{2s}}
&=\frac{S_{t,n}}{2}C_tJ_{\mathrm{2s}}
\end{aligned}
$$

$$
\boxed{\mathrm{QROW}}
$$

$$
U_{m,k,n}
=\mathcal R_{16}\left(X_{m,k}S_{k,h(n)}\right)
$$

$$
Y_{m,n}
=\sum_t\sum_{k\in\mathcal T_t}
U_{m,k,n}\left(W_{k,n}-Z_{k,h(n)}\right)
$$

$$
U_{m,k,n}
\approx A_{m,k,n}C_t
\approx\left(R_{m,k,n}2^\Delta\right)C_t
$$

$$
I_W=\sum_{k\in\mathcal T_t}A_{m,k,n}W_{k,n},
\qquad
A_\Sigma=\sum_{k\in\mathcal T_t}A_{m,k,n}
$$

$$
R_Z
=\sum_{k\in\mathcal T_t}R_{m,k,n}Z_{k,h(n)}
$$

$$
\begin{aligned}
J_{\mathrm{real}}
&=I_W-2^\Delta R_Z,\\
Q_{m,n,t}^{\mathrm{real}}
&=C_tJ_{\mathrm{real}}
\end{aligned}
$$

$$
I_Q=\sum_{k\in\mathcal T_t}A_{m,k,n}Q_{k,n}^{\mathrm{odd}}
$$

$$
\begin{aligned}
J_{\mathrm{odd}}
&=2I_Q+A_\Sigma-2^\Delta R_Z,\\
Q_{m,n,t}^{\mathrm{odd}}
&=C_tJ_{\mathrm{odd}}
\end{aligned}
$$

$$
R_{1+2Z}
=\sum_{k\in\mathcal T_t}
R_{m,k,n}\left(1+2Z_{k,h(n)}\right)
$$

$$
\begin{aligned}
J_{\mathrm{2s}}
&=2I_W+A_\Sigma-2^\Delta R_{1+2Z},\\
Q_{m,n,t}^{\mathrm{2s}}
&=\frac{1}{2}C_tJ_{\mathrm{2s}}
\end{aligned}
$$
