#!/usr/bin/env python3
"""Render finite-coverage scale sweeps and standalone nonfinite heatmaps."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def percent(value):
    return f"{100 * value:.4f}%"


def plot(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    config = result["config"]
    maxima, ks = config["scale_exp_max_values"], config["k_values"]
    candidates = ("rounded_reference", "gpu_true", "gpu_false", "fpint")
    labels = ("Rounded FP64 reference", "GPU reduction=True", "GPU reduction=False", "FPINT")
    fig, axes = plt.subplots(1, 4, figsize=(16, max(6, len(maxima) * .24)), layout="constrained")
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#eeeeee")
    denominator = config["m"] * config["n"] * config["trials"]
    for ax, candidate, label in zip(axes, candidates, labels):
        values = np.array([
            [1 - result["summary"][str(e)]["by_k"][str(k)]["outputs"][candidate]["finite_fraction"] for k in ks]
            for e in maxima
        ])
        mesh = ax.imshow(
            np.ma.masked_equal(values, 0), aspect="auto", interpolation="nearest",
            cmap=cmap, norm=LogNorm(vmin=1 / denominator, vmax=1),
        )
        ax.set_xticks(range(len(ks)), [str(k) for k in ks], rotation=60, ha="right")
        ax.set_yticks(range(len(maxima)), [str(e) for e in maxima])
        ax.set_xlabel("K")
        ax.set_title(label)
    axes[0].set_ylabel("Scale exponent-field maximum (minimum = 0)")
    fig.colorbar(mesh, ax=axes, label="Non-finite fraction (gray = 0)")
    fig.suptitle(
        f"{config['activation_format'].upper()} scale exponent sweep; fixed activation bounds; "
        f"{config['trials']} trials/K"
    )
    fig.savefig(path, dpi=160)
    plt.close(fig)


def render(results, sources, figures, output):
    lines = [
        "# Scale exponent sweep: fixed activation, signed uniform fields",
        "",
        "[문서 목차](README.md) · [현재 GEMM 설정](mxu128_scaled_gemm_experiment.md)",
        "",
        "이 문서는 범위 탐색 기록이다. BF16 finite 경계 EXP=241과 현재 정확도 실험의 상한 EXP=127은 구분한다.",
        "",
        "Activation의 기존 K별 exponent 상한을 고정하고 scale exponent field의 상한만 넓힌 실험이다.",
        "Sign과 mantissa는 독립 uniform raw field로 샘플링하며 음수·양수·0·subnormal scale을 포함한다.",
        "이 scale은 양수 quantization scale 분포를 재현하기 위한 것이 아니라 산술 finite 범위를 시험하기 위한 것이다.",
        "",
        "## 공통 설정",
        "",
        "- Scale exponent field: discrete uniform [0, Emax], Inf/NaN 입력 field는 제외",
        "- Scale sign: uniform {0,1}; mantissa: FP16 [0,1023], BF16 [0,127]",
        "- Activation과 scale은 같은 dtype; signed INT4 [-8,7] uniform; zero-point=0; bias 없음",
        "- MXU rows=128, group size=128, extra bits=19/10; scale은 output channel × K-group마다 독립",
        "- 같은 K/trial은 모든 Emax에서 동일 activation/weight 및 scale sign/mantissa를 사용",
        "- Scale exponent도 동일 U~Uniform[0,1)에서 floor(U × (Emax+1))로 매핑",
        "- Scale 입력은 finite여도 dequantization, tile scaling, accumulation, output cast에서 Inf/NaN이 발생할 수 있음",
        "- FP64 reference, dtype-rounded reference, GPU reduction True/False, FPINT의 finite/±Inf/NaN을 각각 기록",
        "- 공통 finite는 위 다섯 출력의 교집합; finite mask로 원소를 제외하지 않으며 전체 출력 수가 항상 분모",
        "- 99.9% 기준은 이전 실험과 같은 K별 pooled-trial 기준이며 sweep 중단 조건이 아님",
        "- RMSE/ULP 순위가 아니라 finite coverage를 측정하는 실험",
        "",
        "## 측정 결과",
        "",
    ]
    lines.extend([
        "| Dtype | Largest tested Emax meeting target for every K | Worst-K common at that Emax | Next tested Emax | Worst-K common at next Emax |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ])
    for result in results:
        summary = result["summary"]
        maxima = result["config"]["scale_exp_max_values"]
        passing = [e for e in maxima if summary[str(e)]["all_k_meet_target"]]
        last = max(passing) if passing else None
        following = min((e for e in maxima if last is not None and e > last), default=None)
        worst = lambda e: percent(min(row["common_finite_fraction"] for row in summary[str(e)]["by_k"].values())) if e is not None else "none"
        lines.append(
            f"| {result['config']['activation_format'].upper()} | {last} | {worst(last)} | {following} | {worst(following)} |"
        )
    lines.append("")
    for result, source, figure in zip(results, sources, figures):
        config, env = result["config"], result["environment"]
        dtype = config["activation_format"]
        bias = 15 if dtype == "fp16" else 127
        maxima = config["scale_exp_max_values"]
        summary = result["summary"]
        passing = [e for e in maxima if summary[str(e)]["all_k_meet_target"]]
        checks = sum(row["overall"]["reference_checks"] for row in summary.values())
        passed = sum(row["overall"]["reference_checks_passed"] for row in summary.values())
        lines.extend([
            f"### {dtype.upper()}", "",
            f"- 실행 상태: {result['status']}; GPU: {env['device']}",
            f"- PyTorch {env['torch']}, CUDA {env['cuda']}; 완료 UTC: {env['completed_at_utc']}",
            f"- M={config['m']}, N={config['n']}; K={config['k_values']}; K당 {config['trials']} trials; base seed={config['base_seed']}",
            f"- Activation exponent max: {config['activation_exp_max_by_k']}",
            f"- Emax sweep: {maxima}; 총 {len(result['records'])} cases",
            f"- 모든 case의 FP64 reference finite: {all(row['outputs']['fp64']['finite'] == row['total'] for row in result['records'])}",
            f"- 모든 K에서 common finite ≥{percent(config['finite_target'])}인 시험 상한: {passing}",
            f"- CUDA/QCOL reference 검사: {passed}/{checks}; 각 Emax/K의 첫 {config['reference_trials']} trial 검사",
            "- 검사는 finite 값 exact equality, NaN 위치 일치, Inf 부호 일치 기준; NaN payload bit는 비교하지 않음",
            "",
            "| Scale Emax | Max normal exponent Emax−bias | Rounded ref finite | GPU True finite | GPU False finite | FPINT finite | Common finite | Worst-K common |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for exponent in maxima:
            row = summary[str(exponent)]
            overall = row["overall"]
            cells = [
                percent(overall["outputs"][key]["finite_fraction"])
                for key in ("rounded_reference", "gpu_true", "gpu_false", "fpint")
            ]
            cells += [
                percent(overall["common_finite_fraction"]),
                percent(min(r["common_finite_fraction"] for r in row["by_k"].values())),
            ]
            lines.append(f"| {exponent} | {exponent - bias} | " + " | ".join(cells) + " |")
        lines.extend([
            "",
            "전체 평균과 함께 worst-K를 확인한다. 아래 경계는 시험한 Emax에 대한 관측값이며 미측정 범위의 보장은 아니다.",
            "",
            "| K | Largest tested Emax with common finite ≥ target | First tested Emax below target |",
            "| ---: | ---: | ---: |",
        ])
        for k in config["k_values"]:
            good = [e for e in maxima if summary[str(e)]["by_k"][str(k)]["finite_target_met"]]
            bad = [e for e in maxima if not summary[str(e)]["by_k"][str(k)]["finite_target_met"]]
            lines.append(f"| {k} | {max(good) if good else 'none'} | {min(bad) if bad else 'none'} |")
        lines.extend([
            "",
            f"![{dtype.upper()} nonfinite fractions]({os.path.relpath(figure, output.parent)})",
            "",
            "그림은 nonfinite 비율을 log 색상으로 표시한다. 회색은 nonfinite=0이다. 세로축 간격은 시험한 field 값별 동일 간격이다.",
            "",
            f"Raw [JSON]({os.path.relpath(source, output.parent)}), "
            f"[case CSV]({os.path.relpath(source.with_suffix('.csv'), output.parent)}), "
            f"[K별 summary CSV]({os.path.relpath(source.with_name(source.stem + '-summary.csv'), output.parent)}).",
            "",
        ])
    lines.extend([
        "## 재현",
        "",
        "저장소 루트에서:",
        "",
        chr(96) * 3 + "bash",
        "conda activate spinquant",
        "CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 bash scripts/run_fpint_scale_sweep.sh",
        chr(96) * 3,
        "",
        "개별 범위는 measure_fpint_scale_sweep.py의 --scale-exp-max-values로 지정한다.",
        "예: --activation-format fp16 --scale-exp-max-values 15,16,17,18,19,20.",
        "Activation의 K별 상한은 변경하지 않는다. --reference-trials 30이면 모든 case를 QCOL reference와 검사한다.",
        "Raw 파일은 로컬 results/ 아래에 있으며 gitignore 대상이다. 실행 시 사용한 소스 SHA256은 JSON environment에 기록했다.",
        "",
        "## 해석",
        "",
        "- Emax는 IEEE 저장 exponent field 상한이다. Emax−bias는 정상수의 최대 실제 exponent이며 mantissa 때문에 최대 크기는 2^(Emax−bias+1) 미만이다.",
        "- BF16/FP16은 raw exponent 하한 0의 실제 값이 다르므로 동일 실수 입력에 대한 dtype 비교가 아니다.",
        "- 범위를 넓히면 scale 분포도 달라진다. 동일 sign/mantissa를 유지해도 cancellation 때문에 finite 비율의 단조성은 보장되지 않는다.",
        "- BF16 FPINT는 scale 곱과 K-tile 누적을 FP32로 수행한다. 최종 정확한 합이 finite여도 중간 단계는 overflow할 수 있다.",
        "- Rounded FP64 reference의 nonfinite는 출력 dtype 범위 초과를 보여준다. Candidate의 추가 nonfinite에는 중간 연산 순서의 영향도 포함된다.",
        "- Raw CSV는 ±Inf와 NaN을 분리하고, baseline dequantized weight의 nonfinite 수도 기록한다.",
        "- 99.9% 기준 충족은 이 seed/shape/분포에 대한 관측이며, 모든 입력의 finite 보장이 아니다.",
        "",
        "현재 사용할 범위는 [최종 설정](mxu128_scaled_gemm_experiment.md)에 정리했다.",
        "기존 양수 log-uniform 실험은 [archive](archive/log-uniform-20260922/mxu128_scaled_gemm_experiment.md)에 보존한다.",
        "",
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = [json.loads(source.read_text()) for source in args.inputs]
    figures = []
    for result in results:
        figure = args.output.with_name(args.output.stem + "_" + result["config"]["activation_format"] + ".png")
        plot(result, figure)
        figures.append(figure)
    args.output.write_text(render(results, args.inputs, figures, args.output))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
