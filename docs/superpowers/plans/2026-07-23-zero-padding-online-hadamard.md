# Zero-Padding Online Hadamard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-preserving `factorized`/`zero_padding` online R4 Hadamard mode, run W4A8KV4 PTQ for three Llama models with both modes, and summarize accuracy deltas.

**Architecture:** A single mode-aware dispatcher in `utils/hadamard_utils.py` owns transform selection and normalization. PTQ and rotation-training paths pass the parsed mode to both the offline `down_proj` weight transform and the online activation wrapper. Separate utilities run mode-isolated experiments and summarize lm-eval JSON without changing the existing evaluation payload.

**Tech Stack:** Python 3.10, PyTorch, CUDA, `fast-hadamard-transform`, Hugging Face Transformers, pytest, Bash, lm-evaluation-harness.

## Global Constraints

- `--online_had_mode` accepts exactly `factorized` and `zero_padding`.
- The default is exactly `factorized`; existing commands retain their behavior.
- Both modes normalize by `sqrt(n)`, where `n` is the original transform dimension.
- The zero-padding mode uses the fast CUDA kernel's implicit padding and cropping; it does not restore FP equivalence.
- Offline `down_proj` weights and online `down_proj` activations always use the same selected mode.
- The first comparison reuses the same optimized R1/R2 checkpoints for both modes.
- W4A8KV4 settings remain W4, A8, K4, V4, GPTQ with weight clipping, asymmetric A/K/V, activation group size `-1`, and K/V group size `128`.
- Existing user changes in `ptq.py`, `requirement.txt`, `utils/process_args.py`, experiment scripts, and result files must be preserved.
- GPU kernel tests skip explicitly if CUDA or `fast-hadamard-transform` is unavailable.

---

### Task 1: Add the mode contract and transform dispatcher

**Files:**
- Create: `tests/test_hadamard_modes.py`
- Modify: `utils/process_args.py:47-83`
- Modify: `utils/hadamard_utils.py:14-113`

**Interfaces:**
- Produces: `ONLINE_HAD_MODE_FACTORIZED: str`
- Produces: `ONLINE_HAD_MODE_ZERO_PADDING: str`
- Produces: `ONLINE_HAD_MODES: tuple[str, str]`
- Produces: `matmul_hadU_cuda(X: torch.Tensor, hadK: torch.Tensor | None = None, K: int | None = None, mode: str = "factorized") -> torch.Tensor`
- Preserves: existing positional calls `matmul_hadU_cuda(X, hadK, K)`

- [ ] **Step 1: Write failing parser and dispatcher contract tests**

Create `tests/test_hadamard_modes.py`:

```python
import sys

import pytest
import torch

from utils import hadamard_utils
from utils.process_args import parser_gen


def parse_spinquant_args(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["ptq.py", *args])
    parsed, unknown = parser_gen()
    assert unknown == []
    return parsed


def test_online_hadamard_mode_defaults_to_factorized(monkeypatch):
    args = parse_spinquant_args(monkeypatch)
    assert args.online_had_mode == "factorized"


def test_online_hadamard_mode_accepts_zero_padding(monkeypatch):
    args = parse_spinquant_args(
        monkeypatch, "--online_had_mode", "zero_padding"
    )
    assert args.online_had_mode == "zero_padding"


def test_online_hadamard_mode_rejects_unknown_value(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["ptq.py", "--online_had_mode", "unknown"],
    )
    with pytest.raises(SystemExit):
        parser_gen()


def test_factorized_dispatch_matches_legacy_call():
    x = torch.randn(2, 12)
    expected = hadamard_utils.matmul_hadU(x)
    actual = hadamard_utils.matmul_hadU_dispatch(x, mode="factorized")
    torch.testing.assert_close(actual, expected)


def test_dispatch_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported online Hadamard mode"):
        hadamard_utils.matmul_hadU_dispatch(
            torch.randn(2, 12), mode="unknown"
        )
```

- [ ] **Step 2: Run the focused tests and confirm contract failures**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_hadamard_modes.py -k "mode_defaults or mode_accepts or dispatch" -v
```

Expected: FAIL because `online_had_mode`, `matmul_hadU_dispatch`, and the mode constants do not exist.

- [ ] **Step 3: Add the CLI enum without disturbing existing result-path changes**

In `utils/process_args.py`, immediately after `--fp32_had`, add:

```python
    parser.add_argument(
        "--online_had_mode",
        type=str,
        default="factorized",
        choices=["factorized", "zero_padding"],
        help=(
            "Online R4 Hadamard implementation for down_proj: "
            "factorized (default) or fast-kernel implicit zero padding"
        ),
    )
```

Keep the existing `--results_path` argument unchanged.

- [ ] **Step 4: Add mode constants, validation, and CPU dispatch**

In `utils/hadamard_utils.py`, add:

```python
ONLINE_HAD_MODE_FACTORIZED = "factorized"
ONLINE_HAD_MODE_ZERO_PADDING = "zero_padding"
ONLINE_HAD_MODES = (
    ONLINE_HAD_MODE_FACTORIZED,
    ONLINE_HAD_MODE_ZERO_PADDING,
)


def validate_online_had_mode(mode):
    if mode not in ONLINE_HAD_MODES:
        raise ValueError(
            f"Unsupported online Hadamard mode {mode!r}; "
            f"expected one of {ONLINE_HAD_MODES}"
        )


def matmul_hadU_dispatch(X, mode=ONLINE_HAD_MODE_FACTORIZED):
    validate_online_had_mode(mode)
    if mode == ONLINE_HAD_MODE_ZERO_PADDING:
        if not X.is_cuda:
            raise RuntimeError(
                "zero_padding online Hadamard requires a CUDA tensor "
                "and fast-hadamard-transform"
            )
        n = X.shape[-1]
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(
            n, device=X.device, dtype=torch.float32
        ).sqrt()
    return matmul_hadU(X)
```

This helper gives CPU tests a stable dispatch seam. CUDA call sites will use the
mode-aware `matmul_hadU_cuda` below.

- [ ] **Step 5: Make the CUDA transform mode-aware while preserving old calls**

Replace the existing `matmul_hadU_cuda` body with:

```python
@profile("matmul_hadU_cuda")
def matmul_hadU_cuda(
    X,
    hadK=None,
    K=None,
    mode=ONLINE_HAD_MODE_FACTORIZED,
):
    validate_online_had_mode(mode)
    n = X.shape[-1]

    if mode == ONLINE_HAD_MODE_ZERO_PADDING:
        if not X.is_cuda:
            raise RuntimeError(
                "zero_padding online Hadamard requires a CUDA tensor "
                "and fast-hadamard-transform"
            )
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(
            n, device=X.device, dtype=torch.float32
        ).sqrt()

    if K is None:
        hadK, K = get_hadK(n)
    if K == 1:
        return HadamardTransform.apply(X.contiguous()) / torch.tensor(
            n, device=X.device, dtype=torch.float32
        ).sqrt()

    input = X.view(-1, K, n // K)
    input = HadamardTransform.apply(input.contiguous()) / torch.tensor(
        n, device=X.device, dtype=torch.float32
    ).sqrt()
    input = hadK.to(input.device).to(input.dtype) @ input
    return input.reshape(X.shape)
```

Keep `matmul_hadUt_cuda` behavior unchanged except for forwarding named
arguments if its currently invalid `transpose` call is encountered by tests;
do not refactor unrelated paths.

- [ ] **Step 6: Run CPU contract tests**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_hadamard_modes.py -k "not cuda" -v
```

Expected: all selected tests PASS.

- [ ] **Step 7: Commit only Task 1 hunks**

Because `utils/process_args.py` already contains user changes, inspect and stage
only the new mode hunk:

```bash
git diff -- tests/test_hadamard_modes.py utils/process_args.py utils/hadamard_utils.py
git add tests/test_hadamard_modes.py utils/hadamard_utils.py
git add -p utils/process_args.py
git diff --cached --check
git commit -m "feat: add online hadamard mode dispatcher"
```

---

### Task 2: Verify fast-kernel zero padding numerically

**Files:**
- Modify: `tests/test_hadamard_modes.py`

**Interfaces:**
- Consumes: `matmul_hadU_cuda(..., mode: str) -> torch.Tensor`
- Verifies: non-power-of-two direct kernel behavior and power-of-two mode equality

- [ ] **Step 1: Add GPU reference tests**

Append to `tests/test_hadamard_modes.py`:

```python
CUDA_READY = torch.cuda.is_available()


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
@pytest.mark.parametrize("dim", [137, 11008, 14336])
def test_zero_padding_matches_direct_fast_kernel(dim):
    from fast_hadamard_transform import hadamard_transform

    torch.manual_seed(0)
    x = torch.randn(2, dim, device="cuda", dtype=torch.float32)
    expected = hadamard_transform(x.contiguous()) / torch.tensor(
        dim, device=x.device, dtype=torch.float32
    ).sqrt()
    actual = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
@pytest.mark.parametrize("dim", [128, 8192])
def test_modes_match_for_power_of_two_dimensions(dim):
    torch.manual_seed(0)
    x = torch.randn(2, dim, device="cuda", dtype=torch.float32)
    factorized = hadamard_utils.matmul_hadU_cuda(
        x, mode="factorized"
    )
    zero_padding = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    torch.testing.assert_close(zero_padding, factorized)


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
def test_non_power_of_two_modes_are_distinct():
    torch.manual_seed(0)
    x = torch.randn(2, 11008, device="cuda", dtype=torch.float32)
    factorized = hadamard_utils.matmul_hadU_cuda(
        x, mode="factorized"
    )
    zero_padding = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    assert not torch.allclose(zero_padding, factorized)
```

- [ ] **Step 2: Run the GPU tests**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_hadamard_modes.py -k "kernel or power_of_two" -v
```

Expected: all selected tests PASS; if no CUDA is visible they are SKIPPED with
the explicit reason.

- [ ] **Step 3: Commit GPU verification**

```bash
git add tests/test_hadamard_modes.py
git diff --cached --check
git commit -m "test: verify zero-padding hadamard kernel"
```

---

### Task 3: Wire the mode through offline weights and online activations

**Files:**
- Modify: `utils/hadamard_utils.py:139-185`
- Modify: `utils/quant_utils.py:238-285`
- Modify: `eval_utils/rotation_utils.py:78-113`
- Modify: `eval_utils/main.py:24-42`
- Modify: `train_utils/apply_r3_r4.py:23-47`
- Modify: `train_utils/main.py:20-37`
- Modify: `tests/test_hadamard_modes.py`

**Interfaces:**
- Consumes: `args.online_had_mode: str`
- Produces: `ActQuantWrapper.online_had_mode: str`
- Produces: `apply_exact_had_to_linear(..., online_had_mode: str = "factorized")`
- Guarantees: the selected mode reaches both offline R4 and online R4

- [ ] **Step 1: Write failing wrapper and weight-transform tests**

Append to `tests/test_hadamard_modes.py`:

```python
from unittest import mock

from utils import quant_utils


def test_act_quant_wrapper_defaults_to_factorized():
    wrapper = quant_utils.ActQuantWrapper(torch.nn.Linear(12, 4))
    assert wrapper.online_had_mode == "factorized"


@pytest.mark.skipif(not CUDA_READY, reason="CUDA is required")
def test_weight_and_activation_use_zero_padding():
    torch.manual_seed(0)
    linear = torch.nn.Linear(12, 4, bias=False).cuda().float()
    wrapper = quant_utils.ActQuantWrapper(linear)
    wrapper.online_full_had = True
    wrapper.online_had_mode = "zero_padding"

    x = torch.randn(2, 12, device="cuda")
    expected_x = hadamard_utils.matmul_hadU_cuda(
        x, mode="zero_padding"
    )
    expected_output = linear(expected_x)
    actual_output = wrapper(x)
    torch.testing.assert_close(actual_output, expected_output)

    original_weight = linear.weight.detach().clone()
    expected_weight = hadamard_utils.matmul_hadU_cuda(
        original_weight, mode="zero_padding"
    )
    linear.weight.data.copy_(original_weight)
    hadamard_utils.apply_exact_had_to_linear(
        linear, online_had_mode="zero_padding"
    )
    torch.testing.assert_close(linear.weight, expected_weight)
```

- [ ] **Step 2: Run tests and confirm missing wiring**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_hadamard_modes.py -k "wrapper or weight_and_activation" -v
```

Expected: FAIL because `online_had_mode` is not stored or forwarded.

- [ ] **Step 3: Add mode support to offline weight transformation**

Change the signature in `utils/hadamard_utils.py`:

```python
def apply_exact_had_to_linear(
    module,
    had_dim=-1,
    output=False,
    R2=None,
    online_had_mode=ONLINE_HAD_MODE_FACTORIZED,
):
```

At the beginning validate `online_had_mode`. In the `had_dim == -1` branches,
initialize factorization metadata only for factorized mode:

```python
        had_K, K = (None, None)
        if online_had_mode == ONLINE_HAD_MODE_FACTORIZED:
            dimension = out_features if output else in_features
            had_K, K = get_hadK(dimension)
```

Then call:

```python
            W_ = matmul_hadU_cuda(
                W_.t(),
                had_K,
                K,
                mode=online_had_mode,
            ).t()
```

or:

```python
            W_ = matmul_hadU_cuda(
                W_,
                had_K,
                K,
                mode=online_had_mode,
            )
```

Leave the `had_dim != -1` R2 path unchanged.

- [ ] **Step 4: Store and use the mode in the activation wrapper**

In `ActQuantWrapper.__init__` in `utils/quant_utils.py` add:

```python
    self.online_had_mode = hadamard_utils.ONLINE_HAD_MODE_FACTORIZED
```

In both FP32 and FP16 `online_full_had` calls, pass:

```python
mode=self.online_had_mode
```

- [ ] **Step 5: Wire evaluation rotation and activation setup**

In `eval_utils/rotation_utils.py`, change:

```python
def rotate_mlp_output(layer, R1, online_had_mode):
```

and call:

```python
  apply_exact_had_to_linear(
      W,
      had_dim=-1,
      output=False,
      online_had_mode=online_had_mode,
  )
```

In `rotate_model`, pass `args.online_had_mode` to `rotate_mlp_output`.

In `eval_utils/main.py`, configure each `down_proj`:

```python
                qlayers[name].online_had_mode = args.online_had_mode
                if (
                    args.online_had_mode
                    == hadamard_utils.ONLINE_HAD_MODE_FACTORIZED
                ):
                    had_K, K = hadamard_utils.get_hadK(
                        model.config.intermediate_size
                    )
                    qlayers[name].had_K = had_K
                    qlayers[name].K = K
```

Set `online_full_had` and `fp32_had` as before. Do not call `get_hadK` in the
zero-padding branch.

- [ ] **Step 6: Wire rotation-training preparation**

In `train_utils/apply_r3_r4.py`, change:

```python
def R4_rotate_down_proj_weights(layer, online_had_mode):
    apply_exact_had_to_linear(
        layer.mlp.down_proj,
        had_dim=-1,
        output=False,
        online_had_mode=online_had_mode,
    )
```

Pass `args.online_had_mode` from `rotate_model`.

In `train_utils/main.py`, mirror the mode assignment and factorized-only
metadata initialization from `eval_utils/main.py`.

- [ ] **Step 7: Run all focused mode tests**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_hadamard_modes.py -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit the wiring**

```bash
git add utils/hadamard_utils.py utils/quant_utils.py \
  eval_utils/rotation_utils.py eval_utils/main.py \
  train_utils/apply_r3_r4.py train_utils/main.py \
  tests/test_hadamard_modes.py
git diff --cached --check
git commit -m "feat: apply selected hadamard mode to R4"
```

---

### Task 4: Add lm-eval comparison summarization

**Files:**
- Create: `summarize_hadamard_comparison.py`
- Create: `tests/test_summarize_hadamard_comparison.py`

**Interfaces:**
- Produces: `load_metrics(path: pathlib.Path) -> dict[str, float]`
- Produces: `build_rows(factorized_dir: pathlib.Path, zero_padding_dir: pathlib.Path, models: list[str]) -> list[dict[str, object]]`
- Produces: Markdown table on stdout
- Metric selection: normalized accuracy for ARC/HellaSwag/OpenBookQA, accuracy for Winogrande, word perplexity for WikiText

- [ ] **Step 1: Write failing fixture-based summary tests**

Create `tests/test_summarize_hadamard_comparison.py`:

```python
import json

import pytest

from summarize_hadamard_comparison import build_rows, load_metrics


def write_result(path, offset):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "results": {
            "hellaswag": {"acc_norm,none": 0.70 + offset},
            "arc_easy": {"acc_norm,none": 0.60 + offset},
            "arc_challenge": {"acc_norm,none": 0.40 + offset},
            "winogrande": {"acc,none": 0.65 + offset},
            "openbookqa": {"acc_norm,none": 0.45 + offset},
            "wikitext": {"word_perplexity,none": 9.0 + offset},
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_metrics_selects_expected_fields(tmp_path):
    result = tmp_path / "model.json"
    write_result(result, 0.0)
    assert load_metrics(result) == {
        "HellaSwag": 0.70,
        "ARC Easy": 0.60,
        "ARC Challenge": 0.40,
        "Winogrande": 0.65,
        "OpenBookQA": 0.45,
        "WikiText PPL": 9.0,
    }


def test_build_rows_computes_zero_minus_factorized(tmp_path):
    factorized = tmp_path / "factorized"
    zero_padding = tmp_path / "zero_padding"
    write_result(factorized / "llama2-7b.json", 0.0)
    write_result(zero_padding / "llama2-7b.json", 0.02)
    rows = build_rows(
        factorized,
        zero_padding,
        ["llama2-7b"],
    )
    assert rows[0]["model"] == "llama2-7b"
    assert rows[0]["metric"] == "HellaSwag"
    assert rows[0]["delta"] == pytest.approx(0.02)


def test_load_metrics_reports_missing_metric(tmp_path):
    result = tmp_path / "broken.json"
    result.write_text('{"results": {}}', encoding="utf-8")
    with pytest.raises(ValueError, match="hellaswag"):
        load_metrics(result)
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_summarize_hadamard_comparison.py -v
```

Expected: collection ERROR with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the summary utility**

Create `summarize_hadamard_comparison.py` with:

```python
#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


METRICS = {
    "HellaSwag": ("hellaswag", "acc_norm,none"),
    "ARC Easy": ("arc_easy", "acc_norm,none"),
    "ARC Challenge": ("arc_challenge", "acc_norm,none"),
    "Winogrande": ("winogrande", "acc,none"),
    "OpenBookQA": ("openbookqa", "acc_norm,none"),
    "WikiText PPL": ("wikitext", "word_perplexity,none"),
}
DEFAULT_MODELS = ["llama2-7b", "llama3.1-8b", "llama3.2-3b"]


def load_metrics(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Result JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, dict):
        raise ValueError(f"{path}: missing object field 'results'")

    values = {}
    for label, (task, field) in METRICS.items():
        try:
            values[label] = float(results[task][field])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: missing numeric metric results.{task}.{field}"
            ) from exc
    return values


def build_rows(factorized_dir, zero_padding_dir, models):
    rows = []
    for model in models:
        factorized = load_metrics(Path(factorized_dir) / f"{model}.json")
        zero_padding = load_metrics(
            Path(zero_padding_dir) / f"{model}.json"
        )
        for metric in METRICS:
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "factorized": factorized[metric],
                    "zero_padding": zero_padding[metric],
                    "delta": zero_padding[metric] - factorized[metric],
                }
            )
    return rows


def markdown_table(rows):
    lines = [
        "| Model | Metric | Factorized | Zero padding | Delta |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['metric']} | "
            f"{row['factorized']:.6f} | {row['zero_padding']:.6f} | "
            f"{row['delta']:+.6f} |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factorized-dir", type=Path, required=True)
    parser.add_argument("--zero-padding-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = parser.parse_args()
    print(
        markdown_table(
            build_rows(
                args.factorized_dir,
                args.zero_padding_dir,
                args.models,
            )
        )
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run summary tests**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_summarize_hadamard_comparison.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit the summarizer**

```bash
git add summarize_hadamard_comparison.py \
  tests/test_summarize_hadamard_comparison.py
git diff --cached --check
git commit -m "feat: summarize hadamard accuracy comparison"
```

---

### Task 5: Add a resumable, mode-isolated PTQ comparison runner

**Files:**
- Create: `scripts/run_compare_online_hadamard.sh`
- Modify: `README.md:54-83`
- Create: `tests/test_run_compare_online_hadamard.py`

**Interfaces:**
- Consumes: local model paths and existing optimized R1/R2 checkpoints
- Consumes: `ONLINE_HAD_MODES`, `CUDA_DEVICE`, `FORCE_REQUANTIZE`, and optional model/rotation environment variables
- Produces: `saved_models/online-had-comparison/<mode>/<model>/w4-gptq.pt`
- Produces: `results/online-had-comparison/<mode>/<model>.json`
- Produces: `logs/online-had-comparison/<mode>/<model>.log`
- Supports: `DRY_RUN=1` for command/path verification

- [ ] **Step 1: Write a failing dry-run test**

Create `tests/test_run_compare_online_hadamard.py`:

```python
import os
import subprocess
from pathlib import Path


SPINQUANT_DIR = Path(__file__).resolve().parents[1]


def test_dry_run_isolates_modes_and_passes_cli_option():
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["ONLINE_HAD_MODES"] = "factorized zero_padding"
    result = subprocess.run(
        ["bash", "scripts/run_compare_online_hadamard.sh"],
        cwd=SPINQUANT_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    output = result.stdout
    assert "--online_had_mode factorized" in output
    assert "--online_had_mode zero_padding" in output
    assert "online-had-comparison/factorized/llama2-7b.json" in output
    assert "online-had-comparison/zero_padding/llama2-7b.json" in output
```

- [ ] **Step 2: Run the dry-run test and confirm the script is missing**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_run_compare_online_hadamard.py -v
```

Expected: FAIL because `scripts/run_compare_online_hadamard.sh` does not exist.

- [ ] **Step 3: Implement the comparison runner**

Create `scripts/run_compare_online_hadamard.sh` using the same torchrun
arguments and environment variables as `scripts/run_ptq_w4a8kv4.sh`. Define:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SPINQUANT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${SPINQUANT_DIR}"

CUDA_DEVICE=${CUDA_DEVICE:-0}
read -r -a HAD_MODES <<< "${ONLINE_HAD_MODES:-factorized zero_padding}"

MODELS=(
  "llama2-7b|${LLAMA2_7B_MODEL:-./models/llama2-7b}|${LLAMA2_7B_ROTATION:-rotation_llama-2-7b/a16w4kv4-vasym/R.bin}"
  "llama3.1-8b|${LLAMA31_8B_MODEL:-./models/llama3.1-8b}|${LLAMA31_8B_ROTATION:-rotation_llama-3.1-8b/a16w4kv4-vasym/R.bin}"
  "llama3.2-3b|${LLAMA32_3B_MODEL:-./models/llama3.2-3b}|${LLAMA32_3B_ROTATION:-rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin}"
)
```

For each mode/model pair, generate these exact paths:

```bash
checkpoint="saved_models/online-had-comparison/${mode}/${name}/w4-gptq.pt"
result="results/online-had-comparison/${mode}/${name}.json"
log="logs/online-had-comparison/${mode}/${name}.log"
```

Skip a pair if its non-empty result already exists. Load its non-empty
checkpoint unless `FORCE_REQUANTIZE=1`; otherwise save a new checkpoint. Pass
the existing W4A8KV4 arguments plus:

```bash
--online_had_mode "${mode}"
--results_path "${result}"
```

When `DRY_RUN=1`, print one shell-escaped torchrun command per pair without
checking model/rotation existence or launching work. Otherwise use
`CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"`, `torchrun --standalone`, and
`tee -a "${log}"`. After each real run, fail if `${result}` is empty.

- [ ] **Step 4: Add concise README usage**

After the existing PTQ evaluation instructions in `README.md`, add:

```markdown
### Compare online Hadamard implementations

The R4 transform used by `down_proj` defaults to the original factorized
implementation. Select the fast-kernel implicit zero-padding implementation
with `--online_had_mode zero_padding`.

Run the W4A8KV4 comparison for Llama-2 7B, Llama-3.1 8B, and Llama-3.2 3B:

```bash
bash scripts/run_compare_online_hadamard.sh
```

Resume only zero-padding runs with:

```bash
ONLINE_HAD_MODES=zero_padding bash scripts/run_compare_online_hadamard.sh
```

Summarize completed JSON results with:

```bash
python summarize_hadamard_comparison.py \
  --factorized-dir results/online-had-comparison/factorized \
  --zero-padding-dir results/online-had-comparison/zero_padding
```
```

- [ ] **Step 5: Run syntax and dry-run tests**

Run:

```bash
bash -n scripts/run_compare_online_hadamard.sh
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest \
  tests/test_run_compare_online_hadamard.py -v
```

Expected: shell syntax check succeeds and the pytest passes.

- [ ] **Step 6: Commit the runner and documentation**

```bash
git add scripts/run_compare_online_hadamard.sh \
  tests/test_run_compare_online_hadamard.py README.md
git diff --cached --check
git commit -m "feat: run online hadamard PTQ comparison"
```

---

### Task 6: Run regression verification

**Files:**
- Verify only; modify failing implementation/tests only when the failure is caused by this feature

**Interfaces:**
- Verifies all earlier task outputs together

- [ ] **Step 1: Check imports and Python syntax**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m compileall -q \
  utils eval_utils train_utils tests \
  summarize_hadamard_comparison.py
```

Expected: exit status 0.

- [ ] **Step 2: Run the complete focused suite**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jaeyongjang/.conda/envs/spinquant/bin/python -m pytest tests -v
```

Expected: all tests PASS, with only explicit CUDA-dependent skips if CUDA is
unavailable.

- [ ] **Step 3: Verify default factorized numerical behavior**

Run a short Python check:

```bash
CUDA_VISIBLE_DEVICES=0 \
/home/jaeyongjang/.conda/envs/spinquant/bin/python - <<'PY'
import torch
from utils.hadamard_utils import get_hadK, matmul_hadU_cuda

torch.manual_seed(0)
x = torch.randn(2, 11008, device="cuda")
had_k, k = get_hadK(11008)
legacy = matmul_hadU_cuda(x, had_k, k)
explicit = matmul_hadU_cuda(x, had_k, k, mode="factorized")
torch.testing.assert_close(explicit, legacy)
print("factorized default regression: PASS")
PY
```

Expected: `factorized default regression: PASS`.

- [ ] **Step 4: Check the worktree diff**

Run:

```bash
git status --short
git diff --check
git log --oneline -6
```

Expected: no whitespace errors; pre-existing user changes remain present and
unmodified except for the deliberately shared `utils/process_args.py` hunk.

---

### Task 7: Run PTQ and generate the comparison report

**Files:**
- Generated: `saved_models/online-had-comparison/<mode>/<model>/w4-gptq.pt`
- Generated: `results/online-had-comparison/<mode>/<model>.json`
- Generated: `logs/online-had-comparison/<mode>/<model>.log`
- Generated: `results/online-had-comparison/summary.md`

**Interfaces:**
- Consumes: three local models and three existing optimized R1/R2 checkpoints
- Produces: six model/mode evaluations and a Markdown comparison table

- [ ] **Step 1: Verify all model and rotation inputs**

Run:

```bash
for path in \
  models/llama2-7b \
  models/llama3.1-8b \
  models/llama3.2-3b \
  rotation_llama-2-7b/a16w4kv4-vasym/R.bin \
  rotation_llama-3.1-8b/a16w4kv4-vasym/R.bin \
  rotation_llama-3.2-3b/a16w4kv4-vasym/R.bin
do
  test -e "${path}" || { echo "missing: ${path}" >&2; exit 1; }
done
```

Expected: exit status 0 with no missing paths.

- [ ] **Step 2: Launch resumable W4A8KV4 comparisons**

Run:

```bash
CUDA_DEVICE=0 bash scripts/run_compare_online_hadamard.sh
```

Expected: each missing model/mode pair produces a non-empty checkpoint, JSON,
and log; already completed JSON pairs are skipped.

If an existing factorized W4A8KV4 result is reused to avoid an unnecessary
repeat, copy it into the isolated result directory only after verifying its
quantization configuration matches the runner exactly, and record that
provenance in the final report.

- [ ] **Step 3: Generate the Markdown summary**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python \
  summarize_hadamard_comparison.py \
  --factorized-dir results/online-had-comparison/factorized \
  --zero-padding-dir results/online-had-comparison/zero_padding \
  > results/online-had-comparison/summary.md
```

Expected: a table with 18 rows: 3 models multiplied by 6 metrics.

- [ ] **Step 4: Validate the 3B control**

Run:

```bash
/home/jaeyongjang/.conda/envs/spinquant/bin/python - <<'PY'
from pathlib import Path
from summarize_hadamard_comparison import build_rows

rows = build_rows(
    Path("results/online-had-comparison/factorized"),
    Path("results/online-had-comparison/zero_padding"),
    ["llama3.2-3b"],
)
for row in rows:
    print(row)
PY
```

Expected: all six deltas are zero or small enough to be explained by the
evaluation's deterministic/numerical tolerance. Any material difference is
investigated before reporting success.

- [ ] **Step 5: Report results and deferred decision**

Report:

- factorized and zero-padding metrics with deltas;
- whether Llama-3.2 3B behaved as the power-of-two control;
- PTQ/log paths;
- any reused baseline result provenance; and
- whether zero-padding accuracy is sufficient or a separate rotation search
  should be designed.
