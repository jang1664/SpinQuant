# Matrix-output numerical accuracy: workload extension plan

## Goal

Extend the AQP8-vs-AQP16 matrix-output experiment beyond its current WikiText
input set without changing the comparison definition:

- `error_A`: FP16 vs AQP8 W4/KV4, where A/Q/P are all 8 bit.
- `error_B`: FP16 vs AQP16 W4/KV4, where A/Q/P are all 16 bit.
- Report only the outputs of **Linear**, raw **QK** (`Q @ K^T` before
  scale/mask/softmax), and **PV** (`P @ V` before the output projection).
- The headline is `error_A - error_B` and `error_A / error_B`; do not add
  logits, token-distribution, generation-quality, or task-accuracy metrics to
  this experiment.

The present implementation, `measure_matrix_output_accuracy.py`, is
WikiText-specific: it reads `samples.wikitext[*].target` from an existing
lm-eval result JSON.  The extension must make the *input workload* pluggable
while retaining the same model, rotation, quantization, observer, and metric
code paths.

## Proposed representative suite

The suite should cover distinct input/attention characteristics, rather than
add many highly correlated benchmarks.

| Family | Representative workload | Why it is included | Matrix-input unit |
| --- | --- | --- | --- |
| Zero-shot MCQ: scientific knowledge / difficult short reasoning | `arc_challenge` | Difficult science questions; a compact prompt and a small set of answer continuations. | One lm-eval log-likelihood request per answer choice. |
| Zero-shot MCQ: commonsense continuation / relatively long natural prompt | `hellaswag` | Natural-language context plus plausible long endings; tests a different prompt/continuation length mix from ARC. | One log-likelihood request per answer choice. |
| Zero-shot MCQ: minimal-pair coreference | `winogrande` | Short, high-sensitivity disambiguation prompts; complements ARC and HellaSwag without another broad knowledge test. | One log-likelihood request per answer choice. |
| Autoregressive PPL: encyclopedic/formal text | `wikitext` | Existing baseline; preserve it unchanged for continuity. | Rolling next-token windows over each document. |
| Autoregressive PPL: long-form narrative text | `pg19` | Chosen second PPL corpus.  Long documents and literary narrative differ materially from WikiText and naturally create many full-length attention windows. | Rolling next-token windows over each document. |
| Long-context attention stress | RULER **NIAH single-needle retrieval** | The most direct additional stressor for Q/P quantization: one answer-bearing token sequence is embedded in a long, structured distractor context.  It isolates long-context attention from ordinary PPL/MCQ distributional effects. | Prompt-only forward on a fixed-length context, with no generated answer/logit comparison. |

### Why these are the recommended “remaining” workloads

The three zero-shot tasks already exist in the repository's supported
five-task set (`arc_easy`, `arc_challenge`, `hellaswag`, `openbookqa`, and
`winogrande`).  The selected three maximize variety:

- Do **not** choose both `arc_easy` and `arc_challenge`: they are closely
  related; retain the harder `arc_challenge`.
- Do **not** choose both `openbookqa` and ARC for the small suite: both are
  science/fact MCQ; ARC-Challenge is the broader representative.
- Use `winogrande` to retain a short, contrastive language-understanding
  case, and `hellaswag` for natural continuation length/commonsense.

For the second PPL corpus, choose **PG19** rather than another encyclopedic
web-like corpus: the goal is workload diversity and long contiguous documents,
not a second near-WikiText estimate.  Before execution, pin the exact Hugging
Face dataset revision and confirm that its validation split/license is usable
in the execution environment.  If PG19 cannot be pinned or accessed, use a
fixed, revision-pinned C4 validation shard as the only fallback; do not report
both as the nominal second PPL workload.

RULER NIAH is the recommended extra family because A/Q/P quantization changes
attention-side operands.  Run it at 2K and 4K context lengths when every model
supports the length; if a model cannot support 4K, report its 2K result only
and never mix context lengths in a single cross-model aggregate.

## Workload representation and adapters

Create `utils/matrix_workloads.py` and define a normalized immutable example:

```python
@dataclass(frozen=True)
class MatrixWorkloadExample:
    workload: str
    split: str
    example_id: str
    input_ids: torch.Tensor       # rank-1, already tokenized, no padding
    kind: Literal["rolling_lm", "choice", "prompt_only"]
    choice_index: int | None      # set for MCQ request instances
    source_metadata: dict[str, str]
```

Adapters produce fully tokenized examples, so both AQP conditions receive
byte-for-byte identical `input_ids` and the measurement runner does not need
to understand dataset-specific schemas.

1. **Rolling-LM adapter** (`wikitext`, `pg19`)
   - Load a pinned dataset split/revision.
   - Convert each document to rolling windows exactly as the current WikiText
     path does: prepend EOS (or prior token) and evaluate windows of at most
     `--sequence-length` tokens.
   - Retain document ID/hash, split, dataset revision, tokenizer revision, and
     window start offset in `source_metadata`.

2. **lm-eval multiple-choice adapter** (`arc_challenge`, `hellaswag`,
   `winogrande`)
   - Use the same pinned `lm-evaluation-harness` task version/configuration
     used by the existing zero-shot runner (`num_fewshot=0`).
   - Materialize the exact prompt-plus-one-continuation token sequence for
     every answer-choice log-likelihood request.  This is intentional: it
     measures the matrix multiplications the evaluator actually executes.
   - Keep each choice as a separate example and store the task document ID,
     choice index, prompt hash, and continuation hash.  Do not concatenate all
     choices into one synthetic sequence.
   - Report both `examples` (request instances) and `source_documents`
     (questions), since one question produces multiple requests.

3. **RULER NIAH adapter**
   - Generate/cache a deterministic prompt manifest from a pinned RULER
     revision, fixed seed, fixed needle template, and fixed distractor source.
   - Store the generated prompt text hash and target context length in the
     manifest; tokenize it once for the selected model tokenizer.
   - Use `kind="prompt_only"`: forward the context but do not perform answer
     generation or score correctness.  This preserves the matrix-output-only
     scope.

## CLI and manifest design

Replace the implicit `--fp-results-path` workload dependency with:

```text
--workload-manifest workloads/matrix_accuracy_v1.yaml
--workloads wikitext,pg19,arc_challenge,hellaswag,winogrande,ruler_niah
--sequence-length 2048
--ruler-lengths 2048,4096
--max-examples-per-workload N       # smoke/debug only
--dataset-cache-dir ...             # optional explicit cache location
```

`workloads/matrix_accuracy_v1.yaml` should pin, per workload:

- adapter name, dataset/task name, split, revision/version, seed;
- few-shot count and all prompt/template settings;
- document/request cap (full evaluation by default; an explicitly named fixed
  subset only when a full set is impractical);
- RULER context length(s), needle template, and generator revision;
- a stable manifest hash recorded in every result.

Keep `--fp-results-path` temporarily as a backwards-compatible alias for the
legacy WikiText adapter.  Deprecate it after the manifest path produces an
identical WikiText result on the old 62 documents.

## Measurement and result changes

1. Refactor `measure_condition()` to consume an iterator of
   `MatrixWorkloadExample` instead of raw WikiText document dictionaries.
2. Preserve the per-example order: FP16 reference forward, AQP16 candidate
   forward, then AQP8 candidate forward.  Keep the existing checks that A/B
   have equal chunks and equal element counts in all three groups.
3. Reset/aggregate metrics per workload and per group.  Never combine task
   families into one unweighted global metric; their sequence/request lengths
   differ substantially.
4. Write one result JSON per `(model, workload, context-length)` and record:
   - workload manifest/revision/hash and adapter configuration;
   - tokenizer/model/rotation/W4 checkpoint hashes;
   - source-document count, request-example count, chunks, token count, and
     length distribution;
   - `error_A`, `error_B`, and `A_vs_B` for only Linear/QK/PV.
5. Extend `summarize_matrix_output_accuracy.py` to make a table grouped by
   workload family first, then model and Linear/QK/PV.  For RULER, context
   length is a required table column.

## Execution phases

1. **Selection lock**
   - Confirm the six proposed entries above and pin dataset/task revisions.
   - Verify PG19 availability; use the defined C4 fallback only if blocked.
   - Freeze one RULER seed and the supported common length set.

2. **Adapter implementation**
   - Add the normalized example type, manifest parser, and the three adapter
     classes.
   - Refactor the current WikiText path into the rolling-LM adapter without
     changing its token-window semantics.

3. **Instrumentation integration**
   - Feed adapter examples into the existing FP16/AQP16/AQP8 observer flow.
   - Retain eager attention and the existing hard-workload quantization setup:
     W4/KV4, A/Q/P=8 for AQP8; A/Q/P=16 for AQP16.

4. **Validation**
   - Unit test each adapter using a tiny local fixture and assert stable IDs,
     input IDs, token count, and metadata.
   - For each adapter, run one example on a small Llama model and assert the
     only metric groups are Linear/QK/PV, all A/B element counts match, and
     all reported metrics are finite.
   - Regression test legacy WikiText: the manifest adapter must reproduce the
     old document/chunk counts and metrics within deterministic FP reduction
     tolerance.

5. **Full execution and reporting**
   - Run every workload for Llama-2 7B, Llama-3.1 8B, and Llama-3.2 3B.
   - Schedule RULER separately because QK cost grows quadratically with
     context length; do not silently shorten its prompts.
   - Publish the per-workload table and explicitly identify where AQP8 has
     higher/lower error than AQP16.  Treat an AQP8 advantage in any individual
     metric as an observation, not as a configuration failure.

## Acceptance criteria

- The exact input IDs for FP16, AQP16, and AQP8 match for every measured
  example.
- AQP8 verification shows A/Q/P=8; AQP16 verification shows A/Q/P=16;
  W/K/V remain 4 bits for both.
- Each output contains exactly Linear, QK, and PV metrics and no logits or
  task-score comparison.
- Each workload result is reproducible from its pinned manifest and reports
  all provenance/count fields.
- The final report separates zero-shot MCQ, PPL, and long-context stress
  results rather than masking them in a single aggregate.

## Execution record (2026-09-10)

Implemented and executed the representative workload expansion for all three
target models (Llama-2 7B, Llama-3.1 8B, and Llama-3.2 3B).

- The selected zero-shot inputs are the first 16 validation source documents
  from each pinned task revision.  This produces 64 ARC-Challenge requests,
  64 HellaSwag requests, and 32 WinoGrande requests because every answer
  choice is measured as its own prompt-plus-continuation request.
- PG19 could not be loaded by the installed `datasets` version because its
  legacy loading script is unsupported.  The planned fallback was therefore
  used: eight non-empty documents from the pinned `allenai/c4` English
  validation revision.
- RULER uses NVIDIA RULER revision `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`,
  task `niah_single_1`, `noise` haystack, seed 42, and four generated 2K
  prompts.  It is reported solely as an input/matrix-output stress workload;
  no retrieval accuracy is calculated.
- The legacy full WikiText result remains the PPL baseline (62 documents,
  162/186 chunks depending on model).  New workload outputs and the combined
  Markdown report are under `results/aqp-matrix-workloads/` and
  `results/aqp-matrix-output-accuracy/SUMMARY.md`, respectively.
- For every new result, validation confirmed A/Q/P=8 for AQP8,
  A/Q/P=16 for AQP16, W/K/V=4 in both cases, equal A/B element counts, finite
  values, and exactly the Linear/QK/PV output groups.
