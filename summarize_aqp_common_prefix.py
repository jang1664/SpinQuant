"""Verify compact raw artifacts and report free-generation common-prefix std."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from measure_aqp_runtime_fidelity import sha256, write_json
from utils.common_prefix_metrics import cluster_summary


def summarize(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    m = json.loads((root / 'manifest.json').read_text())
    corpus = [json.loads(line) for line in (root / 'corpus.jsonl').read_text().splitlines()]
    if sha256(root / 'corpus.jsonl') != m['corpus_sha256']:
        raise ValueError('corpus hash mismatch')
    docs = [r['source_document_id'] for r in corpus]
    expected_ids = {r['sample_id'] for r in corpus}
    rows, index = [], {}
    for complete_path in sorted(root.glob('*/r*/complete.json')):
        complete = json.loads(complete_path.read_text())
        if {a['sample_id'] for a in complete['artifacts']} != expected_ids or len(complete['artifacts']) != len(corpus):
            raise ValueError('incomplete sample coverage')
        index[str(complete_path.relative_to(root))] = complete
        for artifact in complete['artifacts']:
            path = complete_path.parent / (artifact['sample_id'] + '.json')
            if sha256(path) != artifact['sha256']:
                raise ValueError(f'artifact hash mismatch: {path}')
            row = json.loads(path.read_text())
            stds = [float(np.std(p['probabilities'],ddof=1)) for p in row['positions']]
            computed = float(np.mean(stds)) if stds else None
            if computed != row['avg_std_top1_prob']:
                raise ValueError('saved metric does not match probabilities')
            rows.append(row)
    expected_runs = len(m['conditions']) * m['repeats']
    result = dict(status='complete' if len(index) == expected_runs else 'partial',
        raw_root=str(root), completed_runs=len(index), expected_runs=expected_runs, conditions={})
    for condition in m['conditions']:
        selected = [r for r in rows if r['condition'] == condition]
        if not selected:
            continue
        stats = cluster_summary(selected, docs, m['bootstrap']['draws'], m['bootstrap']['seed'])
        lengths = [r['common_length'] for r in selected]
        stats.update(mean_common_length=float(np.mean(lengths)), median_common_length=float(np.median(lengths)),
            zero_common_length=sum(x == 0 for x in lengths), stop_reasons=dict(Counter(r['stop_reason'] for r in selected)),
            per_repeat={str(r): cluster_summary([s for s in selected if s['repeat']==r],docs,
                m['bootstrap']['draws'],m['bootstrap']['seed']) for r in sorted({s['repeat'] for s in selected})})
        result['conditions'][condition] = stats
    # Existing teacher-forced result is a separately labelled secondary comparator.
    fixed_path = Path(m['source_experiment']) / 'final-analysis/runtime_std.csv'
    fixed = {}
    if fixed_path.exists():
        with fixed_path.open() as stream:
            fixed_rows = list(csv.DictReader(stream))
        for condition in m['conditions']:
            values = [dict(source_document_id=r['source_document_id'], repeat=int(r['repeat']),
                avg_std_top1_prob=float(r['own_c0_fixed_prefix_avg_std_top1_prob']))
                for r in fixed_rows if r['condition'] == condition]
            if values:
                fixed[condition] = cluster_summary(values, docs, m['bootstrap']['draws'], m['bootstrap']['seed'])
    result['teacher_forced_secondary'] = fixed
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / 'summary.json', result)
    write_json(destination / 'artifact_index.json', dict(raw_root=str(root), runs=index))
    shutil.copy2(root / 'manifest.json', destination / 'manifest.json')
    for source in [Path(__file__), Path(__file__).parent / 'utils/common_prefix_metrics.py']:
        shutil.copy2(source, destination / source.name)
    fieldnames = ['condition','repeat','sample_id','source_document_id','common_length',
                  'stop_reason','divergence_index','avg_std_top1_prob']
    with (destination / 'per_window.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)
    lines = ['# Free-generation common-prefix probability variation', '',
        f"Status: **{result['status']}**; {len(index)}/{expected_runs} condition/repeat runs.", '',
        f"Original frozen corpus: {m['windows']} windows, {len(set(docs))} source articles, "
        f"{m['prompt_length']} prompt tokens, at most {m['positions']} generated tokens. "
        f"Batch sizes {m['batch_sizes']}, {m['repeats']} independent model loads, GPU 0 RTX A6000, TP=1, eager attention.", '',
        'Metric: sample standard deviation (ddof=1) of the common top-1 token probability across four batch settings, '
        'averaged over strictly matching generated positions, then eligible windows, then repeats. '
        'Probabilities use float64 softmax of model logits. Repeats are not additional runtime configurations. '
        '95% intervals resample whole source articles (2,000 draws).', '',
        'The first differing token is excluded. A common EOS is included and terminates measurement. '
        'Zero-length prefixes have undefined means and are excluded, with counts below. '
        'Cap endings are right-censored, not evidence of unlimited agreement. '
        'Inactive rows continue their own greedy decoding to retain batch shapes until their block finishes. '
        'Output-length variability and accuracy are not measured by this runner.', '',
        '| Condition | Avg_Std@top1_prob | 95% CI | ×10⁻⁴ | Mean common length | Zero-prefix / all window-repeats |',
        '|---|---:|---|---:|---:|---:|']
    for condition, stats in result['conditions'].items():
        value, ci = stats['mean'], stats['ci95']
        lines.append(f"| {condition} | {value:.8g} | [{ci[0]:.8g}, {ci[1]:.8g}] | {value*1e4:.4f} | "
            f"{stats['mean_common_length']:.3f} | {stats['zero_common_length']} / {stats['total_window_repeats']} |"
            if value is not None else f'| {condition} | undefined | — | — | 0 | all |')
    lines += ['', 'Population-SD sensitivity (ddof=0) equals each reported mean and interval endpoint multiplied by √(3/4). '
        'Full values, stop counts, and per-repeat estimates are in summary.json.', '',
        'These condition-specific common prefixes differ in length and content. Comparing their means is descriptive; '
        'it does not isolate a quantization effect on an identical set of prefixes.', '',
        '## Existing teacher-forced comparison (different prefix rule)', '',
        '| Condition | Fixed-prefix std | 95% CI |', '|---|---:|---|']
    for condition, stats in fixed.items():
        ci = stats['ci95']
        lines.append(f"| {condition} | {stats['mean']:.8g} | [{ci[0]:.8g}, {ci[1]:.8g}] |")
    lines += ['', '## Definition and provenance', '',
        f"Definition adapted from [Yuan et al., NeurIPS 2025, §3.1]({m['paper_url']}). "
        'The public author repository at commit 09ed20f0d9d92a466ac2e457bf572fa3c5f14f10 provides generation code; '
        'an aggregation implementation was not located. The sample-SD, empty-prefix, and EOS conventions are explicit here. '
        'This applies the common-prefix metric to our model/corpus/runtime sweep; it is not a reproduction of the paper’s '
        '12 GPU-type/GPU-count/batch configurations or reasoning benchmarks.', '',
        f'Raw artifacts: `{root}`. Per-window hashes are in artifact_index.json; model, checkpoint, corpus, runtime, '
        'and execution-source hashes are in manifest.json. Layer/quantizer-call/cache validation is saved per run.', '']
    (destination / 'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    summarize(args.experiment, args.output)
