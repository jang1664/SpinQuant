"""Greedy generation across batches 1/2/4/8, stopping measurement at divergence.

Each runtime has independent KV caches and consumes its own argmax outputs.
Inactive rows continue generation to preserve kernel shapes for active rows.
Only complete blocks stop early; output length is not measured by this runner.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.special import logsumexp
import torch

import measure_aqp_runtime_fidelity as base
from utils.common_prefix_metrics import observe, finish


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_bytes():
    return dict(base.SOURCE_SNAPSHOT, **{name: (base.ROOT / name).read_bytes() for name in
        ['measure_aqp_common_prefix.py', 'utils/common_prefix_metrics.py']})


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    old = json.loads((source / 'manifest.json').read_text())
    if base.sha256(source / 'corpus.jsonl') != old['corpus_sha256']:
        raise ValueError('source corpus hash mismatch')
    if old['batch_sizes'] != [1, 2, 4, 8] or old['windows'] % 8:
        raise ValueError('requires aligned complete blocks and batches 1/2/4/8')
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source / 'corpus.jsonl', output / 'corpus.jsonl')
    generation = json.loads((Path(old['paths']['model']) / 'generation_config.json').read_text())
    eos = generation['eos_token_id']
    m = dict(old, experiment_id=output.name, source_experiment=str(source),
        source_manifest_sha256=base.sha256(source / 'manifest.json'),
        created=stamp(), runtime=base.runtime_environment(),
        mode='independent greedy cached decode; aligned runtime steps',
        eos_policy='include common EOS probability, then stop measuring that row',
        eos_ids=eos if isinstance(eos, list) else [eos],
        sampling='argmax raw logits; no sampling, logits processors, or temperature scaling',
        aggregation='mean over common positions per window, equal eligible windows per repeat, equal repeats',
        storage='common-prefix token IDs/probabilities and divergence boundary; no full-vocabulary arrays',
        empty_prefix='undefined; exclude from metric and explicitly count',
        ddof=1, stop_boundary='exclude first differing token; zero-based divergence_index equals common_length',
        cap_policy='positions from source manifest; no divergence through cap is right-censored',
        execution='independent KV caches for every batch group; inactive rows keep own greedy tokens; fixed batch shapes',
        paper_url='https://proceedings.neurips.cc/paper_files/paper/2025/file/f80094a824ba5912d4a2de169c404a40-Paper-Conference.pdf',
        paper_implementation_note='Section 3.1 common-token definition. Public repository at 09ed20f0d9d92a466ac2e457bf572fa3c5f14f10 exposes generation but no located aggregation implementation; ddof=1 and empty/EOS conventions explicitly specified here. ddof=0 sensitivity also reported.')
    snapshot = output / 'code'
    for name, data in source_bytes().items():
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    m['source_sha256'] = {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes().items()}
    base.write_json(output / 'manifest.json', m)
    print(json.dumps(dict(prepared=str(output), windows=m['windows'], cap=m['positions'])), flush=True)


def generate_block(model, samples, m, forward_counter, *, device='cuda'):
    """Keep the original batch grouping and independent greedy histories."""
    size = len(samples)
    if size != 8:
        raise ValueError('full blocks of eight required')
    rows = [dict(sample_id=s['sample_id'], source_document_id=s['source_document_id'],
                 positions=[], stop_reason=None, divergence_index=None, boundary=None) for s in samples]
    groups = {b: [dict(start=start, ids=torch.tensor([s['prompt_ids'] for s in samples[start:start+b]],
                        device=device), cache=None) for start in range(0, size, b)] for b in m['batch_sizes']}
    trace = []
    with torch.inference_mode():
        for step in range(m['positions']):
            length = m['prompt_length'] + step
            runtime_logits = []
            for batch in m['batch_sizes']:
                blocks = []
                for group in groups[batch]:
                    ids = group['ids']
                    positions = torch.arange(length - ids.shape[1], length, device=device).unsqueeze(0).expand(batch, -1)
                    result = model(input_ids=ids,
                        attention_mask=torch.ones((batch, length), dtype=torch.long, device=device),
                        position_ids=positions, past_key_values=group['cache'], use_cache=True, return_dict=True)
                    group['cache'] = result.past_key_values
                    if group['cache'] is None or base.cache_length(group['cache']) != length:
                        raise ValueError('KV cache length mismatch')
                    logits = result.logits[:, -1, :].float().cpu().numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError('nonfinite logits')
                    blocks.append(logits)
                    group['ids'] = torch.tensor(logits.argmax(axis=-1)[:, None], device=device)
                    forward_counter[0] += 1
                    del result
                runtime_logits.append(np.concatenate(blocks, axis=0))
            logits = np.stack(runtime_logits).astype(np.float64)
            winners = logits.argmax(axis=-1)
            normalizers = logsumexp(logits, axis=-1)
            top_prob = np.exp(np.take_along_axis(logits, winners[..., None], axis=-1)[..., 0] - normalizers)
            ref_prob = np.exp(np.take_along_axis(logits,
                np.broadcast_to(winners[0][None, :, None], (*winners.shape, 1)), axis=-1)[..., 0] - normalizers)
            for i, row in enumerate(rows):
                observe(row, step, winners[:, i], top_prob[:, i], ref_prob[:, i].tolist(), m['eos_ids'])
            trace.append(dict(step=step, cache_length=length, input_tokens=m['prompt_length'] if step == 0 else 1))
            if all(row['stop_reason'] is not None for row in rows):
                break
    return [finish(row) for row in rows], trace


def run(output, conditions=None, repeats=None):
    out = Path(output).resolve()
    m = json.loads((out / 'manifest.json').read_text())
    if base.sha256(out / 'corpus.jsonl') != m['corpus_sha256']:
        raise ValueError('corpus changed')
    if base.runtime_environment() != m['runtime']:
        raise ValueError('runtime differs from frozen manifest')
    if {name: hashlib.sha256(data).hexdigest() for name, data in source_bytes().items()} != m['source_sha256']:
        raise ValueError('source changed after preparation')
    if base.sha256(m['paths']['checkpoint']) != m['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    base.validate_checkpoint_weights(m, out / 'weight_validation.json')
    corpus = [json.loads(line) for line in (out / 'corpus.jsonl').read_text().splitlines()]
    conditions = conditions or list(m['conditions'])
    repeats = list(range(m['repeats'])) if repeats is None else repeats
    if not set(conditions) <= set(m['conditions']) or not set(repeats) <= set(range(m['repeats'])):
        raise ValueError('requested run outside manifest')
    for repeat in repeats:
        for condition in conditions:
            dest = out / condition / f'r{repeat}'
            if (dest / 'complete.json').exists():
                continue
            dest.mkdir(parents=True, exist_ok=False)
            base.write_json(out / 'status.json', dict(state='running', condition=condition, repeat=repeat,
                finished_windows=0, total_windows=len(corpus), pid=os.getpid(), updated=stamp()))
            torch.manual_seed(m['seed'])
            torch.cuda.manual_seed_all(m['seed'])
            print(f'LOAD {condition} r{repeat}', flush=True)
            model, validation = base.load_condition(m, condition)
            counts, handles = base.attach_calls(model)
            counter = [0]
            artifacts, cache_trace = [], []
            torch.cuda.reset_peak_memory_stats()
            started = time.monotonic()
            try:
                for start in range(0, len(corpus), 8):
                    rows, trace = generate_block(model, corpus[start:start+8], m, counter)
                    cache_trace.append(dict(block_start=start, steps=trace))
                    for row in rows:
                        row.update(condition=condition, repeat=repeat)
                        path = dest / (row['sample_id'] + '.json')
                        base.write_json(path, row)
                        artifacts.append(dict(sample_id=row['sample_id'], sha256=base.sha256(path)))
                    elapsed = time.monotonic() - started
                    print(f'PROGRESS {condition} r{repeat} {start+8}/{len(corpus)} '
                          f'common_lengths={[r["common_length"] for r in rows]} elapsed={elapsed:.1f}s', flush=True)
                    base.write_json(out / 'status.json', dict(state='running', condition=condition, repeat=repeat,
                        finished_windows=start+8, total_windows=len(corpus), elapsed_seconds=elapsed,
                        pid=os.getpid(), updated=stamp()))
                for name, record in counts.items():
                    expected = counter[0] if record['bits'] < 16 else 0
                    if record['calls'] != expected:
                        raise ValueError(f'quantizer call mismatch {name}: {record}, expected={expected}')
                base.write_json(dest / 'condition_validation.json', dict(validation, quantizer_calls=counts,
                    expected_forward_calls=counter[0], cache_trace=cache_trace, actual_batch_sizes=m['batch_sizes'],
                    grouping=[s['sample_id'] for s in corpus]))
                base.write_json(dest / 'complete.json', dict(condition=condition, repeat=repeat, artifacts=artifacts,
                    seconds=time.monotonic()-started, peak_memory_bytes=torch.cuda.max_memory_allocated()))
            except Exception as error:
                base.write_json(dest / 'failure.json', dict(type=type(error).__name__, message=str(error)))
                base.write_json(out / 'status.json', dict(state='failed', updated=stamp(), error=str(error)))
                raise
            finally:
                for handle in handles:
                    handle.remove()
                del model
                gc.collect()
                torch.cuda.empty_cache()
    complete = len(list(out.glob('*/r*/complete.json')))
    base.write_json(out / 'status.json', dict(state='generation_complete' if complete == len(m['conditions'])*m['repeats'] else 'partial',
        completed_condition_repeats=complete, expected_condition_repeats=len(m['conditions'])*m['repeats'], updated=stamp()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--source', required=True)
    prep.add_argument('--output', required=True)
    runp = sub.add_parser('run')
    runp.add_argument('--output', required=True)
    runp.add_argument('--conditions')
    runp.add_argument('--repeats')
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.action == 'prepare':
        prepare(args.source, args.output)
    else:
        run(args.output, args.conditions.split(',') if args.conditions else None,
            [int(x) for x in args.repeats.split(',')] if args.repeats else None)


if __name__ == '__main__':
    main()
