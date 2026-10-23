"""Compare saved zero-shot answer-choice probabilities against unquantized FP16.

Primary distribution: softmax of each choice's summed continuation log-likelihood.
Secondary distribution: softmax of the score used for reported task accuracy
(character-normalized scores for acc_norm, raw scores for Winogrande).
These are relative choice-score probabilities, not next-token vocabulary logits.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.special import softmax

TASKS = ['hellaswag', 'arc_easy', 'arc_challenge', 'winogrande', 'openbookqa']
CONDITIONS = ['fp_base', 'aqp16', 'aqp8']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def key(row):
    return str(row['doc_id']), row['doc_hash'], row['prompt_hash'], row['target_hash']


def align_samples(reference, other):
    indexed = {key(r): r for r in other}
    ref_keys = [key(r) for r in reference]
    if len(indexed) != len(other) or len(set(ref_keys)) != len(reference):
        raise ValueError('duplicate sample IDs/hashes')
    if set(indexed) != set(ref_keys):
        raise ValueError('sample IDs/hashes do not match')
    aligned = [indexed[k] for k in ref_keys]
    for left, right in zip(reference, aligned):
        if any(left[k] != right[k] for k in ['arguments', 'target', 'doc', 'metrics']):
            raise ValueError('prompt/choice/document/target mismatch')
    return aligned


def scores(sample):
    values = np.array([r[0] for r in sample['filtered_resps']], dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('invalid choice scores')
    if len(values) != len(sample['arguments']):
        raise ValueError('score and choice counts differ')
    return values


def choice_lengths(sample):
    choices = sample['doc']['choices']
    if isinstance(choices, dict):
        choices = choices['text']
    if len(choices) != len(sample['arguments']):
        raise ValueError('choice count mismatch')
    # Exactly lm-eval character-length normalization, without delimiter stripping.
    lengths = np.array([len(choice) for choice in choices], dtype=np.float64)
    if np.any(lengths == 0):
        raise ValueError('empty normalized choice')
    return lengths


def row_metrics(choice_scores):
    z = np.asarray(choice_scores, dtype=np.float64)
    if z.ndim != 2 or z.shape[0] != 3 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError('three finite equal-length choice-score vectors required')
    p = softmax(z, axis=1)
    top = z.argmax(axis=1)  # ties select lowest choice index
    ref = int(top[0])
    row = {'reference_choice_index': ref, 'p_fp_reference': float(p[0,ref])}
    for i, label in [(1,'aqp16'),(2,'aqp8')]:
        delta = float(p[i,ref] - p[0,ref])
        row.update({f'{label}_choice_index':int(top[i]), f'p_{label}_reference':float(p[i,ref]),
            f'{label}_signed_error':delta, f'{label}_absolute_error':abs(delta),
            f'{label}_squared_error':delta**2, f'p_{label}_own_top1':float(p[i,top[i]]),
            f'{label}_own_top1_absolute_error':float(abs(p[i,top[i]] - p[0,ref])),
            f'{label}_agrees_with_fp':int(top[i] == ref)})
    row['absolute_error_delta_8_minus_16'] = row['aqp8_absolute_error'] - row['aqp16_absolute_error']
    return row


def summarize_rows(rows):
    fields = ['p_fp_reference','p_aqp16_reference','p_aqp8_reference',
        'p_aqp16_own_top1','p_aqp8_own_top1','aqp16_signed_error','aqp8_signed_error',
        'aqp16_absolute_error','aqp8_absolute_error','aqp16_own_top1_absolute_error',
        'aqp8_own_top1_absolute_error','aqp16_agrees_with_fp','aqp8_agrees_with_fp']
    result = dict(samples=len(rows), **{name:float(np.mean([r[name] for r in rows])) for name in fields})
    for label in ['aqp16','aqp8']:
        result[label+'_rmse'] = float(np.sqrt(np.mean([r[label+'_squared_error'] for r in rows])))
    diff = np.array([r['absolute_error_delta_8_minus_16'] for r in rows])
    result.update(mae_delta_8_minus_16=float(diff.mean()),
        aqp16_closer_fraction=float(np.mean(diff > 0)), aqp8_closer_fraction=float(np.mean(diff < 0)),
        exact_tie_fraction=float(np.mean(diff == 0)),
        mae_ratio_8_over_16=result['aqp8_absolute_error']/result['aqp16_absolute_error']
            if result['aqp16_absolute_error'] else None)
    return result


def paired_bootstrap(groups, draws, seed):
    """Resample matched questions within each task, retaining original task sizes."""
    rng = np.random.default_rng(seed)
    totals = np.zeros((draws,2),dtype=np.float64)
    n = 0
    for rows in groups:
        values = np.array([[r['aqp16_absolute_error'],r['aqp8_absolute_error']] for r in rows])
        n += len(rows)
        for first in range(0,draws,32):
            count = min(32,draws-first)
            indices = rng.integers(len(rows),size=(count,len(rows)))
            totals[first:first+count] += values[indices].sum(axis=1)
    means = totals/n
    ratios = np.divide(means[:,1],means[:,0],out=np.full(draws,np.nan),where=means[:,0]!=0)
    valid = ratios[np.isfinite(ratios)]
    return dict(draws=draws, seed=seed, unit='matched question, stratified by task',
        mae_delta_8_minus_16_ci95=np.quantile(means[:,1]-means[:,0],[.025,.975]).tolist(),
        mae_ratio_8_over_16_ci95=np.quantile(valid,[.025,.975]).tolist() if len(valid) else None)


def validate_metadata(data):
    required = {'fp_base':(16,16,16,16,16,16), 'aqp16':(4,4,4,16,16,16), 'aqp8':(4,4,4,8,8,8)}
    fields = ['weight_bits','key_bits','value_bits','activation_bits','query_bits','probability_bits']
    common = ['model','compute_dtype','attention_backend','linear_backend','model_max_length',
              'lm_eval_batch_size','num_fewshot','seed']
    q = {label:data[label]['spinquant_quantization'] for label in CONDITIONS}
    for label in CONDITIONS:
        if tuple(q[label][k] for k in fields) != required[label]:
            raise ValueError(f'{label}: bit configuration mismatch')
        if q[label]['compute_dtype'] != 'fp16' or data[label]['config']['limit'] is not None:
            raise ValueError('requires complete FP16 evaluation')
        if any(data[label]['n-shot'][task] != 0 for task in TASKS):
            raise ValueError('requires zero-shot outputs')
        for name in common:
            if q[label][name] != q['fp_base'][name]:
                raise ValueError(f'{label}: runtime mismatch: {name}')
    for name in ['quantized_checkpoint_sha256','rotation_checkpoint_sha256']:
        if not q['aqp16'].get(name) or q['aqp16'][name] != q['aqp8'][name]:
            raise ValueError(f'quantized conditions do not share {name}')
    # Every other quantization setting must match, except actual timing and
    # the intended operand widths. Coverage metadata is also checked if present.
    allowed = {'activation_bits','query_bits','probability_bits','evaluation_seconds'}
    differences = [k for k in set(q['aqp16']) | set(q['aqp8'])
                   if q['aqp16'].get(k) != q['aqp8'].get(k) and k not in allowed]
    if differences:
        raise ValueError(f'unexpected quantized configuration differences: {differences}')
    return {label:{k:v for k,v in q[label].items() if k!='fpint_linear_coverage'} for label in CONDITIONS}


def run(input_root, output, draws=2000, seed=20261019):
    root, out = Path(input_root).resolve(), Path(output).resolve()
    paths = {label:root/(label+'.json') for label in CONDITIONS}
    data = {label:json.loads(path.read_text()) for label,path in paths.items()}
    metadata = validate_metadata(data)
    modes = {'raw_likelihood':[], 'task_decision':[]}
    validation = {}
    for task in TASKS:
        samples = {'fp_base':data['fp_base']['samples'][task]}
        for label in CONDITIONS[1:]:
            samples[label] = align_samples(samples['fp_base'],data[label]['samples'][task])
        normalized = 'acc_norm' in samples['fp_base'][0]['metrics']
        metric = 'acc_norm' if normalized else 'acc'
        for label in CONDITIONS:
            counts = data[label]['n-samples'][task]
            if counts['original'] != counts['effective'] or counts['effective'] != len(samples[label]):
                raise ValueError('incomplete task sample coverage')
            if data[label]['configs'][task] != data['fp_base']['configs'][task]:
                raise ValueError('task configuration mismatch')
        for index, reference in enumerate(samples['fp_base']):
            z = np.stack([scores(samples[label][index]) for label in CONDITIONS])
            decision = z/choice_lengths(reference) if normalized else z
            gold = int(reference['doc']['answer'])-1 if task == 'winogrande' else int(reference['target'])
            for i,label in enumerate(CONDITIONS):
                saved = samples[label][index]
                if float(int(z[i].argmax()) == gold) != saved['acc']:
                    raise ValueError('reconstructed raw accuracy differs from stored sample')
                if float(int(decision[i].argmax()) == gold) != saved[metric]:
                    raise ValueError('reconstructed task-decision accuracy differs from stored sample')
            for mode,values in [('raw_likelihood',z),('task_decision',decision)]:
                row = dict(mode=mode,task=task,doc_id=reference['doc_id'],doc_hash=reference['doc_hash'],
                    prompt_hash=reference['prompt_hash'],target_hash=reference['target_hash'],
                    choices=z.shape[1],**row_metrics(values))
                modes[mode].append(row)
        validation[task] = dict(samples=len(samples['fp_base']), decision_metric=metric,
            matched_document_prompt_target_hashes=True, matched_arguments=True,
            reconstructed_all_three_models_accuracy=True)
        print(f'ALIGNED {task}: {len(samples["fp_base"])} questions',flush=True)
    summary = dict(model=metadata['fp_base']['model'], runtime=metadata,
        probability='float64 softmax over choice scores; temperature 1; not calibrated correctness probability or vocabulary token probability',
        reference='FP16 argmax choice, separately selected within each score definition; never remove flips or incorrect answers',
        modes={},validation=validation,
        sources={label:dict(path=str(path),sha256=sha256(path)) for label,path in paths.items()},
        source_code_sha256=sha256(__file__))
    for mode,rows in modes.items():
        groups = [[r for r in rows if r['task']==task] for task in TASKS]
        per_task = {task:dict(summarize_rows(group),bootstrap=paired_bootstrap([group],draws,seed))
                    for task,group in zip(TASKS,groups)}
        overall = dict(summarize_rows(rows),bootstrap=paired_bootstrap(groups,draws,seed))
        summary['modes'][mode] = dict(tasks=per_task,overall_micro=overall,
            task_macro_mae={label:float(np.mean([v[label+'_absolute_error'] for v in per_task.values()]))
                            for label in ['aqp16','aqp8']})
    out.mkdir(parents=True,exist_ok=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    shutil.copy2(__file__,out/Path(__file__).name)
    for mode,rows in modes.items():
        with (out/(mode+'_per_question.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_report(summary,out)
    for mode,entry in summary['modes'].items():
        print(mode,json.dumps(entry['overall_micro'],indent=2),flush=True)
    return summary


def write_report(summary,out):
    lines=['# Zero-shot top-1 choice probability fidelity','',
        f"Model: **{summary['model']}**.",
        f"All three saved runs: FP16 compute, eager attention, batch size 4, maximum length 2048, zero-shot, "
        f"all {summary['modes']['raw_likelihood']['overall_micro']['samples']:,} questions. "
        'FP base has W/A/Q/K/V/P=16 and no rotation; AQP16/AQP8 share the same W4/K4/V4 checkpoint and rotation. '
        'Only A/Q/P widths differ between the two quantized runs. This script analyzes saved per-choice scores.','',
        'For each question and condition, p(c)=softmax(s)[c]. Primary s is the summed continuation log-likelihood. '
        'Define c*=argmax_c s_FP(c), and error_q=p_q(c*)−p_FP(c*). '
        'MAE averages |error_q| per question; it is not the absolute difference between mean probabilities. '
        'RMSE=sqrt(mean(error_q²)). Smaller errors mean closer to the unquantized reference.','',
        'These are relative probabilities over the supplied answer-choice scores, not next-token vocabulary probabilities '
        'or calibrated probabilities of correctness. Winogrande scores the same continuation under different filled contexts. '
        'The secondary task-decision distribution uses the exact character-normalized scores for acc_norm tasks, '
        'and raw scores for Winogrande; its probability scale differs from the primary distribution.','',
        'All questions are retained, including wrong answers and top-1 flips. Sample IDs, document/prompt/target hashes, '
        'prompt/choice arguments, task configurations, full counts, and reconstructed accuracies were checked across all three runs. '
        'The lowest choice index breaks ties.','']
    for mode,title in [('raw_likelihood','Primary: raw likelihood choice distribution'),
                       ('task_decision','Secondary: benchmark decision-score distribution')]:
        entry=summary['modes'][mode]
        lines += ['## '+title,'',
            'Probability columns are percentages; absolute errors are percentage points (pp).','',
            '| Task | N | FP p(c*) | AQP16 p(c*) | AQP8 p(c*) | AQP16 MAE | AQP8 MAE | MAE ratio 8/16 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for task,v in list(entry['tasks'].items())+[('All (sample weighted)',entry['overall_micro'])]:
            lines.append(f"| {task} | {v['samples']} | {100*v['p_fp_reference']:.4f} | "
                f"{100*v['p_aqp16_reference']:.4f} | {100*v['p_aqp8_reference']:.4f} | "
                f"{100*v['aqp16_absolute_error']:.4f} | {100*v['aqp8_absolute_error']:.4f} | {v['mae_ratio_8_over_16']:.4f} |")
        lines += ['', '| Task | AQP16 RMSE (pp) | AQP8 RMSE (pp) | 95% CI for MAE₈−MAE₁₆ (pp) | AQP16 closer (%) |',
            '|---|---:|---:|---|---:|']
        for task,v in list(entry['tasks'].items())+[('All (sample weighted)',entry['overall_micro'])]:
            ci=np.array(v['bootstrap']['mae_delta_8_minus_16_ci95'])*100
            lines.append(f"| {task} | {100*v['aqp16_rmse']:.4f} | {100*v['aqp8_rmse']:.4f} | "
                f"[{ci[0]:.4f}, {ci[1]:.4f}] | {100*v['aqp16_closer_fraction']:.2f} |")
        ci=entry['overall_micro']['bootstrap']['mae_ratio_8_over_16_ci95']
        lines += ['',f'Overall MAE ratio 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]. '
            'Intervals use 2,000 paired question-bootstrap draws, stratified by task for the overall result. '
            'These intervals describe question-sampling variation, not new model-run variability.', '',
            '### Each model’s own top-1 confidence (secondary diagnostic)', '',
            'Here choices may differ between models; confidence similarity alone does not establish prediction fidelity.','',
            '| Task | FP max p (%) | AQP16 max p (%) | AQP8 max p (%) | Own-max MAE16 (pp) | Own-max MAE8 (pp) | Top-1 agreement16 (%) | Top-1 agreement8 (%) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for task,v in list(entry['tasks'].items())+[('All (sample weighted)',entry['overall_micro'])]:
            fields=['p_fp_reference','p_aqp16_own_top1','p_aqp8_own_top1',
                'aqp16_own_top1_absolute_error','aqp8_own_top1_absolute_error',
                'aqp16_agrees_with_fp','aqp8_agrees_with_fp']
            lines.append('| '+task+' | '+' | '.join(f'{100*v[f]:.4f}' for f in fields)+' |')
        macro=entry['task_macro_mae']
        lines += ['',f"Equal-task macro MAE (pp): AQP16={100*macro['aqp16']:.4f}, AQP8={100*macro['aqp8']:.4f}.",'']
    lines += ['## Reproducibility','', 'Source paths, SHA256 hashes, complete precision/runtime metadata, '
        'validation results, confidence intervals, and signed errors are in summary.json. '
        'Per-question probabilities, chosen indices, errors, and identity hashes are in the two CSV files. '
        'The analysis source is copied alongside the report.','']
    (out/'REPORT.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    run(args.input_root,args.output)
