"""Fresh Llama-3.1 zero-shot likelihoods using frozen full benchmark requests.

Uses lm-eval's installed HFLM likelihood path without downloading or rebuilding
datasets. All scores are recomputed; the source JSON supplies only test inputs,
task configuration and identifiers. Each task is batched independently.
"""
from __future__ import annotations
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from transformers import LlamaTokenizerFast
from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM
import lm_eval.api.model
import lm_eval.models.huggingface

import measure_aqp_runtime_fidelity as base
from summarize_zero_shot_top1_fidelity import TASKS, CONDITIONS, scores, choice_lengths, row_metrics

COUNTS = dict(hellaswag=10042,arc_easy=2376,arc_challenge=1172,winogrande=1267,openbookqa=500)
MODEL_CONDITIONS = dict(fp_base='F16',aqp16='AQP16',aqp8='AQP8')


def now():
    return datetime.now(timezone.utc).isoformat()


def sources():
    result = dict(base.SOURCE_SNAPSHOT)
    for name in ['measure_aqp_zero_shot.py','summarize_zero_shot_top1_fidelity.py']:
        result[name]=(base.ROOT/name).read_bytes()
    for module in [lm_eval.api.model,lm_eval.models.huggingface]:
        result['external/'+module.__name__.replace('.','/')+'.py']=Path(module.__file__).read_bytes()
    return result


def prepare(source_manifest, source_results, output, limit=None):
    out=Path(output).resolve()
    original=json.loads(Path(source_manifest).read_text())
    corpus=json.loads(Path(source_results).read_text())
    for task in TASKS:
        if corpus['n-shot'][task] != 0 or len(corpus['samples'][task]) != COUNTS[task]:
            raise ValueError('source is not the complete expected zero-shot corpus')
        if corpus['n-samples'][task]['effective'] != COUNTS[task]:
            raise ValueError('source task count mismatch')
    out.mkdir(parents=True,exist_ok=False)
    frozen={k:{task:corpus[k][task] for task in TASKS} for k in ['configs','versions','n-shot','n-samples']}
    # Deliberately remove old predictions from the evaluation input.
    frozen['samples']={task:[{k:v for k,v in s.items() if k not in
        ['resps','filtered_resps','acc','acc_norm']} for s in corpus['samples'][task]] for task in TASKS}
    base.write_json(out/'corpus.json',frozen)
    m={k:original[k] for k in ['paths','checkpoint_sha256','rotation_sha256','rotation_usage',
        'model_files_sha256','model_revision','tokenizer_revision','seed','quantization']}
    m.update(experiment_id=out.name, created=now(),
        model_name='meta-llama/Llama-3.1-8B', source_manifest=str(Path(source_manifest).resolve()),
        source_manifest_sha256=base.sha256(source_manifest), source_results=str(Path(source_results).resolve()),
        source_results_sha256=base.sha256(source_results),corpus_sha256=base.sha256(out/'corpus.json'),
        prompt_length=2048, positions=0, mode='zero-shot choice continuation likelihood; KV cache disabled',
        batch_size=4, batch_sizes=[4], repeats=1, max_length=2048, eval_limit=limit, task_counts=COUNTS,
        conditions={k:original['conditions'][k] for k in ['F16','AQP16','AQP8']},
        runtime=base.runtime_environment(),
        lm_eval_version=importlib.metadata.version('lm_eval'),
        grouping='each task separately; HFLM sorts by tokenized length; context-logit reuse enabled',
        source_sha256={name:hashlib.sha256(data).hexdigest() for name,data in sources().items()})
    for name,data in sources().items():
        p=out/'code'/name; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(data)
    base.write_json(out/'manifest.json',m)
    print(json.dumps(dict(prepared=str(out),limit=limit,questions=sum(COUNTS.values()) if limit is None else limit*5)),flush=True)


def quantization_metadata(m,condition):
    bits=m['conditions'][MODEL_CONDITIONS[condition]]
    return dict(model=m['model_name'],weight_bits=bits['w_bits'],key_bits=bits['k_bits'],value_bits=bits['v_bits'],
        activation_bits=bits['a_bits'],query_bits=bits['q_bits'],probability_bits=bits['p_bits'],
        compute_dtype='fp16',attention_backend='eager',linear_backend='standard',model_max_length=2048,
        lm_eval_batch_size='4',num_fewshot=0,seed=m['seed'],query_groupsize=128,probability_groupsize=-1,
        quantized_checkpoint_sha256=m['checkpoint_sha256'] if condition!='fp_base' else None,
        rotation_checkpoint_sha256=m['rotation_sha256'] if condition!='fp_base' else None,
        rotation_usage='embedded in checkpoint, not reapplied' if condition!='fp_base' else 'none')


def run(output,condition):
    out=Path(output).resolve(); m=json.loads((out/'manifest.json').read_text())
    dest=out/condition
    if (dest/'complete.json').exists():
        print(f'SKIP complete {condition}',flush=True); return
    if {k:hashlib.sha256(v).hexdigest() for k,v in sources().items()} != m['source_sha256']:
        raise ValueError('execution source differs from frozen preparation')
    if base.runtime_environment()!=m['runtime'] or importlib.metadata.version('lm_eval')!=m['lm_eval_version']:
        raise ValueError('runtime differs from manifest')
    if base.sha256(out/'corpus.json')!=m['corpus_sha256']:
        raise ValueError('corpus hash mismatch')
    if base.sha256(m['paths']['checkpoint'])!=m['checkpoint_sha256']:
        raise ValueError('checkpoint hash mismatch')
    for name,expected in m['model_files_sha256'].items():
        if base.sha256(Path(m['paths']['model'])/name)!=expected:
            raise ValueError(f'original model artifact hash mismatch: {name}')
    base.validate_checkpoint_weights(m,out/'weight_validation.json')
    dest.mkdir(exist_ok=False)
    torch.manual_seed(m['seed']); torch.cuda.manual_seed_all(m['seed'])
    print(f'LOAD {condition}',flush=True)
    model,validation=base.load_condition(m,MODEL_CONDITIONS[condition])
    model.config.use_cache=False
    tokenizer=LlamaTokenizerFast.from_pretrained(m['paths']['model'],local_files_only=True,
        model_max_length=2048,padding_side='right',add_eos_token=False,add_bos_token=False)
    lm=HFLM(pretrained=model,tokenizer=tokenizer,max_length=2048,batch_size=4,device='cuda',
            backend='causal',add_bos_token=False,logits_cache=True)
    source=json.loads((out/'corpus.json').read_text())
    result={k:copy.deepcopy(source[k]) for k in ['configs','versions','n-shot']}
    result.update(samples={},results={},**{'n-samples':{}},
        config=dict(model='hf',model_dtype='torch.float16',batch_size='4',batch_sizes=[],
            device='cuda',limit=m['eval_limit'],max_length=2048),
        spinquant_quantization=quantization_metadata(m,condition),
        fresh_evaluation=dict(started=now(),input_corpus_sha256=m['corpus_sha256'],
            lm_eval_version=m['lm_eval_version'],runtime=m['runtime'],source_sha256=m['source_sha256']))
    quantizer_calls,handles=base.attach_calls(model)
    state=dict(task=None,forward_calls=0,last_print=0,batch_shapes=Counter(),task_forward_calls=0)
    started=time.monotonic(); torch.cuda.reset_peak_memory_stats()

    def forward_hook(module,args):
        ids=args[0]
        if ids.shape[0] > 4 or ids.shape[1] > 2048:
            raise ValueError('unexpected batch or sequence shape')
        state['forward_calls']+=1; state['task_forward_calls']+=1
        state['batch_shapes'][f'{ids.shape[0]}x{ids.shape[1]}']+=1
        if time.monotonic()-state['last_print'] >= 60:
            elapsed=time.monotonic()-started
            record=dict(state='running',condition=condition,task=state['task'],
                task_forward_batches=state['task_forward_calls'],forward_calls=state['forward_calls'],
                completed_tasks=list(result['samples']),elapsed_seconds=elapsed,pid=os.getpid(),updated=now())
            base.write_json(out/'status.json',record)
            print('PROGRESS '+json.dumps(record),flush=True); state['last_print']=time.monotonic()
    handles.append(model.register_forward_pre_hook(forward_hook))
    try:
        for task in TASKS:
            state['task']=task; state['task_forward_calls']=0; state['last_print']=0
            samples=copy.deepcopy(source['samples'][task][:m['eval_limit']])
            requests=[]
            for sample in samples:
                for i,args in enumerate(sample['arguments']):
                    requests.append(Instance(request_type='loglikelihood',doc=sample['doc'],arguments=tuple(args),
                        idx=i,metadata=(task,sample['doc_id'],1)))
            tick=time.monotonic()
            responses=lm.loglikelihood(requests,disable_tqdm=True)
            if len(responses)!=len(requests):
                raise ValueError('missing likelihood outputs')
            offset=0
            for sample in samples:
                count=len(sample['arguments'])
                values=[[float(v),bool(greedy)] for v,greedy in responses[offset:offset+count]]
                sample['filtered_resps']=values; sample['resps']=[[r] for r in values]
                z=scores(sample)
                gold=int(sample['doc']['answer'])-1 if task=='winogrande' else int(sample['target'])
                sample['acc']=float(int(z.argmax())==gold)
                if 'acc_norm' in sample['metrics']:
                    sample['acc_norm']=float(int((z/choice_lengths(sample)).argmax())==gold)
                offset+=count
            result['samples'][task]=samples
            result['n-samples'][task]=dict(original=COUNTS[task],effective=len(samples))
            result['results'][task]={metric+',none':float(np.mean([s[metric] for s in samples]))
                for metric in samples[0]['metrics']}
            base.write_json(dest/(task+'.json'),dict(samples=samples,results=result['results'][task],
                seconds=time.monotonic()-tick,requests=len(requests),forward_batches=state['task_forward_calls']))
            print(f'COMPLETE {condition} {task}: {len(samples)} questions, {len(requests)} requests, '
                  f'{time.monotonic()-tick:.1f}s',flush=True)
        for name,record in quantizer_calls.items():
            expected=state['forward_calls'] if record['bits']<16 else 0
            if record['calls']!=expected:
                raise ValueError(f'quantizer forward count mismatch: {name}')
        result['spinquant_quantization']['evaluation_seconds']=time.monotonic()-started
        result['fresh_evaluation']['finished']=now()
        base.write_json(dest/'validation.json',dict(validation,quantizer_calls=quantizer_calls,
            forward_calls=state['forward_calls'],batch_shapes=dict(state['batch_shapes']),
            tokenizer_class=type(tokenizer).__name__,add_bos_token=False,max_length=2048))
        result_path=out/(condition+'.json'); base.write_json(result_path,result)
        base.write_json(dest/'complete.json',dict(condition=condition,finished=now(),
            limit=m['eval_limit'],questions=sum(len(s) for s in result['samples'].values()),
            seconds=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),
            output_sha256=base.sha256(result_path)))
        base.write_json(out/'status.json',dict(state='condition_complete',condition=condition,updated=now()))
    except Exception as error:
        base.write_json(dest/'failure.json',dict(type=type(error).__name__,message=str(error)))
        base.write_json(out/'status.json',dict(state='failed',condition=condition,error=str(error),updated=now()))
        raise
    finally:
        for handle in handles: handle.remove()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for arg in ['source-manifest','source-results','output']: prep.add_argument('--'+arg,required=True)
    prep.add_argument('--limit',type=int)
    runp=sub.add_parser('run'); runp.add_argument('--output',required=True)
    runp.add_argument('--condition',required=True,choices=CONDITIONS)
    args=p.parse_args(); torch.set_num_threads(4)
    if args.action=='prepare':
        if args.limit is not None and args.limit<1: p.error('limit must be positive')
        prepare(args.source_manifest,args.source_results,args.output,args.limit)
    else: run(args.output,args.condition)
