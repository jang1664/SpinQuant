"""Fixed-prefix cached decode; explicit A/Q/P ablation and batch-size sweep.

prepare freezes corpus/provenance; run creates immutable per-condition outputs.
Full vocabulary arrays are losslessly stored as float16 when exactly representable,
otherwise float32. Identical repeated arrays are hard-linked to repeat zero.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

STAGES = {'AQP16': (16,16,16), 'A8': (8,16,16), 'AQ8': (8,8,16), 'AQP8': (8,8,8),
          'F16': (16,16,16), 'FB16': (16,16,16)}
ROOT = Path(__file__).resolve().parent
SOURCE_SNAPSHOT = {name:(ROOT/name).read_bytes() for name in (
    'measure_aqp_runtime_fidelity.py', 'measure_matrix_output_accuracy.py',
    'result_analysis/load_model.py', 'eval_utils/main.py', 'eval_utils/modeling_llama.py',
    'eval_utils/rotation_utils.py', 'utils/quant_utils.py', 'utils/hadamard_utils.py', 'global_params.py')}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def runtime_environment():
    return {
        'python':sys.version, 'executable':sys.executable,
        'libraries':{n:importlib.metadata.version(n) for n in ['torch','transformers','datasets','numpy','scipy']},
        'cuda':torch.version.cuda, 'cudnn':torch.backends.cudnn.version(),
        'gpu':command('nvidia-smi','--query-gpu=uuid,name,driver_version','--format=csv,noheader'),
        'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','CUBLAS_WORKSPACE_CONFIG','ZP_INT8','SIGNED_KV','ZP_CLAMP','SCALE_NO_UPCAST']},
        'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,
        'tf32_cudnn':torch.backends.cudnn.allow_tf32,
        'fp16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
        'bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        'deterministic':torch.are_deterministic_algorithms_enabled(),
        'cudnn_benchmark':torch.backends.cudnn.benchmark,
        'float32_matmul_precision':torch.get_float32_matmul_precision(), 'autocast':False,
        'attention_backend':'eager', 'tensor_parallel':1,
    }


def prepare(args):
    from datasets import Dataset, load_dataset
    from transformers import AutoTokenizer, AutoConfig
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.input_model, local_files_only=True)
    config = AutoConfig.from_pretrained(args.input_model, local_files_only=True)
    if args.prompt_length + args.positions > config.max_position_embeddings:
        raise ValueError('workload exceeds model context')
    if args.dataset_arrow:
        data = Dataset.from_file(args.dataset_arrow)
    else:
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', cache_dir=args.dataset_cache)
    documents, lines = [], []
    # WikiText rows are paragraphs, not independent source documents. Top-level
    # article headings delimit bootstrap clusters; subheadings remain inside them.
    for text in data['text']:
        if re.match(r'^\s*= [^=].*[^=] =\s*$', text) and lines:
            documents.append(''.join(lines)); lines=[]
        lines.append(text)
    if lines:
        documents.append(''.join(lines))
    candidates=[]
    width=args.prompt_length+args.positions
    for doc_id, text in enumerate(documents):
        tokens=tokenizer(text, add_special_tokens=False)['input_ids']
        for offset in range(0,len(tokens)-width+1,width):
            candidates.append(dict(source_document_id=f'wikitext-test-article-{doc_id}',
                                   source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                                   token_offset=offset, prompt_ids=tokens[offset:offset+args.prompt_length],
                                   continuation_ids=tokens[offset+args.prompt_length:offset+width]))
    if len(candidates)<args.windows:
        raise ValueError(f'only {len(candidates)} nonoverlapping windows available')
    picks=np.random.default_rng(args.seed).choice(len(candidates),args.windows,replace=False)
    corpus=[dict(sample_id=f'window-{i:04d}',**candidates[int(j)]) for i,j in enumerate(picks)]
    with (out/'corpus.jsonl').open('w') as f:
        for row in corpus:
            f.write(json.dumps(row)+'\n')
    paths={'model':str(Path(args.input_model).resolve()), 'checkpoint':str(Path(args.checkpoint).resolve()),
           'rotation':str(Path(args.rotation).resolve())}
    print('Hashing checkpoint and model artifacts',flush=True)
    model_files={str(p.relative_to(paths['model'])):sha256(p) for p in sorted(Path(paths['model']).iterdir())
                 if p.is_file() and p.suffix in ('.json','.safetensors','.bin','.model')}
    manifest=dict(schema_version=1, experiment_id=out.name, scope=args.scope, paths=paths,
                  checkpoint_sha256=sha256(args.checkpoint),rotation_sha256=sha256(args.rotation),
                  rotation_usage='embedded in quantized checkpoint; not reapplied', model_files_sha256=model_files,
                  model_revision=Path(paths['model']).name,tokenizer_revision=Path(paths['model']).name,
                  corpus_sha256=sha256(out/'corpus.jsonl'), dataset_fingerprint=data._fingerprint,
                  dataset='wikitext/wikitext-2-raw-v1/test', dataset_cache=args.dataset_cache,
                  dataset_arrow=args.dataset_arrow, dataset_arrow_sha256=sha256(args.dataset_arrow) if args.dataset_arrow else None,
                  selection='seeded sample without replacement of nonoverlapping within-article windows',
                  seed=args.seed,windows=args.windows,prompt_length=args.prompt_length,positions=args.positions,
                  batch_sizes=[int(x) for x in args.batch_sizes.split(',')], repeats=args.repeats,
                  conditions={k:dict(a_bits=v[0],q_bits=v[1],p_bits=v[2],w_bits=16 if k.startswith('F') else 4,
                     k_bits=16 if k.startswith('F') else 4,v_bits=16 if k.startswith('F') else 4,
                     compute_dtype='bf16' if k=='FB16' else 'fp16') for k,v in STAGES.items()},
                  quantization=dict(a_groupsize=-1,o_proj_a_groupsize=128,q_groupsize=128,p_groupsize=-1,
                     k_groupsize=128,v_groupsize=128,a_asym=True,q_asym=False,p_asym=True,k_asym=True,v_asym=True,
                     clip_ratio=1.0,linear_backend='standard',w_clip=True,w_groupsize=-1,w_asym=False,
                     online_had_mode='factorized',fp32_had=False,ptq_seed=0),
                  mode='cached_decode', first_position='last prompt output predicts first continuation token',
                  eos_policy='replay fixed text through cap, no generated EOS stopping',padding='none: equal fixed lengths',
                  analysis_dtype='float64', aggregation='equal windows, MSE before sqrt',
                  bootstrap=dict(unit='source article',draws=2000,seed=args.seed),
                  tie_rule='lowest token ID', runtime=runtime_environment(),
                  code_commit=command('git','rev-parse','HEAD'),code_status=command('git','status','--short'),
                  storage='full vocabulary, exact float16 or float32 npz; identical repeats hardlinked')
    (out/'code.diff').write_text(command('git','diff','HEAD'))
    # Include untracked experiment implementation, which git diff omits.
    sources=[Path(__file__),ROOT/'utils/paired_fidelity_metrics.py',ROOT/'summarize_aqp_runtime_fidelity.py',
             ROOT/'measure_matrix_output_accuracy.py',ROOT/'tests/test_paired_fidelity.py']
    snapshot=out/'code'; snapshot.mkdir()
    for p in sources:
        if p.exists():
            (snapshot/p.name).write_bytes(p.read_bytes())
    manifest['source_sha256']={p.name:sha256(p) for p in sources if p.exists()}
    write_json(out/'manifest.json',manifest)
    print(json.dumps({'prepared':str(out),'windows':len(corpus),'documents':len(set(x['source_document_id'] for x in corpus))}),flush=True)


def quantizer_config(q):
    return {k:getattr(q,k,None) for k in ('bits','groupsize','sym','clip_ratio')}


def load_condition(manifest, condition):
    from result_analysis.load_model import load_model
    from measure_matrix_output_accuracy import validate_quantized_configuration
    c=manifest['conditions'][condition]
    # The legacy loader logs entire argparse objects; omit those to keep logs
    # concise and avoid recording any future credential-bearing arguments.
    logging.getLogger('spinquant').setLevel(logging.WARNING)
    quantized=not condition.startswith('F')
    model,_=load_model(input_model=manifest['paths']['model'],
        load_qmodel_path=manifest['paths']['checkpoint'] if quantized else None,
        optimized_rotation_path=None, rotate=quantized, device='cuda',
        model_max_length=manifest['prompt_length']+manifest['positions'],
        q_groupsize=128,p_groupsize=-1,q_asym=False,p_asym=True,
        attention_backend='eager', **c)
    model.eval(); model.config.use_cache=True
    if quantized:
        validation=validate_quantized_configuration(model,a_bits=c['a_bits'],q_bits=c['q_bits'],p_bits=c['p_bits'])
    else:
        validation={'unquantized':True}
    validation['layers']={}
    from utils.quant_utils import ActQuantWrapper
    for name,mod in model.named_modules():
        if isinstance(mod,ActQuantWrapper):
            validation['layers'][name]={'A':quantizer_config(mod.quantizer),'output':quantizer_config(mod.out_quantizer),
                'W_bits':c['w_bits'] if '.layers.' in name else 16,
                'linear_backend':mod.linear_backend,'online_had_mode':getattr(mod,'online_had_mode',None),
                'fp32_had':getattr(mod,'fp32_had',None)}
            if not quantized and (mod.quantizer.bits!=16 or mod.out_quantizer.bits!=16):
                raise ValueError('floating control has an active quantizer')
            if 'v_proj' in name and mod.out_quantizer.bits != c['v_bits']:
                raise ValueError('V bit mismatch')
    for i,layer in enumerate(model.model.layers):
        attn=layer.self_attn
        qk=getattr(attn,'apply_rotary_pos_emb_qk_rotation_wrapper',None)
        if quantized and (qk is None or qk.k_bits!=4):
            raise ValueError('K bit mismatch')
        if not quantized and (qk is not None or attn.p_quantizer is not None):
            raise ValueError('floating control has Q/K rotation or P quantization')
        validation['layers'][f'attention.{i}']={
            'Q':None if qk is None else quantizer_config(qk.q_quantizer),
            'K':None if qk is None else quantizer_config(qk.k_quantizer),
            'Q_head_group_size':None if qk is None else qk.q_groupsize,
            'K_head_group_size':None if qk is None else qk.k_groupsize,
            'P':None if attn.p_quantizer is None else quantizer_config(attn.p_quantizer)}
    validation['weight_source']='checkpoint SHA256 in manifest' if quantized else 'original model SHA256 in manifest'
    validation['batch_scale_policy']='A: per token/group; Q/K: per head row; P: per probability row; V: head group'
    return model,validation


def attach_calls(model):
    from utils.quant_utils import ActQuantizer
    counts,handles={},[]
    for name,mod in model.named_modules():
        if isinstance(mod,ActQuantizer):
            counts[name]={'bits':mod.bits,'calls':0}
            def hook(module,inputs,output,name=name):
                counts[name]['calls']+=1
                if module.bits==16 and not torch.equal(inputs[0],output):
                    raise ValueError('bypass changed values')
            handles.append(mod.register_forward_hook(hook))
    return counts,handles


def cache_length(cache):
    if hasattr(cache,'get_seq_length'):
        return cache.get_seq_length()
    return cache[0][0].shape[-2]


def validate_checkpoint_weights(manifest, output):
    checkpoint=torch.load(manifest['paths']['checkpoint'],map_location='cpu',weights_only=False,mmap=True)
    quantizers=checkpoint.get('w_quantizers',{})
    if not quantizers:
        raise ValueError('checkpoint lacks per-layer weight quantization metadata')
    evidence={name:{'bits':int(q.bits),'symmetric':bool(q.sym),
                    'weight_shape':list(checkpoint['model'][name+'.weight'].shape)}
              for name,q in quantizers.items()}
    if any(q['bits']!=4 for q in evidence.values()):
        raise ValueError('checkpoint contains non-W4 projection')
    projection_weights={k[:-7] for k in checkpoint['model'] if k.endswith('.module.weight') and '.layers.' in k}
    if set(evidence)!=projection_weights:
        raise ValueError('weight quantizer metadata does not cover every layer projection')
    write_json(output,{'checkpoint_sha256':manifest['checkpoint_sha256'],'layers':evidence,
                       'coverage':len(evidence),'source':'saved w_quantizers metadata; no requantization'})


def run(args):
    out=Path(args.output_dir); m=json.loads((out/'manifest.json').read_text())
    source_hashes={name:hashlib.sha256(data).hexdigest() for name,data in SOURCE_SNAPSHOT.items()}
    source_dir=out/'execution_sources'/source_hashes['measure_aqp_runtime_fidelity.py']
    source_dir.mkdir(parents=True,exist_ok=True)
    for name,data in SOURCE_SNAPSHOT.items():
        target=source_dir/name; target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists(): target.write_bytes(data)
    if sha256(out/'corpus.jsonl')!=m['corpus_sha256']:
        raise ValueError('corpus hash changed')
    if sha256(m['paths']['checkpoint'])!=m['checkpoint_sha256']:
        raise ValueError('checkpoint hash changed')
    validate_checkpoint_weights(m,out/'weight_validation.json')
    if runtime_environment()!=m['runtime']:
        raise ValueError('runtime environment differs from frozen manifest')
    corpus=[json.loads(line) for line in (out/'corpus.jsonl').read_text().splitlines()]
    conditions=args.conditions.split(',') if args.conditions else list(STAGES)
    batches=[int(x) for x in args.batch_sizes.split(',')] if args.batch_sizes else m['batch_sizes']
    repeats=[int(x) for x in args.repeat_ids.split(',')] if args.repeat_ids else list(range(m['repeats']))
    if not set(conditions)<=m['conditions'].keys() or not set(batches)<=set(m['batch_sizes']) or not set(repeats)<=set(range(m['repeats'])):
        raise ValueError('requested conditions outside manifest')
    # The first invocation should request --batch-sizes 1 for all four stages.
    for repeat in repeats:
        for condition in conditions:
            pending=[b for b in batches if not (out/f'{condition}/r{repeat}/b{b}/complete.json').exists()]
            if not pending:
                continue
            torch.manual_seed(m['seed']); torch.cuda.manual_seed_all(m['seed'])
            print(f'LOAD {condition} repeat={repeat}',flush=True)
            model,validation=load_condition(m,condition)
            for batch_size in pending:
                dest=out/f'{condition}/r{repeat}/b{batch_size}'
                dest.mkdir(parents=True,exist_ok=False)
                write_json(dest/'execution_source.json',{'source_sha256':source_hashes,
                    'source_directory':str(source_dir.resolve())})
                torch.cuda.reset_peak_memory_stats()
                started=time.monotonic(); artifacts=[]; timings=[]
                counts,handles=attach_calls(model)
                expected_calls=0; cache_trace=[]
                try:
                    with torch.inference_mode():
                        for start in range(0,len(corpus),batch_size):
                            if shutil.disk_usage(out).free < 10*1024**3:
                                raise OSError('less than 10 GiB free: stopping before next shard')
                            samples=corpus[start:start+batch_size]
                            if len(samples)!=batch_size:
                                raise ValueError('window count must be divisible by actual batch size')
                            ids=torch.tensor([s['prompt_ids'] for s in samples],device='cuda')
                            cache=None; values=[]
                            for step in range(m['positions']):
                                length=m['prompt_length']+step
                                positions=torch.arange(length-ids.shape[1],length,device='cuda').unsqueeze(0).expand(len(samples),-1)
                                torch.cuda.synchronize(); tick=time.monotonic()
                                result=model(input_ids=ids,attention_mask=torch.ones((len(samples),length),device='cuda',dtype=torch.long),
                                    position_ids=positions,past_key_values=cache,use_cache=True,return_dict=True)
                                torch.cuda.synchronize()
                                timings.append({'batch_start':start,'step':step,'seconds':time.monotonic()-tick,
                                                'allocated_bytes':torch.cuda.memory_allocated()})
                                cache=result.past_key_values
                                if cache is None or cache_length(cache)!=length:
                                    raise ValueError(f'KV cache length mismatch at step {step}')
                                if start==0:
                                    cache_trace.append({'step':step,'input_tokens':ids.shape[1],'cache_length':int(cache_length(cache))})
                                logits=result.logits[:,-1,:].float().cpu().numpy()
                                if not np.isfinite(logits).all():
                                    raise ValueError('nonfinite model output')
                                compact=logits.astype(np.float16)
                                values.append(compact if np.array_equal(compact.astype(np.float32),logits) else logits)
                                ids=torch.tensor([[s['continuation_ids'][step]] for s in samples],device='cuda')
                                expected_calls+=1
                                del result
                            array=np.stack(values,axis=1)
                            for index,sample in enumerate(samples):
                                path=dest/f"{sample['sample_id']}.npz"
                                old=out/f"{condition}/r0/b{batch_size}/{path.name}"
                                identical=False
                                if repeat and old.exists():
                                    with np.load(old) as f:
                                        identical=np.array_equal(f['logits'],array[index])
                                if identical:
                                    os.link(old,path)
                                else:
                                    np.savez_compressed(path,logits=array[index])
                                artifacts.append(dict(sample_id=sample['sample_id'],sha256=sha256(path),
                                                      shape=list(array[index].shape),dtype=str(array.dtype),
                                                      identical_to_repeat_zero=identical if repeat else None))
                            print(f'PROGRESS {condition} r{repeat} b{batch_size} {start+len(samples)}/{len(corpus)} elapsed={time.monotonic()-started:.1f}s',flush=True)
                            del cache,values,array
                    for name,record in counts.items():
                        expected=expected_calls if record['bits']<16 else 0
                        if record['calls']!=expected:
                            raise ValueError(f'quantizer call mismatch {name}: {record}, expected={expected}')
                    write_json(dest/'condition_validation.json',dict(validation,quantizer_calls=counts,cache_trace=cache_trace,
                        expected_forward_calls=expected_calls,actual_batch_size=batch_size,grouping=[s['sample_id'] for s in corpus]))
                    write_json(dest/'timing.json',timings)
                    write_json(dest/'complete.json',dict(condition=condition,repeat=repeat,batch_size=batch_size,
                        seconds=time.monotonic()-started,peak_memory_bytes=torch.cuda.max_memory_allocated(),artifacts=artifacts))
                except Exception as error:
                    write_json(dest/'failure.json',{'type':type(error).__name__,'message':str(error)})
                    raise
                finally:
                    for handle in handles: handle.remove()
            del model
            gc.collect(); torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='action',required=True)
    prep=sub.add_parser('prepare')
    for name in ['input-model','checkpoint','rotation','output-dir']:
        prep.add_argument('--'+name,required=True)
    prep.add_argument('--windows',type=int,default=128)
    prep.add_argument('--prompt-length',type=int,default=512)
    prep.add_argument('--positions',type=int,default=128)
    prep.add_argument('--seed',type=int,default=20261015)
    prep.add_argument('--scope',choices=['smoke','main'],required=True)
    prep.add_argument('--repeats',type=int,default=3)
    prep.add_argument('--batch-sizes',default='1,2,4,8')
    prep.add_argument('--dataset-cache',default='/data/hf_cache/datasets')
    prep.add_argument('--dataset-arrow',help='Read an existing cached Arrow file without cache locks')
    runp=sub.add_parser('run')
    runp.add_argument('--output-dir',required=True)
    runp.add_argument('--conditions')
    runp.add_argument('--batch-sizes')
    runp.add_argument('--repeat-ids')
    args=parser.parse_args()
    torch.set_num_threads(4)
    if args.action=='prepare':
        if min(args.windows,args.prompt_length,args.positions,args.repeats)<1:
            parser.error('sizes must be positive')
        bs=[int(x) for x in args.batch_sizes.split(',')]
        if 1 not in bs or len(bs)!=len(set(bs)) or any(b<1 or args.windows%b for b in bs):
            parser.error('unique positive batches including 1 must divide window count')
        prepare(args)
    else:
        run(args)


if __name__=='__main__':
    main()
