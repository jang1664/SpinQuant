"""Analyze full-vocabulary cached-decode shards without pooling contexts or repeats."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from utils.paired_fidelity_metrics import paired_metrics, probabilities, cluster_ratio, MARGIN_EDGES

STAGES=['AQP16','A8','AQ8','AQP8']


def write_json(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def summarize_group(g):
    per=g.groupby('sample_id',sort=True)
    result={'windows':len(per),'positions':len(g)}
    for field in ['logit_mae','logit_mse','centered_logit_mse','probability_mae','probability_mse',
                  'probability_max_abs','flip','baseline_tie','test_tie','winner_outside_top2',
                  'margin_logit','margin_probability','delta_margin']:
        macro=float(per[field].mean().mean())
        result[field.replace('_mse','_rmse')]=float(np.sqrt(macro)) if field.endswith('_mse') else macro
    result['top1_disagreement_rate']=result.pop('flip')
    result['top1_agreement_rate']=1-result['top1_disagreement_rate']
    d=g.selected_probability_delta
    result['selected_signed_mean']=float(per.selected_probability_delta.mean().mean())
    result['selected_mean_abs']=float(g.assign(abs_delta=d.abs()).groupby('sample_id').abs_delta.mean().mean())
    result['selected_rms']=float(np.sqrt(g.assign(d2=d*d).groupby('sample_id').d2.mean().mean()))
    result['token_weighted_selected_rms']=float(np.sqrt(np.mean(d*d)))
    result['token_weighted_flip_rate']=float(g.flip.mean())
    for q in [.5,.9,.99]:
        # All planned windows have equal positions; this empirical distribution
        # therefore assigns equal total weight to each window.
        result[f'selected_abs_p{int(q*100)}']=float(d.abs().quantile(q))
        result[f'margin_probability_p{int(q*100)}']=float(g.margin_probability.quantile(q))
    return result


def analyze(args):
    root=Path(args.experiment); out=root/args.analysis_id
    out.mkdir(exist_ok=False)
    (out/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    metrics_source=Path(__file__).parent/'utils/paired_fidelity_metrics.py'
    (out/'paired_fidelity_metrics.py').write_bytes(metrics_source.read_bytes())
    m=json.loads((root/'manifest.json').read_text())
    corpus=[json.loads(line) for line in (root/'corpus.jsonl').read_text().splitlines()]
    if sha256(root/'corpus.jsonl')!=m['corpus_sha256']: raise ValueError('corpus hash mismatch')
    expected={(c,r,b) for c in m['conditions'] for r in range(m['repeats']) for b in m['batch_sizes']}
    completed={}; missing=[]; repeat_evidence=[]; artifact_hashes={}
    validations={}
    if (root/'weight_validation.json').exists():
        validations['checkpoint_weights']=json.loads((root/'weight_validation.json').read_text())
    for c,r,b in sorted(expected):
        path=root/f'{c}/r{r}/b{b}'
        if not (path/'complete.json').exists():
            missing.append([c,r,b]); continue
        done=json.loads((path/'complete.json').read_text())
        if len(done['artifacts'])!=len(corpus) or {a['sample_id'] for a in done['artifacts']}!={s['sample_id'] for s in corpus}:
            raise ValueError(f'incomplete samples in {path}')
        for art in done['artifacts']:
            if sha256(path/f"{art['sample_id']}.npz")!=art['sha256']:
                raise ValueError('logits checksum mismatch')
            artifact_hashes[c,r,b,art['sample_id']]=art['sha256']
        completed[c,r,b]=path
        repeat_evidence.append(dict(condition=c,repeat=r,batch=b,
            identical_windows_to_repeat_zero=sum(a.get('identical_to_repeat_zero') is True for a in done['artifacts']),
            windows=len(done['artifacts'])))
        validations[f'{c}/r{r}/b{b}']=json.loads((path/'condition_validation.json').read_text())
    if not completed: raise ValueError('no completed measurements')
    write_json(out/'condition_validation.json',validations)
    pairs=[]
    for r in range(m['repeats']):
        for b in m['batch_sizes']:
            for test in STAGES[1:]:
                if ('AQP16',r,b) in completed and (test,r,b) in completed:
                    pairs.append(('cumulative','AQP16',test,r,b,b))
            for base,test in zip(STAGES,STAGES[1:]):
                if (base,r,b) in completed and (test,r,b) in completed:
                    pairs.append(('incremental',base,test,r,b,b))
            if b!=1:
                for c in m['conditions']:
                    if (c,r,1) in completed and (c,r,b) in completed:
                        pairs.append(('runtime',c,c,r,1,b))
    std_rows=[]; cancellation=[]; writer=None
    with (out/'paired_metrics.csv').open('w') as handle:
        for sample in corpus:
            sid=sample['sample_id']; arrays={}; content_arrays={}; metric_cache={}
            for key,path in completed.items():
                digest=artifact_hashes[(*key,sid)]
                if digest not in content_arrays:
                    with np.load(path/f'{sid}.npz') as f: content_arrays[digest]=f['logits']
                arrays[key]=content_arrays[digest]
                if arrays[key].shape[0]!=m['positions']: raise ValueError('position count differs')
            if ('AQP16',0,1) not in arrays: raise ValueError('missing global reference AQP16,r0,c0')
            ref=arrays['AQP16',0,1].argmax(-1)
            for kind,base,test,r,bb,tb in pairs:
                # Identical outputs permit exact calculation reuse, but every
                # repeat/comparison still gets its own position records below.
                cache_key=(artifact_hashes[base,r,bb,sid],artifact_hashes[test,r,tb,sid])
                if cache_key not in metric_cache:
                    metric_cache[cache_key]=paired_metrics(arrays[base,r,bb],arrays[test,r,tb],ref,args.chunk_size)
                rows=[dict(row) for row in metric_cache[cache_key]]
                for pos,row in enumerate(rows):
                    row.update(sample_id=sid,source_document_id=sample['source_document_id'],token_position=pos,
                        absolute_token_position=m['prompt_length']+pos,comparison_kind=kind,
                        baseline_condition=base,test_condition=test,repeat=r,baseline_batch=bb,test_batch=tb,
                        mode='cached_decode',compute_dtype=m['conditions'][test]['compute_dtype'])
                    if writer is None: writer=csv.DictWriter(handle,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row)
            # One runtime standard deviation per repeat, never across repeats.
            for r in range(m['repeats']):
                for c in m['conditions']:
                    if not all((c,r,b) in arrays for b in m['batch_sizes']): continue
                    own_ref=arrays[c,r,1].argmax(-1)
                    ps=[]; own=[]
                    for b in m['batch_sizes']:
                        p=probabilities(arrays[c,r,b]); ix=np.arange(len(p))
                        ps.append(p[ix,ref]); own.append(p[ix,own_ref])
                    std_rows.append(dict(sample_id=sid,source_document_id=sample['source_document_id'],condition=c,repeat=r,
                        fixed_prefix_avg_std_top1_prob=float(np.std(ps,axis=0,ddof=1).mean()),
                        own_c0_fixed_prefix_avg_std_top1_prob=float(np.std(own,axis=0,ddof=1).mean())))
                for b in m['batch_sizes']:
                    if not all((c,r,b) in arrays for c in STAGES): continue
                    selected=[probabilities(arrays[c,r,b])[np.arange(len(ref)),ref] for c in STAGES]
                    steps=np.diff(selected,axis=0); total=selected[-1]-selected[0]
                    residual=np.max(np.abs(steps.sum(0)-total))
                    if residual>1e-12: raise ValueError('signed steps fail telescoping identity')
                    winners=[arrays[c,r,b].argmax(-1) for c in STAGES]
                    for i in [1,2]:
                        returns=(winners[i]!=winners[0]) & (winners[i+1]==winners[0])
                        cancellation.append(dict(sample_id=sid,repeat=r,batch=b,stage=STAGES[i+1],
                            return_to_reference_rate=float(returns.mean()),
                            previous_disagreements=int((winners[i]!=winners[0]).sum()),
                            returns=int(returns.sum()),telescoping_max_residual=float(residual),
                            signed_cancellation_fraction=float((np.abs(total)+1e-15<np.abs(steps).sum(0)).mean())))
            print(f'ANALYZED {sid}',flush=True)
    if writer is None: raise ValueError('no complete pairs')
    df=pd.read_csv(out/'paired_metrics.csv')
    groups=['comparison_kind','baseline_condition','test_condition','repeat','baseline_batch','test_batch']
    summaries=[]; margins=[]
    for key,g in df.groupby(groups,sort=True):
        meta=dict(zip(groups,key)); summaries.append(dict(meta,**summarize_group(g)))
        for low,high in zip(MARGIN_EDGES,MARGIN_EDGES[1:]):
            bin_rows=g[(g.margin_probability>=low)&(g.margin_probability<high)]
            margins.append(dict(meta,lower=low,upper=min(high,1),positions=len(bin_rows),
                                windows=bin_rows.sample_id.nunique(),
                                flip_rate=None if bin_rows.empty else float(bin_rows.groupby('sample_id').flip.mean().mean()),
                                token_weighted_flip_rate=None if bin_rows.empty else float(bin_rows.flip.mean())))
    summary=pd.DataFrame(summaries); summary.to_csv(out/'summary.csv',index=False)
    pd.DataFrame(margins).to_csv(out/'margin_flip_rates.csv',index=False)
    pd.DataFrame(std_rows).to_csv(out/'runtime_std.csv',index=False)
    pd.DataFrame(cancellation).to_csv(out/'cancellation.csv',index=False)
    ratios=[]; overlap=[]
    # Use precisely matched runtime/position/repeat sets for numerator and denominator.
    runtime=df[(df.comparison_kind=='runtime')&(df.test_condition=='AQP16')]
    keys=['sample_id','source_document_id','token_position','repeat','test_batch']
    for (kind,base,test),g in df[(df.comparison_kind!='runtime')&(df.test_batch!=1)].groupby(groups[:3]):
        merged=g.merge(runtime[keys+['selected_probability_delta','flip']],on=keys,suffixes=('_quant','_runtime'),validate='one_to_one')
        if len(merged)!=len(g): continue
        for rep in ['all']+sorted(merged.repeat.unique().tolist()):
            part=merged if rep=='all' else merged[merged.repeat==rep]
            sample_stats=part.assign(n2=part.selected_probability_delta_quant**2,d2=part.selected_probability_delta_runtime**2).groupby(['sample_id','source_document_id'])[['n2','d2']].mean()
            result=cluster_ratio(sample_stats.n2,sample_stats.d2,sample_stats.index.get_level_values(1),seed=m['seed'])
            ratios.append(dict(comparison_kind=kind,baseline_condition=base,test_condition=test,repeat=rep,
                               runtime_batches=sorted(part.test_batch.unique().tolist()),**result))
        q=merged.selected_probability_delta_quant.abs(); d=merged.selected_probability_delta_runtime.abs()
        qbig=(q>=q.quantile(.9))&(q>0); dbig=(d>=d.quantile(.9))&(d>0)
        overlap.append(dict(comparison_kind=kind,baseline_condition=base,test_condition=test,
                            abs_spearman=None if q.nunique()<2 or d.nunique()<2 else float(q.corr(d,method='spearman')),
                            both_top_decile_fraction=float((qbig&dbig).mean()),
                            both_flip_fraction=float((merged.flip_quant&merged.flip_runtime).mean())))
    ratios.sort(key=lambda row:(row['comparison_kind'],STAGES.index(row['test_condition']),
                                -1 if row['repeat']=='all' else row['repeat']))
    ablation=summary[summary.comparison_kind!='runtime'].copy()
    for bit in ['a_bits','q_bits','p_bits']:
        ablation[bit]=ablation.test_condition.map(lambda c:m['conditions'][c][bit])
    ratio_frame=pd.DataFrame([{k:v for k,v in r.items() if k in ['comparison_kind','baseline_condition','test_condition','repeat','ratio','ratio_ci','runtime_rms','quantization_rms']} for r in ratios if r['repeat']!='all'])
    if len(ratio_frame):
        ablation=ablation.merge(ratio_frame,on=['comparison_kind','baseline_condition','test_condition','repeat'],how='left')
        # Ratios summarize non-c0 runtimes jointly; never label a c0 row with them.
        ablation.loc[ablation.test_batch==1,['ratio','ratio_ci','runtime_rms','quantization_rms']]=None
    ablation.to_csv(out/'ablation_summary.csv',index=False)
    runtime_std_summary=[]
    if std_rows:
        for (condition,repeat),g in pd.DataFrame(std_rows).groupby(['condition','repeat']):
            runtime_std_summary.append(dict(condition=condition,repeat=int(repeat),windows=len(g),
                fixed_prefix_avg_std_top1_prob=float(g.fixed_prefix_avg_std_top1_prob.mean()),
                own_c0_fixed_prefix_avg_std_top1_prob=float(g.own_c0_fixed_prefix_avg_std_top1_prob.mean())))
    primary_margin_bins=[]
    for kind,condition in [('cumulative','AQP8'),('runtime','AQP16')]:
        g=df[(df.comparison_kind==kind)&(df.test_condition==condition)&(df.test_batch!=1)]
        if g.empty: continue
        for low,high in zip(MARGIN_EDGES,MARGIN_EDGES[1:]):
            rows=g[(g.margin_probability>=low)&(g.margin_probability<high)]
            primary_margin_bins.append(dict(comparison_kind=kind,test_condition=condition,
                lower=low,upper=min(high,1),positions=len(rows),windows=rows.sample_id.nunique(),
                flip_rate=None if rows.empty else float(rows.groupby('sample_id').flip.mean().mean())))
    state=dict(status='complete' if not missing else 'partial',completed_runs=len(completed),expected_runs=len(expected),
               missing=missing,ratios=ratios,overlap=overlap,repeat_evidence=repeat_evidence,
               manifest_sha256=sha256(root/'manifest.json'),analysis_source_sha256=sha256(__file__),
               metrics_source_sha256=sha256(metrics_source),runtime_std_summary=runtime_std_summary,
               primary_margin_bins=primary_margin_bins,
               aggregation='equal-window means; sqrt after mean MSE; repeated runs retained separately',
               primary_reference_token='AQP16 repeat 0 batch 1 argmax, fixed across all comparisons',
               schema='CSV: token IDs/positions/repeat/batches integers; measurements float64; flags boolean',
               full_vocabulary=True,positions_per_window=m['positions'],windows=m['windows'])
    write_json(out/'summary.json',state)
    make_figures(out,df,summary,pd.DataFrame(margins),ratios)
    report(out,m,state,summary,cancellation)
    print(json.dumps({'analysis':str(out),'status':state['status'],'completed_runs':len(completed)}),flush=True)


def make_figures(out,df,summary,margins,ratios):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figs=out/'figures'; figs.mkdir()
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind in ['cumulative','incremental']:
        g=summary[(summary.comparison_kind==kind)&(summary.test_batch==1)]
        means=g.assign(selected_mse=g.selected_rms**2).groupby('test_condition')[['selected_mse','top1_disagreement_rate']].mean().reindex(STAGES[1:])
        means['selected_rms']=np.sqrt(means.selected_mse)
        means=means[['selected_rms','top1_disagreement_rate']]
        for ax,col in zip(axes,means): ax.plot(STAGES[1:],means[col],marker='o',label=kind); ax.set_ylabel(col); ax.legend()
    fig.tight_layout(); fig.savefig(figs/'ablation.png',dpi=180); plt.close(fig)
    rows=[r for r in ratios if r['repeat']=='all']
    if rows:
        fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(len(rows)); width=.35
        for shift,metric,label in [(-width/2,'quantization','quantization'),(width/2,'runtime','AQP16 runtime')]:
            heights=np.array([r[metric+'_rms'] for r in rows])
            ci=np.array([r[metric+'_ci'] for r in rows])
            # Percentile CIs can exclude a point estimate; draw intervals directly.
            ax.bar(x+shift,heights,width,label=label)
            ax.vlines(x+shift,ci[:,0],ci[:,1],color='black',linewidth=1)
        ax.set_xticks(x,[r['comparison_kind']+'\n'+r['test_condition'] for r in rows]); ax.set_ylabel('Fixed-token probability RMS')
        ax.legend(loc='upper center',bbox_to_anchor=(.5,1.18),ncol=2)
        fig.tight_layout(); fig.savefig(figs/'quantization_runtime.png',dpi=180); plt.close(fig)
    chosen=margins[(margins.comparison_kind=='cumulative')&(margins.test_condition=='AQP8')]
    if len(chosen):
        fig,ax=plt.subplots(); chosen.groupby('lower').flip_rate.mean().plot.bar(ax=ax)
        ax.set_ylabel('Flip rate'); ax.set_xlabel('Baseline probability margin bin lower bound')
        fig.tight_layout(); fig.savefig(figs/'margin_flips.png',dpi=180); plt.close(fig)
    keys=['sample_id','token_position','repeat','test_batch']
    q=df[(df.comparison_kind=='cumulative')&(df.test_condition=='AQP8')&(df.test_batch!=1)]
    d=df[(df.comparison_kind=='runtime')&(df.test_condition=='AQP16')]
    both=q.merge(d,on=keys,suffixes=('_q','_r'))
    if len(both):
        fig,ax=plt.subplots(); ax.scatter(both.selected_probability_delta_r.abs(),both.selected_probability_delta_q.abs(),s=2,alpha=.25)
        ax.set_xlabel('|runtime fixed-token probability change|'); ax.set_ylabel('|AQP8 fixed-token probability change|')
        fig.tight_layout(); fig.savefig(figs/'paired_scatter.png',dpi=180); plt.close(fig)


def report(out,m,state,summary,cancellation):
    lines=['# A/Q/P numerical fidelity report','',f"Scope: **{m['scope']}**; model `{m['paths']['model']}`.",
        f"Status: **{state['status']}** ({state['completed_runs']}/{state['expected_runs']} runs).",
        f"Fixed WikiText corpus: {m['windows']} windows, {m['prompt_length']} prompt tokens, {m['positions']} evaluated positions.",
        '', '## Measurement and validation','',
        'Every completed run validates layer-wise A/Q/P/K/V settings, actual quantizer calls, and growing KV caches. W4 weights come from the identical hashed checkpoint. When present, weight_validation.json verifies the saved w_quantizers metadata covers every projection and declares four bits for each. No weights or rotations are reoptimized.',
        'Every condition replays identical original-text tokens and maintains its own cache. Equal fixed lengths require no padding. The last prefill logit is position 0; positions 1 onward use cached one-token decode. Logits are retained for the full vocabulary, and analysis uses float64.',
        'Runtime batches are compared within dtype. Repeats are independent model loads and are never pooled as additional runtime conditions. Reference tokens are fixed by AQP16/r0/b1; pair-specific top-1/top-2 IDs are stored separately.',
        '', '## Quantization versus runtime','',
        '| Comparison | Stage | Quantization RMS | Runtime RMS | Ratio | 95% paired ratio CI |',
        '|---|---|---:|---:|---:|---|']
    for r in state['ratios']:
        if r['repeat']!='all': continue
        ratio='unstable' if r['ratio'] is None else f"{r['ratio']:.4g}"
        ci='not estimable' if r['ratio_ci'] is None else str([round(v,4) for v in r['ratio_ci']])
        lines.append(f"| {r['comparison_kind']} | {r['test_condition']} | {r['quantization_rms']:.6g} | {r['runtime_rms']:.6g} | {ratio} | {ci} |")
    lines+=['','RMS uses the same selected-token probability difference and the same non-c0 runtime set. All stages use the AQP16 runtime denominator. Bootstrap resamples source articles 2,000 times, preserving windows within each article. Per-repeat estimates and absolute-value intervals are in summary.json.',
            '', '## Direct cumulative and incremental differences at batch 1','',
            '| Kind | Baseline → test | Logit RMSE | Probability RMSE | Selected-token RMS | Flip rate |',
            '|---|---|---:|---:|---:|---:|']
    # Show repeat zero rather than an arithmetic average of repeat RMSE values.
    c0=summary[(summary.test_batch==1)&(summary.repeat==0)].copy()
    c0['_stage']=c0.test_condition.map(lambda c:STAGES.index(c) if c in STAGES else 99)
    for _,r in c0.sort_values(['comparison_kind','_stage']).iterrows():
        lines.append(f"| {r.comparison_kind} | {r.baseline_condition} → {r.test_condition} | {r.logit_rmse:.6g} | {r.probability_rmse:.6g} | {r.selected_rms:.6g} | {r.top1_disagreement_rate:.4%} |")
    lines+=['','The table shows repeat 0; summary.csv retains every repeat. Incremental differences are measured directly, not obtained by subtracting RMSE values. Signed A, Q, P changes telescope to the cumulative change, but their RMS and flip rates do not add.',
            '', '## Runtime variation and independent repeats','',
            '| Condition | Repeat | Own-c0-token fixed-prefix average probability std |',
            '|---|---:|---:|']
    for row in state['runtime_std_summary']:
        lines.append(f"| {row['condition']} | {row['repeat']} | {row['own_c0_fixed_prefix_avg_std_top1_prob']:.6g} |")
    repeats=[row for row in state['repeat_evidence'] if row['repeat']>0]
    if repeats:
        identical=sum(row['identical_windows_to_repeat_zero'] for row in repeats)
        count=sum(row['windows'] for row in repeats)
        lines+=['',f'{identical}/{count} repeated window outputs are exactly equal to repeat 0 at every vocabulary entry and position. This equality check precedes lossless storage deduplication. Per-repeat paired metrics and standard deviations are preserved separately.']
    lines+=['','The std uses sample standard deviation over batch configurations (ddof=1), averaged over fixed-prefix positions and equally weighted windows. Each condition uses its own c0 token in this table. F16 and FB16 are separate within-dtype runtime controls; their difference is not interpreted as pure operation-order error.',
            '', '## Margins, cancellation, and reference returns','']
    if state['primary_margin_bins']:
        lines+=['| Effect | Baseline probability margin | Position observations | Equal-window flip rate |',
                '|---|---|---:|---:|']
        for row in state['primary_margin_bins']:
            rate='no observations' if row['flip_rate'] is None else f"{row['flip_rate']:.3%}"
            lines.append(f"| {row['comparison_kind']} {row['test_condition']} | [{row['lower']:g}, {row['upper']:g}{']' if row['upper']==1 else ')'} | {row['positions']} | {rate} |")
        lines+=['','These counts include non-c0 runtime conditions and repeated observations; they are not independent sample counts. Rates first average observations within each participating window, then weight windows equally. Each effect uses its own pair baseline margin.','']
    if cancellation:
        c=pd.DataFrame(cancellation)
        lines.append(f"Signed cancellation occurs at {c.signed_cancellation_fraction.mean():.2%} of positions averaged over runs. Maximum telescoping residual: {c.telescoping_max_residual.max():.3g}.")
        for stage,g in c.groupby('stage'):
            lines.append(f"Adding {stage}: {g.returns.sum()} returns to the same-runtime AQP16 token among {g.previous_disagreements.sum()} previous disagreements ({g.return_to_reference_rate.mean():.3%} of all positions, averaged over runs). These are reference returns, not corrections to ground truth.")
    lines+=['','margin_flip_rates.csv reports counts and actual flips in the four prespecified probability-margin bins. paired_metrics.csv includes exact ties and winners outside the baseline top two. summary.json records overlap between large quantization/runtime changes. Figures use linear axes and preserve zeros; no stacked contribution bars are used.',
            '', '## Limits and reproduction','',
            'This is a controlled fixed-text-prefix experiment, not a reproduction of the paper’s free-generation common-prefix stopping rule. fixed_prefix_avg_std_top1_prob is reported separately from the RMS ratio. External paper values are not denominators. Results do not establish task accuracy loss or generalize to other models, backends, or reasoning workloads.',
            'Optional free generation, full-sequence teacher forcing, tensor parallelism, and extra backends are not included. The exploratory smoke corpus is separate from the main corpus.',
            f"Missing runs: `{state['missing']}`.",
            '', '```sh',
            f"env CUDA_VISIBLE_DEVICES=0 /home/jaeyongjang/.conda/envs/spinquant/bin/python measure_aqp_runtime_fidelity.py run --output-dir {out.parent}",
            f"/home/jaeyongjang/.conda/envs/spinquant/bin/python summarize_aqp_runtime_fidelity.py --experiment {out.parent} --analysis-id analysis-next",
            '```','',
            'Run from my_works/spinquant; pipe stdout through tee and inspect exit status/failure records. Existing run directories are immutable. manifest.json records model/tokenizer revisions, file hashes, runtime flags, corpus selection, and code provenance. code.diff and code/ retain implementation snapshots. The source Arrow cache was read without modification.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment',required=True)
    p.add_argument('--analysis-id',default='analysis')
    p.add_argument('--chunk-size',type=int,default=16)
    analyze(p.parse_args())
