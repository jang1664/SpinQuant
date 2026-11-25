import numpy as np
import pytest

from utils.paired_fidelity_metrics import (
    paired_metrics, probabilities, align_positions, macro_mean, cluster_ratio,
)


def test_identity_and_shift():
    z=np.array([[1.,2.,3.],[2.,2.,-1.]])
    for row in paired_metrics(z,z,[2,0]):
        assert row['logit_mse']==0 and row['probability_mse']==0 and not row['flip']
    for row in paired_metrics(z,z+10,[2,0]):
        assert row['logit_mae']==10 and row['centered_logit_mse']==0
        assert row['probability_mse']<1e-28


def test_near_tie_and_third_winner():
    rows=paired_metrics([[1,.999,0],[1,.9,.8]],[[.998,1,0],[1,.9,1.1]],[0,0])
    assert all(r['flip'] for r in rows)
    assert rows[1]['winner_outside_top2'] and rows[1]['delta_margin']==0
    assert paired_metrics([[1,1,0]],[[1,1,0]],[0])[0]['baseline_top1']==0


def test_alignment_and_rejection():
    a=[dict(sample_id='a',token_position=0),dict(sample_id='b',token_position=1)]
    b=a[::-1]+[dict(valid=False)]
    assert len(align_positions(a,b))==2
    with pytest.raises(ValueError): align_positions(a,b[1:])
    with pytest.raises(ValueError): align_positions(a,a+a)


def test_macro_std_and_bootstrap():
    assert macro_mean([0,0,3],['a','a','b'])==1.5
    assert np.mean([0,0,3])==1
    assert np.std([0,2],ddof=1)==pytest.approx(2**.5)
    result=cluster_ratio([4,16,36],[1,4,9],['a','a','b'])
    assert result['ratio']==2 and result['ratio_ci']==[2,2]
    assert cluster_ratio([1,2],[0,0],['a','b'])['ratio'] is None


def test_chunks_and_cancellation():
    rng=np.random.default_rng(5); z=rng.normal(size=(7,11)); t=z+rng.normal(size=z.shape)*.1
    assert paired_metrics(z,t,z.argmax(1),1)==paired_metrics(z,t,z.argmax(1),7)
    p=[probabilities(v)[:,0] for v in (z,t,z+.5,t-.2)]
    steps=[p[i+1]-p[i] for i in range(3)]
    np.testing.assert_allclose(sum(steps),p[-1]-p[0],atol=1e-15)
    assert sum(np.sqrt(np.mean(d*d)) for d in steps)>np.sqrt(np.mean((p[-1]-p[0])**2))


@pytest.mark.parametrize('groupsize',[-1,8])
def test_quantizer_scales_do_not_reduce_batch_axis(groupsize):
    import torch
    from utils.quant_utils import ActQuantizer
    torch.manual_seed(19)
    x=torch.randn(2,3,16)
    x[1]*=100
    quantizer=ActQuantizer(); quantizer.configure(8,groupsize=groupsize,sym=False)
    quantizer.find_params(x); batched=quantizer(x)
    individual=[]
    for i in range(2):
        quantizer.find_params(x[i:i+1]); individual.append(quantizer(x[i:i+1]))
    torch.testing.assert_close(batched,torch.cat(individual),rtol=0,atol=0)


@pytest.mark.parametrize('bits',[(16,16,16),(8,16,16),(8,8,16),(8,8,8)])
def test_actual_quantizers_and_cached_decode(bits):
    import torch
    from transformers import LlamaConfig
    from eval_utils.modeling_llama import LlamaForCausalLM
    from eval_utils.main import ptq_model
    from tests.test_aqp_quantization import ptq_args
    from measure_matrix_output_accuracy import validate_quantized_configuration
    from measure_aqp_runtime_fidelity import attach_calls, cache_length
    config=LlamaConfig(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=1,
                       num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=32)
    config._attn_implementation='eager'
    args=ptq_args(16); args.a_bits,args.q_bits,args.p_bits=bits
    args.k_bits=4; args.k_groupsize=8; args.q_groupsize=8
    args.k_asym=True; args.k_clip_ratio=1.; args.k_pre_rope=False
    model=ptq_model(args,LlamaForCausalLM(config)).eval()
    validate_quantized_configuration(model,a_bits=bits[0],q_bits=bits[1],p_bits=bits[2])
    calls,handles=attach_calls(model)
    with torch.inference_mode():
        pre=model(torch.tensor([[1,2,3]]),use_cache=True)
        dec=model(torch.tensor([[4]]),past_key_values=pre.past_key_values,use_cache=True)
    assert cache_length(dec.past_key_values)==4
    for r in calls.values():
        assert r['calls']==(2 if r['bits']<16 else 0)
    for h in handles: h.remove()


def test_analysis_complete_and_missing_shard(tmp_path):
    import json
    from types import SimpleNamespace
    from measure_aqp_runtime_fidelity import STAGES, sha256
    from summarize_aqp_runtime_fidelity import analyze
    corpus=[dict(sample_id=f's{i}',source_document_id=f'd{i}') for i in range(2)]
    (tmp_path/'corpus.jsonl').write_text('\n'.join(json.dumps(s) for s in corpus))
    conditions={c:dict(a_bits=v[0],q_bits=v[1],p_bits=v[2],compute_dtype='fp16') for c,v in STAGES.items()}
    m=dict(conditions=conditions,repeats=1,batch_sizes=[1,2],windows=2,prompt_length=3,positions=2,
           seed=1,scope='smoke',paths={'model':'synthetic'},corpus_sha256=sha256(tmp_path/'corpus.jsonl'))
    (tmp_path/'manifest.json').write_text(json.dumps(m))
    for ci,c in enumerate(conditions):
        for b in [1,2]:
            path=tmp_path/f'{c}/r0/b{b}'; path.mkdir(parents=True)
            arts=[]
            for i,s in enumerate(corpus):
                z=np.array([[1.,.99,0],[0,1.,.99]])
                z[:,0]+=.01*ci+.001*b*(i+1)
                p=path/f"{s['sample_id']}.npz"; np.savez(p,logits=z)
                arts.append(dict(sample_id=s['sample_id'],sha256=sha256(p)))
            (path/'complete.json').write_text(json.dumps(dict(artifacts=arts)))
            (path/'condition_validation.json').write_text('{}')
    analyze(SimpleNamespace(experiment=str(tmp_path),analysis_id='analysis',chunk_size=1))
    summary=json.loads((tmp_path/'analysis/summary.json').read_text())
    assert summary['status']=='complete' and len(summary['ratios'])==12
    assert (tmp_path/'analysis/figures/paired_scatter.png').exists()
    shard=tmp_path/'AQ8/r0/b2/s0.npz'; shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='checksum'):
        analyze(SimpleNamespace(experiment=str(tmp_path),analysis_id='bad',chunk_size=1))


def test_supervisor_checks_program_exit_not_tee(tmp_path,monkeypatch):
    import sys
    import time
    from types import SimpleNamespace
    from scripts import run_aqp_runtime_fidelity_plan as supervisor
    args=SimpleNamespace(smoke=str(tmp_path/'smoke'),main=str(tmp_path/'main'),
        progress=str(tmp_path/'progress.md'),status=str(tmp_path/'status.json'),logs=str(tmp_path/'logs'))
    sleep=time.sleep
    monkeypatch.setattr(supervisor.time,'sleep',lambda seconds:sleep(.01))
    with pytest.raises(RuntimeError,match='program exit=7, tee exit=0'):
        supervisor.run_logged(args,'intentional_failure',[sys.executable,'-c',
            'import sys; print("test stdout"); print("test stderr",file=sys.stderr); sys.exit(7)'])
    assert (tmp_path/'logs/intentional_failure.stdout.log').read_text()=='test stdout\n'
    assert (tmp_path/'logs/intentional_failure.stderr.log').read_text()=='test stderr\n'
