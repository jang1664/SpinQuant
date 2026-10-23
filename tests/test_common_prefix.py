import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.common_prefix_metrics import observe, finish, cluster_summary
from measure_aqp_common_prefix import generate_block


def new_row():
    return dict(positions=[], stop_reason=None, divergence_index=None, boundary=None)


def test_common_token_sample_std_and_excluded_boundary():
    row = new_row()
    observe(row, 0, [2]*4, [.1, .2, .3, .4], [.1, .2, .3, .4], [])
    observe(row, 1, [2, 3, 2, 2], [.2]*4, [.2, .1, .2, .2], [])
    observe(row, 2, [2]*4, [.8]*4, [.8]*4, [])
    result = finish(row)
    assert result['common_length'] == 1
    assert result['divergence_index'] == 1
    assert result['avg_std_top1_prob'] == pytest.approx(np.std([.1,.2,.3,.4], ddof=1))


def test_zero_length_eos_and_cap():
    row = new_row()
    observe(row, 0, [1,2,1,1], [.4]*4, [.4]*4, [])
    assert finish(row)['avg_std_top1_prob'] is None
    row = new_row()
    observe(row, 0, [3]*4, [.4]*4, [.4]*4, [3])
    assert finish(row)['stop_reason'] == 'common_eos'
    assert row['common_length'] == 1
    assert row['avg_std_top1_prob'] == 0
    row = new_row()
    observe(row, 0, [3]*4, [.4]*4, [.4]*4, [])
    assert finish(row)['stop_reason'] == 'cap'


def test_article_bootstrap_retains_window_weight_and_undefined():
    rows = [dict(source_document_id=d, repeat=r, avg_std_top1_prob=v)
            for r in range(3) for d,v in [('a',1.), ('a',3.), ('b',8.), ('c',None)]]
    result = cluster_summary(rows, ['a','b','c'], draws=200)
    assert result['mean'] == 4
    assert result['eligible_window_repeats'] == 9
    assert result['total_window_repeats'] == 12
    assert result['ci95'][0] <= result['mean'] <= result['ci95'][1]
    assert result['bootstrap_valid_draws'] < 200


class FakeModel:
    """Logits depend on each row's history; batch 8 diverges at step 2."""
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, attention_mask, position_ids, past_key_values, **kwargs):
        b, width = input_ids.shape
        previous = past_key_values[0][0] if past_key_values else None
        history = input_ids if previous is None else torch.cat([previous[:,0,:,0].long(), input_ids], dim=1)
        length = history.shape[1]
        assert attention_mask.shape == (b,length)
        assert position_ids[0].tolist() == list(range(length-width,length))
        if previous is not None:
            # Real autoregressive inputs must be the preceding own prediction.
            old_history = history[:,:-1]
            expected = (old_history.sum(-1) % 6 + (b == 8 and old_history.shape[1] == 4)) % 6
            assert torch.equal(input_ids[:,0], expected)
        winners = (history.sum(-1) % 6 + (b == 8 and length == 4)) % 6
        logits = torch.zeros((b, width, 6))
        logits[torch.arange(b), -1, winners] = 2 + b / 10
        cache_tensor = history[:,None,:,None]
        self.calls.append((b,length))
        return SimpleNamespace(logits=logits, past_key_values=((cache_tensor,cache_tensor),))


def test_interleaved_caches_match_independent_greedy_and_keep_batch_shapes():
    samples = [dict(sample_id=str(i), source_document_id='a', prompt_ids=[i%6,1]) for i in range(8)]
    m = dict(positions=6, prompt_length=2, batch_sizes=[1,2,4,8], eos_ids=[])
    model = FakeModel()
    counter = [0]
    rows, trace = generate_block(model, samples, m, counter, device='cpu')
    assert counter[0] == 45  # three steps times 8+4+2+1 groups
    assert len(trace) == 3
    assert all(r['common_length'] == 2 and r['stop_reason'] == 'diverged' for r in rows)
    independent = []
    for b in m['batch_sizes']:
        full = []
        for start in range(0,8,b):
            ids = torch.tensor([s['prompt_ids'] for s in samples[start:start+b]])
            cache = None
            group = []
            for step in range(3):
                length = 2+step
                result = FakeModel()(ids, torch.ones((b,length)),
                    torch.arange(length-ids.shape[1],length)[None].expand(b,-1), cache)
                cache = result.past_key_values
                probs = torch.softmax(result.logits[:,-1].double(), dim=-1)
                ids = probs.argmax(-1)[:,None]
                group.append(probs.max(-1).values.numpy())
            full.extend(np.stack(group, axis=1).tolist())
        independent.append(full)
    independent = np.array(independent)
    for i,row in enumerate(rows):
        assert row['avg_std_top1_prob'] == pytest.approx(independent[:,i,:2].std(axis=0,ddof=1).mean())


def test_inactive_rows_keep_running_until_whole_block_stops():
    samples = [dict(sample_id=str(i), source_document_id='a', prompt_ids=[i%6,1]) for i in range(8)]
    m = dict(positions=6, prompt_length=2, batch_sizes=[1,2,4,8], eos_ids=[1])
    counter = [0]
    rows, _ = generate_block(FakeModel(), samples, m, counter, device='cpu')
    assert rows[0]['stop_reason'] == 'common_eos'
    assert rows[0]['common_length'] == 1
    assert any(r['stop_reason'] == 'diverged' for r in rows)
    assert counter[0] == 45


def test_report_recomputes_metric_and_rejects_changed_artifacts(tmp_path):
    from measure_aqp_runtime_fidelity import sha256, write_json
    from summarize_aqp_common_prefix import summarize
    root, dest = tmp_path / 'raw', tmp_path / 'report'
    run = root / 'AQP8/r0'
    run.mkdir(parents=True)
    corpus = dict(sample_id='x', source_document_id='doc')
    (root / 'corpus.jsonl').write_text(json.dumps(corpus)+'\n')
    write_json(root / 'manifest.json', dict(corpus_sha256=sha256(root/'corpus.jsonl'),
        conditions={'AQP8': {}}, repeats=1, bootstrap=dict(draws=20,seed=0),
        source_experiment=str(tmp_path/'old'), windows=1, prompt_length=2, positions=3,
        batch_sizes=[1,2,4,8], paper_url='https://example.org'))
    row = new_row()
    observe(row, 0, [1]*4, [.1,.2,.3,.4], [.1,.2,.3,.4], [])
    finish(row)
    row.update(corpus, condition='AQP8', repeat=0)
    artifact = run / 'x.json'
    write_json(artifact, row)
    write_json(run / 'complete.json', dict(condition='AQP8',repeat=0,
        artifacts=[dict(sample_id='x',sha256=sha256(artifact))]))
    result = summarize(root,dest)
    assert result['status'] == 'complete'
    assert result['conditions']['AQP8']['mean'] == row['avg_std_top1_prob']
    assert (dest/'REPORT.md').exists()
    artifact.write_text('{}')
    with pytest.raises(ValueError, match='hash mismatch'):
        summarize(root,dest)
