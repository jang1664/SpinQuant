import copy
import numpy as np
import pytest

from summarize_zero_shot_top1_fidelity import align_samples, row_metrics, choice_lengths, summarize_rows, paired_bootstrap


def sample(i):
    return dict(doc_id=i,doc_hash=f'd{i}',prompt_hash=f'p{i}',target_hash=f't{i}',
        arguments=[['question',' A'],['question',' B']],target=0,doc={},metrics=['acc'])


def test_alignment_reorders_and_rejects_mismatched_or_duplicate_questions():
    reference=[sample(0),sample(1)]
    assert align_samples(reference,list(reversed(reference))) == reference
    changed=copy.deepcopy(reference); changed[0]['arguments'][0][1]=' C'
    with pytest.raises(ValueError,match='mismatch'):
        align_samples(reference,changed)
    with pytest.raises(ValueError,match='duplicate'):
        align_samples(reference,[sample(0),sample(0)])
    changed=copy.deepcopy(reference); changed[0]['prompt_hash']='wrong'
    with pytest.raises(ValueError,match='hashes'):
        align_samples(reference,changed)


def test_same_confidence_different_winner_is_not_same_candidate_fidelity():
    row=row_metrics(np.log([[.9,.1],[.9,.1],[.1,.9]]))
    assert row['aqp16_absolute_error'] == 0
    assert row['aqp8_absolute_error'] == pytest.approx(.8)
    assert row['aqp8_own_top1_absolute_error'] == 0
    assert row['aqp8_agrees_with_fp'] == 0


def test_absolute_errors_do_not_cancel_and_paired_bootstrap_is_question_level():
    rows=[dict(task='t',**row_metrics(np.log(p))) for p in
          [[[.6,.4],[.5,.5],[.4,.6]],[[.6,.4],[.7,.3],[.8,.2]]]]
    result=summarize_rows(rows)
    assert result['aqp16_signed_error'] == pytest.approx(0,abs=1e-14)
    assert result['aqp16_absolute_error'] == pytest.approx(.1)
    assert result['aqp8_absolute_error'] == pytest.approx(.2)
    ci=paired_bootstrap([rows],100,0)
    assert ci['mae_delta_8_minus_16_ci95'] == pytest.approx([.1,.1])


def test_character_normalization_uses_choice_text_without_target_delimiter():
    row=sample(0); row['doc']['choices']={'text':[' A','BB']}
    assert choice_lengths(row).tolist() == [2,2]


def test_reject_nonfinite_scores():
    with pytest.raises(ValueError,match='finite'):
        row_metrics([[0,1],[0,float('nan')],[0,1]])
