"""Greedy common-prefix probability statistics, independent of model execution."""
from __future__ import annotations

import numpy as np


def observe(row, step, token_ids, probabilities, reference_probabilities, eos_ids):
    """Consume one aligned position; exclude the first differing token position."""
    if row['stop_reason'] is not None:
        return
    ids = np.asarray(token_ids)
    probs = np.asarray(probabilities, dtype=np.float64)
    if ids.ndim != 1 or len(ids) < 2 or probs.shape != ids.shape:
        raise ValueError('one token and probability per runtime configuration required')
    if not np.isfinite(probs).all() or np.any((probs < 0) | (probs > 1)):
        raise ValueError('invalid probabilities')
    if not np.all(ids == ids[0]):
        row['stop_reason'] = 'diverged'
        row['divergence_index'] = step
        row['boundary'] = dict(step=step, token_ids=ids.tolist(),
            own_top1_probabilities=probs.tolist(),
            b1_token_probabilities=list(reference_probabilities))
        return
    row['positions'].append(dict(step=step, token_id=int(ids[0]),
        probabilities=probs.tolist(), std_ddof1=float(np.std(probs, ddof=1))))
    if int(ids[0]) in eos_ids:
        row['stop_reason'] = 'common_eos'


def finish(row):
    if row['stop_reason'] is None:
        row['stop_reason'] = 'cap'
    row['common_length'] = len(row['positions'])
    row['avg_std_top1_prob'] = (float(np.mean([p['std_ddof1'] for p in row['positions']]))
                              if row['positions'] else None)
    return row


def cluster_summary(rows, documents, draws=2000, seed=20261015):
    """Equal-window mean within repeats, then equal-repeat mean; bootstrap articles.

    Empty prefixes are undefined, never silently replaced with zero. Resampling
    includes all source articles, including articles with no eligible windows.
    """
    if not rows:
        raise ValueError('no rows')
    docs = sorted(set(documents))
    repeats = sorted({int(r['repeat']) for r in rows})
    sums = np.zeros((len(repeats), len(docs)))
    counts = np.zeros_like(sums)
    for row in rows:
        value = row['avg_std_top1_prob']
        if value is not None:
            i, j = repeats.index(int(row['repeat'])), docs.index(row['source_document_id'])
            sums[i, j] += value
            counts[i, j] += 1

    def estimate(weights):
        numerator, denominator = sums @ weights, counts @ weights
        if np.any(denominator == 0):
            return None
        return float(np.mean(numerator / denominator))

    value = estimate(np.ones(len(docs)))
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        weights = np.bincount(rng.integers(len(docs), size=len(docs)), minlength=len(docs))
        result = estimate(weights)
        if result is not None:
            estimates.append(result)
    return dict(mean=value, ci95=np.quantile(estimates, [.025, .975]).tolist() if estimates else None,
        bootstrap_draws=draws, bootstrap_valid_draws=len(estimates),
        eligible_window_repeats=int(counts.sum()), total_window_repeats=len(rows),
        source_documents=len(docs), ddof=1,
        ddof0_mean=None if value is None else value * np.sqrt(3 / 4))
