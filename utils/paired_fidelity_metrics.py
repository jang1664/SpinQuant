"""Float64, position-paired metrics. All reductions retain sample identity."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp

MARGIN_EDGES = [0, 1e-4, 1e-3, 1e-2, 1.00000000001]


def probabilities(logits):
    z = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(z).all():
        raise ValueError("nonfinite logits")
    return np.exp(z - logsumexp(z, axis=-1, keepdims=True))


def paired_metrics(baseline, test, reference_tokens, chunk_size=16):
    """Return one row per unpadded position; ties select the lowest token ID."""
    x, y = np.asarray(baseline), np.asarray(test)
    if x.shape != y.shape or x.ndim != 2 or len(reference_tokens) != len(x):
        raise ValueError("paired arrays must have identical [position, vocabulary] shapes")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    rows = []
    for start in range(0, len(x), chunk_size):
        b, t = x[start:start+chunk_size].astype('float64'), y[start:start+chunk_size].astype('float64')
        pb, pt = probabilities(b), probabilities(t)
        dz, dp = t-b, pt-pb
        a, winner = b.argmax(-1), t.argmax(-1)
        order = np.arange(len(b))
        runners = b.copy()
        runners[order, a] = -np.inf
        second = runners.argmax(-1)
        mz = b[order, a] - b[order, second]
        mp = pb[order, a] - pb[order, second]
        r = np.asarray(reference_tokens[start:start+len(b)])
        arrays = {
            'logit_mae': np.abs(dz).mean(-1), 'logit_mse': (dz*dz).mean(-1),
            'centered_logit_mse': ((dz-dz.mean(-1, keepdims=True))**2).mean(-1),
            'probability_mae': np.abs(dp).mean(-1), 'probability_mse': (dp*dp).mean(-1),
            'probability_max_abs': np.abs(dp).max(-1),
            'reference_token': r, 'selected_probability_delta': dp[order, r],
            'baseline_top1': a, 'baseline_top2': second, 'test_top1': winner,
            'margin_logit': mz, 'margin_probability': mp,
            'delta_margin': t[order,a]-t[order,second]-mz,
            'flip': winner != a, 'baseline_tie': mz == 0,
            'test_tie': (t == t.max(-1, keepdims=True)).sum(-1) > 1,
            'winner_outside_top2': (winner != a) & (winner != second),
            'test_winner_gap_over_baseline': t[order,winner]-t[order,a],
        }
        rows.extend({k:v[i].item() for k,v in arrays.items()} for i in range(len(b)))
    return rows


def align_positions(left, right):
    """Join arbitrary regrouped/padded records without silently dropping positions."""
    def indexed(records):
        result = {}
        for row in records:
            if not row.get('valid', True):
                continue
            key = (row['sample_id'], row['token_position'])
            if key in result:
                raise ValueError(f'duplicate position {key}')
            result[key] = row
        return result
    a, b = indexed(left), indexed(right)
    if a.keys() != b.keys():
        raise ValueError('position sets differ')
    return [(a[k], b[k]) for k in sorted(a)]


def macro_mean(values, sample_ids):
    values, sample_ids = np.asarray(values), np.asarray(sample_ids)
    return float(np.mean([values[sample_ids == s].mean() for s in np.unique(sample_ids)]))


def cluster_ratio(numerator_mse, denominator_mse, documents, seed=20261015, draws=2000):
    """Resample source documents, retaining their constituent equal-weight windows."""
    n, d, docs = np.asarray(numerator_mse), np.asarray(denominator_mse), np.asarray(documents)
    if not (n.shape == d.shape == docs.shape) or len(n) == 0:
        raise ValueError('unaligned bootstrap inputs')
    clusters = np.unique(docs)
    indices = [np.flatnonzero(docs == c) for c in clusters]
    ns, ds = np.array([n[i].sum() for i in indices]), np.array([d[i].sum() for i in indices])
    counts = np.array([len(i) for i in indices])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), (draws, len(clusters)))
    nb = np.sqrt(ns[picks].sum(1)/counts[picks].sum(1))
    db = np.sqrt(ds[picks].sum(1)/counts[picks].sum(1))
    nr, dr = float(np.sqrt(n.mean())), float(np.sqrt(d.mean()))
    unstable = dr <= 1e-12 or np.quantile(db, .025) <= 1e-12
    return dict(quantization_rms=nr, runtime_rms=dr,
                quantization_ci=np.quantile(nb,[.025,.975]).tolist(),
                runtime_ci=np.quantile(db,[.025,.975]).tolist(),
                ratio=None if unstable else nr/dr,
                ratio_ci=None if unstable else np.quantile(nb/db,[.025,.975]).tolist(),
                ratio_status='unstable_denominator' if unstable else 'estimated',
                documents=len(clusters), windows=len(n), bootstrap_draws=draws)
