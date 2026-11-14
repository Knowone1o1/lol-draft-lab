"""Paired bootstrap intervals and ranking metrics for the backtest (judge finding M4)."""
from __future__ import annotations

import numpy as np


def _cluster_sums(v, clusters):
    _, inv = np.unique(clusters, return_inverse=True)
    return np.bincount(inv, weights=v), np.bincount(inv)


def boot_mean(v, clusters=None, B: int = 2000, seed: int = 0):
    """Mean of v with a 95% bootstrap interval; resamples whole clusters (e.g. games) if given."""
    v = np.asarray(v, float)
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    clusters = np.arange(len(v)) if clusters is None else np.asarray(clusters)
    s, n = _cluster_sums(v, clusters)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), size=(B, len(s)))
    stats = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def paired_diff(a, b, clusters=None, B: int = 2000, seed: int = 0):
    """Mean of (a - b) over the same units, with a paired 95% bootstrap interval."""
    return boot_mean(np.asarray(a, float) - np.asarray(b, float), clusters, B, seed)


def ranks(scores_list):
    """Rank of the chosen candidate (index 0... or y) in each score array; ties count half."""
    out = []
    for s, y in scores_list:
        s = np.asarray(s, float)
        out.append(1 + (s > s[y]).sum() + 0.5 * ((s == s[y]).sum() - 1))
    return np.array(out)


def logloss_terms(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def fmt_ci(m, lo, hi, pct=False, digits=4, sign=False):
    if pct:
        f = (lambda x: f"{x * 100:+.1f}") if sign else (lambda x: f"{x * 100:.1f}")
        return f"{f(m)}% [{f(lo)}, {f(hi)}]"
    f = (lambda x: f"{x:+.{digits}f}") if sign else (lambda x: f"{x:.{digits}f}")
    return f"{f(m)} [{f(lo)}, {f(hi)}]"
