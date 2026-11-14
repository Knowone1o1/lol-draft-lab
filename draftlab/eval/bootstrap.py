"""Bootstrap 95% intervals that resample whole SERIES (games in a series are not independent).

mean_ci    for per-game terms (log-loss, Brier, accuracy): fast, by summing within series.
stat_ci    for statistics that need the whole sample (AUC, calibration slope, ECE): recomputed per resample.
paired     model A minus model B on the same games, resampling series once for both (paired).
"""
from __future__ import annotations

import numpy as np


def _series_index(series):
    uniq, inv = np.unique(np.asarray(series), return_inverse=True)
    return len(uniq), inv


def mean_ci(v, series, B: int = 2000, seed: int = 0):
    v = np.asarray(v, float)
    k, inv = _series_index(series)
    s, n = np.bincount(inv, weights=v, minlength=k), np.bincount(inv, minlength=k)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, k, size=(B, k))
    stats = s[draws].sum(1) / n[draws].sum(1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(v.mean()), float(lo), float(hi)


def paired(a, b, series, B: int = 2000, seed: int = 0):
    """Mean of (a - b) per game with a paired, series-level 95% interval."""
    return mean_ci(np.asarray(a, float) - np.asarray(b, float), series, B, seed)


def stat_ci(fn, y, p, series, B: int = 300, seed: int = 0):
    y, p = np.asarray(y), np.asarray(p)
    k, inv = _series_index(series)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv, minlength=k))[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, k, size=k)
        idx = np.concatenate([members[j] for j in pick])
        v = fn(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(fn(y, p)), float(lo), float(hi)
