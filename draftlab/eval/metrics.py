"""Probability metrics. Headline: log-loss and Brier. Secondary: accuracy, AUC. Calibration: slope, ECE."""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

EPS = 1e-6


def logloss_terms(y, p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    y = np.asarray(y, float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_terms(y, p):
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def accuracy_terms(y, p):
    return ((np.asarray(p) > 0.5) == (np.asarray(y) == 1)).astype(float)


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan")


def calibration_slope(y, p):
    """Slope (and intercept) of a logistic fit of the outcome on logit(p). 1 = well calibrated; <1 = the
    model is overconfident (predictions too extreme); >1 = underconfident."""
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    z = np.log(p / (1 - p))[:, None]
    y = np.asarray(y)
    if not 0 < y.mean() < 1:
        return float("nan"), float("nan")
    m = LogisticRegression(C=1e6, max_iter=1000).fit(z, y)
    return float(m.coef_[0][0]), float(m.intercept_[0])


def reliability(y, p, bins: int = 10):
    """Equal-width bins: (bin lower, bin upper, n, mean predicted, observed rate)."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append((edges[b], edges[b + 1], int(m.sum()), float(p[m].mean()) if m.any() else np.nan,
                     float(y[m].mean()) if m.any() else np.nan))
    return rows


def ece(y, p, bins: int = 10):
    """Expected calibration error: sample-weighted mean |observed - predicted| over equal-width bins."""
    n = len(y)
    return float(sum(r[2] / n * abs(r[4] - r[3]) for r in reliability(y, p, bins) if r[2]))


def summary(y, p) -> dict:
    slope, intercept = calibration_slope(y, p)
    return {"log_loss": float(logloss_terms(y, p).mean()), "brier": float(brier_terms(y, p).mean()),
            "accuracy": float(accuracy_terms(y, p).mean()), "auc": auc(y, p), "cal_slope": slope,
            "cal_intercept": intercept, "ece": ece(y, p), "n": int(len(y))}
