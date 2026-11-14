"""Models: conditional-logit pick/ban imitation models and a logistic win-probability model."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression


class ConditionalLogit:
    """Multinomial choice model: P(choose c | state) = softmax_c(x_c · w).

    Each training example is one draft action: a candidate matrix X (one row per available
    champion) and the index of the champion actually chosen. Weights are interpretable:
    a positive weight means teams prefer champions with more of that feature.
    """

    def __init__(self, feature_names, l2: float = 1.0):
        self.feature_names = list(feature_names)
        self.l2 = l2
        self.w = None
        self.mu = None
        self.sd = None

    def fit(self, groups):
        X = np.vstack([x for x, _ in groups]).astype(np.float64)
        sizes = np.array([len(x) for x, _ in groups])
        starts = np.r_[0, np.cumsum(sizes)[:-1]]
        chosen = starts + np.array([y for _, y in groups])
        self.mu = X.mean(0)
        self.sd = X.std(0) + 1e-9
        Xs = (X - self.mu) / self.sd
        l2 = self.l2

        def f(w):
            s = Xs @ w
            m = np.maximum.reduceat(s, starts)
            e = np.exp(s - np.repeat(m, sizes))
            Z = np.add.reduceat(e, starts)
            loss = (m + np.log(Z) - s[chosen]).sum() + 0.5 * l2 * w @ w
            p = e / np.repeat(Z, sizes)
            grad = Xs.T @ p - Xs[chosen].sum(0) + l2 * w
            return loss, grad

        res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B")
        self.w = res.x
        return self

    def scores(self, X):
        return ((np.asarray(X, dtype=np.float64) - self.mu) / self.sd) @ self.w

    def proba(self, X, temperature: float = 1.0):
        s = self.scores(X) / max(temperature, 1e-6)
        s -= s.max()
        e = np.exp(s)
        return e / e.sum()

    def coefficients(self):
        return sorted(zip(self.feature_names, self.w), key=lambda t: -abs(t[1]))


class WinModel:
    def __init__(self, feature_names, C: float = 1.0):
        self.feature_names = list(feature_names)
        self.model = LogisticRegression(C=C, max_iter=1000)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def proba(self, X):
        return self.model.predict_proba(np.atleast_2d(X))[:, 1]

    def coefficients(self):
        return list(zip(self.feature_names, self.model.coef_[0])) + [("blue_side (intercept)", self.model.intercept_[0])]


def heuristic_pick_scores(X, names):
    """Ban-Teemo-style fixed-weight baseline (an approximation, for comparison only).

    Weighted percentile ranks: meta 25%, strength 20%, counter 25%, role fit 15%, proficiency 15%,
    multiplied by a synergy factor.
    """
    idx = {n: i for i, n in enumerate(names)}

    def pct(col):
        v = X[:, idx[col]]
        return v.argsort().argsort() / max(len(v) - 1, 1)

    base = (0.25 * pct("meta_presence") + 0.20 * pct("champ_strength") + 0.25 * pct("lane_counter")
            + 0.15 * pct("role_fit") + 0.15 * pct("player_comfort"))
    syn = np.clip(1 + 5 * X[:, idx["synergy"]], 0.75, 1.25)
    return base * syn


def rank_metrics(score_lists):
    """score_lists: list of (scores array, chosen index). Returns top-1/top-5/top-10 accuracy and mean rank."""
    ranks = []
    for s, y in score_lists:
        ranks.append(int((s > s[y]).sum()) + 1)
    r = np.array(ranks)
    if len(r) == 0:
        return {"n": 0}
    return {"n": len(r), "top1": float((r <= 1).mean()), "top5": float((r <= 5).mean()),
            "top10": float((r <= 10).mean()), "mean_rank": float(r.mean())}
