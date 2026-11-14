"""Baselines every evaluation report includes.

always_blue  P(blue wins) = blue side's win rate over all earlier days (the side advantage alone).
elo          logistic regression on the pre-game Elo difference (team ratings from win/loss, K = 24),
             refitted daily on earlier days. This is the current app's strength input.
elo_league   the same with league-strength offsets added to each rating; the offsets are refitted each
             month on games before that month (draftlab/league.py), so they never see the future.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .base import WinModel, daily_refit, day_of


class AlwaysBlue(WinModel):
    name = "always_blue"

    @property
    def features(self):
        return ["side"]

    def walk_forward(self, games, predict_idx):
        y = games["blue_win"].to_numpy(dtype=float)
        days = day_of(games["date"])
        uniq, inv = np.unique(days, return_inverse=True)
        wins = np.bincount(inv, weights=y, minlength=len(uniq))
        n = np.bincount(inv, minlength=len(uniq))
        cw, cn = np.cumsum(wins) - wins, np.cumsum(n) - n          # strictly earlier days
        p_day = (cw + 1) / (cn + 2)
        return p_day[inv[predict_idx]]


def _logit_fit(X, y):
    return LogisticRegression(C=1e6, max_iter=1000).fit(X, y)


def _logit_predict(m, X):
    return m.predict_proba(X)[:, 1]


class EloLogit(WinModel):
    name = "elo"

    def __init__(self, k: float = 24.0, **params):
        super().__init__(k=k, **params)

    @property
    def features(self):
        return ["elo_diff/400 (pre-game, win/loss Elo)", "intercept = side"]

    def _ratings(self, games):
        from ..data import add_elo
        g2, _ = add_elo(games[["Blue_teamname", "Red_teamname", "blue_win", "date", "league"]].copy(), k=self.params["k"])
        return g2["blue_elo"].to_numpy(), g2["red_elo"].to_numpy()

    def walk_forward(self, games, predict_idx):
        be, re_ = self._ratings(games)
        X = ((be - re_) / 400.0)[:, None]
        return daily_refit(games, predict_idx, X, _logit_fit, _logit_predict)


class EloLeagueLogit(EloLogit):
    name = "elo_league"

    @property
    def features(self):
        return ["(elo + league offset) diff/400, offsets refitted monthly on past games", "intercept = side"]

    def walk_forward(self, games, predict_idx):
        from ..league import monthly_adjusted
        be, re_ = self._ratings(games)
        g = games.assign(blue_elo=be, red_elo=re_)
        ab, ar = monthly_adjusted(g, be, re_)
        X = ((ab - ar) / 400.0)[:, None]
        return daily_refit(games, predict_idx, X, _logit_fit, _logit_predict)
