"""One interface for every pre-game win model, so the evaluation harness and the app can switch by config.

A model predicts P(blue side wins) for a set of games WALK-FORWARD: the prediction for a game on day d may
only use (a) model parameters fitted on games from days before d, and (b) features built from games that
ended before that game (e.g. a pre-game rating that already includes game 1 of the same series). Nothing
from the game itself or later games may be used. tests/test_eval.py checks this with a sentinel test:
flipping every later result must not change any earlier prediction.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class WinModel(ABC):
    name: str = "base"

    def __init__(self, **params):
        self.params = params

    @property
    def features(self) -> list[str]:
        return []

    @abstractmethod
    def walk_forward(self, games: pd.DataFrame, predict_idx: np.ndarray) -> np.ndarray:
        """P(blue wins) for rows `predict_idx` of `games` (date-sorted), using only earlier information."""

    @property
    def caveats(self) -> list[str]:
        """Plain-language limits of this model that every report must show."""
        return []

    def fitted(self) -> dict:
        """Interpretable coefficients from the most recent fit (empty if not applicable)."""
        return {}

    def describe(self) -> dict:
        return {"name": self.name, "params": self.params, "features": self.features,
                "caveats": self.caveats, "fitted": self.fitted()}


def day_of(dates: pd.Series) -> np.ndarray:
    return dates.dt.floor("D").to_numpy()


def daily_refit(games: pd.DataFrame, predict_idx: np.ndarray, X: np.ndarray, fit, predict,
                min_train: int = 200) -> np.ndarray:
    """Refit once per prediction day on every game from earlier days, then predict that day's games.

    X: feature matrix for ALL rows (each row's features must already be pre-game / leakage-free).
    fit(X_train, y_train) -> fitted object; predict(fitted, X_rows) -> probabilities.
    """
    y = games["blue_win"].to_numpy(dtype=float)
    days = day_of(games["date"])
    out = np.full(len(predict_idx), np.nan)
    order = np.argsort(days[predict_idx], kind="stable")
    cur_day, fitted = None, None
    for j in order:
        i = predict_idx[j]
        if days[i] != cur_day:
            cur_day = days[i]
            train = days < cur_day
            fitted = fit(X[train], y[train]) if train.sum() >= min_train else None
        out[j] = predict(fitted, X[i:i + 1])[0] if fitted is not None else 0.5
    return out
