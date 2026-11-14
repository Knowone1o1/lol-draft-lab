"""Does the draft add value on top of team strength? (Track A, step 4)

Offset approach: the best strength model's walk-forward logit enters with its coefficient FIXED at 1, and only the
draft coefficients are fitted on top:

    logit P(blue wins) = strength_logit (offset, pre-game, walk-forward) + beta . draft_features

so the draft can only add what strength does not already explain. Draft coefficients are refitted MONTHLY on
games before the month (from TRAIN_START), ridge-penalised, on features standardised with training rows only.
Draft features come from data/draft_features.pkl (scripts/build_draft_features.py, leakage-free).

Feature sets
  champ       the app's 5 champion features (champion strength, player comfort, synergy, lane counter, team matchups)
  comp        tag-sheet timing difference + timing mismatch size + style matchup, and the data early score
  first_sel   2026 First Selection: which team had it (games 2+: loser of the previous game; game 1 unknown) and
              what that team ended up with. Whether it CHOSE side or pick order is not identifiable from the data.
  interact    adds every champion feature x |strength gap| (absolute offset logit): does draft matter more or less
              in close matchups? and timing, high-variance share and style matchup x SIGNED strength gap (offset
              logit, centred on its training mean; > 0 = blue favoured). Hypothesis: underdogs gain from early /
              high-variance comps, favourites from safe / scaling comps -> timing_diff x gap > 0, hv_diff x gap < 0.
  stage       "full" (default), "b1" (after blue's first pick) or "p1" (after pick phase 1) for champion features
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .base import WinModel

TRAIN_START = "2023-01-01"
CHAMP = ["champ_strength", "player_comfort", "synergy", "lane_counter", "team_matchups"]
COMP = ["timing_diff", "timing_gap", "style_adv", "hv_diff", "early_diff"]
SIGNED_INTER = ["timing_diff", "hv_diff", "style_adv"]   # x signed strength gap (A4 hypothesis test)
FIRST_SEL = ["fs_blue", "fs_got_side_and_pick", "fs_got_one", "fs_got_neither"]
_BASE_CACHE: dict = {}
FEATURES_OVERRIDE = None     # tests inject a synthetic table here


def _features_table():
    if FEATURES_OVERRIDE is not None:
        return FEATURES_OVERRIDE
    from ..config import ROOT
    return pd.read_pickle(ROOT / "data" / "draft_features.pkl")


def first_selection(games: pd.DataFrame) -> pd.DataFrame:
    """2026 games 2+ of a series: +1/-1 if blue/red had First Selection (loser of the previous game), else 0,
    and what the First Selection team ended up with (blue side and/or first pick), signed toward that team."""
    out = pd.DataFrame(0.0, index=games.index, columns=FIRST_SEL)
    if not {"series_id", "game", "blue_first_pick"} <= set(games.columns):
        return out
    g = games[["series_id", "game", "Blue_teamname", "Red_teamname", "blue_win", "date", "blue_first_pick"]]
    prev = g.sort_values(["series_id", "game"]).groupby("series_id").shift(1)
    loser = np.where(prev["blue_win"] == 1, prev["Red_teamname"], prev["Blue_teamname"])
    loser = pd.Series(loser, index=prev.index).where(prev["blue_win"].notna()).reindex(g.index)
    ok = (g["date"].dt.year >= 2026) & loser.notna()
    fs_blue = (loser == g["Blue_teamname"]) & ok
    fs_red = (loser == g["Red_teamname"]) & ok
    sgn = np.where(fs_blue, 1.0, np.where(fs_red, -1.0, 0.0))
    has_side = np.where(fs_blue, 1, 0)                         # FS team is on blue side
    has_pick = np.where(fs_blue, g["blue_first_pick"] == 1, g["blue_first_pick"] == 0).astype(int)
    n = has_side + has_pick
    out["fs_blue"] = sgn
    out["fs_got_side_and_pick"] = sgn * (n == 2)
    out["fs_got_one"] = sgn * (n == 1)
    out["fs_got_neither"] = sgn * (n == 0)
    return out


def add_interactions(X, names, off, rows, dates, predict_idx):
    """Champion features x |gap| and SIGNED_INTER x signed gap (centred on the mean offset of games before the
    first predicted day, i.e. training data only)."""
    first = dates.to_numpy()[predict_idx].min()
    train = rows[dates.to_numpy()[rows] < first]
    centre = float(np.nanmean(off[train])) if len(train) else 0.0
    gap = off - centre
    ci = [i for i, n in enumerate(names) if any(n.endswith(c) for c in CHAMP)]
    si = [names.index(c) for c in SIGNED_INTER if c in names]
    X = np.column_stack([X, X[:, ci] * np.abs(gap)[:, None], X[:, si] * gap[:, None]])
    return X, names + [names[i] + " x |gap|" for i in ci] + [names[i] + " x gap" for i in si]


class DraftOffset(WinModel):
    name = "draft"
    feature_sets: tuple = ("champ",)

    def __init__(self, base: str = "strength_g15_tier", lam: float = 1.0, stage: str = "full",
                 interact: bool = False, **params):
        super().__init__(base=base, lam=lam, stage=stage, interact=interact, **params)
        self.coef_ = {}

    @property
    def features(self):
        f = [f"OFFSET: {self.params['base']} walk-forward logit (coefficient fixed at 1)"]
        for fs in self.feature_sets:
            cols = self._cols(fs)
            f.append(f"{fs}: {', '.join(cols)}")
        if self.params["interact"]:
            f.append("champion features x |strength gap|; timing, high-variance share, style matchup x signed gap")
        f.append(f"draft coefficients refitted monthly from {TRAIN_START}, ridge lam={self.params['lam']} (standardised)")
        return f

    @property
    def caveats(self):
        c = ["2026 red-first-pick games have no partial-draft (b1/p1) features until the C0 pick-order fix."]
        if "first_sel" in self.feature_sets:
            c.append("First Selection: who had it is known only for 2026 games 2+ (loser of the previous game); "
                     "whether that team chose side or pick order is NOT identifiable from Oracle's Elixir data, so the "
                     "features describe what it ended up with.")
        return c

    def fitted(self):
        return self.coef_

    def _cols(self, fs):
        if fs == "champ":
            pre = "" if self.params["stage"] == "full" else self.params["stage"] + "_"
            return [pre + c for c in CHAMP]
        return {"comp": COMP, "first_sel": FIRST_SEL}[fs]

    # ------------------------------------------------------------------ base offset (shared, cached)
    def _base_logit(self, games, rows):
        from . import get_model
        from ..config import REPORT_DIR
        name = self.params["base"]
        f = Path(REPORT_DIR) / "runs" / f"{name}_tuned.json"
        prm = json.loads(f.read_text())["params"] if f.exists() else {}
        key = (name, json.dumps(prm, sort_keys=True), len(games), games["date"].iloc[-1], rows[0], rows[-1], len(rows))
        if key not in _BASE_CACHE:
            p = np.clip(get_model(name, **prm).walk_forward(games, rows), 1e-6, 1 - 1e-6)
            _BASE_CACHE[key] = np.log(p / (1 - p))
        return _BASE_CACHE[key]

    def _design(self, games):
        tab = _features_table()
        parts = []
        for fs in self.feature_sets:
            if fs == "first_sel":
                parts.append(first_selection(games))
            else:
                parts.append(tab.reindex(games["gameid"])[self._cols(fs)].set_axis(games.index))
        return pd.concat(parts, axis=1)

    def walk_forward(self, games, predict_idx):
        dates = games["date"]
        lo = np.searchsorted(dates.to_numpy(), np.datetime64(pd.Timestamp(TRAIN_START)))
        hi = int(predict_idx.max()) + 1
        rows = np.arange(lo, hi)
        off = np.full(len(games), np.nan)
        off[rows] = self._base_logit(games, rows)
        X = self._design(games).to_numpy(dtype=float)
        names = list(self._design(games.iloc[:1]).columns)
        if self.params["interact"]:
            X, names = add_interactions(X, names, off, rows, dates, predict_idx)
        y = games["blue_win"].to_numpy(dtype=float)
        month = dates.dt.to_period("M").to_numpy()
        out = np.empty(len(predict_idx))
        beta = np.zeros(X.shape[1])
        for mth in pd.unique(month[predict_idx]):
            start = pd.Period(mth).to_timestamp()
            tr = rows[(dates.to_numpy()[rows] < np.datetime64(start))]
            tr = tr[~np.isnan(X[tr]).any(axis=1)]
            if len(tr) >= 500:
                mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
                Z, o, yt, lam = (X[tr] - mu) / sd, off[tr], y[tr], self.params["lam"]

                def f(b):
                    z = o + Z @ b
                    return (np.sum(np.logaddexp(0, z) - yt * z) + 0.5 * lam * b @ b,
                            Z.T @ (1 / (1 + np.exp(-z)) - yt) + lam * b)
                b_std = minimize(f, beta, jac=True, method="L-BFGS-B").x
                beta, scale = b_std, (mu, sd)
                self._beta, self._scale = beta, scale
                self.coef_ = {n: round(float(v), 4) for n, v in zip(names, b_std)}   # per 1 SD of the feature
            sel = np.nonzero(month[predict_idx] == mth)[0]
            for j in sel:
                i = predict_idx[j]
                z = off[i]
                if len(tr) >= 500 and not np.isnan(X[i]).any():
                    z += ((X[i] - scale[0]) / scale[1]) @ beta
                out[j] = 1 / (1 + np.exp(-z))
        return out

    def level_logit(self, X_row: np.ndarray) -> float:
        """Draft-only logit at LEVEL strength (gap 0, side-neutral): sum of beta_j * x_j / sd_j over the main
        draft features (not centred, so two identical drafts give exactly 50%); interaction terms vanish because the
        gap is 0. For the app (A6)."""
        if not hasattr(self, "_scale"):
            raise RuntimeError("fit the model first (walk_forward)")
        mu, sd = self._scale
        k = len(X_row)
        return float((np.asarray(X_row, float) / sd[:k]) @ self._beta[:k])


def level_strength_prob(model: "DraftOffset", X_row) -> float:
    """P(blue wins) from the draft alone when both teams are equally strong (no side term)."""
    return 1 / (1 + np.exp(-model.level_logit(X_row)))


class DraftChamp(DraftOffset):
    name, feature_sets = "draft_champ", ("champ",)


class DraftComp(DraftOffset):
    name, feature_sets = "draft_comp", ("comp",)


class DraftChampComp(DraftOffset):
    name, feature_sets = "draft_champ_comp", ("champ", "comp")


class DraftInteract(DraftOffset):
    name, feature_sets = "draft_interact", ("champ", "comp")

    def __init__(self, **params):
        params.setdefault("interact", True)
        super().__init__(**params)


class DraftFirstSel(DraftOffset):
    name, feature_sets = "draft_first_sel", ("champ", "first_sel")


class DraftB1(DraftOffset):
    name, feature_sets = "draft_b1", ("champ",)

    def __init__(self, **params):
        params.setdefault("stage", "b1")
        super().__init__(**params)


class DraftP1(DraftOffset):
    name, feature_sets = "draft_p1", ("champ",)

    def __init__(self, **params):
        params.setdefault("stage", "p1")
        super().__init__(**params)


DRAFT_MODELS = (DraftChamp, DraftComp, DraftChampComp, DraftInteract, DraftFirstSel, DraftB1, DraftP1)
