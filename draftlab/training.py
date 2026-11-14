"""Turn games into training examples with leakage-free, monthly-refreshed snapshots."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DRAFT_ORDER
from .features import (StateContext, actual_roles, ban_matrix, comp_features, pick_matrix,
                       state_from_game)
from .stats import Snapshot


def month_starts(dates: pd.Series) -> list:
    first = dates.min().to_period("M").to_timestamp()
    last = dates.max().to_period("M").to_timestamp()
    return list(pd.date_range(first, last, freq="MS"))


def draft_examples(row: dict, snap: Snapshot, n_neg: int | None, rng: np.random.Generator):
    """Yield (kind, X, chosen_index, meta) for each recorded pick/ban of a game.

    kind is "pick", "ban1" (first ban phase) or "ban2" (second ban phase).
    If n_neg is set, candidates are subsampled to the chosen champion + n_neg random others
    (valid for conditional logit thanks to its uniform-conditioning property).
    meta["pr"] holds each candidate's pick rate (for the role-aware baselines, judge M9).
    """
    universe = snap.champions
    full = state_from_game(row, 20)
    for i, (side, kind, _) in enumerate(full.order):      # follows the side that picked first (2026)
        chosen = full.actions[i]
        if not chosen:
            continue
        state = state_from_game(row, i)
        unavailable = state.unavailable()
        if chosen in unavailable:
            continue
        cands = [c for c in universe if c not in unavailable and c != chosen]
        if n_neg is not None and len(cands) > n_neg:
            cands = list(rng.choice(cands, size=n_neg, replace=False))
        cands = [chosen] + cands
        ctx = StateContext(state, snap)
        if kind == "pick":
            X = pick_matrix(state, side, cands, snap, ctx)
            k = "pick"
        else:
            X = ban_matrix(state, side, cands, snap, ctx)
            k = "ban1" if i < 6 else "ban2"
        yield k, X, 0, {"gameid": row.get("gameid"), "step": i, "side": side, "team": state.team(side),
                        "cands": cands, "pr": np.array([snap.pick_rate(c) for c in cands], np.float32)}


# Draft stages for the per-stage win-model check (judge M13): number of draft actions done.
STAGES = {"bans only": 6, "1v1": 8, "2v2": 10, "3v3": 12, "4v4": 18, "5v5": 20}


def build_examples(games: pd.DataFrame, train_mask, months: list, n_neg_train: int = 60,
                   full_months: set | None = None, holdout_start=None, seed: int = 0,
                   params: dict | None = None, index=None, log=print):
    """Examples for every month in `months`, each using a snapshot of all games before that month.

    Within a month, games are grouped by patch and each group gets a meta snapshot targeted at its
    own patch (the patch is known before the draft, so this is fair).

    Draft examples of months in `full_months` and of holdout games (date >= holdout_start) keep every
    candidate (for top-k scoring); other months are subsampled to n_neg_train negatives.
    Returns {"draft": {kind: [(X, y, meta)]}, "win": {...arrays...}, "stage": {gameid: {stage: x}}};
    every meta / win row carries its month and a holdout flag so callers can split freely.
    """
    from .fast_stats import StatsIndex
    from .params import load_params, snapshot_kwargs
    kw = dict(snapshot_kwargs(params or load_params()))
    kw["adjusted_strength"] = False  # build.py swaps strength variants in afterwards (same snapshot month)
    idx = index or StatsIndex(games)
    rng = np.random.default_rng(seed)
    full_months = full_months or set()
    hs = pd.Timestamp(holdout_start) if holdout_start is not None else None
    out = {"draft": {"pick": [], "ban1": [], "ban2": []},
           "win": {"X": [], "y": [], "league": [], "gameid": [], "month": [], "holdout": []},
           "stage": {}}
    for m in months:
        nxt = m + pd.offsets.MonthBegin(1)
        month_games = games[(games["date"] >= m) & (games["date"] < nxt) & train_mask]
        if month_games.empty:
            continue
        patches = month_games["patch_idx"].fillna(-1)
        n_rows, n_snap, n_hold = 0, 0, 0
        for p, grp in month_games.groupby(patches):
            snap = idx.snapshot(m, target_patch=None if p < 0 else p, with_rosters=False, **kw)
            if snap.n_games == 0:
                continue
            n_rows += len(grp)
            n_snap = snap.n_games
            for row in grp.to_dict("records"):
                hold = hs is not None and row["date"] >= hs
                n_hold += hold
                full = hold or m in full_months
                _game_examples(row, snap, m, hold, full, n_neg_train, rng, out)
        tag = "full" if m in full_months else "sub"
        log(f"  {m:%Y-%m} [{tag}{', holdout ' + str(n_hold) if n_hold else ''}] games={n_rows} "
            f"patches={patches.nunique()} snapshot={n_snap} games")
    return out


def _game_examples(row, snap, month, hold, full, n_neg_train, rng, out):
    st = state_from_game(row, 20)
    x = comp_features(st, snap, row["blue_elo"], row["red_elo"], role_hint=actual_roles(row))
    w = out["win"]
    w["X"].append(x)
    w["y"].append(row["blue_win"])
    w["league"].append(row.get("league"))
    w["gameid"].append(row.get("gameid"))
    w["month"].append(month)
    w["holdout"].append(hold)
    if full and row["has_order"]:
        # Mid-draft states with inferred roles, as the app sees them (M13).
        out["stage"][row.get("gameid")] = {
            name: comp_features(state_from_game(row, n), snap, row["blue_elo"], row["red_elo"])
            for name, n in STAGES.items()}
    if not row["has_order"]:
        return
    for kind, X, y, meta in draft_examples(row, snap, None if full else n_neg_train, rng):
        meta["month"], meta["holdout"], meta["full"] = month, hold, full
        out["draft"][kind].append((X.astype(np.float16) if full else X, y, meta))


def subsample(examples, n_neg: int, seed: int = 1):
    """Reduce full-candidate examples to chosen + n_neg negatives (for final refit)."""
    rng = np.random.default_rng(seed)
    out = []
    for X, y, meta in examples:
        if len(X) - 1 > n_neg:
            idx = np.r_[y, rng.choice(np.delete(np.arange(len(X)), y), size=n_neg, replace=False)]
            out.append((X[idx], 0, meta))
        else:
            out.append((X, y, meta))
    return out
