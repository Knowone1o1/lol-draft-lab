"""Part 3: test the early-game rules on past games, walk-forward, never touching the frozen holdout.

For every month M in the test window:
  - early-game stats (champion / matchup / player gold @15) are rebuilt from games played BEFORE M;
  - the lane gold-at-15 forecast (the same Ridge model as the app) is refit on the 12 months before M;
  - each game's facts are computed with the SAME code the plan card uses (draftlab/rules/draft_facts.py),
    and each rule's condition is evaluated -> did the rule fire for this team?
Team strength is the walk-forward strength_g15_tier_cal logit (A3/A4), used as a fixed offset for win outcomes
(coefficient 1, as in A4) and as a control for gold outcomes. CIs resample whole series.

Champion tags (comp timing, primary style) are the current sheet, not as-of-date labels: a small hindsight risk.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import HOLDOUT_START, REPORT_DIR, ROLES, TRAIN_LEAGUES
from .condition import evaluate, substitute
from .draft_facts import PRIO_PCT, StatsLookups, draft_facts

LANES3 = ("top", "mid", "bot")
SIDES = ("Blue", "Red")


# ----------------------------------------------------------------------------- facts table
def build_fact_table(start="2024-01-01", rules=None, history_months=12, log=print) -> pd.DataFrame:
    """One row per team per game (2 per game) in [start, HOLDOUT_START): rule firings, facts used by the tests,
    outcomes and the walk-forward strength logit."""
    from sklearn.linear_model import Ridge

    from ..comp import champ_tag, comp_profile
    from ..data import game_actions
    from ..early import EarlyStats, attach_early, game_lane_matrix, lane_long, load_early_raw, team_long
    from ..eval.dataset import load_eval_games
    from ..training import month_starts

    hold = pd.Timestamp(HOLDOUT_START)
    games = load_eval_games()
    team_raw, pl_raw = load_early_raw()
    g = attach_early(games, team_raw, pl_raw)
    g = g[g["date"] < hold].reset_index(drop=True)          # the holdout never enters, not even as history
    lanes, teams = lane_long(g), team_long(g)
    t0 = pd.Timestamp(start)
    months = [m for m in month_starts(g["date"]) if m >= t0 - pd.DateOffset(months=history_months)]
    fit_mask = g["league"].isin(TRAIN_LEAGUES).to_numpy()

    # lane matrices for every game from (start - history) on, each with stats from before its own month
    LX, ES = {}, {}
    for m in months:
        nxt = m + pd.offsets.MonthBegin(1)
        idx = np.nonzero(((g["date"] >= m) & (g["date"] < nxt)).to_numpy())[0]
        if not len(idx):
            continue
        es = EarlyStats(lanes, teams, m)
        ES[m] = es
        for i in idx:
            row = g.iloc[i]
            picks = {s: {r: row[f"{s}_{r}_champ"] for r in ROLES} for s in SIDES}
            players = {s: {r: row[f"{s}_{r}_player"] for r in ROLES} for s in SIDES}
            LX[i] = game_lane_matrix(es, picks, players, (row["Blue_teamname"], row["Red_teamname"]),
                                     row["blue_elo"] - row["red_elo"])
        log(f"  stats {m:%Y-%m}: {len(idx)} games")

    test_idx = np.nonzero((g["date"] >= t0).to_numpy())[0]
    strength = strength_logit(games, g, test_idx)

    rows, CUTS = [], {}
    for m in [m for m in months if m >= t0]:
        nxt = m + pd.offsets.MonthBegin(1)
        lo = m - pd.DateOffset(months=history_months)
        train = [i for i in LX if lo <= g.at[i, "date"] < m and fit_mask[i]]
        models = {}
        for k, r in enumerate(ROLES):
            ys = [(LX[i][k], g.at[i, f"blue_{r}_gd15"]) for i in train if np.isfinite(g.at[i, f"blue_{r}_gd15"])]
            if len(ys) < 200:
                continue
            models[r] = Ridge(alpha=1.0).fit(np.array([x for x, _ in ys]), np.array([y for _, y in ys]))
        cuts = None
        if len(models) == 5:
            # per-lane cuts: this month's model applied to the training window's games (all before M)
            P = {r: models[r].predict(np.array([LX[i][k] for i in train])) for k, r in enumerate(ROLES)}
            cuts = {r: float(np.percentile(np.abs(P[r]), PRIO_PCT)) for r in ("top", "mid", "bot")}
            cuts["midjng"] = float(np.percentile(np.abs(P["mid"] + P["jng"]), PRIO_PCT))
            CUTS[m] = cuts
        look = StatsLookups(ES.get(m))
        idx = [i for i in test_idx if m <= g.at[i, "date"] < nxt]
        for i in idx:
            row = g.iloc[i].to_dict()
            by_role = {s: {r: row[f"{s}_{r}_champ"] for r in ROLES} for s in SIDES}
            rosters = {s: {r: row[f"{s}_{r}_player"] for r in ROLES} for s in SIDES}
            lane_gold = ({r: float(models[r].predict(LX[i][k:k + 1])[0]) for k, r in enumerate(ROLES)}
                         if len(models) == 5 else None)
            pick_index = None
            if row.get("has_order"):
                picks = [c for (_, kind, c) in game_actions(row) if kind == "pick" and c]
                pick_index = {c: n for n, c in enumerate(picks)}
            timing = {s: comp_profile([c for c in by_role[s].values() if c])["timing"] for s in SIDES}
            for side in SIDES:
                sg = 1 if side == "Blue" else -1
                f, _ = draft_facts(side, by_role, champ_tag, lane_gold, pick_index, rosters, look, timing, cuts)
                rec = {"gameid": row["gameid"], "date": row["date"], "league": row["league"], "tier": row["tier"],
                       "series_id": row["series_id"], "side": side,
                       "win": float(row["blue_win"] if side == "Blue" else 1 - row["blue_win"]),
                       "strength": sg * strength.get(i, np.nan),
                       "gd15": sg * row.get("blue_gd15", np.nan),
                       "plates_diff": sg * row.get("blue_plates_diff", np.nan),
                       "first_dragon": _side_bin(row.get("blue_first_dragon"), side),
                       "grubs": _side_bin(row.get("blue_grubs"), side),
                       "fc_team": None if lane_gold is None else sg * sum(lane_gold.values())}
                for r in ROLES:
                    rec[f"gd15_{r}"] = sg * row.get(f"blue_{r}_gd15", np.nan)
                    rec[f"fc_{r}"] = None if lane_gold is None else sg * lane_gold[r]
                for k in ("our.timing", "their.timing", "our.jng.facilitator", "their.jng.carry", "our.midjng",
                          "our.strong_lanes", "our.top.prio", "our.mid.prio", "our.bot.prio"):
                    rec[k] = f.get(k)
                for lane in LANES3:
                    for k in (f"our.{lane}.blind", f"our.{lane}.matchup_dependent", f"their.{lane}.star",
                              f"their.{lane}.blind"):
                        rec[k] = f.get(k)
                for r in (rules or {}).get("rules", []):
                    if r["kind"] != "draft":
                        continue
                    for lane in r["for_each"].get("lane") or [None]:
                        tree = substitute(r["when"], {"lane": lane}) if lane else r["when"]
                        res, _ = evaluate(tree, f)
                        rec[f"fires.{r['id']}" + (f".{lane}" if lane else "")] = res
                rows.append(rec)
        log(f"  facts {m:%Y-%m}: {len(idx)} games; cuts {cuts}")
    out = pd.DataFrame(rows)
    out.attrs["prio_cuts"] = {f"{m:%Y-%m}": c for m, c in CUTS.items()}
    return out


def _side_bin(v, side):
    if v is None or not np.isfinite(v):
        return np.nan
    return float(v) if side == "Blue" else 1.0 - float(v)


def strength_logit(games_all, g, test_idx) -> dict:
    """Walk-forward strength_g15_tier_cal logit for blue, keyed by row of g (same order as games_all up to holdout)."""
    from ..winmodels import get_model
    f = REPORT_DIR / "runs" / "strength_g15_tier_cal_tuned.json"
    prm = json.loads(f.read_text())["params"]
    games_all = games_all[games_all["date"] < pd.Timestamp(HOLDOUT_START)]
    games_all = games_all.sort_values("date", kind="stable").reset_index(drop=True)
    pos = {gid: k for k, gid in enumerate(games_all["gameid"])}
    rows = np.array([pos[gid] for gid in g.loc[test_idx, "gameid"]])
    assert (games_all.loc[rows, "date"] < pd.Timestamp(HOLDOUT_START)).all()
    p = np.clip(get_model("strength_g15_tier_cal", **prm).walk_forward(games_all, rows), 1e-6, 1 - 1e-6)
    return dict(zip(test_idx, np.log(p / (1 - p))))


# ----------------------------------------------------------------------------- fitting
def logit_fit(X, y, offset, iters=25):
    """Logistic regression with a fixed offset (coefficient 1), Newton steps, tiny ridge for stability."""
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(offset + X @ b)))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def ols_fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def series_bootstrap(fit, data: pd.DataFrame, B=1000, seed=0):
    """Refit `fit(df) -> effect` on resampled series; returns (effect, lo, hi)."""
    est = fit(data)
    ser = data["series_id"].to_numpy()
    uniq, inv = np.unique(ser, return_inverse=True)
    order = np.argsort(inv, kind="stable")
    members = np.split(order, np.cumsum(np.bincount(inv))[:-1])
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[j] for j in pick])
        try:
            v = fit(data.iloc[idx])
        except np.linalg.LinAlgError:
            continue
        if np.isfinite(v):
            vals.append(v)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(est), float(lo), float(hi)


def label(effect_lo, effect_hi, direction, n, min_n=150):
    """Supported: 95% CI excludes 0 in the rule's direction. Contradicted: excludes 0 the other way. Else Unclear."""
    if n < min_n:
        return "Unclear"
    if effect_lo > 0 or effect_hi < 0:
        return "Supported" if (effect_lo > 0) == (direction > 0) else "Contradicted"
    return "Unclear"
