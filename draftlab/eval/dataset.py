"""Games prepared for evaluation: date-sorted, with series ids, league tier and roster-change flags."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import ROLES
from ..league import INTERNATIONAL, MAJORS, PARENT

MAJOR_EVENTS = {"WLDs", "MSI", "FST", "EWC"}   # top-tier international events count as "major"
ERL_PARENT = "LEC"                             # EMEA regional leagues (+ EMEA Masters) = "ERL"


def tier_of(league: str) -> str:
    if league in MAJORS or league in MAJOR_EVENTS:
        return "major"
    if PARENT.get(league) == ERL_PARENT or league == "EM":
        return "erl"
    return "other"


def roster_changes(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """For each game and side: how many of the team's 5 players differ from its previous game."""
    last: dict = {}
    out = {s: np.zeros(len(g), dtype=int) for s in ("Blue", "Red")}
    teams = {s: g[f"{s}_teamname"].to_numpy() for s in ("Blue", "Red")}
    players = {s: g[[f"{s}_{r}_player" for r in ROLES]].to_numpy() for s in ("Blue", "Red")}
    for i in range(len(g)):
        for s in ("Blue", "Red"):
            t, cur = teams[s][i], set(players[s][i])
            prev = last.get(t)
            out[s][i] = 0 if prev is None else len(cur - prev)
            last[t] = cur
    return out["Blue"], out["Red"]


def load_eval_games(paths=None, elo_k: float = 24.0) -> pd.DataFrame:
    from ..data import load_games
    g, _, _ = load_games(paths, elo_k=elo_k)
    g = g.sort_values("date", kind="stable").reset_index(drop=True)
    g["tier"] = g["league"].map(tier_of)
    g["international"] = g["league"].isin(INTERNATIONAL)
    rb, rr = roster_changes(g)
    g["roster_changed_blue"], g["roster_changed_red"] = rb, rr
    g["roster_change_3plus"] = (rb >= 3) | (rr >= 3)
    if "series_id" not in g:
        g["series_id"] = np.arange(len(g))
    g["blue_first_pick"] = g["gameid"].map(first_pick_map(paths)).fillna(1.0)
    m = margin_table(paths)
    for c in MARGIN_COLS:
        g[c] = g["gameid"].map(m[c]) if len(m) else np.nan
    return g


MARGIN_COLS = ["m_gold15", "m_gold25", "m_length"]


def margin_table(paths=None) -> pd.DataFrame:
    """Post-game margins from blue's point of view, indexed by gameid. Used ONLY to update ratings after a game.

    m_gold15  blue gold lead at 15 min, in thousands (NaN where the league has no timeline data, e.g. much of LPL)
    m_gold25  blue gold lead at 25 min, in thousands; games that ended before 25 min use the final gold lead
    m_length  +/- (winner side) x quickness: (50 - minutes) / 20, clipped to [0.25, 2] -> a 30-min win = 1.0,
              a 45-min grind = 0.25. Fast stomps count more than long, even games.
    """
    from ..config import RAW_DIR
    paths = list(paths) if paths else sorted(RAW_DIR.glob("*.csv"))
    parts = []
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        need = ["gameid", "position", "side", "result", "golddiffat15", "golddiffat25", "gamelength", "totalgold"]
        if not set(need) <= set(cols):
            continue
        d = pd.read_csv(p, usecols=need, low_memory=False)
        t = d[d["position"] == "team"]
        b = t[t["side"] == "Blue"].set_index("gameid")
        r = t[t["side"] == "Red"].set_index("gameid")
        b = b[~b.index.duplicated()]
        r = r[~r.index.duplicated()].reindex(b.index)
        minutes = b["gamelength"] / 60
        g25 = b["golddiffat25"].where(b["golddiffat25"].notna() | (minutes >= 25), b["totalgold"] - r["totalgold"])
        quick = ((50 - minutes) / 20).clip(0.25, 2)
        parts.append(pd.DataFrame({"m_gold15": b["golddiffat15"] / 1000, "m_gold25": g25 / 1000,
                                   "m_length": np.where(b["result"] == 1, quick, -quick)}, index=b.index))
    return pd.concat(parts) if parts else pd.DataFrame(columns=MARGIN_COLS)


def first_pick_map(paths=None) -> dict:
    """gameid -> 1.0 if blue side had first pick, 0.0 if red did. Before 2026 blue always picked first; from
    2026 side and first pick can be chosen separately (about 30% of 2026 games have red first pick)."""
    from ..config import RAW_DIR
    paths = list(paths) if paths else sorted(RAW_DIR.glob("*.csv"))
    out = {}
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        if "firstPick" not in cols:
            continue
        d = pd.read_csv(p, usecols=["gameid", "position", "side", "firstPick"], low_memory=False)
        t = d[(d["position"] == "team") & (d["firstPick"] == 1)]
        out.update(dict(zip(t["gameid"], (t["side"] == "Blue").astype(float))))
    return out
