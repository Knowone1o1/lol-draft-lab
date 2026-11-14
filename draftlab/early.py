"""Early-game module: forecast the state of the game at 15 minutes from the draft and the players.

Targets (blue side's point of view):
  - gold difference at 15 in each lane (each player vs the opposite laner, from Oracle's Elixir player rows)
  - team gold difference at 15
  - first dragon, void grubs majority, first blood, first tower (yes/no)

Statistics are recency-weighted aggregates of games played before a reference date (no look-ahead):
  - champion lane-phase strength: average gold lead at 15 of a champion in a role (shrunk to 0)
  - lane matchup lift: how much better a champion does than expected against a specific opponent
  - player early rating: a player's average gold lead at 15 in their role
  - team early rating and objective habits (first dragon / grubs / first blood / first tower rates)
  - jungler champion objective rates
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RAW_DIR, ROLES, SIDES

OBJECTIVES = ["first_dragon", "grubs", "first_blood", "first_tower"]
TEAM_COLS = {"golddiffat10": "gd10", "golddiffat15": "gd15", "golddiffat20": "gd20", "xpdiffat15": "xpd15",
             "firstdragon": "first_dragon", "firstblood": "first_blood", "firsttower": "first_tower",
             "void_grubs": "grubs_n", "opp_void_grubs": "opp_grubs_n", "turretplates": "plates",
             "opp_turretplates": "opp_plates"}
PLAYER_COLS = {"golddiffat10": "gd10", "golddiffat15": "gd15", "csdiffat15": "csd15", "xpdiffat15": "xpd15"}


def load_early_raw(paths=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team-level and player-level early stats keyed by gameid (+ side, + position)."""
    paths = list(paths) if paths else sorted(RAW_DIR.glob("*.csv"))
    want = {"gameid", "side", "position"} | set(TEAM_COLS) | set(PLAYER_COLS)
    frames = []
    for p in paths:
        cols = pd.read_csv(p, nrows=0).columns
        frames.append(pd.read_csv(p, usecols=[c for c in cols if c in want], low_memory=False))
    df = pd.concat(frames, ignore_index=True).drop_duplicates(["gameid", "side", "position"])
    team = df[df["position"] == "team"].rename(columns=TEAM_COLS)
    team = team[["gameid", "side"] + [c for c in TEAM_COLS.values() if c in team.columns]]
    pl = df[df["position"].isin(ROLES)].rename(columns=PLAYER_COLS)
    pl = pl[["gameid", "side", "position"] + [c for c in PLAYER_COLS.values() if c in pl.columns]]
    return team, pl


def attach_early(games: pd.DataFrame, team: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    """Add blue-perspective early targets to the games table."""
    g = games.copy()
    bt = team[team["side"] == "Blue"].set_index("gameid")
    for c in ("gd10", "gd15", "gd20", "xpd15"):
        if c in bt:
            g[f"blue_{c}"] = g["gameid"].map(bt[c])
    for c in ("first_dragon", "first_blood", "first_tower"):
        if c in bt:
            g[f"blue_{c}"] = g["gameid"].map(bt[c])
    if "grubs_n" in bt and "opp_grubs_n" in bt:
        diff = bt["grubs_n"] - bt["opp_grubs_n"]
        g["blue_grubs"] = g["gameid"].map(np.sign(diff).replace(0, np.nan).map({1.0: 1.0, -1.0: 0.0}))
    if "plates" in bt and "opp_plates" in bt:
        g["blue_plates_diff"] = g["gameid"].map(bt["plates"] - bt["opp_plates"])
    bp = pl[pl["side"] == "Blue"].set_index(["gameid", "position"])
    for r in ROLES:
        for c in ("gd15", "gd10", "csd15"):
            if c in bp:
                s = bp[c].xs(r, level="position")
                g[f"blue_{r}_{c}"] = g["gameid"].map(s)
    return g


def lane_long(g: pd.DataFrame) -> pd.DataFrame:
    """Two rows per lane per game (one per side): champ, opponent, player, team and that side's gd15."""
    parts = []
    for r_i, r in enumerate(ROLES):
        col = f"blue_{r}_gd15"
        if col not in g:
            continue
        for side, opp, sign in (("Blue", "Red", 1.0), ("Red", "Blue", -1.0)):
            parts.append(pd.DataFrame({
                "gi": np.arange(len(g)), "date": g["date"].values, "role": r,
                "champ": g[f"{side}_{r}_champ"].values, "opp": g[f"{opp}_{r}_champ"].values,
                "player": g[f"{side}_{r}_player"].values, "team": g[f"{side}_teamname"].values,
                "gd15": sign * g[col].values,
            }))
    out = pd.concat(parts, ignore_index=True)
    return out[out["gd15"].notna()]


def team_long(g: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for side, sign in (("Blue", 1.0), ("Red", -1.0)):
        d = {"gi": np.arange(len(g)), "date": g["date"].values, "team": g[f"{side}_teamname"].values,
             "jng": g[f"{side}_jng_champ"].values,
             "gd15": sign * g["blue_gd15"].values if "blue_gd15" in g else np.nan}
        for o in OBJECTIVES:
            col = f"blue_{o}"
            if col in g:
                v = g[col].values.astype(float)
                d[o] = v if side == "Blue" else 1.0 - v
        parts.append(pd.DataFrame(d))
    return pd.concat(parts, ignore_index=True)


class EarlyStats:
    """Recency-weighted early-game aggregates from games before ref_date."""

    def __init__(self, lanes: pd.DataFrame, teams: pd.DataFrame, ref_date, half_life=120.0,
                 matchup_half_life=540.0, k_champ=15.0, k_matchup=10.0, k_player=10.0, k_team=6.0, k_obj=12.0):
        ref = pd.Timestamp(ref_date)
        self.ref_date = ref
        L = lanes[lanes["date"] < ref]
        T = teams[teams["date"] < ref]
        dl = (ref - L["date"]).dt.total_seconds().to_numpy() / 86400
        dt = (ref - T["date"]).dt.total_seconds().to_numpy() / 86400
        L = L.assign(w=np.power(0.5, dl / half_life), wl=np.power(0.5, dl / matchup_half_life))
        T = T.assign(w=np.power(0.5, dt / half_life))

        def shrunk_mean(df, keys, val, wcol, k, prior=0.0):
            x = df.assign(_wv=df[wcol] * df[val])
            s = x.groupby(keys)[["_wv", wcol]].sum()
            return ((s["_wv"] + k * prior) / (s[wcol] + k)).to_dict(), s[wcol].to_dict()

        self.champ, self.champ_n = shrunk_mean(L, ["champ", "role"], "gd15", "w", k_champ)
        self.player, self.player_n = shrunk_mean(L, ["player", "role"], "gd15", "w", k_player)
        # Matchup lift: residual after both champions' (long-memory) lane strength.
        ch_long, _ = shrunk_mean(L, ["champ", "role"], "gd15", "wl", k_champ)
        exp = [ch_long.get((c, r), 0.0) - ch_long.get((o, r), 0.0) for c, o, r in zip(L["champ"], L["opp"], L["role"])]
        L = L.assign(resid=L["gd15"].to_numpy() - np.array(exp))
        self.matchup, self.matchup_n = shrunk_mean(L, ["champ", "opp", "role"], "resid", "wl", k_matchup)
        Tg = T[T["gd15"].notna()]
        self.team_gd, self.team_n = shrunk_mean(Tg, ["team"], "gd15", "w", k_team)
        self.team_obj, self.jng_obj = {}, {}
        for o in OBJECTIVES:
            if o not in T:
                continue
            To = T[T[o].notna()]
            self.team_obj[o], _ = shrunk_mean(To, ["team"], o, "w", k_obj, prior=0.5)
            self.jng_obj[o], _ = shrunk_mean(To, ["jng"], o, "w", k_obj * 2, prior=0.5)

    # lookups
    def champ_lane(self, c, r):
        return self.champ.get((c, r), 0.0) if c else 0.0

    def matchup_lift(self, c, o, r):
        return self.matchup.get((c, o, r), 0.0) if c and o else 0.0

    def player_lane(self, p, r):
        return self.player.get((p, r), 0.0) if p else 0.0

    def team_early(self, t):
        return self.team_gd.get(t, 0.0)

    def team_objective(self, t, o):
        return self.team_obj.get(o, {}).get(t, 0.5)

    def jungle_objective(self, c, o):
        return self.jng_obj.get(o, {}).get(c, 0.5) if c else 0.5


LANE_FEATURES = ["champ_strength_diff", "matchup_lift", "player_diff", "team_early_diff", "elo_diff"]
TEAM_FEATURES = ["champ_strength_sum", "matchup_sum", "player_sum", "team_early_diff", "elo_diff"]
OBJ_FEATURES = ["team_rate_diff", "jungler_rate_diff", "side_lanes_strength", "team_early_diff", "elo_diff"]
# Which lanes matter most for each objective (dragon = bot side, grubs = top side).
OBJ_LANES = {"first_dragon": ["bot", "sup", "jng", "mid"], "grubs": ["top", "jng", "mid"],
             "first_blood": ROLES, "first_tower": ROLES}


def lane_features(es: EarlyStats, role, blue_c, red_c, blue_p, red_p, blue_t, red_t, elo_diff):
    m = es.matchup_lift(blue_c, red_c, role) - es.matchup_lift(red_c, blue_c, role)
    return np.array([
        (es.champ_lane(blue_c, role) - es.champ_lane(red_c, role)) / 1000,
        m / 2000,
        (es.player_lane(blue_p, role) - es.player_lane(red_p, role)) / 1000,
        (es.team_early(blue_t) - es.team_early(red_t)) / 1000,
        elo_diff / 400,
    ])


def game_lane_matrix(es: EarlyStats, picks: dict, players: dict, teams: tuple, elo_diff: float):
    """picks/players: {side: {role: name}} (missing roles allowed). Returns (5, n_feat)."""
    return np.vstack([lane_features(es, r, picks["Blue"].get(r), picks["Red"].get(r), players["Blue"].get(r),
                                    players["Red"].get(r), teams[0], teams[1], elo_diff) for r in ROLES])


def team_features(lane_X):
    return np.array([lane_X[:, 0].sum(), lane_X[:, 1].sum(), lane_X[:, 2].sum(), lane_X[0, 3], lane_X[0, 4]])


def objective_features(es: EarlyStats, o, picks, teams, lane_X):
    lanes = [ROLES.index(r) for r in OBJ_LANES[o]]
    return np.array([
        es.team_objective(teams[0], o) - es.team_objective(teams[1], o),
        es.jungle_objective(picks["Blue"].get("jng"), o) - es.jungle_objective(picks["Red"].get("jng"), o),
        lane_X[lanes, 0].sum() + lane_X[lanes, 2].sum(),
        lane_X[0, 3],
        lane_X[0, 4],
    ])
