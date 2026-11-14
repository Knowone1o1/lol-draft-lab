"""Recency-weighted statistics snapshot built from games played before a reference date.

Everything the models use (champion strength, role distribution, player comfort,
team habits, synergy, lane counters) comes from a Snapshot, so building one from
only *past* games gives leakage-free features for backtesting.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import (FEARLESS_GAME_WEIGHTS, HALF_LIFE_DAYS, K_CHAMP, K_LANE, K_PAIR,
                     MATCHUP_HALF_LIFE_DAYS, PATCH_BOOST, ROLES, SIDES)
from .data import is_missing

INTERNATIONAL = {"WLDs", "MSI", "FST", "EWC", "ASI", "KeSPA Cup", "AmCh"}


class Snapshot:
    # Shrinkage strengths (pseudo-games); class defaults so they can be tuned per instance.
    k_champ = K_CHAMP
    k_pair = K_PAIR
    k_lane = K_LANE
    # Team-adjusted champion strength {champ: effect on win probability} (fast_stats.adjusted_strength);
    # None = raw recency-weighted win rate.
    str_adj = None

    """Two memories:

    - *meta* tables (presence, champion strength, player comfort, team habits) use a short
      half-life, so old patches/seasons barely count;
    - *matchup* tables (lane counters, synergies) use a long half-life across all seasons. They
      are measured as lift beyond each champion's strength over the same long window, so a
      champion simply being strong in an old meta doesn't make it look like a counter.
    """

    def __init__(self, games: pd.DataFrame, ref_date, half_life: float = HALF_LIFE_DAYS,
                 matchup_half_life: float = MATCHUP_HALF_LIFE_DAYS, target_patch=None,
                 patch_boost: float = PATCH_BOOST, matchup_from: "Snapshot | None" = None):
        """target_patch: patch index (see data.patch_index) the meta should be most relevant to.
        matchup_from: reuse the matchup memory of another snapshot with the same ref_date (fast)."""
        self.ref_date = pd.Timestamp(ref_date)
        self.half_life = half_life
        self.matchup_half_life = matchup_half_life
        self.target_patch = target_patch
        self.patch_boost = patch_boost
        self.champ_games_long = defaultdict(float)
        self.champ_wins_long = defaultdict(float)
        self.champ_games = defaultdict(float)
        self.champ_wins = defaultdict(float)
        self.str_games = defaultdict(float)          # strength: recency-only weights
        self.str_wins = defaultdict(float)
        self.str_total = 0.0
        self.champ_bans = defaultdict(float)
        self.champ_role = defaultdict(lambda: np.zeros(5))
        self.player_champ = defaultdict(float)       # (pkey, champ) -> weighted games
        self.player_champ_wins = defaultdict(float)
        self.player_games = defaultdict(float)
        self.player_role = defaultdict(lambda: np.zeros(5))
        self.player_name = {}
        self.team_pick = defaultdict(float)          # (team, champ)
        self.team_ban = defaultdict(float)
        self.team_games = defaultdict(float)
        self.team_roster = {}                        # team -> {role: pkey} from latest game
        self.team_last = {}
        self.team_league = {}
        self.team_league_w = {}
        # Pick patterns (meta memory): how often two champions appear together / against each other.
        self.copick = defaultdict(float)             # (a, b) sorted, same team
        self.vspick = defaultdict(float)             # (a, b) sorted, opposite teams
        self.pair = defaultdict(float)               # (a, b) sorted, same team
        self.pair_wins = defaultdict(float)
        self.vs_any = defaultdict(float)             # (a, b): a's team vs b anywhere on enemy team
        self.vs_any_wins = defaultdict(float)
        self.lane = defaultdict(float)               # (a, b, role_idx): a vs b in that lane
        self.lane_wins = defaultdict(float)
        self.total = 0.0
        self.n_games = 0

        if matchup_from is not None:
            for k in ("champ_games_long", "champ_wins_long", "pair", "pair_wins", "lane", "lane_wins",
                      "vs_any", "vs_any_wins"):
                setattr(self, k, getattr(matchup_from, k))

        if len(games):
            past = games[games["date"] < self.ref_date]
            days = ((self.ref_date - past["date"]).dt.total_seconds() / 86400.0).clip(lower=0).to_numpy()
            gw = fearless_game_weights(past)
            if matchup_from is None:
                keep = days < 10 * matchup_half_life
                w_long = np.power(0.5, days[keep] / matchup_half_life) * gw[keep]
                for row, wl in zip(past[keep].to_dict("records"), w_long):
                    self._add_matchups(row, float(wl))
            # Meta: recency x patch relevance. Games older than 6 half-lives (<2%) are skipped.
            keep = days < 6 * half_life
            meta = past[keep]
            w_rec = np.power(0.5, days[keep] / half_life) * gw[keep]
            w_meta = w_rec
            if target_patch is not None and "patch_idx" in meta.columns:
                dist = np.abs(meta["patch_idx"].to_numpy(dtype=float) - float(target_patch))
                dist = np.nan_to_num(dist, nan=99.0)
                w_meta = w_rec * (1.0 + patch_boost * np.power(0.5, dist))
            # Patch weighting sharpens *priority* (presence, bans, habits, pick patterns), but makes
            # win rates noisier (smaller effective sample), so champion strength uses recency only.
            for row, wm, wr_ in zip(meta.to_dict("records"), w_meta, w_rec):
                self._add(row, float(wm), float(wr_))
        self._role_cache = {}

    # ------------------------------------------------------------------ build
    def _add_matchups(self, row: dict, w: float):
        champs = {}
        for side in SIDES:
            won = row["blue_win"] if side == "Blue" else 1 - row["blue_win"]
            cs = [row[f"{side}_{r}_champ"] for r in ROLES]
            for c in cs:
                self.champ_games_long[c] += w
                self.champ_wins_long[c] += w * won
            for i in range(5):
                for j in range(i + 1, 5):
                    key = tuple(sorted((cs[i], cs[j])))
                    self.pair[key] += w
                    self.pair_wins[key] += w * won
            champs[side] = cs
        bw = row["blue_win"]
        # Whole-team matchups: a vs b anywhere on the enemy team (e.g. Sylas vs Malphite).
        for a in champs["Blue"]:
            for b in champs["Red"]:
                self.vs_any[(a, b)] += w
                self.vs_any_wins[(a, b)] += w * bw
                self.vs_any[(b, a)] += w
                self.vs_any_wins[(b, a)] += w * (1 - bw)
        for ri in range(5):
            a, b = champs["Blue"][ri], champs["Red"][ri]
            self.lane[(a, b, ri)] += w
            self.lane_wins[(a, b, ri)] += w * bw
            self.lane[(b, a, ri)] += w
            self.lane_wins[(b, a, ri)] += w * (1 - bw)

    def _add(self, row: dict, w: float, w_strength: float | None = None):
        ws = w if w_strength is None else w_strength
        self.str_total += ws
        self.total += w
        self.n_games += 1
        blue = [row[f"Blue_{r}_champ"] for r in ROLES]
        red = [row[f"Red_{r}_champ"] for r in ROLES]
        for cs in (blue, red):
            for i in range(5):
                for j in range(i + 1, 5):
                    self.copick[tuple(sorted((cs[i], cs[j])))] += w
        for a in blue:
            for b in red:
                self.vspick[tuple(sorted((a, b)))] += w
        for side in SIDES:
            won = row["blue_win"] if side == "Blue" else 1 - row["blue_win"]
            team = row[f"{side}_teamname"]
            self.team_games[team] += w
            lg = row.get("league")
            if lg not in INTERNATIONAL:
                self.team_league_w.setdefault(team, {})
                self.team_league_w[team][lg] = self.team_league_w[team].get(lg, 0.0) + w
            if team not in self.team_last or row["date"] >= self.team_last[team]:
                self.team_last[team] = row["date"]
                # Home league = league with the most (recency-weighted) domestic games.
                lw = self.team_league_w.get(team)
                self.team_league[team] = max(lw, key=lw.get) if lw else lg
                self.team_roster[team] = {r: row[f"{side}_{r}_player"] for r in ROLES}
            for ri, r in enumerate(ROLES):
                c = row[f"{side}_{r}_champ"]
                p = row[f"{side}_{r}_player"]
                self.champ_games[c] += w
                self.champ_wins[c] += w * won
                self.str_games[c] += ws
                self.str_wins[c] += ws * won
                self.champ_role[c][ri] += w
                self.player_champ[(p, c)] += w
                self.player_champ_wins[(p, c)] += w * won
                self.player_games[p] += w
                self.player_role[p][ri] += w
                self.player_name[p] = row.get(f"{side}_{r}_pname") or p
                self.team_pick[(team, c)] += w
            for i in range(1, 6):
                b = row.get(f"{side}_ban{i}")
                if not is_missing(b):
                    self.champ_bans[b] += w
                    self.team_ban[(team, b)] += w

    # ---------------------------------------------------------------- queries
    @property
    def champions(self) -> list:
        return sorted(set(self.champ_games) | set(self.champ_bans))

    def presence(self, c) -> float:
        return (self.champ_games[c] + self.champ_bans[c]) / max(self.total, 1e-9)

    def pick_rate(self, c) -> float:
        return self.champ_games[c] / max(self.total, 1e-9)

    def ban_rate(self, c) -> float:
        return self.champ_bans[c] / max(self.total, 1e-9)

    def wr(self, c) -> float:
        """Champion strength: recency-weighted win rate (not patch-boosted), shrunk towards 50%.
        With str_adj set: 50% + the champion's effect after adjusting for both teams' ratings (M8)."""
        if self.str_adj is not None:
            return 0.5 + self.str_adj.get(c, 0.0)
        return (self.str_wins.get(c, 0.0) + self.k_champ * 0.5) / (self.str_games.get(c, 0.0) + self.k_champ)

    def raw_games(self, c) -> float:
        return self.champ_games[c]

    def role_p(self, c) -> np.ndarray:
        rp = self._role_cache.get(c)
        if rp is None:
            counts = self.champ_role.get(c)
            if counts is None or counts.sum() == 0:
                rp = np.full(5, 0.2)
            else:
                rp = (counts + 0.01) / (counts.sum() + 0.05)
            self._role_cache[c] = rp
        return rp

    def player_games_on(self, p, c) -> float:
        return self.player_champ.get((p, c), 0.0) if p else 0.0

    def player_wr_on(self, p, c) -> float:
        g = self.player_champ.get((p, c), 0.0)
        return (self.player_champ_wins.get((p, c), 0.0) + 5 * self.wr(c)) / (g + 5)

    def player_share(self, p, c) -> float:
        if not p:
            return 0.0
        return self.player_champ.get((p, c), 0.0) / max(self.player_games.get(p, 0.0), 1.0)

    def player_main_role(self, p):
        rc = self.player_role.get(p)
        return ROLES[int(np.argmax(rc))] if rc is not None and rc.sum() > 0 else None

    def _team_rate(self, c) -> float:
        """Share of team-games (2 per game) in which champion c was picked."""
        return self.champ_games.get(c, 0.0) / max(2 * self.total, 1e-9)

    def copick_lift(self, a, b, alpha: float = 3.0) -> float:
        """log(observed / expected) of a and b on the same team; > 0 = pros pair them (e.g. Orianna + Nocturne)."""
        n = self.copick.get(tuple(sorted((a, b))), 0.0)
        e = 2 * self.total * self._team_rate(a) * self._team_rate(b)
        return math.log((n + alpha) / (e + alpha))

    def vspick_lift(self, a, b, alpha: float = 3.0) -> float:
        """log(observed / expected) of a and b on opposite teams; > 0 = commonly picked into each other."""
        n = self.vspick.get(tuple(sorted((a, b))), 0.0)
        # each game has two orientations (a blue / b red and vice versa)
        e = 2 * self.total * self._team_rate(a) * self._team_rate(b)
        return math.log((n + alpha) / (e + alpha))

    def wr_long(self, c) -> float:
        """Champion win rate over the long (multi-season) matchup window."""
        return (self.champ_wins_long.get(c, 0.0) + self.k_champ * 0.5) / (self.champ_games_long.get(c, 0.0) + self.k_champ)

    def syn(self, a, b) -> float:
        """Win-rate lift of a and b on the same team beyond their individual (long-window) strength."""
        key = tuple(sorted((a, b)))
        exp = 0.5 * (self.wr_long(a) + self.wr_long(b))
        g = self.pair.get(key, 0.0)
        return (self.pair_wins.get(key, 0.0) + self.k_pair * exp) / (g + self.k_pair) - exp

    def counter(self, a, b, ri: int) -> float:
        """Win-rate lift of champion a facing b in lane `ri`, beyond their individual (long-window) strength."""
        exp = 0.5 + 0.5 * (self.wr_long(a) - self.wr_long(b))
        g = self.lane.get((a, b, ri), 0.0)
        return (self.lane_wins.get((a, b, ri), 0.0) + self.k_lane * exp) / (g + self.k_lane) - exp

    def vs_team(self, a, b) -> float:
        """Win-rate lift of a when b is anywhere on the enemy team, beyond both champions' strength."""
        exp = 0.5 + (self.wr_long(a) - 0.5) - (self.wr_long(b) - 0.5)
        g = self.vs_any.get((a, b), 0.0)
        return (self.vs_any_wins.get((a, b), 0.0) + self.k_pair * exp) / (g + self.k_pair) - exp

    def vs_team_games(self, a, b) -> float:
        return self.vs_any.get((a, b), 0.0)

    def lane_games(self, a, b, ri: int) -> float:
        return self.lane.get((a, b, ri), 0.0)

    def team_pick_count(self, team, c) -> float:
        return self.team_pick.get((team, c), 0.0)

    def team_ban_count(self, team, c) -> float:
        return self.team_ban.get((team, c), 0.0)

    def drop_cache(self):
        self._role_cache = {}

    def __getstate__(self):
        d = self.__dict__.copy()
        for k in ("champ_role", "player_role"):
            d[k] = dict(d[k])
        for k, v in list(d.items()):
            if isinstance(v, defaultdict):
                d[k] = dict(v)
        d["_role_cache"] = {}
        return d

    def __setstate__(self, d):
        d.setdefault("str_games", d.get("champ_games", {}))
        d.setdefault("str_wins", d.get("champ_wins", {}))
        d.setdefault("str_total", d.get("total", 0.0))
        d.setdefault("vs_any", {})
        d.setdefault("vs_any_wins", {})
        d.setdefault("copick", {})
        d.setdefault("vspick", {})
        d.setdefault("champ_games_long", {})
        d.setdefault("champ_wins_long", {})
        for k in ("champ_games_long", "champ_wins_long", "champ_games", "champ_wins", "champ_bans", "player_champ", "player_champ_wins",
                  "player_games", "team_pick", "team_ban", "team_games", "pair", "pair_wins",
                  "copick", "vspick", "str_games", "str_wins", "vs_any", "vs_any_wins",
                  "lane", "lane_wins"):
            d[k] = defaultdict(float, d[k])
        d["champ_role"] = defaultdict(lambda: np.zeros(5), d["champ_role"])
        d["player_role"] = defaultdict(lambda: np.zeros(5), d["player_role"])
        self.__dict__.update(d)


def fearless_game_weights(games: pd.DataFrame) -> np.ndarray:
    """Per-game weight: later games of a fearless series count less (see config.FEARLESS_GAME_WEIGHTS)."""
    n = len(games)
    if n == 0 or "fearless" not in games.columns or "game" not in games.columns:
        return np.ones(n)
    gn = pd.to_numeric(games["game"], errors="coerce").fillna(1).clip(1, 5).astype(int)
    w = gn.map(FEARLESS_GAME_WEIGHTS).fillna(1.0).to_numpy(dtype=float)
    return np.where(games["fearless"].to_numpy(dtype=bool), w, 1.0)


def log1p(x: float) -> float:
    return math.log1p(max(x, 0.0))
