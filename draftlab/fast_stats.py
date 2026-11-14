"""Vectorised Snapshot builder.

Every statistic in a Snapshot is a weighted sum over games. StatsIndex precomputes, once, which
table cells each game contributes to; any weighting (half-lives, patch boost, fearless decay) or any
set of excluded games then becomes a numpy bincount — milliseconds instead of a Python loop over
tens of thousands of games. The result is an ordinary Snapshot, so all downstream code is unchanged.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import (FEARLESS_GAME_WEIGHTS, HALF_LIFE_DAYS, MATCHUP_HALF_LIFE_DAYS, PATCH_BOOST, ROLES,
                     SIDES)
from .data import is_missing
from .stats import INTERNATIONAL, Snapshot


class StatsIndex:
    def __init__(self, games: pd.DataFrame):
        g = games.reset_index(drop=True)
        self.games = g
        n = len(g)
        self.n = n
        self.date = g["date"].to_numpy(dtype="datetime64[ns]")
        self.patch = g["patch_idx"].to_numpy(dtype=float) if "patch_idx" in g else np.full(n, np.nan)
        self.fearless = g["fearless"].to_numpy(dtype=bool) if "fearless" in g else np.zeros(n, bool)
        self.gamenum = pd.to_numeric(g.get("game", pd.Series(1, index=g.index)), errors="coerce").fillna(1).clip(1, 5).astype(int).to_numpy()
        bw = g["blue_win"].to_numpy(dtype=float)

        champ_cols = [f"{s}_{r}_champ" for s in SIDES for r in ROLES]
        ban_cols = [f"{s}_ban{i}" for s in SIDES for i in range(1, 6) if f"{s}_ban{i}" in g]
        names = set(pd.unique(g[champ_cols].to_numpy().ravel()))
        for c in ban_cols:
            names |= {x for x in pd.unique(g[c]) if not is_missing(x)}
        self.champs = sorted(names)
        cid = {c: i for i, c in enumerate(self.champs)}
        C = self.C = len(self.champs)

        players = pd.unique(g[[f"{s}_{r}_player" for s in SIDES for r in ROLES]].to_numpy().ravel())
        self.players = list(players)
        pid = {p: i for i, p in enumerate(self.players)}
        teams = pd.unique(g[["Blue_teamname", "Red_teamname"]].to_numpy().ravel())
        self.teams = list(teams)
        tid = {t: i for i, t in enumerate(self.teams)}

        # champion ids, shape (n, side, role)
        ch = np.zeros((n, 2, 5), dtype=np.int64)
        pl = np.zeros((n, 2, 5), dtype=np.int64)
        for si, s in enumerate(SIDES):
            for ri, r in enumerate(ROLES):
                ch[:, si, ri] = g[f"{s}_{r}_champ"].map(cid).to_numpy()
                pl[:, si, ri] = g[f"{s}_{r}_player"].map(pid).to_numpy()
        tm = np.stack([g["Blue_teamname"].map(tid).to_numpy(), g["Red_teamname"].map(tid).to_numpy()], axis=1)
        win = np.stack([bw, 1 - bw], axis=1)                      # (n, side)
        self.ch, self.pl, self.tm, self.win = ch, pl, tm, win

        # --- per player-slot rows (n*10)
        self.s_game = np.repeat(np.arange(n), 10)
        self.s_champ = ch.reshape(-1)
        self.s_player = pl.reshape(-1)
        self.s_role = np.tile(np.arange(5), 2 * n)
        self.s_team = np.repeat(tm.reshape(-1), 5)
        self.s_win = np.repeat(win.reshape(-1), 5)

        # --- bans
        bg, bt, bc = [], [], []
        for si, s in enumerate(SIDES):
            for i in range(1, 6):
                col = f"{s}_ban{i}"
                if col not in g:
                    continue
                v = g[col].map(cid)
                ok = v.notna().to_numpy()
                bg.append(np.nonzero(ok)[0])
                bt.append(tm[ok, si])
                bc.append(v[ok].astype(np.int64).to_numpy())
        self.b_game = np.concatenate(bg) if bg else np.zeros(0, np.int64)
        self.b_team = np.concatenate(bt) if bt else np.zeros(0, np.int64)
        self.b_champ = np.concatenate(bc) if bc else np.zeros(0, np.int64)

        # --- same-team pairs (sorted ids => sorted names, since champs list is sorted)
        iu, ju = np.triu_indices(5, 1)
        a = ch[:, :, iu]; b = ch[:, :, ju]                       # (n, 2, 10)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        self.p_game = np.repeat(np.arange(n), 20)
        self.p_key = (lo * C + hi).reshape(-1)
        self.p_win = np.repeat(win, 10, axis=1).reshape(-1)

        # --- opposing pairs, ordered: champion of side s vs champion of the other side (n*2*25)
        A = ch[:, :, :, None]                                     # (n,2,5,1)
        B = ch[:, ::-1, None, :]                                  # (n,2,1,5) other side
        self.v_game = np.repeat(np.arange(n), 50)
        self.v_key = (A * C + B).reshape(-1)
        self.v_win = np.repeat(win, 25, axis=1).reshape(-1)
        # unordered (sorted) once per game for pick patterns: blue x red
        a2, b2 = ch[:, 0, :, None], ch[:, 1, None, :]
        lo2, hi2 = np.minimum(a2, b2), np.maximum(a2, b2)
        self.u_game = np.repeat(np.arange(n), 25)
        self.u_key = (lo2 * C + hi2).reshape(-1)

        # --- lane matchups, ordered (n*2*5): key = (a*C + b)*5 + role
        self.l_game = np.repeat(np.arange(n), 10)
        self.l_key = ((ch * C + ch[:, ::-1, :]) * 5 + np.arange(5)).reshape(-1)
        self.l_win = np.repeat(win, 5, axis=1).reshape(-1)

        # --- team-game rows (n*2)
        self.t_game = np.repeat(np.arange(n), 2)
        self.t_team = tm.reshape(-1)
        self.league = g["league"].to_numpy() if "league" in g else np.full(n, None)
        # Pre-game rating difference (blue - red), the offset for team-adjusted champion strength (M8).
        self.elo_diff = ((g["blue_elo"] - g["red_elo"]).to_numpy(dtype=float) if "blue_elo" in g
                         else np.zeros(n))

    # ------------------------------------------------------------------ M8
    def adjusted_strength(self, ref_date, half_life=HALF_LIFE_DAYS, fearless_weights=None, k_champ=None,
                          elo_diff=None) -> dict:
        """Champion strength adjusted for who played it (judge finding M8).

        Weighted logistic regression on games before ref_date (same recency weights as the raw win rate):
            logit P(blue wins) = a + b * elo_diff/400 + sum_blue beta_c - sum_red beta_c
        Both teams' ratings enter as a fitted offset, so a champion favoured by strong teams no longer
        inherits their win rate. L2 penalty k_champ/4 on beta = the same pull to 50% as k_champ pseudo-games
        in the raw win rate. Returns {champion: beta/4}, the effect on the win-probability scale near 50%.
        """
        from scipy import sparse
        from scipy.optimize import minimize
        from .config import K_CHAMP
        k = K_CHAMP if k_champ is None else k_champ
        ed = (self.elo_diff if elo_diff is None else np.asarray(elo_diff, float)) / 400.0
        ref = np.datetime64(pd.Timestamp(ref_date))
        days = (ref - self.date) / np.timedelta64(1, "D")
        w = np.where((days > 0) & (days < 6 * half_life), np.power(0.5, np.clip(days, 0, None) / half_life), 0.0)
        w = w * self.game_weights(fearless_weights)
        rows = np.nonzero(w > 0)[0]
        if len(rows) < 50:
            return {}
        n, C = len(rows), self.C
        ch = self.ch[rows]                                       # (n, 2, 5)
        r_idx = np.repeat(np.arange(n), 10)
        c_idx = ch.reshape(-1)
        val = np.tile(np.r_[np.ones(5), -np.ones(5)], n)
        X = sparse.csr_matrix((val, (r_idx, c_idx)), shape=(n, C))
        y = self.win[rows, 0]
        ww = w[rows]
        e = ed[rows]
        lam = k / 4.0

        def f(theta):
            a, b, beta = theta[0], theta[1], theta[2:]
            z = a + b * e + X @ beta
            loss = np.sum(ww * (np.logaddexp(0, z) - y * z)) + 0.5 * lam * beta @ beta
            r = ww * (1 / (1 + np.exp(-z)) - y)
            grad = np.r_[r.sum(), r @ e, X.T @ r + lam * beta]
            return loss, grad

        res = minimize(f, np.zeros(C + 2), jac=True, method="L-BFGS-B")
        beta = res.x[2:]
        used = np.bincount(c_idx, minlength=C) > 0
        return {self.champs[i]: float(beta[i] / 4.0) for i in np.nonzero(used)[0]}

    # ------------------------------------------------------------------ weights
    def game_weights(self, fearless_weights: dict | None = None) -> np.ndarray:
        fw = fearless_weights or FEARLESS_GAME_WEIGHTS
        w = np.array([fw.get(int(k), 1.0) for k in range(6)])[self.gamenum]
        return np.where(self.fearless, w, 1.0)

    def snapshot(self, ref_date, half_life=HALF_LIFE_DAYS, matchup_half_life=MATCHUP_HALF_LIFE_DAYS,
                 target_patch=None, patch_boost=PATCH_BOOST, fearless_weights=None,
                 include: np.ndarray | None = None, k_champ=None, k_pair=None, k_lane=None,
                 with_rosters: bool = True, win_only: bool = False, adjusted_strength: bool = False) -> Snapshot:
        """Build a Snapshot from games before ref_date (and in `include`, if given).

        win_only: build just the tables the win model reads (strength, player comfort, synergy,
        counters, whole-team matchups) — used by the tuner for speed.
        """
        ref = np.datetime64(pd.Timestamp(ref_date))
        days = (ref - self.date) / np.timedelta64(1, "D")
        valid = days > 0
        if include is not None:
            valid &= include
        days = np.clip(days, 0, None)
        gw = self.game_weights(fearless_weights) * valid
        w_long = np.where(days < 10 * matchup_half_life, np.power(0.5, days / matchup_half_life), 0.0) * gw
        w_rec = np.where(days < 6 * half_life, np.power(0.5, days / half_life), 0.0) * gw
        w_meta = w_rec
        if target_patch is not None:
            dist = np.nan_to_num(np.abs(self.patch - float(target_patch)), nan=99.0)
            w_meta = w_rec * (1.0 + patch_boost * np.power(0.5, dist))

        C, P, T = self.C, len(self.players), len(self.teams)
        names = self.champs
        bc = np.bincount

        snap = Snapshot.__new__(Snapshot)
        snap.ref_date = pd.Timestamp(ref_date)
        snap.half_life, snap.matchup_half_life = half_life, matchup_half_life
        snap.target_patch, snap.patch_boost = target_patch, patch_boost
        if k_champ is not None: snap.k_champ = k_champ
        if k_pair is not None: snap.k_pair = k_pair
        if k_lane is not None: snap.k_lane = k_lane

        def cdict(arr):
            nz = np.nonzero(arr)[0]
            return defaultdict(float, {names[i]: float(arr[i]) for i in nz})

        def pairdict(arr):
            nz = np.nonzero(arr)[0]
            return defaultdict(float, {(names[k // C], names[k % C]): float(arr[k]) for k in nz})

        # ---- meta memory (slot rows)
        sw = w_meta[self.s_game]
        sr = w_rec[self.s_game]
        snap.champ_games = cdict(bc(self.s_champ, sw, C))
        snap.champ_wins = cdict(bc(self.s_champ, sw * self.s_win, C))
        snap.str_games = cdict(bc(self.s_champ, sr, C))
        snap.str_wins = cdict(bc(self.s_champ, sr * self.s_win, C))
        pc_key = self.s_player * C + self.s_champ
        pc = bc(pc_key, sw, P * C)
        pcw = bc(pc_key, sw * self.s_win, P * C)
        nz = np.nonzero(pc)[0]
        snap.player_champ = defaultdict(float, {(self.players[k // C], names[k % C]): float(pc[k]) for k in nz})
        snap.player_champ_wins = defaultdict(float, {(self.players[k // C], names[k % C]): float(pcw[k]) for k in nz})
        pg = bc(self.s_player, sw, P)
        snap.player_games = defaultdict(float, {self.players[i]: float(pg[i]) for i in np.nonzero(pg)[0]})
        if win_only:
            empty = lambda: defaultdict(float)
            snap.champ_role, snap.player_role = defaultdict(lambda: np.zeros(5)), defaultdict(lambda: np.zeros(5))
            snap.team_pick, snap.champ_bans, snap.team_ban, snap.team_games = empty(), empty(), empty(), empty()
            snap.copick, snap.vspick = empty(), empty()
        else:
            cr = bc(self.s_champ * 5 + self.s_role, sw, C * 5).reshape(C, 5)
            snap.champ_role = defaultdict(lambda: np.zeros(5), {names[i]: cr[i].copy() for i in np.nonzero(cr.sum(1))[0]})
            prr = bc(self.s_player * 5 + self.s_role, sw, P * 5).reshape(P, 5)
            snap.player_role = defaultdict(lambda: np.zeros(5), {self.players[i]: prr[i].copy() for i in np.nonzero(prr.sum(1))[0]})
            tp = bc(self.s_team * C + self.s_champ, sw, T * C)
            snap.team_pick = defaultdict(float, {(self.teams[k // C], names[k % C]): float(tp[k]) for k in np.nonzero(tp)[0]})
            bw_ = w_meta[self.b_game]
            snap.champ_bans = cdict(bc(self.b_champ, bw_, C))
            tb = bc(self.b_team * C + self.b_champ, bw_, T * C)
            snap.team_ban = defaultdict(float, {(self.teams[k // C], names[k % C]): float(tb[k]) for k in np.nonzero(tb)[0]})
            tg = bc(self.t_team, w_meta[self.t_game], T)
            snap.team_games = defaultdict(float, {self.teams[i]: float(tg[i]) for i in np.nonzero(tg)[0]})
            snap.copick = pairdict(bc(self.p_key, w_meta[self.p_game], C * C))
            snap.vspick = pairdict(bc(self.u_key, w_meta[self.u_game], C * C))
        snap.total = float(w_meta.sum())
        snap.str_total = float(w_rec.sum())
        snap.n_games = int((w_meta > 0).sum())

        # ---- matchup memory
        slw = w_long[self.s_game]
        snap.champ_games_long = cdict(bc(self.s_champ, slw, C))
        snap.champ_wins_long = cdict(bc(self.s_champ, slw * self.s_win, C))
        pw = w_long[self.p_game]
        snap.pair = pairdict(bc(self.p_key, pw, C * C))
        snap.pair_wins = pairdict(bc(self.p_key, pw * self.p_win, C * C))
        vw = w_long[self.v_game]
        snap.vs_any = pairdict(bc(self.v_key, vw, C * C))
        snap.vs_any_wins = pairdict(bc(self.v_key, vw * self.v_win, C * C))
        lw = w_long[self.l_game]
        lg = bc(self.l_key, lw, C * C * 5)
        lgw = bc(self.l_key, lw * self.l_win, C * C * 5)
        nz = np.nonzero(lg)[0]
        snap.lane = defaultdict(float, {(names[k // 5 // C], names[k // 5 % C], int(k % 5)): float(lg[k]) for k in nz})
        snap.lane_wins = defaultdict(float, {(names[k // 5 // C], names[k // 5 % C], int(k % 5)): float(lgw[k]) for k in nz})

        # ---- rosters / names / leagues (only needed by the app)
        snap.player_name, snap.team_roster, snap.team_last, snap.team_league, snap.team_league_w = {}, {}, {}, {}, {}
        if with_rosters:
            self._fill_rosters(snap, w_meta)
        snap._role_cache = {}
        snap.str_adj = (self.adjusted_strength(ref_date, half_life, fearless_weights, snap.k_champ)
                        if adjusted_strength else None)
        return snap

    def _fill_rosters(self, snap, w_meta):
        g = self.games
        used = np.nonzero(w_meta > 0)[0]
        for s in SIDES:
            for r in ROLES:
                for p, nm in zip(g[f"{s}_{r}_player"].to_numpy()[used], g[f"{s}_{r}_pname"].to_numpy()[used]):
                    snap.player_name[p] = nm if isinstance(nm, str) and nm else p
        for i in used:  # games are date-sorted, so the last write per team is its latest game
            row_league = self.league[i]
            for si, s in enumerate(SIDES):
                team = self.teams[self.tm[i, si]]
                if row_league not in INTERNATIONAL:
                    lw = snap.team_league_w.setdefault(team, {})
                    lw[row_league] = lw.get(row_league, 0.0) + float(w_meta[i])
                snap.team_last[team] = pd.Timestamp(self.date[i])
                snap.team_roster[team] = {r: self.players[self.pl[i, si, ri]] for ri, r in enumerate(ROLES)}
        for team in snap.team_last:
            lw = snap.team_league_w.get(team)
            snap.team_league[team] = max(lw, key=lw.get) if lw else None
