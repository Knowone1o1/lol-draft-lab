"""League strength offsets and league-adjusted Elo (judge finding M1).

Plain Elo has one closed pool per league: every league starts at 1500, so an ERL team that dominates its
league ends up rated above Gen.G. Each team gets a home league (its most common non-international league),
and each league an offset in Elo points, fitted on games between teams from different home leagues
(internationals, EMEA Masters, KeSPA Cup ...) with a prior: majors 0, an academy/ERL league = -200.

Leakage-free use in training: `monthly_adjusted` refits the offsets every month using only games played
*before* that month, and adds them to each team's pre-game Elo. `scripts/build_league_elo.py` fits the
same offsets on all data for the app (models/league_elo.json).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

INTERNATIONAL = {"WLDs", "MSI", "FST", "EWC", "ASI", "KeSPA Cup", "AmCh", "EM", "LAS", "Asia Masters"}
MAJORS = {"LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP", "CBLOL"}
PARENT = {  # academy / regional leagues -> the major league their teams feed
    "LCKC": "LCK", "LDL": "LPL", "NACL": "LCS", "LJL": "LCP", "VCS": "LCP", "PCS": "LCP",
    "LFL": "LEC", "PRM": "LEC", "LVP SL": "LEC", "TCL": "LEC", "NLC": "LEC", "EBL": "LEC", "HM": "LEC",
    "LIT": "LEC", "AL": "LEC", "ROL": "LEC", "HLL": "LEC", "LPLOL": "LEC", "RL": "LEC", "LES": "LEC",
    "CD": "CBLOL", "LRN": "LTA N", "LRS": "LTA S",
}
MINOR_GAP = 200.0     # prior: an academy/ERL league is this many Elo points below its parent
PRIOR_SD = 120.0      # how far the data may move a league from its prior (Elo points)
HOME_WINDOW_DAYS = 730


def league_prior(lg) -> float:
    if lg in MAJORS:
        return 0.0
    if lg in PARENT and PARENT[lg] in MAJORS:
        return -MINOR_GAP
    return -MINOR_GAP - 50.0  # unknown small league


def home_leagues(g: pd.DataFrame) -> dict:
    """team -> most common non-international league in `g`."""
    parts = [g[[f"{s}_teamname", "league"]].set_axis(["team", "league"], axis=1) for s in ("Blue", "Red")]
    t = pd.concat(parts, ignore_index=True)
    t = t[~t["league"].isin(INTERNATIONAL)]
    if t.empty:
        return {}
    c = t.groupby(["team", "league"]).size().reset_index(name="n").sort_values(["team", "n"])
    return dict(zip(c["team"], c["league"]))  # last row per team = most games


def fit_offsets(g: pd.DataFrame, home: dict, blue_elo=None, red_elo=None) -> tuple[dict, int]:
    """League offsets (Elo points, average major = 0) from cross-league games in `g`."""
    blue_elo = g["blue_elo"].to_numpy() if blue_elo is None else np.asarray(blue_elo)
    red_elo = g["red_elo"].to_numpy() if red_elo is None else np.asarray(red_elo)
    hb = g["Blue_teamname"].map(home).to_numpy()
    hr = g["Red_teamname"].map(home).to_numpy()
    ok = pd.notna(hb) & pd.notna(hr) & (hb != hr)
    leagues = sorted(set(home.values()))
    if not leagues:
        return {}, 0
    p0 = np.array([league_prior(lg) for lg in leagues])
    if ok.sum() == 0:
        return dict(zip(leagues, p0)), 0
    idx = {lg: i for i, lg in enumerate(leagues)}
    bi = np.array([idx[x] for x in hb[ok]])
    ri = np.array([idx[x] for x in hr[ok]])
    d = (blue_elo - red_elo)[ok]
    y = g["blue_win"].to_numpy(dtype=float)[ok]
    s = math.log(10) / 400

    def nll(o):
        z = s * (d + o[bi] - o[ri])
        loss = np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(((o - p0) / PRIOR_SD) ** 2)
        pz = 1 / (1 + np.exp(-z))
        grad = np.zeros_like(o)
        np.add.at(grad, bi, s * (pz - y))
        np.add.at(grad, ri, -s * (pz - y))
        return loss, grad + (o - p0) / PRIOR_SD ** 2

    res = minimize(nll, p0.copy(), jac=True, method="L-BFGS-B")
    off = dict(zip(leagues, res.x))
    majors = [off[m] for m in MAJORS if m in off]
    shift = float(np.mean(majors)) if majors else 0.0  # anchor: average major league = 0
    return {k: float(v - shift) for k, v in off.items()}, int(ok.sum())


def _team_offset(team, game_league, home, off):
    lg = home.get(team)
    if lg is None and game_league not in INTERNATIONAL:
        lg = game_league
    if lg is None:
        return -MINOR_GAP - 50.0
    return off.get(lg, league_prior(lg))


def monthly_adjusted(g: pd.DataFrame, blue_elo=None, red_elo=None) -> tuple[np.ndarray, np.ndarray]:
    """League-adjusted pre-game ratings, leakage-free: the offsets used for a game in month m are fitted
    only on games played before m (home leagues from the 2 years before m)."""
    blue_elo = g["blue_elo"].to_numpy(dtype=float) if blue_elo is None else np.asarray(blue_elo, float)
    red_elo = g["red_elo"].to_numpy(dtype=float) if red_elo is None else np.asarray(red_elo, float)
    dates = g["date"]
    month = dates.dt.to_period("M").dt.to_timestamp().to_numpy()
    ob, or_ = np.zeros(len(g)), np.zeros(len(g))
    lgs = g["league"].to_numpy()
    bt, rt = g["Blue_teamname"].to_numpy(), g["Red_teamname"].to_numpy()
    for m in np.unique(month):
        m = pd.Timestamp(m)
        past = ((dates < m) & (dates >= m - pd.Timedelta(days=HOME_WINDOW_DAYS))).to_numpy()
        home = home_leagues(g[past]) if past.any() else {}
        off, _ = fit_offsets(g[past], home, blue_elo[past], red_elo[past]) if past.any() else ({}, 0)
        sel = np.nonzero(month == np.datetime64(m))[0]
        for i in sel:
            ob[i] = _team_offset(bt[i], lgs[i], home, off)
            or_[i] = _team_offset(rt[i], lgs[i], home, off)
    return blue_elo + ob, red_elo + or_
