"""Load Oracle's Elixir match CSVs and reshape them into one row per game.

Oracle's Elixir has 12 rows per game: 10 player rows (position top/jng/mid/bot/sup)
and 2 team rows (position == "team"). Team rows carry bans (ban1..ban5, in ban order)
and the team's picks in pick order (pick1..pick5).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd

from .config import RAW_DIR, ROLES, SIDES

# Standard tournament draft order: (side, kind, index within that side's list)
DRAFT_ORDER = [
    ("Blue", "ban", 0), ("Red", "ban", 0), ("Blue", "ban", 1), ("Red", "ban", 1),
    ("Blue", "ban", 2), ("Red", "ban", 2),
    ("Blue", "pick", 0), ("Red", "pick", 0), ("Red", "pick", 1), ("Blue", "pick", 1),
    ("Blue", "pick", 2), ("Red", "pick", 2),
    ("Red", "ban", 3), ("Blue", "ban", 3), ("Red", "ban", 4), ("Blue", "ban", 4),
    ("Red", "pick", 3), ("Blue", "pick", 3), ("Blue", "pick", 4), ("Red", "pick", 4),
]
# Riot's 2026 "First Selection" decouples side and draft order: the team that picks first takes the whole
# first-pick sequence (first ban, first pick), even on red side. Same sequence with the side labels swapped.
# (Assumption from press coverage: the first-pick team also bans first; Riot's own diagram was not found.)
DRAFT_ORDER_RED_FIRST = [("Red" if s == "Blue" else "Blue", k, i) for s, k, i in DRAFT_ORDER]


def draft_order(blue_first: bool = True) -> list:
    """The 20-step draft sequence when blue (default, and always before 2026) or red picks first."""
    return DRAFT_ORDER if blue_first else DRAFT_ORDER_RED_FIRST


def blue_first_of(row: dict) -> bool:
    v = row.get("blue_first_pick", 1.0)
    return not (v is not None and not (isinstance(v, float) and math.isnan(v)) and float(v) == 0.0)


def other(side: str) -> str:
    return "Red" if side == "Blue" else "Blue"


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x)) or x == ""


def load_raw(paths=None) -> pd.DataFrame:
    paths = list(paths) if paths else sorted(RAW_DIR.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files in {RAW_DIR}. Download Oracle's Elixir match data "
            "(see README) and put the files there."
        )
    frames = []
    for p in paths:
        # patch as text: as a float, "16.10" would become 16.1 and merge with patch 16.1
        f = pd.read_csv(p, low_memory=False, dtype={"patch": str})
        if "gameid" not in f.columns:
            raise ValueError(f"{p} doesn't look like an Oracle's Elixir file (no 'gameid' column)")
        frames.append(f)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.drop_duplicates(subset=["gameid", "participantid"] if "participantid" in df else None)
    return df


def build_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game, wide columns like Blue_top_champ, Red_ban3, Blue_pick2."""
    pl = df[df["position"].isin(ROLES)].copy()
    if "playerid" in pl.columns:
        pl["pkey"] = pl["playerid"].where(pl["playerid"].notna() & (pl["playerid"] != ""), pl["playername"])
    else:
        pl["pkey"] = pl["playername"]
    tm = df[df["position"] == "team"].copy()

    parts = []
    for col, src in (("champ", "champion"), ("player", "pkey"), ("pname", "playername")):
        p = pl.pivot_table(index="gameid", columns=["side", "position"], values=src, aggfunc="first")
        p.columns = [f"{s}_{r}_{col}" for s, r in p.columns]
        parts.append(p)

    team_cols = ["teamname", "result"] + [f"ban{i}" for i in range(1, 6)] + [f"pick{i}" for i in range(1, 6)]
    team_cols += [c for c in ("golddiffat10", "golddiffat15", "xpdiffat15", "firstblood", "firstdragon", "firstPick")
                  if c in tm.columns]
    team_cols = [c for c in team_cols if c in tm.columns]
    t = tm.drop_duplicates(["gameid", "side"]).set_index(["gameid", "side"])[team_cols].unstack("side")
    t.columns = [f"{s}_{c}" for c, s in t.columns]

    meta_cols = {c: (c, "first") for c in ("league", "year", "split", "playoffs", "date", "game", "patch") if c in tm.columns}
    meta = tm.groupby("gameid").agg(**meta_cols)

    g = meta.join(t, how="inner")
    for p in parts:
        g = g.join(p, how="inner")

    required = [f"{s}_{r}_champ" for s in SIDES for r in ROLES] + ["Blue_teamname", "Red_teamname", "Blue_result", "date"]
    missing = [c for c in required if c not in g.columns]
    if missing:
        raise ValueError(f"Data is missing expected columns: {missing}")
    g = g.dropna(subset=required)
    g["blue_win"] = g["Blue_result"].astype(float).round().astype(int)
    for s in SIDES:
        for r in ROLES:
            g[f"{s}_{r}_player"] = g[f"{s}_{r}_player"].fillna(g[f"{s}_{r}_pname"])

    # which side picked first (Oracle's Elixir "firstPick" on the team rows); before 2026 always blue
    fp = g["Blue_firstPick"] if "Blue_firstPick" in g.columns else pd.Series(np.nan, index=g.index)
    g["blue_first_pick"] = pd.to_numeric(fp, errors="coerce").fillna(1.0)
    g["patch_idx"] = g["patch"].map(patch_index) if "patch" in g.columns else None
    g["has_order"] = [_has_valid_order(row) for row in g.to_dict("records")]
    g = g.sort_values(["date", "game"] if "game" in g.columns else ["date"]).reset_index()
    return g


def patch_index(p):
    """'16.17' -> sequential number so that patch distance works across seasons (15.24 -> 16.01 = 1 apart)."""
    if is_missing(p):
        return np.nan
    try:
        major, minor = str(p).strip().split(".")[:2]
        return int(major) * 24 + int(minor)
    except ValueError:
        return np.nan


def patch_label(idx) -> str:
    idx = int(idx)
    major, minor = divmod(idx, 24)
    if minor == 0:
        major, minor = major - 1, 24
    return f"{major}.{minor:02d}"


def _has_valid_order(row: dict) -> bool:
    for s in SIDES:
        picks = [row.get(f"{s}_pick{i}") for i in range(1, 6)]
        if any(is_missing(p) for p in picks):
            return False
        if set(picks) != {row[f"{s}_{r}_champ"] for r in ROLES}:
            return False
    return True


def game_actions(row: dict) -> list:
    """The 20 draft actions of a game, in order: list of (side, kind, champion or None).
    Follows the side that actually picked first (2026 First Selection: red can pick first)."""
    out = []
    for side, kind, idx in draft_order(blue_first_of(row)):
        champ = row.get(f"{side}_{kind}{idx + 1}")
        out.append((side, kind, None if is_missing(champ) else champ))
    return out


def annotate_series(g: pd.DataFrame, fearless_threshold: float = 0.05) -> tuple[pd.DataFrame, set]:
    """Group games into series and detect fearless draft per league/year.

    In fearless leagues, champions picked earlier in the series are unavailable;
    the set is stored in the `locked` column (empty set for non-fearless leagues).
    """
    g = g.sort_values("date").reset_index(drop=True)
    recs = g.to_dict("records")
    last: dict = {}
    sid = 0
    series_ids, prior_sets = [], []
    for row in recs:
        key = (row.get("league"), tuple(sorted((row["Blue_teamname"], row["Red_teamname"]))))
        gn = row.get("game")
        gn = None if is_missing(gn) else int(gn)
        st = last.get(key)
        new = (
            st is None
            or (row["date"] - st["date"]).total_seconds() > 10 * 3600
            or (gn is not None and st["game"] is not None and gn <= st["game"])
            or gn == 1
        )
        if new:
            sid += 1
            st = {"sid": sid, "champs": set()}
        series_ids.append(st["sid"])
        prior_sets.append(frozenset(st["champs"]))
        champs = {row[f"{s}_{r}_champ"] for s in SIDES for r in ROLES}
        st = {"sid": st["sid"], "champs": st["champs"] | champs, "date": row["date"], "game": gn}
        last[key] = st
    g["series_id"] = series_ids
    g["prior_champs"] = prior_sets

    # A league-year is "fearless" if games 2+ of a series (almost) never reuse a champion.
    fearless = set()
    for (league, year), grp in g.groupby(["league", g["date"].dt.year]):
        later = grp[grp["prior_champs"].map(len) > 0]
        if len(later) < 10:
            continue
        viol = np.mean([
            bool(row["prior_champs"] & {row[f"{s}_{r}_champ"] for s in SIDES for r in ROLES})
            for row in later.to_dict("records")
        ])
        if viol < fearless_threshold:
            fearless.add((league, int(year)))
    g["fearless"] = [(lg, int(d.year)) in fearless for lg, d in zip(g["league"], g["date"])]
    g["locked"] = [p if f else frozenset() for p, f in zip(g["prior_champs"], g["fearless"])]
    return g, fearless


def add_elo(g: pd.DataFrame, k: float = 24.0, regress: float = 0.0, gap_days: float = 30.0) -> tuple[pd.DataFrame, dict]:
    """Pre-game team Elo (sequential, no look-ahead). Returns the frame and final ratings.

    regress: between splits (a team's first game after a break of more than `gap_days`), pull its rating
    this share of the way back to the mean of its home league's active teams (judge finding M1). The
    returned final ratings already include the pending pull for teams currently on a break.
    """
    from .league import INTERNATIONAL as INTL
    elo = defaultdict(lambda: 1500.0)
    last, league_of, members = {}, {}, defaultdict(set)
    gap = np.timedelta64(int(gap_days * 86400), "s")
    year = np.timedelta64(365, "D")

    def pulled(t, d):
        lg = league_of.get(t)
        peers = [elo[x] for x in members.get(lg, ()) if x != t and d - last[x] < year] if lg else []
        mean = float(np.mean(peers)) if peers else 1500.0
        return mean + (1.0 - regress) * (elo[t] - mean)

    be, re_ = [], []
    if "date" not in g:
        regress = 0.0
    dates = g["date"].to_numpy(dtype="datetime64[s]") if "date" in g else np.zeros(len(g), "datetime64[s]")
    leagues = g["league"].to_numpy() if "league" in g else np.full(len(g), None)
    for b, r, w, d, lg in zip(g["Blue_teamname"], g["Red_teamname"], g["blue_win"], dates, leagues):
        if regress:
            for t in (b, r):
                if t in last and d - last[t] > gap:
                    elo[t] = pulled(t, d)
        eb, er = elo[b], elo[r]
        be.append(eb)
        re_.append(er)
        exp = 1.0 / (1.0 + 10 ** ((er - eb) / 400.0))
        elo[b] = eb + k * (w - exp)
        elo[r] = er - k * (w - exp)
        for t in (b, r):
            last[t] = d
            if lg not in INTL and league_of.get(t) != lg:
                if t in league_of:
                    members[league_of[t]].discard(t)
                league_of[t] = lg
                members[lg].add(t)
    if regress and len(dates):
        now = dates.max()
        for t in list(elo):
            if now - last[t] > gap:
                elo[t] = pulled(t, now)
    g = g.copy()
    g["blue_elo"] = be
    g["red_elo"] = re_
    return g, dict(elo)


def load_games(paths=None, elo_k: float | None = None):
    """Full pipeline: raw CSVs -> games table with series, fearless locks and Elo.

    `blue_elo`/`red_elo` are the ratings the models use (league-adjusted if config.ELO_LEAGUE_ADJUST);
    the returned dict holds plain team ratings (the app adds models/league_elo.json offsets)."""
    if elo_k is None:
        from .params import load_params
        elo_k = load_params()["elo_k"]
    from .config import ELO_LEAGUE_ADJUST, ELO_REGRESS, ELO_REGRESS_GAP_DAYS
    g = build_games(load_raw(paths))
    g, fearless = annotate_series(g)
    g, elo = add_elo(g, k=elo_k, regress=ELO_REGRESS, gap_days=ELO_REGRESS_GAP_DAYS)
    g["blue_elo_raw"], g["red_elo_raw"] = g["blue_elo"], g["red_elo"]
    if ELO_LEAGUE_ADJUST:
        # League-adjusted pre-game ratings, offsets refitted monthly on earlier games only (M1).
        from .league import monthly_adjusted
        g["blue_elo"], g["red_elo"] = monthly_adjusted(g)
    return g, elo, fearless
