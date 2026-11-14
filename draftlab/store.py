"""DuckDB data layer (Phase 1). `series_id` / `game_id` are the keys everywhere.

Tables
  series           series_id, league, start_time, team_a, team_b, best_of, fearless, games_played, wins_a, wins_b, source
  games            game_id, series_id, game_number, date, league, patch, blue_team, red_team, blue_win, source
  draft_actions    game_id, step (0-19, tournament draft order), side, kind (ban/pick), champion
  game_players     game_id, side, role, player_key, player_name, champion
  events           game_id, game_time_s, event_type (kill/objective/ward/tower/item), subtype, team, player, x, y, detail, source
  vod_sync         game_id, video_url, source, created
  vod_sync_points  game_id, game_time_s, video_time_s, method   (several points per game, so pauses are handled)

The pro-data store (data/draftlab.duckdb) is written by scripts/ingest.py and read by the app read-only.
Practice series created in the app live in a separate file (data/practice.duckdb), so the app never needs
a write lock on the pro store.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from .config import ROLES, ROOT, SIDES

DB_PATH = ROOT / "data" / "draftlab.duckdb"
PRACTICE_DB = ROOT / "data" / "practice.duckdb"

SCHEMA = """
CREATE TABLE IF NOT EXISTS series (
  series_id VARCHAR PRIMARY KEY, league VARCHAR, start_time TIMESTAMP, team_a VARCHAR, team_b VARCHAR,
  best_of INTEGER, fearless BOOLEAN, games_played INTEGER, wins_a INTEGER, wins_b INTEGER, source VARCHAR);
CREATE TABLE IF NOT EXISTS games (
  game_id VARCHAR PRIMARY KEY, series_id VARCHAR, game_number INTEGER, date TIMESTAMP, league VARCHAR,
  patch VARCHAR, blue_team VARCHAR, red_team VARCHAR, blue_win INTEGER, source VARCHAR);
CREATE TABLE IF NOT EXISTS draft_actions (
  game_id VARCHAR, step INTEGER, side VARCHAR, kind VARCHAR, champion VARCHAR, PRIMARY KEY (game_id, step));
CREATE TABLE IF NOT EXISTS game_players (
  game_id VARCHAR, side VARCHAR, role VARCHAR, player_key VARCHAR, player_name VARCHAR, champion VARCHAR,
  PRIMARY KEY (game_id, side, role));
CREATE TABLE IF NOT EXISTS events (
  game_id VARCHAR, game_time_s INTEGER, event_type VARCHAR, subtype VARCHAR, team VARCHAR, player VARCHAR,
  x DOUBLE, y DOUBLE, detail VARCHAR, source VARCHAR);
CREATE TABLE IF NOT EXISTS vod_sync (
  game_id VARCHAR PRIMARY KEY, video_url VARCHAR, source VARCHAR, created TIMESTAMP);
CREATE TABLE IF NOT EXISTS vod_sync_points (
  game_id VARCHAR, game_time_s DOUBLE, video_time_s DOUBLE, method VARCHAR, PRIMARY KEY (game_id, game_time_s));
"""


# ----------------------------------------------------------------------------- connection
@contextmanager
def connect(path: Path | str | None = None, read_only: bool = False):
    path = Path(path or DB_PATH)
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path), read_only=read_only)
    try:
        if not read_only:
            con.execute(SCHEMA)
        yield con
    finally:
        con.close()


# ----------------------------------------------------------------------------- series grouping
def make_series_id(league, team_a, team_b, start: datetime) -> str:
    a, b = sorted([str(team_a), str(team_b)])
    h = hashlib.sha1(f"{league}|{a}|{b}|{start:%Y-%m-%dT%H:%M}".encode()).hexdigest()[:8]
    return f"{(league or 'X').replace(' ', '')}-{start:%Y%m%d}-{h}"


def best_of_from(wins_a: int, wins_b: int, games_played: int) -> int:
    """Infer the series format from the winner's score (1 -> Bo1, 2 -> Bo3, 3 -> Bo5)."""
    w = max(wins_a, wins_b)
    if w >= 3 or games_played >= 4:
        return 5
    if w == 2 or games_played >= 2:
        return 3
    return 1


def group_series(games: list[dict], gap_hours: float = 10.0) -> list[dict]:
    """Assign series_id to games (same league + team pair, consecutive game numbers, < gap_hours apart)."""
    games = sorted(games, key=lambda g: (g["date"], g.get("game_number") or 0))
    open_series: dict = {}
    for g in games:
        key = (g.get("league"), tuple(sorted((g["blue_team"], g["red_team"]))))
        gn = g.get("game_number")
        st = open_series.get(key)
        new = (st is None or (g["date"] - st["last"]).total_seconds() > gap_hours * 3600
               or gn == 1 or (gn is not None and st["gn"] is not None and gn <= st["gn"]))
        if new:
            st = {"id": make_series_id(g.get("league"), *key[1], g["date"]), "start": g["date"]}
        g["series_id"] = st["id"]
        st.update(last=g["date"], gn=gn)
        open_series[key] = st
    return games


def detect_fearless(games: list[dict], threshold: float = 0.05) -> set:
    """League-years where games 2+ of a series (almost) never reuse a picked champion."""
    by_series: dict = {}
    for g in games:
        by_series.setdefault(g["series_id"], []).append(g)
    stats: dict = {}
    for gs in by_series.values():
        gs.sort(key=lambda g: (g["date"], g.get("game_number") or 0))
        used: set = set()
        for i, g in enumerate(gs):
            picks = {c for _, k, c in g["actions"] if k == "pick" and c}
            if i > 0:
                ly = (g.get("league"), g["date"].year)
                n, v = stats.get(ly, (0, 0))
                stats[ly] = (n + 1, v + bool(picks & used))
            used |= picks
    return {ly for ly, (n, v) in stats.items() if n >= 10 and v / n < threshold}


# ----------------------------------------------------------------------------- ingest
def ingest_matches(con, source, replace: bool = True) -> dict:
    """Load games, drafts and players from a MatchSource; build series rows. Returns counts."""
    games = group_series(list(source.games()))
    fearless = detect_fearless(games)
    if replace:
        for t in ("draft_actions", "game_players", "games", "series"):  # children first (they look up games)
            con.execute(f"DELETE FROM {t} WHERE " + ("source = ?" if t in ("series", "games") else
                        "game_id IN (SELECT game_id FROM games WHERE source = ?)"), [source.name])
    g_rows, a_rows, p_rows = [], [], []
    for g in games:
        g_rows.append((g["game_id"], g["series_id"], g.get("game_number"), g["date"], g.get("league"),
                       None if g.get("patch") is None else str(g["patch"]), g["blue_team"], g["red_team"],
                       int(g["blue_win"]), source.name))
        for step, (side, kind, champ) in enumerate(g["actions"]):
            a_rows.append((g["game_id"], step, side, kind, champ))
        for side in SIDES:
            for role in ROLES:
                pk, pn, ch = g["players"][side][role]
                p_rows.append((g["game_id"], side, role, pk, pn, ch))
    gdf = pd.DataFrame(g_rows, columns=["game_id", "series_id", "game_number", "date", "league", "patch",
                                        "blue_team", "red_team", "blue_win", "source"])
    adf = pd.DataFrame(a_rows, columns=["game_id", "step", "side", "kind", "champion"])
    pdf = pd.DataFrame(p_rows, columns=["game_id", "side", "role", "player_key", "player_name", "champion"])
    con.execute("INSERT OR REPLACE INTO games SELECT * FROM gdf")
    con.execute("INSERT OR REPLACE INTO draft_actions SELECT * FROM adf")
    con.execute("INSERT OR REPLACE INTO game_players SELECT * FROM pdf")
    # series rows
    s_rows = []
    for sid, grp in gdf.groupby("series_id"):
        grp = grp.sort_values("date")
        a, b = sorted({*grp["blue_team"], *grp["red_team"]})[:2] if len({*grp["blue_team"], *grp["red_team"]}) >= 2 \
            else (grp["blue_team"].iloc[0], grp["red_team"].iloc[0])
        wins_a = int(((grp["blue_team"] == a) & (grp["blue_win"] == 1)).sum() + ((grp["red_team"] == a) & (grp["blue_win"] == 0)).sum())
        wins_b = len(grp) - wins_a
        start = grp["date"].iloc[0]
        lg = grp["league"].iloc[0]
        s_rows.append((sid, lg, start, a, b, best_of_from(wins_a, wins_b, len(grp)),
                       (lg, pd.Timestamp(start).year) in fearless, len(grp), wins_a, wins_b, source.name))
    sdf = pd.DataFrame(s_rows, columns=["series_id", "league", "start_time", "team_a", "team_b", "best_of",
                                        "fearless", "games_played", "wins_a", "wins_b", "source"])
    con.execute("INSERT OR REPLACE INTO series SELECT * FROM sdf")
    return {"games": len(gdf), "series": len(sdf), "draft_actions": len(adf), "game_players": len(pdf),
            "fearless_league_years": len(fearless)}


def ingest_events(con, source, game_ids: Iterable[str] | None = None) -> int:
    """Load events from an EventSource for the given games (default: all it has). Replaces existing rows."""
    n = 0
    for gid in (game_ids or source.available_games()):
        rows = list(source.events(gid))
        con.execute("DELETE FROM events WHERE game_id = ? AND source = ?", [gid, source.name])
        if rows:
            df = pd.DataFrame(rows)[["game_id", "game_time_s", "event_type", "subtype", "team", "player", "x", "y", "detail"]]
            df["source"] = source.name
            con.execute("INSERT INTO events SELECT * FROM df")
            n += len(df)
    return n


# ----------------------------------------------------------------------------- VOD sync
def set_vod(con, game_id: str, video_url: str, points: list[tuple[float, float]], method: str = "manual",
            source: str = "manual"):
    """Store a game's video and its sync points [(game_time_s, video_time_s), ...]."""
    con.execute("INSERT OR REPLACE INTO vod_sync VALUES (?, ?, ?, ?)", [game_id, video_url, source, datetime.now()])
    con.execute("DELETE FROM vod_sync_points WHERE game_id = ?", [game_id])
    for gt, vt in points:
        con.execute("INSERT INTO vod_sync_points VALUES (?, ?, ?, ?)", [game_id, float(gt), float(vt), method])


def video_time(con, game_id: str, game_time_s: float) -> float | None:
    """Map in-game time to video time by interpolating between sync points.

    Between two points the mapping is linear; a pause shows up as video time advancing faster than game
    time between two points, so syncing at intervals (not one offset) absorbs pauses. Outside the points,
    extrapolate at 1 video second per game second from the nearest point.
    """
    pts = con.execute("SELECT game_time_s, video_time_s FROM vod_sync_points WHERE game_id = ? ORDER BY game_time_s",
                      [game_id]).fetchall()
    if not pts:
        return None
    if game_time_s <= pts[0][0]:
        return pts[0][1] - (pts[0][0] - game_time_s)
    if game_time_s >= pts[-1][0]:
        return pts[-1][1] + (game_time_s - pts[-1][0])
    for (g0, v0), (g1, v1) in zip(pts, pts[1:]):
        if g0 <= game_time_s <= g1:
            # a pause sits inside this segment: keep 1:1 speed and push the extra video time to the end
            # of the segment (pauses happen at a moment, not spread over the segment)
            return min(v0 + (game_time_s - g0), v1)
    return None


# ----------------------------------------------------------------------------- series state
def series_state(con, series_id: str) -> dict | None:
    """Score, format, games played and the champions locked by fearless (picks of either team)."""
    s = con.execute("SELECT * FROM series WHERE series_id = ?", [series_id]).fetchdf()
    if s.empty:
        return None
    s = s.iloc[0].to_dict()
    games = con.execute("SELECT game_id, game_number, date, blue_team, red_team, blue_win FROM games "
                        "WHERE series_id = ? ORDER BY date, game_number", [series_id]).fetchdf()
    picks = con.execute("SELECT d.game_id, d.side, d.champion FROM draft_actions d JOIN games g USING (game_id) "
                        "WHERE g.series_id = ? AND d.kind = 'pick' AND d.champion IS NOT NULL", [series_id]).fetchdf()
    used = {t: set() for t in (s["team_a"], s["team_b"])}
    out_games = []
    for g in games.to_dict("records"):
        gp = picks[picks["game_id"] == g["game_id"]]
        for side, team in (("Blue", g["blue_team"]), ("Red", g["red_team"])):
            used.setdefault(team, set()).update(gp[gp["side"] == side]["champion"])
        out_games.append({**g, "date": str(g["date"]), "winner": g["blue_team"] if g["blue_win"] else g["red_team"],
                          "picks": {side: list(gp[gp["side"] == side]["champion"]) for side in SIDES}})
    locked = sorted(set().union(*used.values())) if s["fearless"] else []
    need = s["best_of"] // 2 + 1
    return {**{k: (str(v) if k == "start_time" else v) for k, v in s.items()},
            "games": out_games, "locked": locked, "used_by_team": {k: sorted(v) for k, v in used.items()},
            "finished": max(s["wins_a"], s["wins_b"]) >= need, "games_remaining": max(0, s["best_of"] - s["games_played"])}


def remaining_pool(state: dict, pool: Iterable[str]) -> list[str]:
    """Champions from a player's/team's pool still available in this series (fearless locks removed)."""
    locked = set(state.get("locked") or [])
    return [c for c in pool if c not in locked]


# ----------------------------------------------------------------------------- practice series (app)
def create_practice_series(con, team_a: str, team_b: str, best_of: int, fearless: bool) -> str:
    start = datetime.now()
    sid = "PRACTICE-" + make_series_id("P", team_a, team_b, start).split("-", 1)[1]
    con.execute("INSERT OR REPLACE INTO series VALUES (?, 'PRACTICE', ?, ?, ?, ?, ?, 0, 0, 0, 'practice')",
                [sid, start, *sorted([team_a, team_b]), best_of, fearless])
    return sid


def record_practice_game(con, series_id: str, blue_team: str, red_team: str, actions: list, blue_win: int) -> str:
    """Store one drafted practice game (20 actions as champion names or None) and update the series score."""
    n = con.execute("SELECT count(*) FROM games WHERE series_id = ?", [series_id]).fetchone()[0]
    gid = f"{series_id}-G{n + 1}"
    con.execute("INSERT INTO games VALUES (?, ?, ?, ?, 'PRACTICE', NULL, ?, ?, ?, 'practice')",
                [gid, series_id, n + 1, datetime.now(), blue_team, red_team, int(blue_win)])
    from .data import DRAFT_ORDER
    for step, ((side, kind, _), champ) in enumerate(zip(DRAFT_ORDER, actions)):
        con.execute("INSERT INTO draft_actions VALUES (?, ?, ?, ?, ?)", [gid, step, side, kind, champ])
    winner = blue_team if blue_win else red_team
    a, b = con.execute("SELECT team_a, team_b FROM series WHERE series_id = ?", [series_id]).fetchone()
    con.execute("UPDATE series SET games_played = games_played + 1, wins_a = wins_a + ?, wins_b = wins_b + ? "
                "WHERE series_id = ?", [int(winner == a), int(winner == b), series_id])
    return gid
