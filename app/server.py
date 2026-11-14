"""Draft practice web app.

Run:  .venv\\Scripts\\python -m uvicorn app.server:app --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from draftlab.config import ROLES
from draftlab import store
from draftlab.brief import BriefEngine
from draftlab.early_engine import EarlyEngine
from draftlab.engine import DraftEngine
from draftlab.features import DraftState
from draftlab.matchups import best_answers

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="LoL Draft Lab")
_engine: DraftEngine | None = None


def engine() -> DraftEngine:
    global _engine
    if _engine is None:
        try:
            _engine = DraftEngine()
        except FileNotFoundError:
            raise HTTPException(503, "No trained model found. Run: python scripts/build.py")
    return _engine


class StatePayload(BaseModel):
    blue_team: str
    red_team: str
    blue_roster: dict[str, str]
    red_roster: dict[str, str]
    actions: list[str | None] = []
    locked: list[str] = []
    temperature: float = 0.7
    games_remaining: int = 0     # fearless series: max games still to play after this one
    brief_side: str = "Blue"     # whose 15-minute game plan to build
    sort: str = "blend"          # recommendation order: blend | win | pro
    ai_mode: str = "imitate"     # opponent: imitate (like the real team) | smart (uses the recommendations)
    blue_first: bool = True      # 2026 First Selection: False when red took the first pick


def to_state(p: StatePayload) -> DraftState:
    if len(p.actions) > 20:
        raise HTTPException(400, "too many actions")
    st = DraftState(p.blue_team, p.red_team, dict(p.blue_roster), dict(p.red_roster),
                    list(p.actions), frozenset(p.locked), blue_first=p.blue_first)
    seen = [c for c in p.actions if c]
    if len(seen) != len(set(seen)) or set(seen) & set(p.locked):
        raise HTTPException(400, "a champion was used twice")
    return st


@app.get("/api/setup")
def setup():
    e = engine()
    return {"teams": e.teams(), "players": e.players(), "champions": e.champion_info(),
            "roles": ROLES, "info": e.info}


@app.post("/api/state")
def state(p: StatePayload):
    e = engine()
    st = to_state(p)
    nxt = st.next_action()
    return {
        "next": None if nxt is None else {"side": nxt[0], "kind": nxt[1], "index": nxt[2]},
        "win_prob_blue": e.win_prob(st),
        "win_prob_blue_level": e.level_win_prob(st),    # draft alone at equal strength (None on the legacy model)
        "roles": e.roles(st),
        "recommendations": e.recommend(st, games_remaining=p.games_remaining, sort=p.sort) if nxt else [],
    }


@app.post("/api/ai_move")
def ai_move(p: StatePayload):
    e = engine()
    st = to_state(p)
    if st.next_action() is None:
        raise HTTPException(400, "draft is complete")
    champ, prob = e.ai_move(st, temperature=p.temperature, mode=p.ai_mode)
    return {"champion": champ, "prob": prob}


_early: EarlyEngine | None = None


def early_engine() -> EarlyEngine:
    global _early
    if _early is None:
        try:
            _early = EarlyEngine()
        except FileNotFoundError:
            raise HTTPException(503, "No early-game model. Run: python scripts/build_early.py")
    return _early


@app.post("/api/early")
def early(p: StatePayload):
    e = engine()
    st = to_state(p)
    roles = e.roles(st)
    return early_engine().forecast(st.blue_team, st.red_team, {"Blue": st.blue_roster, "Red": st.red_roster},
                                   roles, names=e.name)


_brief: BriefEngine | None = None


@app.post("/api/brief")
def brief(p: StatePayload):
    """15-minute game plan for `p.brief_side`, given both comps' timing and the early forecast."""
    global _brief
    if _brief is None:
        try:
            _brief = BriefEngine()
        except FileNotFoundError:
            raise HTTPException(503, "No game-plan model. Run: python scripts/build_brief.py")
    e = engine()
    st = to_state(p)
    side = p.brief_side if p.brief_side in ("Blue", "Red") else "Blue"
    other = "Red" if side == "Blue" else "Blue"
    try:
        fc = early(p)
    except HTTPException:
        fc = None
    elo = e.team_elo(st.team(side)) - e.team_elo(st.team(other))
    roles = e.roles(st)  # {side: {champion: role}}
    by_role = {s: {r: c for c, r in roles.get(s, {}).items()} for s in ("Blue", "Red")}
    out = _brief.brief(side, by_role[side], by_role[other], elo, fc, team=st.team(side), enemy=st.team(other))
    out["picks"] = {"own": len(st.picks(side)), "enemy": len(st.picks(other))}
    return out


_plan = None


@app.post("/api/plan_card")
def plan_card(p: StatePayload):
    """Early-game plan card for each team after a full draft: rule lines with IDs, why, sources and evidence."""
    global _plan
    e = engine()
    st = to_state(p)
    if len(st.picks("Blue")) < 5 or len(st.picks("Red")) < 5:
        raise HTTPException(400, "the plan card needs a full draft (5 picks per team)")
    try:
        fc = early(p)
    except HTTPException:
        fc = None
    if _plan is None:
        from draftlab.rules.plan import PlanEngine
        try:
            _plan = PlanEngine(early_stats=early_engine().es if fc else None)
        except FileNotFoundError:
            raise HTTPException(503, "No rules file. Run: python scripts/build_rules.py")
    roles = e.roles(st)
    by_role = {s: {r: c for c, r in roles.get(s, {}).items()} for s in ("Blue", "Red")}
    pick_index, n = {}, 0
    for (s, k, _), c in zip(st.order, st.actions):
        if k == "pick" and c:
            pick_index[c] = n
            n += 1
    briefs = {}
    for side in ("Blue", "Red"):
        try:
            briefs[side] = brief(p.model_copy(update={"brief_side": side}))
        except HTTPException:
            pass
    return {"cards": _plan.cards({"Blue": st.blue_team, "Red": st.red_team}, by_role,
                                 {"Blue": st.blue_roster, "Red": st.red_roster}, pick_index, fc, briefs)}


# ----------------------------------------------------------------------------- Phase 1: data layer
def _pro_db():
    if not store.DB_PATH.exists():
        raise HTTPException(503, "No data store yet. Run: python scripts/ingest.py")
    return store.connect(store.DB_PATH, read_only=True)


@app.get("/api/series")
def list_series(team: str | None = None, league: str | None = None, limit: int = 20):
    """Recent pro series (keyed by series_id), optionally for one team or league."""
    q, args = "SELECT * FROM series WHERE 1=1", []
    if team:
        q += " AND (team_a = ? OR team_b = ?)"
        args += [team, team]
    if league:
        q += " AND league = ?"
        args.append(league)
    q += " ORDER BY start_time DESC LIMIT ?"
    args.append(max(1, min(limit, 200)))
    with _pro_db() as con:
        df = con.execute(q, args).fetchdf()
    df["start_time"] = df["start_time"].astype(str)
    return df.to_dict("records")


@app.get("/api/series/{series_id}")
def get_series(series_id: str):
    """Series state: format, score, games with picks, and champions locked by fearless."""
    db = store.PRACTICE_DB if series_id.startswith("PRACTICE-") else store.DB_PATH
    if not db.exists():
        raise HTTPException(404, "series not found")
    with store.connect(db, read_only=True) as con:
        st = store.series_state(con, series_id)
    if st is None:
        raise HTTPException(404, "series not found")
    return st


@app.get("/api/games/{game_id}")
def get_game(game_id: str):
    """One game: draft (20 actions), players, event counts by type, and VOD sync points if any."""
    with _pro_db() as con:
        g = con.execute("SELECT * FROM games WHERE game_id = ?", [game_id]).fetchdf()
        if g.empty:
            raise HTTPException(404, "game not found")
        draft = con.execute("SELECT step, side, kind, champion FROM draft_actions WHERE game_id = ? ORDER BY step", [game_id]).fetchdf()
        players = con.execute("SELECT side, role, player_name, champion FROM game_players WHERE game_id = ?", [game_id]).fetchdf()
        ev = con.execute("SELECT event_type, count(*) AS n FROM events WHERE game_id = ? GROUP BY 1", [game_id]).fetchall()
        vod = con.execute("SELECT video_url FROM vod_sync WHERE game_id = ?", [game_id]).fetchone()
        pts = con.execute("SELECT game_time_s, video_time_s FROM vod_sync_points WHERE game_id = ? ORDER BY 1", [game_id]).fetchall()
    out = g.iloc[0].to_dict()
    out["date"] = str(out["date"])
    return {**out, "draft": draft.to_dict("records"), "players": players.to_dict("records"),
            "events": dict(ev), "vod": {"url": vod[0], "sync_points": pts} if vod else None}


@app.get("/api/clips")
def clips(q: str = "", game_id: str | None = None, limit: int = 20):
    """Phase 2 clip finder: events matching a short query, as clips with YouTube ?t= links or cut commands."""
    from draftlab.vod import find_clips
    query = q if q else {}
    if game_id:
        query = {**(dict() if isinstance(query, str) else query), **({"game_id": game_id} if not q else {})}
        if q:
            query = f"{q} in {game_id}"
    with _pro_db() as con:
        return [c.to_dict() for c in find_clips(con, query, limit=max(1, min(limit, 100)))]


class PracticeSeries(BaseModel):
    team_a: str
    team_b: str
    best_of: int = 3
    fearless: bool = True


class PracticeGame(BaseModel):
    blue_team: str
    red_team: str
    actions: list[str | None]
    blue_win: int


@app.post("/api/practice/series")
def new_practice_series(p: PracticeSeries):
    if p.best_of not in (1, 3, 5):
        raise HTTPException(400, "best_of must be 1, 3 or 5")
    with store.connect(store.PRACTICE_DB) as con:
        sid = store.create_practice_series(con, p.team_a, p.team_b, p.best_of, p.fearless)
        return store.series_state(con, sid)


@app.post("/api/practice/series/{series_id}/games")
def add_practice_game(series_id: str, g: PracticeGame):
    if len(g.actions) != 20:
        raise HTTPException(400, "a game needs all 20 draft actions")
    with store.connect(store.PRACTICE_DB) as con:
        st = store.series_state(con, series_id)
        if st is None:
            raise HTTPException(404, "series not found")
        if st["finished"]:
            raise HTTPException(400, "series is already finished")
        clash = set(st["locked"]) & {c for c in g.actions if c}
        if clash:
            raise HTTPException(400, f"locked by fearless in this series: {', '.join(sorted(clash))}")
        store.record_practice_game(con, series_id, g.blue_team, g.red_team, g.actions, g.blue_win)
        return store.series_state(con, series_id)


@app.get("/api/matchups")
def matchups(champ: str, role: str | None = None, min_games: int = 8):
    e = engine()
    if e.matchups is None:
        raise HTTPException(503, "No matchup table in the model. Run: python scripts/build.py --refresh-stats")
    norm = lambda x: "".join(ch for ch in x.lower() if ch.isalnum())
    champ = next((c for c in e.champions if norm(c) == norm(champ)), champ)  # "azir" -> "Azir"
    into, beats = best_answers(e.matchups, champ, role or None, min_games=min_games)
    return {"champion": champ, "good_into_it": into, "it_beats": beats}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
