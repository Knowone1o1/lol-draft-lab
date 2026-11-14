"""Phase 1 data layer: series/game keys, Bo-format, fearless, series state, events, VOD sync, practice series."""
import json

import pytest

from draftlab import store
from draftlab.sources import GridSource, OraclesElixirSource, RiotTimelineSource, SourceNotConfigured


@pytest.fixture()
def loaded(db, synthetic_games):
    counts = store.ingest_matches(db, OraclesElixirSource(games=synthetic_games))
    return db, counts


def test_ingest_counts_and_keys(loaded, synthetic_games):
    db, counts = loaded
    n = len(synthetic_games)
    assert counts["games"] == n
    assert counts["draft_actions"] == 20 * n
    assert counts["game_players"] == 10 * n
    # every game has a series, and game_id is unique
    assert db.execute("SELECT count(*) FROM games WHERE series_id IS NULL").fetchone()[0] == 0
    assert db.execute("SELECT count(DISTINCT game_id) FROM games").fetchone()[0] == n


def test_ingest_is_idempotent(loaded, synthetic_games):
    db, first = loaded
    second = store.ingest_matches(db, OraclesElixirSource(games=synthetic_games))
    assert second == first
    assert db.execute("SELECT count(*) FROM draft_actions").fetchone()[0] == first["draft_actions"]


def test_series_grouping_and_format(loaded):
    db, counts = loaded
    # the synthetic data plays 2-game series: every series has 2 games and is a best-of-3
    rows = db.execute("SELECT games_played, best_of, wins_a + wins_b FROM series").fetchall()
    assert all(g == 2 and bo == 3 and w == 2 for g, bo, w in rows)


def test_best_of_inference():
    assert store.best_of_from(1, 0, 1) == 1
    assert store.best_of_from(2, 0, 2) == 3
    assert store.best_of_from(2, 1, 3) == 3
    assert store.best_of_from(3, 1, 4) == 5


def test_fearless_detected_only_where_champions_never_repeat(loaded):
    db, _ = loaded
    flags = dict(db.execute("SELECT league || '-' || CAST(year(start_time) AS VARCHAR), bool_and(fearless) "
                            "FROM series GROUP BY 1").fetchall())
    assert flags["LCK-2026"] is True
    assert flags["LEC-2026"] is False


def test_series_state_tracks_locks_and_pools(loaded):
    db, _ = loaded
    sid = db.execute("SELECT series_id FROM series WHERE league = 'LCK' AND year(start_time) = 2026 LIMIT 1").fetchone()[0]
    st = store.series_state(db, sid)
    picks = set().union(*[set(g["picks"]["Blue"]) | set(g["picks"]["Red"]) for g in st["games"]])
    assert set(st["locked"]) == picks and len(picks) == 20     # 2 games x 10 picks, none repeated
    assert st["games_remaining"] == 1 and st["best_of"] == 3
    pool = list(picks)[:3] + ["NotPickedChamp"]
    assert store.remaining_pool(st, pool) == ["NotPickedChamp"]
    # a non-fearless series locks nothing
    sid2 = db.execute("SELECT series_id FROM series WHERE league = 'LEC' AND year(start_time) = 2026 LIMIT 1").fetchone()[0]
    assert store.series_state(db, sid2)["locked"] == []


def test_series_id_is_stable(synthetic_games):
    src = OraclesElixirSource(games=synthetic_games)
    a = {g["game_id"]: g["series_id"] for g in store.group_series(list(src.games()))}
    b = {g["game_id"]: g["series_id"] for g in store.group_series(list(src.games()))}
    assert a == b


TIMELINE = {"info": {"frames": [
    {"timestamp": 0, "events": [{"type": "WARD_PLACED", "timestamp": 75000, "creatorId": 4}]},
    {"timestamp": 60000, "events": [
        {"type": "CHAMPION_KILL", "timestamp": 185500, "killerId": 7, "victimId": 2, "position": {"x": 7000, "y": 7200}},
        {"type": "ELITE_MONSTER_KILL", "timestamp": 400000, "killerId": 2, "killerTeamId": 200,
         "monsterType": "DRAGON", "monsterSubType": "FIRE_DRAGON", "position": {"x": 9866, "y": 4414}},
        {"type": "BUILDING_KILL", "timestamp": 610000, "killerId": 1, "teamId": 200, "towerType": "OUTER_TURRET",
         "position": {"x": 4318, "y": 13875}},
        {"type": "ITEM_PURCHASED", "timestamp": 30000, "participantId": 9, "itemId": 1055},
        {"type": "SKILL_LEVEL_UP", "timestamp": 1000, "participantId": 1}]}]}}


def test_riot_timeline_events(tmp_path, db):
    (tmp_path / "G1.json").write_text(json.dumps(TIMELINE))
    src = RiotTimelineSource(tmp_path)
    ev = list(src.events("G1"))
    by = {e["event_type"]: e for e in ev}
    assert len(ev) == 5                                       # skill level-ups are ignored
    assert by["kill"]["game_time_s"] == 185 and by["kill"]["team"] == "Red" and by["kill"]["x"] == 7000
    assert by["objective"]["team"] == "Red" and by["objective"]["subtype"] == "fire_dragon"
    assert by["tower"]["team"] == "Blue"                       # Red's tower destroyed -> Blue's event
    assert by["ward"]["team"] == "Blue" and by["item"]["team"] == "Red"
    assert store.ingest_events(db, src) == 5
    assert store.ingest_events(db, src) == 5                   # re-ingest replaces, doesn't duplicate
    assert db.execute("SELECT count(*) FROM events").fetchone()[0] == 5


def test_vod_sync_handles_pauses(db):
    # video starts 100 s before the game; a 60 s pause happens between 600 s and 900 s of game time
    store.set_vod(db, "G1", "https://example.com/vod", [(0, 100), (600, 700), (900, 1060)])
    assert store.video_time(db, "G1", 300) == 400
    assert store.video_time(db, "G1", 750) == 850
    assert store.video_time(db, "G1", 1000) == 1160            # after the pause, offset is 160 s
    assert store.video_time(db, "unknown", 10) is None


def test_practice_series(db):
    sid = store.create_practice_series(db, "T1", "Gen.G", best_of=3, fearless=True)
    actions = [f"C{i}" for i in range(20)]
    store.record_practice_game(db, sid, "T1", "Gen.G", actions, blue_win=1)
    st = store.series_state(db, sid)
    assert st["games_played"] == 1 and st["wins_a"] + st["wins_b"] == 1
    assert len(st["locked"]) == 10 and not st["finished"]


def test_grid_needs_access(monkeypatch):
    monkeypatch.delenv("GRID_API_KEY", raising=False)
    with pytest.raises(SourceNotConfigured):
        list(GridSource().available_games())
