"""Phase 1 endpoints, called directly against temporary DuckDB files."""
import pytest
from fastapi import HTTPException

from draftlab import store
from draftlab.sources import OraclesElixirSource


@pytest.fixture()
def api(tmp_path, monkeypatch, synthetic_games):
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "pro.duckdb")
    monkeypatch.setattr(store, "PRACTICE_DB", tmp_path / "practice.duckdb")
    with store.connect(store.DB_PATH) as con:
        store.ingest_matches(con, OraclesElixirSource(games=synthetic_games))
    import app.server as server
    return server


def test_list_and_get_series(api):
    rows = api.list_series(league="LCK", limit=5)
    assert len(rows) == 5 and all(r["league"] == "LCK" for r in rows)
    st = api.get_series(rows[0]["series_id"])
    assert st["series_id"] == rows[0]["series_id"] and len(st["games"]) == st["games_played"]
    with pytest.raises(HTTPException):
        api.get_series("nope")


def test_get_game(api):
    sid = api.list_series(limit=1)[0]["series_id"]
    gid = api.get_series(sid)["games"][0]["game_id"]
    g = api.get_game(gid)
    assert len(g["draft"]) == 20 and len(g["players"]) == 10 and g["vod"] is None


def test_practice_series_flow_and_fearless_guard(api):
    st = api.new_practice_series(api.PracticeSeries(team_a="T1", team_b="Gen.G", best_of=3, fearless=True))
    sid = st["series_id"]
    acts = [f"C{i}" for i in range(20)]
    st = api.add_practice_game(sid, api.PracticeGame(blue_team="T1", red_team="Gen.G", actions=acts, blue_win=1))
    assert st["games_played"] == 1 and len(st["locked"]) == 10
    with pytest.raises(HTTPException) as e:   # a locked champion can't be used again
        api.add_practice_game(sid, api.PracticeGame(blue_team="Gen.G", red_team="T1", actions=acts, blue_win=0))
    assert "locked by fearless" in str(e.value.detail)
    assert api.get_series(sid)["games_played"] == 1
