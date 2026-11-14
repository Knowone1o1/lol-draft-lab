"""Oracle's Elixir match CSVs as a MatchSource (drafts, players, results; no in-game events)."""
from __future__ import annotations

import math
from typing import Iterable

import pandas as pd

from ..config import ROLES, SIDES
from ..data import build_games, game_actions, load_raw


def _clean(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


class OraclesElixirSource:
    name = "oracles_elixir"

    def __init__(self, paths=None, games: pd.DataFrame | None = None):
        self._paths = paths
        self._games = games

    def frame(self) -> pd.DataFrame:
        if self._games is None:
            self._games = build_games(load_raw(self._paths))
        return self._games

    def games(self) -> Iterable[dict]:
        g = self.frame()
        for r in g.to_dict("records"):
            yield {
                "game_id": str(r["gameid"]),
                "league": _clean(r.get("league")),
                "date": pd.Timestamp(r["date"]).to_pydatetime(),
                "game_number": int(r["game"]) if _clean(r.get("game")) is not None else None,
                "patch": _clean(r.get("patch")),
                "blue_team": r["Blue_teamname"], "red_team": r["Red_teamname"],
                "blue_win": int(r["blue_win"]),
                "gamelength_s": None,
                "actions": game_actions(r),
                "players": {s: {ro: (r[f"{s}_{ro}_player"], _clean(r.get(f"{s}_{ro}_pname")) or r[f"{s}_{ro}_player"],
                                     r[f"{s}_{ro}_champ"]) for ro in ROLES} for s in SIDES},
            }
