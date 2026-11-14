"""Data-source interfaces. Every external source sits behind one of these, so sources can be swapped
(GRID, Riot API, Leaguepedia, gol.gg, local files) without touching the store or the models.

All records are plain dicts with the column names of the DuckDB store (draftlab/store.py).
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

EVENT_TYPES = ("kill", "objective", "ward", "tower", "item")


class SourceNotConfigured(RuntimeError):
    """The source needs access or credentials that are not set up (e.g. GRID without a key)."""


@runtime_checkable
class MatchSource(Protocol):
    """Games with drafts, players and results."""

    name: str

    def games(self) -> Iterable[dict]:
        """One dict per game with keys: game_id, series_key (league, teams, start time), game_number, date,
        league, patch, blue_team, red_team, blue_win, gamelength_s, actions (20 x (side, kind, champion|None)),
        players ({side: {role: (player_key, player_name, champion)}})."""
        ...


@runtime_checkable
class EventSource(Protocol):
    """In-game events with time and (where available) map position."""

    name: str

    def events(self, game_id: str) -> Iterable[dict]:
        """Dicts with keys: game_id, game_time_s, event_type (one of EVENT_TYPES), subtype, team ('Blue'/'Red'),
        player, x, y, detail."""
        ...

    def available_games(self) -> Iterable[str]:
        ...


@runtime_checkable
class AccountSource(Protocol):
    """Which solo-queue accounts belong to which pro player."""

    name: str

    def accounts(self) -> Iterable[dict]:
        """Dicts with keys: player_key, player_name, riot_id, server."""
        ...
