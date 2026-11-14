"""GRID official esports data (EventSource / MatchSource) - placeholder until access exists.

GRID is Riot's exclusive official esports data partner; teams get post-game data through the LoL Data
Portal and commercial clients through the API. When access is available, implement `events()` here so
it yields the same dicts as RiotTimelineSource (GRID's Riot-sourced game data uses the same event types).
The API key is read from the GRID_API_KEY environment variable / .env file and never stored in code.
"""
from __future__ import annotations

import os
from typing import Iterable

from .base import SourceNotConfigured


class GridSource:
    name = "grid"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GRID_API_KEY")

    def _require(self):
        if not self.api_key:
            raise SourceNotConfigured("GRID access is not configured: set GRID_API_KEY (requires an org's GRID / "
                                      "LoL Data Portal access). Until then, load timelines with RiotTimelineSource.")

    def available_games(self) -> Iterable[str]:
        self._require()
        raise NotImplementedError("GRID series listing: implement once the API contract is available.")

    def events(self, game_id: str) -> Iterable[dict]:
        self._require()
        raise NotImplementedError("GRID events: map GRID's game data to the EventSource dict format.")
