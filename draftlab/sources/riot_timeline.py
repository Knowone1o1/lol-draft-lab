"""In-game events from Riot match-v5 timeline JSON files (EventSource).

Riot's documented timeline format (`/lol/match/v5/matches/{id}/timeline`) is what tournament-code games
(e.g. scrims) produce, and GRID distributes Riot-sourced game data for official matches in the same shape.
Put one file per game in a folder as `<game_id>.json` (the timeline payload, i.e. with "info" -> "frames").
An optional `<game_id>.participants.json` maps participantId -> {"team": "Blue"|"Red", "player": name};
otherwise participants 1-5 are Blue and 6-10 Red (Riot's convention).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

# Riot event type -> (our event_type, subtype getter)
_TYPE_MAP = {
    "CHAMPION_KILL": "kill",
    "ELITE_MONSTER_KILL": "objective",
    "BUILDING_KILL": "tower",
    "WARD_PLACED": "ward",
    "WARD_KILL": "ward",
    "ITEM_PURCHASED": "item",
    "TURRET_PLATE_DESTROYED": "tower",
}


def _team_of_id(team_id):
    return {100: "Blue", 200: "Red"}.get(team_id)


class RiotTimelineSource:
    name = "riot_timeline"

    def __init__(self, folder):
        self.folder = Path(folder)

    def available_games(self) -> Iterable[str]:
        for f in sorted(self.folder.glob("*.json")):
            if not f.name.endswith(".participants.json"):
                yield f.stem

    def _participants(self, game_id):
        f = self.folder / f"{game_id}.participants.json"
        if f.exists():
            raw = json.loads(f.read_text(encoding="utf-8"))
            return {int(k): v for k, v in raw.items()}
        return {i: {"team": "Blue" if i <= 5 else "Red", "player": None} for i in range(1, 11)}

    def events(self, game_id: str) -> Iterable[dict]:
        data = json.loads((self.folder / f"{game_id}.json").read_text(encoding="utf-8"))
        frames = data.get("info", data).get("frames", [])
        parts = self._participants(game_id)
        for fr in frames:
            for ev in fr.get("events", []):
                et = _TYPE_MAP.get(ev.get("type"))
                if et is None:
                    continue
                t = int(ev.get("timestamp", 0)) // 1000
                pos = ev.get("position") or {}
                actor = ev.get("killerId") or ev.get("creatorId") or ev.get("participantId")
                team = parts.get(actor, {}).get("team") if actor else None
                if ev.get("type") == "ELITE_MONSTER_KILL":
                    team = _team_of_id(ev.get("killerTeamId")) or team
                    sub = (ev.get("monsterSubType") or ev.get("monsterType") or "").lower()
                elif ev.get("type") in ("BUILDING_KILL", "TURRET_PLATE_DESTROYED"):
                    # teamId is the team that OWNED the building: the destroyer is the other team
                    owner = _team_of_id(ev.get("teamId"))
                    team = {"Blue": "Red", "Red": "Blue"}.get(owner, team)
                    sub = (ev.get("towerType") or ev.get("buildingType") or ev.get("type")).lower()
                elif ev.get("type") == "CHAMPION_KILL":
                    sub = "kill"
                elif et == "ward":
                    sub = ev.get("type").lower()
                else:
                    sub = str(ev.get("itemId", ""))
                yield {"game_id": game_id, "game_time_s": t, "event_type": et, "subtype": sub, "team": team,
                       "player": parts.get(actor, {}).get("player") if actor else None,
                       "x": pos.get("x"), "y": pos.get("y"),
                       "detail": json.dumps({k: ev[k] for k in ("victimId", "laneType", "assistingParticipantIds")
                                             if k in ev}) or None}
