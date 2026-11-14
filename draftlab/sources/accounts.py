"""Pro solo-queue account mapping (AccountSource) from a CSV, e.g. data/soloq/pro_accounts.csv.

The CSV can come from Leaguepedia, a hand-filled sheet, or any other mapping; this keeps the store
independent of where the mapping was found. Expected columns: player_key (or player), player_name,
riot_id, server.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

from ..config import ROOT


class CsvAccountSource:
    name = "accounts_csv"

    def __init__(self, path=None):
        self.path = Path(path or ROOT / "data" / "soloq" / "pro_accounts.csv")

    def accounts(self) -> Iterable[dict]:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                riot_id = r.get("riot_id") or (f"{r.get('game_name')}#{r.get('tag_line')}" if r.get("game_name") else None)
                if not riot_id:
                    continue
                yield {"player_key": r.get("player_key") or r.get("player") or r.get("player_name"),
                       "player_name": r.get("player_name") or r.get("player"),
                       "riot_id": riot_id, "server": r.get("server")}
