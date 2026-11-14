"""Tunable weighting parameters.

Defaults come from config.py; scripts/tune.py writes the tuned values to params.json in the project
root, which then overrides the defaults for build.py and the app.
"""
from __future__ import annotations

import json

from . import config

PARAMS_FILE = config.ROOT / "params.json"

DEFAULTS = {
    "half_life": config.HALF_LIFE_DAYS,                 # meta memory (days)
    "matchup_half_life": config.MATCHUP_HALF_LIFE_DAYS,  # counters / synergies memory (days)
    "patch_boost": config.PATCH_BOOST,                   # extra weight on/near the target patch
    "fearless_decay": None,                              # None = use config.FEARLESS_GAME_WEIGHTS
    "k_champ": config.K_CHAMP,                           # shrinkage, champion win rates
    "k_pair": config.K_PAIR,                             # shrinkage, synergies / whole-team matchups
    "k_lane": config.K_LANE,                             # shrinkage, lane counters
    "elo_k": 24.0,                                       # Elo update size
    "adjusted_strength": config.ADJUSTED_STRENGTH,       # team-adjusted champion strength (M8)
}


def load_params(path=None) -> dict:
    p = dict(DEFAULTS)
    f = path or PARAMS_FILE
    if f.exists():
        p.update(json.loads(f.read_text(encoding="utf-8")).get("params", {}))
    return p


def fearless_weights(p: dict) -> dict:
    d = p.get("fearless_decay")
    if d is None:
        return dict(config.FEARLESS_GAME_WEIGHTS)
    return {n: float(d) ** (n - 1) for n in range(1, 6)}


def snapshot_kwargs(p: dict) -> dict:
    return {"half_life": p["half_life"], "matchup_half_life": p["matchup_half_life"],
            "patch_boost": p["patch_boost"], "fearless_weights": fearless_weights(p),
            "k_champ": p["k_champ"], "k_pair": p["k_pair"], "k_lane": p["k_lane"],
            "adjusted_strength": bool(p.get("adjusted_strength", False))}
