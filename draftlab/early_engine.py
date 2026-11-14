"""Runtime early-game forecast for the draft app (uses models/early.pkl from scripts/build_early.py)."""
from __future__ import annotations

import pickle

import numpy as np

from .config import MODEL_DIR, ROLE_LABELS, ROLES
from .early import OBJECTIVES, game_lane_matrix, objective_features, team_features

# Typical miss of the forecasts on held-out games (reports/early_game.md): mean absolute error in gold.
TYPICAL_ERROR = {"lane": 700, "team": 2000}
LOW_DATA_LEAGUES = {"LPL"}
OBJ_LABEL = {"first_dragon": "First dragon", "grubs": "Void grubs", "first_blood": "First blood",
             "first_tower": "First tower"}


class EarlyEngine:
    def __init__(self, path=None):
        with open(path or MODEL_DIR / "early.pkl", "rb") as f:
            art = pickle.load(f)
        self.es = art["stats"]
        self.lane = art["lane"]
        self.team = art["team"]
        self.obj = art["obj"]
        from .config import league_elo
        lg = league_elo()
        self.elo = {t: v + lg["offsets"].get(lg["home"].get(t), 0.0) for t, v in art["elo"].items()}
        self.home = lg["home"]
        self.conv = art.get("league_conversion")
        self.comeback = art.get("league_comeback")

    def forecast(self, blue_team, red_team, rosters: dict, roles: dict, names=None) -> dict:
        """roles: {side: {champion: role}} (from DraftEngine.roles); rosters: {side: {role: player key}}."""
        names = names or (lambda p: p)
        picks = {s: {r: c for c, r in roles.get(s, {}).items()} for s in ("Blue", "Red")}
        elo_diff = self.elo.get(blue_team, 1500.0) - self.elo.get(red_team, 1500.0)
        LX = game_lane_matrix(self.es, picks, rosters, (blue_team, red_team), elo_diff)
        es = self.es
        lanes = []
        for i, r in enumerate(ROLES):
            b, c = picks["Blue"].get(r), picks["Red"].get(r)
            gold = float(self.lane[r].predict(LX[i:i + 1])[0])
            reasons = []
            for ch in (b, c):
                if ch:
                    s = es.champ_lane(ch, r)
                    if abs(s) >= 100:
                        reasons.append(f"{ch}: averages {s:+,.0f} gold at 15 in {ROLE_LABELS[r].lower()}")
            if b and c:  # one matchup line per lane, from the side that benefits
                m = es.matchup_lift(b, c, r)
                n = es.matchup_n.get((b, c, r), 0)
                if abs(m) >= 150 and n >= 2:
                    win_c, lose_c = (b, c) if m > 0 else (c, b)
                    reasons.append(f"{win_c} does {abs(m):,.0f} better into {lose_c} than their strength suggests (~{n:.0f} weighted games)")
            for side in ("Blue", "Red"):
                p = rosters.get(side, {}).get(r)
                v = es.player_lane(p, r)
                if p and abs(v) >= 150:
                    reasons.append(f"{names(p)}: averages {v:+,.0f} gold at 15")
            lanes.append({"role": r, "label": ROLE_LABELS[r], "blue": b, "red": c, "gold15": round(gold),
                          "complete": bool(b and c), "reasons": reasons[:4]})
        # Team gold shown = sum of the lane forecasts, so the numbers on screen add up (judge report M10).
        team_gold = float(sum(l["gold15"] for l in lanes))
        objectives = []
        for o in OBJECTIVES:
            if o in self.obj:
                x = objective_features(es, o, picks, (blue_team, red_team), LX)
                objectives.append({"key": o, "label": OBJ_LABEL[o], "blue_prob": float(self.obj[o].predict_proba(x[None, :])[0, 1])})
        # Oracle's Elixir has gold-at-15 for only ~15% of LPL games, so LPL forecasts rest on other leagues.
        low = [t for t in (blue_team, red_team) if self.home.get(t) in LOW_DATA_LEAGUES]
        return {"lanes": lanes, "team_gold15": round(team_gold), "objectives": objectives,
                "typical_error": TYPICAL_ERROR, "low_data": low,
                "context": {"conversion": self.conv, "comeback": self.comeback}}
