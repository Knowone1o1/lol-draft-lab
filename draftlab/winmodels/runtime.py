"""A6: the app's A3 + A4 win model at runtime, from models/a3a4_win.json (scripts/build_a3a4.py).

    logit P(blue wins) = scale * (side + first_pick * (+1 / -1) + score[blue] - score[red])     (A3 strength)
                       + sum_j beta_j * (x_j - mu_j) / sd_j                                     (A4 draft layer)

x = the app's 5 champion features, blue minus red (draftlab.features.comp_features without the Elo term).
A team the artifact doesn't know gets its home league's score if known, else 0 (an average team).
"level" = the draft alone at equal strength and no side / pick-order term: sum_j beta_j * x_j / sd_j, uncentred, so
two identical drafts give exactly 50% (A4).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _sig(z):
    return float(1 / (1 + np.exp(-z)))


class A3A4WinModel:
    name = "a3a4"

    def __init__(self, path: Path):
        art = json.loads(Path(path).read_text(encoding="utf-8"))
        self.art = art
        s, d = art["strength"], art["draft"]
        self.scale, self.side, self.fp = s["scale"], s["side"], s["first_pick"]
        self.score, self.home, self.league = s["score"], s.get("home", {}), s.get("league_score", {})
        self.beta, self.mu, self.sd = (np.array(d[k], float) for k in ("beta_per_sd", "mu", "sd"))

    def team_score(self, team: str) -> float:
        if team in self.score:
            return float(self.score[team])
        return float(self.league.get(self.home.get(team, ""), 0.0))

    def strength_logit(self, blue: str, red: str, blue_first: bool = True) -> float:
        return self.scale * (self.side + self.fp * (1.0 if blue_first else -1.0)
                             + self.team_score(blue) - self.team_score(red))

    def draft_logit(self, x) -> float:
        return float(((np.asarray(x, float) - self.mu) / self.sd) @ self.beta)

    def level_logit(self, x) -> float:
        return float((np.asarray(x, float) / self.sd) @ self.beta)

    def prob(self, blue: str, red: str, blue_first: bool, x) -> float:
        return _sig(self.strength_logit(blue, red, blue_first) + self.draft_logit(x))

    def level_prob(self, x) -> float:
        return _sig(self.level_logit(x))

    def describe(self) -> dict:
        s = self.art["strength"]
        return {"name": self.name, "built": self.art["built"], "data_to": self.art["data_to"],
                "strength": f"{s['model']} (side {s['side']:+.3f}, first pick {s['first_pick']:+.3f}, x{s['scale']:.3f})",
                "draft": self.art["draft"]["coef"]}
