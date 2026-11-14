"""15-minute game plan: what a team needs by minute 15, given BOTH comps' timing and style.

Model (fitted by scripts/build_brief.py, which picks the variant that predicts best on recent games):
    timing:            win ~ gold@15 + own timing + enemy timing + gold x own timing + gold x enemy timing + Elo
    timing+style:      ... + style-matchup score from the team's Comp Types sheet (Strong vs / Weak vs)
    timing+style+gold: ... + style score x gold
    timing+learned:    ... + style score + learned weights on the style mix (own minus enemy)
Timing and style mix come from the champion tag sheet (draftlab/comp.py).
Win condition (draftlab/wincon.py): which champions need a lead, game length, objective path.
"""
from __future__ import annotations

import math
import pickle

import numpy as np
import pandas as pd

from .comp import comp_profile, style_advantage, style_matrix
from .config import MODEL_DIR, ROLE_LABELS
from .wincon import STYLES, carries_for, describe_path, style_vector

OBJ_COLS = ["first_tower", "grubs", "first_dragon"]
SCENARIOS = [
    {"key": "bad", "label": "Bad case", "gold": -2500},
    {"key": "even", "label": "Even", "gold": 0},
    {"key": "good", "label": "Good case", "gold": 2000},
    {"key": "best", "label": "Best case", "gold": 4000},
]
TIMING_POINTS = {"Early": 0.0, "Mid / tempo": 1.0, "Late": 2.0}
VARIANTS = {
    "timing": "Gold at 15, both comps' timing, gold × timing, Elo",
    "timing+style": "+ the sheet's style-matchup score",
    "timing+style+gold": "+ style-matchup score and style score × gold",
    "timing+learned": "+ style-matchup score and learned weights on the style mix",
}


def features(df, variant: str = "timing") -> np.ndarray:
    g = np.asarray(df["gd"], float) / 1000
    o = np.asarray(df["own"], float) - 1
    e = np.asarray(df["opp"], float) - 1
    elo = np.asarray(df["elo"], float) / 400
    cols = [g, o, e, g * o, g * e, elo]
    if variant != "timing":
        s = np.asarray(df["sadv"], float)
        cols.append(s)
        if variant == "timing+style+gold":
            cols.append(g * s)
        if variant == "timing+learned":
            diff = np.vstack(df["sv"].to_numpy()) - np.vstack(df["osv"].to_numpy())
            cols += [diff[:, i] for i in range(diff.shape[1])]
    return np.column_stack(cols)


def base_features(df) -> np.ndarray:
    return features(df, "timing")


def obj_features(df, variant: str = "timing") -> np.ndarray:
    return np.column_stack([features(df, variant)] + [np.asarray(df[c], float) for c in OBJ_COLS])


def _row(gold, own, opp, elo, sadv=0.0, sv=None, osv=None):
    z = np.zeros(len(STYLES))
    return pd.DataFrame({"gd": [gold], "own": [own], "opp": [opp], "elo": [elo], "sadv": [sadv],
                         "sv": [z if sv is None else sv], "osv": [z if osv is None else osv]})


def _line(model, variant, own, opp, elo, sadv=0.0, sv=None, osv=None):
    """Log-odds is linear in gold: return (intercept, slope per 1,000 gold)."""
    a = model.decision_function(features(_row(0, own, opp, elo, sadv, sv, osv), variant))[0]
    b = model.decision_function(features(_row(1000, own, opp, elo, sadv, sv, osv), variant))[0] - a
    return a, b


def break_even(model, own, opp, elo_diff=0.0, variant="timing", sadv=0.0, sv=None, osv=None) -> float:
    """Gold at 15 where the win chance is 50%."""
    a, b = _line(model, variant, own, opp, elo_diff, sadv, sv, osv)
    return -a / b * 1000


def gold_for(model, own, opp, elo_diff, p, variant="timing", sadv=0.0, sv=None, osv=None) -> float:
    a, b = _line(model, variant, own, opp, elo_diff, sadv, sv, osv)
    return (math.log(p / (1 - p)) - a) / b * 1000


class BriefEngine:
    def __init__(self, path=None):
        with open(path or MODEL_DIR / "brief.pkl", "rb") as f:
            art = pickle.load(f)
        self.m = art["model"]
        self.mo = art["obj_model"]
        self.variant = art.get("variant", "timing")
        self.carry = art.get("carry_table")
        self.similar = art.get("similar")

    def win_prob(self, gold, own, opp, elo_diff, sadv=0.0, sv=None, osv=None):
        X = features(_row(gold, own, opp, elo_diff, sadv, sv, osv), self.variant)
        return float(self.m.predict_proba(X)[0, 1])

    def objective_value(self, gold, own, opp, elo_diff, sadv=0.0, sv=None, osv=None):
        """Win-chance gain from each objective at this gold, vs not getting it (others at 50%)."""
        out = {}
        for c in OBJ_COLS:
            rows = []
            for v in (1.0, 0.0):
                d = _row(gold, own, opp, elo_diff, sadv, sv, osv)
                for k in OBJ_COLS:
                    d[k] = [v if k == c else 0.5]
                rows.append(self.mo.predict_proba(obj_features(d, self.variant))[0, 1])
            out[c] = rows[0] - rows[1]
        return out

    def brief(self, side: str, own_roles: dict, enemy_roles: dict, elo_diff: float, forecast: dict | None = None,
              team: str = "", enemy: str = "") -> dict:
        """own_roles / enemy_roles: {role: champion}. elo_diff and forecast gold are from `side`'s view."""
        own_champs, enemy_champs = [c for c in own_roles.values() if c], [c for c in enemy_roles.values() if c]
        me, them = comp_profile(own_champs), comp_profile(enemy_champs)
        own, opp = me["timing_score"], them["timing_score"]
        sv, osv = style_vector(me), style_vector(them)
        sadv = style_advantage(sv, osv, STYLES)
        kw = {"variant": self.variant, "sadv": sadv, "sv": sv, "osv": osv}
        be = break_even(self.m, own, opp, elo_diff, **kw)
        comfy = gold_for(self.m, own, opp, elo_diff, 0.65, **kw)
        rel = own - opp                                   # < 0: we are the earlier comp
        scen = []
        for s in SCENARIOS:
            p = self.win_prob(s["gold"], own, opp, elo_diff, sadv, sv, osv)
            scen.append({**s, "win_prob": p, "advice": self._advice(s["key"], p, rel, me, them, be)})
        fc = None
        if forecast:
            fg = forecast["team_gold15"] * (1 if side == "Blue" else -1)
            fc = {"gold": int(fg), "win_prob": self.win_prob(fg, own, opp, elo_diff, sadv, sv, osv)}
        obj_val = self.objective_value(max(0.0, be), own, opp, elo_diff, sadv, sv, osv)
        a, b = _line(self.m, self.variant, own, opp, elo_diff, sadv, sv, osv)
        curve = [{"gold": gg, "p": float(1 / (1 + math.exp(-(a + b * gg / 1000))))} for gg in range(-6000, 6001, 250)]
        return {
            "curve": curve,
            "side": side, "team": team, "enemy": enemy,
            "own_comp": _public(me), "enemy_comp": _public(them),
            "relative": "earlier" if rel < -0.25 else "later" if rel > 0.25 else "similar",
            "style": self._style_text(sv, osv, sadv), "model_variant": self.variant,
            "break_even_gold": int(round(be)), "comfortable_gold": int(round(comfy)),
            "scenarios": scen, "forecast": fc,
            "objective_value": {k: round(float(v), 4) for k, v in obj_val.items()},
            "plan": self._plan(me, them, forecast, side),
            "win_condition": self._wincon(own_roles, own, sv, opp),
        }

    # ----------------------------------------------------------------- win condition
    def _wincon(self, own_roles, own_ts, sv, opp_ts):
        out = {"carries": [], "timing": None, "timing_text": None, "objective_text": None, "objectives": []}
        if self.carry is not None:
            top, _ = carries_for(own_roles, self.carry, top=2)
            out["carries"] = top
        if self.similar is not None:
            out.update(describe_path(self.similar.query(own_ts, sv, opp_ts), self.similar.base))
        return out

    @staticmethod
    def _style_text(sv, osv, sadv):
        styles, M = style_matrix()
        pairs = []
        for i, a in enumerate(STYLES):
            for j, b in enumerate(STYLES):
                if a in styles and b in styles:
                    m = M[styles.index(a), styles.index(b)]
                    c = sv[i] * osv[j] * m
                    if abs(c) > 0.01:
                        pairs.append((c, a, b))
        pairs.sort(key=lambda t: -abs(t[0]))
        good = [f"your {a} vs their {b}" for c, a, b in pairs if c > 0][:2]
        bad = [f"your {a} vs their {b}" for c, a, b in pairs if c < 0][:2]
        return {"score": round(float(sadv), 3), "favoured": good, "unfavoured": bad}

    # ----------------------------------------------------------------- words
    @staticmethod
    def _advice(key, p, rel, me, them, be):
        earlier, later = rel < -0.25, rel > 0.25
        pct = f"{p:.0%}"
        if key == "best":
            if earlier:
                return f"{pct}. Close it: your comp is strongest now, so turn the lead into towers, drakes and Baron and end before their scaling arrives."
            if later:
                return f"{pct}. You are ahead and still scaling: no need to force. Take safe objectives, keep vision and let your item spikes win the fights."
            return f"{pct}. Press the lead at your next item spike: deny their jungle, take drakes 2–3 and set up Baron with priority."
        if key == "good":
            if earlier:
                return f"{pct}. Good but not safe: a late enemy comp keeps getting stronger. Keep the tempo by trading every free objective and set up Baron before 25 minutes."
            if later:
                return f"{pct}. Comfortable for a scaling comp. Avoid coin-flip fights, keep farming, and fight only around drakes when your carries have items."
            return f"{pct}. Play to your 2-item spike and use priority to set up Herald and the next drakes."
        if key == "even":
            if earlier and be > 150:
                return (f"{pct}. Even is a loss of tempo for an earlier comp: you need about {be:+,.0f} gold at 15 to be a coin flip. "
                        "Force plays before 15: level 1–3 jungle pressure, dive the weak lane, take grubs and drake 1.")
            if earlier:
                return (f"{pct}. You are the stronger team, which covers the timing gap at even gold, but the enemy comp gains "
                        "from here: still use your early window for grubs, drake 1 and a tower.")
            if later and p < 0.45:
                return (f"{pct}. Your comp scales better, but the enemy is the stronger team: stay even, avoid early fights, "
                        "and the gap closes as your items come online.")
            if later:
                return (f"{pct}. Even is good for you: the enemy needed a lead and didn't get it. Trade objectives cheaply, avoid fights, "
                        "and wait for your 3-item spike.")
            if p < 0.45:
                return (f"{pct}. Similar comp timing, but the enemy is the stronger team, so even gold leaves you the underdog: "
                        f"you need about {be:+,.0f} at 15. Look for an early edge in your strongest lane and take the first drake.")
            if p > 0.55:
                return (f"{pct}. Similar comp timing and you are the stronger team, so even gold favours you. Hold lanes even "
                        "and fight at the first 2-item spike around drake 2 or Herald.")
            return f"{pct}. A real coin flip. Hold lanes even and fight at the first 2-item spike around drake 2 or Herald."
        # bad
        if earlier:
            return (f"{pct}. Your early window has gone: an early comp that is behind rarely outscales. Look for picks and "
                    "cross-map trades rather than straight 5v5s, and keep the game messy.")
        if later:
            return (f"{pct}. Still playable for a scaling comp. Give up early objectives cheaply, protect your carry's farm, "
                    "and don't take fights before your item spikes.")
        return f"{pct}. Stop the bleeding: concede the side objective, group for the next drake with vision first, and look for a pick."

    @staticmethod
    def _plan(me, them, forecast, side):
        lines = []
        pt, ps = me.get("plan_timing") or {}, me.get("plan_style") or {}
        if pt.get("Early game plan"):
            lines.append(f"{me['timing']} comp: {pt['Early game plan']}.")
        if ps.get("Early game plan") and me.get("main_style"):
            lines.append(f"{me['main_style']}: {ps['Early game plan']}.")
        if ps.get("Win condition"):
            lines.append(f"Win condition: {ps['Win condition']}.")
        ts = them.get("plan_style") or {}
        if them.get("main_style") and ts.get("Weak vs"):
            lines.append(f"Enemy is {them['main_style']} ({them['timing']}); it is weak vs {ts['Weak vs']}.")
        if forecast:
            sign = 1 if side == "Blue" else -1
            lanes = sorted(forecast["lanes"], key=lambda l: -sign * l["gold15"])
            best, worst = lanes[0], lanes[-1]
            if sign * best["gold15"] > 150:
                lines.append(f"Play through {ROLE_LABELS[best['role']].lower()} (forecast {sign * best['gold15']:+,} gold at 15).")
            if sign * worst["gold15"] < -150:
                lines.append(f"Cover {ROLE_LABELS[worst['role']].lower()} (forecast {sign * worst['gold15']:+,}): jungle presence or a safe farm plan.")
            for o in forecast.get("objectives", []):
                pr = o["blue_prob"] if side == "Blue" else 1 - o["blue_prob"]
                if o["key"] in ("grubs", "first_dragon") and pr >= 0.55:
                    lines.append(f"{o['label']}: favoured ({pr:.0%}), so set up for it.")
        return lines


def _public(p):
    return {k: p[k] for k in ("timing_score", "timing", "main_style", "backup_style", "untagged")}
