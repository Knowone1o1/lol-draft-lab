"""Comp labels from the team's champion tag sheet (data/comp_labels.xlsx).

Each champion has a primary style, a secondary style and a timing (Early / Mid / tempo / Late).
A comp's timing score is the average over its five champions (Early = 0, Mid = 1, Late = 2), so
0 is an all-in early comp and 2 a full scaling comp. Its main style is the style with the most
points (primary = 2, secondary = 1). The Comp Types sheet supplies each archetype's win condition
and early-game plan, which the briefing quotes.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache

import pandas as pd

from .config import ROOT

SHEET = ROOT / "data" / "comp_labels.xlsx"
TIMING_SCORE = {"early": 0.0, "mid / tempo": 1.0, "mid": 1.0, "late": 2.0}
EARLY_TEXT = {"lvl1": "Lvl 1", "laning": "Laning (0–10 min)", "spike": "Early spike"}
EARLY_SCORE = "Early score (0–6)"
EARLY_TRAITS = {"Engage": "engage", "Peel": "peel", "Dive threat": "dive", "Poke": "poke", "Catch CC": "catch",
                "Wave clear": "waveclear", "Roam / gank": "roam", "Invade": "invade"}


def _norm(s) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


@lru_cache(maxsize=1)
def load_sheet(path: str | None = None):
    p = path or SHEET
    tags = pd.read_excel(p, sheet_name="Champion Tags")
    tags = tags[tags["Champion"].notna()]
    champ = {}
    # Early-game columns (used by the early-game rules). A yes/no trait column with no "yes" anywhere in the sheet
    # has not been labelled yet, so its values are read as unknown (None) rather than "no".
    labelled = {c: bool(tags[c].fillna(False).astype(bool).any()) for c in EARLY_TRAITS if c in tags}
    for r in tags.to_dict("records"):
        champ[_norm(r["Champion"])] = {
            "name": r["Champion"], "role": r.get("Main role"),
            "primary": r.get("Primary style") if isinstance(r.get("Primary style"), str) else None,
            "secondary": r.get("Secondary style") if isinstance(r.get("Secondary style"), str) else None,
            "timing": r.get("Timing") if isinstance(r.get("Timing"), str) else None,
            "early": {
                **{k: (r.get(col).strip() if isinstance(r.get(col), str) and r.get(col).strip() else None)
                   for k, col in EARLY_TEXT.items()},
                "early_score": (float(r.get(EARLY_SCORE)) if pd.notna(r.get(EARLY_SCORE)) else None),
                **{key: (bool(r.get(col)) if labelled.get(col) and pd.notna(r.get(col)) else None)
                   for col, key in EARLY_TRAITS.items()},
            },
        }
    types = pd.read_excel(p, sheet_name="Comp Types")
    types = types[types["Archetype"].notna()]
    archetypes = {r["Archetype"]: {k: r.get(k) for k in ("Axis", "Win condition", "Early game plan", "Labeling signals",
                                                         "Typical champions", "Strong vs", "Weak vs")}
                  for r in types.to_dict("records")}
    return champ, archetypes


def champ_tag(name):
    champ, _ = load_sheet()
    return champ.get(_norm(name))


def timing_label(score: float) -> str:
    if score < 0.8:
        return "Early"
    if score > 1.2:
        return "Late"
    return "Mid / tempo"


STYLE_KEYWORDS = {
    "Front-to-back": ["front-to-back", "front to back", "teamfight", "grouped 5", "slow teamfight", "slow comps"],
    "Hard engage": ["hard engage", "engage"],
    "Dive": ["dive"],
    "Poke / siege": ["poke", "siege"],
    "Pick": ["pick"],
    "Split push": ["split push", "split-push"],
    "Protect the carry": ["protect the carry", "protect"],
    "Disengage": ["disengage"],
    "Skirmish": ["skirmish"],
}


def _styles_in(text) -> set:
    if not isinstance(text, str):
        return set()
    t = text.lower()
    found = set()
    for s, kws in STYLE_KEYWORDS.items():
        if any(k in t for k in kws):
            # "hard engage" also contains "engage"; "disengage" contains "engage" too
            if s == "Hard engage" and "hard engage" not in t and ("disengage" in t or "engage" not in t.replace("disengage", "")):
                continue
            found.add(s)
    return found


@lru_cache(maxsize=1)
def style_matrix():
    """M[a][b] = +1 if the sheet says style a is strong vs b, -1 if weak vs b, combining both rows
    (a strong vs b and b weak vs a reinforce each other). Returns (styles, numpy matrix)."""
    import numpy as np
    _, arche = load_sheet()
    styles = [s for s, v in arche.items() if v.get("Axis") == "Fight style"]
    idx = {s: i for i, s in enumerate(styles)}
    M = np.zeros((len(styles), len(styles)))
    for s in styles:
        for t in _styles_in(arche[s].get("Strong vs")):
            if t in idx and t != s:
                M[idx[s], idx[t]] += 1
                M[idx[t], idx[s]] -= 1
        for t in _styles_in(arche[s].get("Weak vs")):
            if t in idx and t != s:
                M[idx[s], idx[t]] -= 1
                M[idx[t], idx[s]] += 1
    return styles, np.clip(M, -1, 1)


def style_advantage(sv, osv, styles_order) -> float:
    """Sheet-based style matchup score: own style mix x matrix x enemy style mix (between -1 and +1)."""
    import numpy as np
    styles, M = style_matrix()
    pos = [styles.index(s) if s in styles else None for s in styles_order]
    keep = [i for i, p in enumerate(pos) if p is not None]
    Ms = M[np.ix_([pos[i] for i in keep], [pos[i] for i in keep])]
    return float(np.asarray(sv)[keep] @ Ms @ np.asarray(osv)[keep])


def comp_profile(champs: list) -> dict:
    """Timing score, timing label and main/backup style of a (possibly partial) comp."""
    _, arche = load_sheet()
    scores, styles, missing = [], Counter(), []
    per = []
    for c in champs:
        t = champ_tag(c)
        if not t:
            missing.append(c)
            continue
        if t["timing"] and t["timing"].lower() in TIMING_SCORE:
            scores.append(TIMING_SCORE[t["timing"].lower()])
        if t["primary"]:
            styles[t["primary"]] += 2
        if t["secondary"] and t["secondary"] != "None":
            styles[t["secondary"]] += 1
        per.append({"champion": c, "timing": t["timing"], "primary": t["primary"], "secondary": t["secondary"]})
    score = sum(scores) / len(scores) if scores else 1.0
    ranked = [s for s, _ in styles.most_common()]
    label = timing_label(score)
    return {"timing_score": round(score, 3), "timing": label, "main_style": ranked[0] if ranked else None,
            "backup_style": ranked[1] if len(ranked) > 1 else None, "style_points": dict(styles),
            "champions": per, "untagged": missing,
            "plan_timing": arche.get(label, {}), "plan_style": arche.get(ranked[0], {}) if ranked else {}}
