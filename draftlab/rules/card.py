"""Plan card: apply the early-game rules to one team's facts and lay the result out by section.

Every line carries the rule ID, the plan, the why, the condition in plain words with the actual values that made it
fire, the source IDs and the rule's evidence label (Part 3). Rules whose facts aren't labelled yet are listed as
"needs labels" (with the missing tags), never silently dropped or guessed.
"""
from __future__ import annotations

from .condition import evaluate, plain, substitute
from .facts import LEVELS, fact_spec

LANE_WORD = {"top": "Top", "mid": "Mid", "bot": "Bot"}
EVIDENCE_DEFAULT = {True: "Not tested yet", False: "Not testable from match data"}


def evidence_for(rule: dict, evidence: dict | None):
    """(label, detail) for a rule: from data/rules/evidence.json (Part 3) when present, else a default."""
    ev = (evidence or {}).get(rule["id"])
    if isinstance(ev, str):
        return ev, None
    if isinstance(ev, dict):
        if ev.get("effect") == "–":
            return ev["label"], f"Unclear: too few past games where the rule applied. Test: {ev['test']}"
        if ev.get("effect"):
            return ev["label"], (f"{ev['label']}: {ev['effect']} {ev['ci']} over {ev['n_games']:,} past games. "
                                 f"Test: {ev['test']}")
        return ev.get("label", "Not tested"), ev.get("why")
    return EVIDENCE_DEFAULT[rule["oe_testable"]], None


def _fill(text, lane):
    return text.replace("{lane}", lane) if (text and lane) else text


def _values(tree, facts, why):
    """The facts a condition used, with their values, in words: 'our top lane priority = Weak (forecast -950 ...)'."""
    out, seen = [], set()

    def walk(t):
        if "fact" in t:
            names = [t["fact"]] + ([t["value"]["ref"]] if isinstance(t["value"], dict) else [])
            for n in names:
                if n in seen:
                    continue
                seen.add(n)
                spec = fact_spec(n)
                v = facts.get(n)
                item = {"fact": n, "words": spec[3] if spec else n,
                        "value": "not labelled yet" if v is None else ("yes" if v is True else "no" if v is False else v)}
                if why.get(n):
                    item["how"] = why[n]
                out.append(item)
        for k in ("and", "or"):
            for x in t.get(k, []):
                walk(x)
        if "not" in t:
            walk(t["not"])

    walk(tree)
    return out


def _missing_words(missing):
    return [fact_spec(m)[3] if fact_spec(m) else m for m in missing]


def evaluate_rules(rules_data: dict, facts: dict, why: dict | None = None, evidence: dict | None = None):
    """All draft rules evaluated for one team -> list of line dicts (status: applies | needs_labels | unless | off)."""
    why = why or {}
    evidence = evidence or {}
    lines = []
    for r in rules_data["rules"]:
        if r["kind"] != "draft":
            continue
        lanes = r["for_each"].get("lane") or [None]
        for lane in lanes:
            when = substitute(r["when"], {"lane": lane}) if lane else r["when"]
            res, missing = evaluate(when, facts)
            line = {
                "rule": r["id"], "section": r["section"], "lane": lane,
                "then": (f"{LANE_WORD[lane]} lane: " if lane else "") + r["then"],
                "why": r["why"], "if_text": r["if_text"], "condition": plain(when),
                "values": _values(when, facts, why), "live_check": r["live_check"],
                "approximation": r["approximation"], "sources": r["sources"],
            }
            line["evidence"], line["evidence_detail"] = evidence_for(r, evidence)
            if res is None:
                line.update(status="needs_labels", missing=_missing_words(missing))
            elif res is False:
                line["status"] = "off"
            else:
                line["status"] = "applies"
                if r.get("unless"):
                    unless = substitute(r["unless"], {"lane": lane}) if lane else r["unless"]
                    u, umiss = evaluate(unless, facts)
                    line["unless"] = plain(unless)
                    line["unless_values"] = _values(unless, facts, why)
                    if u is True:
                        line["status"] = "unless"
                    elif u is None:
                        line["unless_missing"] = _missing_words(umiss)
            lines.append(line)
    # overrides: a winner that applies strikes through the loser's lines
    for c in rules_data.get("conflicts", []):
        winners = [l for l in lines if l["rule"] == c["winner"] and l["status"] == "applies"]
        if not winners:
            continue
        for l in lines:
            if l["rule"] == c["loser"] and l["status"] == "applies":
                l["status"] = "overridden"
                l["overridden_by"] = c["winner"]
                l["override_reason"] = (f"{c['winner']} applies (" + "; ".join(w["condition"] for w in winners) +
                                        f"), and the rules sheet says {c['winner']} is an exception to {c['loser']}.")
    return lines


def jungle_plan(facts: dict, lines: list) -> dict:
    """Start side and path in words, from R08 (lanes with priority) or R20 (cover a Weak lane)."""
    def idx(lane):
        v = facts.get(f"our.{lane}.prio")
        return LEVELS.index(v) if v in LEVELS else None

    r20 = [l for l in lines if l["rule"] == "R20" and l["status"] == "applies"]
    if r20:
        weak = [l["lane"] for l in r20]
        side_lanes = [w for w in weak if w in ("top", "bot")]
        if len(side_lanes) == 1:
            s = side_lanes[0]
            return {"rule": "R20", "start": f"{s} side", "path": f"Start {s} side to cover the Weak {s} lane, then path out.",
                    "finish": None}
        if side_lanes:
            return {"rule": "R20", "start": None, "finish": None,
                    "path": "Both side lanes are Weak: pick the lane the enemy jungler is likelier to gank and start there."}
        return {"rule": "R20", "start": None, "finish": "mid",
                "path": "Mid is Weak: path through mid early (river side of your first camps) to cover it."}
    t, b = idx("top"), idx("bot")
    if t is None or b is None:
        return {"rule": "R08", "start": None, "finish": None, "path": "Lane priority unknown: no start side suggested."}
    if t > b:
        return {"rule": "R08", "start": "bot side", "finish": "top side",
                "path": "Start bot side and finish top side, where your lane has priority."}
    if b > t:
        return {"rule": "R08", "start": "top side", "finish": "bot side",
                "path": "Start top side and finish bot side, where your lane has priority."}
    return {"rule": "R08", "start": "top side", "finish": "bot side",
            "path": "No side lane has more priority than the other: start top side and finish bot side, toward "
                    "Dragon (R19's default first objective)."}


def objective_plan(lines: list) -> dict:
    """First objective in one sentence, saying which rule decided it."""
    st = {l["rule"]: l["status"] for l in lines if l["lane"] is None}
    if st.get("R12") == "applies":
        return {"rule": "R12", "choice": "Trade",
                "text": "Don't contest the first objectives: trade on the opposite side and protect your scaling carry."}
    if st.get("R11") == "applies":
        return {"rule": "R11", "choice": "Both",
                "text": "Contest first Grubs and first Dragon: your early comp must convert priority before it falls off."}
    if st.get("R19") == "unless":
        return {"rule": "R19", "choice": "Grubs", "text": "First Grubs: your lane priority favours top side."}
    return {"rule": "R19", "choice": "Dragon", "text": "First Dragon (the default)."}


def build_card(rules_data: dict, side: str, team: str, enemy: str, facts: dict, why: dict, by_role: dict,
               forecast: dict | None = None, brief: dict | None = None, evidence: dict | None = None) -> dict:
    other = "Red" if side == "Blue" else "Blue"
    lines = evaluate_rules(rules_data, facts, why, evidence)
    sections = []
    for key, title in rules_data["sections"].items():
        sec = [l for l in lines if l["section"] == key and l["status"] != "off"]
        sections.append({"key": key, "title": title,
                         "lines": [l for l in sec if l["status"] in ("applies", "overridden", "unless")],
                         "needs_labels": [l for l in sec if l["status"] == "needs_labels"]})
    sign = 1 if side == "Blue" else -1
    lanes = []
    for f_lane in (forecast or {}).get("lanes", []):
        r = f_lane["role"]
        lanes.append({"role": r, "label": f_lane["label"], "ours": by_role.get(side, {}).get(r),
                      "theirs": by_role.get(other, {}).get(r), "gold15": sign * f_lane["gold15"],
                      "prio": facts.get(f"our.{r}.prio"), "how": why.get(f"our.{r}.prio"),
                      "reasons": f_lane.get("reasons", [])})
    objectives = [{"key": o["key"], "label": o["label"], "prob": o["blue_prob"] if side == "Blue" else 1 - o["blue_prob"]}
                  for o in (forecast or {}).get("objectives", []) if o["key"] in ("grubs", "first_dragon")]
    obj = objective_plan(lines)
    probs = {o["key"]: o["prob"] for o in objectives}
    key = {"Dragon": "first_dragon", "Grubs": "grubs"}.get(obj["choice"])
    if key in probs and probs[key] < 0.40:
        label = "first Dragon" if key == "first_dragon" else "Grubs"
        obj["note"] = (f"The forecast gives you only {probs[key]:.0%} for {label} in this draft: "
                       "if you go for it, set up early (vision and lane priority) or trade the other side.")
    applied = {l["rule"] for l in lines if l["status"] == "applies"}
    watch = [{"rule": r["id"], "section": r["section"], "when": r["if_text"], "then": r["then"], "why": r["why"],
              "sources": r["sources"], "evidence": evidence_for(r, evidence)[0]}
             for r in rules_data["rules"] if r["kind"] == "live"]
    watch += [{"rule": l["rule"], "section": l["section"], "when": l["live_check"],
               "then": f"confirms {l['rule']}: {l['then']}", "why": l["why"], "sources": l["sources"],
               "evidence": l["evidence"]}
              for l in lines if l["status"] == "applies" and l["live_check"]]
    scouting = [{"rule": r["id"], "when": r["if_text"], "then": r["then"], "sources": r["sources"]}
                for r in rules_data["rules"] if r["kind"] == "scouting"]
    conflicts = []
    for c in rules_data.get("conflicts", []):
        active = any(l["rule"] == c["loser"] and l["status"] == "overridden" for l in lines)
        conflicts.append({**c, "active": active,
                          "text": (f"{c['winner']} wins over {c['loser']} in this draft." if active else
                                   f"{c['winner']} would win over {c['loser']}, but {c['winner']} doesn't apply here.")})
    unlabeled = sorted({m for l in lines if l["status"] == "needs_labels" for m in l["missing"]})
    out = {
        "side": side, "team": team, "enemy": enemy,
        "summary": {"applies": len(applied), "needs_labels": sum(l["status"] == "needs_labels" for l in lines),
                    "missing_tags": unlabeled},
        "level1": None,
        "jungle": jungle_plan(facts, lines),
        "lanes": lanes,
        "first_objective": {**obj, "forecast": objectives},
        "sections": sections,
        "watch_for": watch,
        "scouting": scouting,
        "conflicts": conflicts,
    }
    if brief:
        out["gold15"] = {"break_even": brief.get("break_even_gold"), "comfortable": brief.get("comfortable_gold"),
                         "forecast": (brief.get("forecast") or {}).get("gold"),
                         "forecast_win_prob": (brief.get("forecast") or {}).get("win_prob"),
                         "comp": brief.get("own_comp"), "enemy_comp": brief.get("enemy_comp")}
        out["win_condition"] = brief.get("win_condition")
    l1 = [l for l in lines if l["section"] == "level1" and l["status"] == "applies"]
    out["level1"] = {"rule": l1[0]["rule"], "text": l1[0]["then"]} if l1 else \
        {"rule": None, "text": "No level-1 rule applies" + (" yet: level-1 tags are not labelled."
                                                              if any(l["section"] == "level1" and l["status"] == "needs_labels" for l in lines)
                                                              else ": play a standard level 1.")}
    return out
