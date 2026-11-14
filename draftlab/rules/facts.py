"""The draft facts that early-game rule conditions can use (Part 1: names, meaning, where each value comes from).

A fact name is `<side>.<thing>` or `<side>.<role>.<thing>`, with side = our | their (relative to the team the card is
for) and role = top | jng | mid | bot | sup. `live.<thing>` facts only exist during a game; rules that need them are
"watch for" items, never triggered from the draft.

Values are computed per draft in Part 2. A fact whose value is unknown (e.g. a tag the sheet has not filled in yet)
makes a condition evaluate to UNKNOWN rather than false, so the card can say "needs labels" instead of guessing.
"""
from __future__ import annotations

import re

SIDES = ("our", "their")
ROLES = ("top", "jng", "mid", "bot", "sup")
ROLE_WORD = {"top": "top laner", "jng": "jungler", "mid": "mid laner", "bot": "ADC", "sup": "support",
             "{lane}": "{lane} laner"}
LANE_WORD = {"top": "top", "mid": "mid", "bot": "bot", "{lane}": "{lane}"}
LANES = ("top", "mid", "bot")
# a rule with "For each: lane=top,bot" writes {lane} in place of the lane; the name is filled in per lane on the card
_R = r"(top|jng|mid|bot|sup|\{lane\})"
_L = r"(top|mid|bot|\{lane\})"
LEVELS = ["Weak", "Even", "Strong"]
SPIKES = ["Level 2", "Level 3", "Level 6", "1 item", "2 items", "3+ items"]     # earliest -> latest
STYLES = ["Front-to-back", "Hard engage", "Dive", "Poke / siege", "Pick", "Split push", "Protect the carry",
          "Disengage", "Skirmish"]
TRAITS = {"engage": "Engage", "peel": "Peel", "dive": "Dive threat", "poke": "Poke", "catch": "Catch CC",
          "waveclear": "Wave clear", "roam": "Roam / gank", "invade": "Invade"}

# pattern -> (type, allowed values or None, source, plain-language description with {side}/{role} placeholders)
_SPECS = {
    # ---- per champion, from the Champion Tags sheet
    r"(our|their)\." + _R + r"\.lvl1": ("enum", LEVELS, "tag: Lvl 1", "{side} {role}'s level-1 strength"),
    r"(our|their)\." + _R + r"\.laning": ("enum", LEVELS, "tag: Laning (0-10 min)", "{side} {role}'s laning tag"),
    r"(our|their)\." + _R + r"\.spike": ("enum", SPIKES, "tag: Early spike", "{side} {role}'s early spike"),
    r"(our|their)\." + _R + r"\.early_score": ("number", None, "tag: Early score (0-6)", "{side} {role}'s early score"),
    r"(our|their)\." + _R + r"\.style": ("enum", STYLES, "tag: Primary style", "{side} {role}'s primary style"),
    r"(our|their)\." + _R + r"\.timing": ("enum", ["Early", "Mid / tempo", "Late"], "tag: Timing", "{side} {role}'s timing"),
    **{rf"(our|their)\.{_R}\.{k}": ("bool", None, f"tag: {v}", f"{{side}} {{role}} has '{v}'")
       for k, v in TRAITS.items()},
    # ---- per lane, from the gold-at-15 lane forecast (matchup + players); the Laning tag overrides when filled
    r"(our|their)\." + _L + r"\.prio": ("enum", LEVELS, "forecast: lane gold @15, top/bottom 10% of recent forecasts for the lane = Strong/Weak; Laning tags override",
                                          "{side} {lane} lane priority"),
    # ---- per lane, from the draft's pick order
    r"(our|their)\." + _L + r"\.blind": ("bool", None, "pick order", "{side} {lane} laner was picked before its lane opponent (blind)"),
    r"(our|their)\." + _L + r"\.matchup_dependent": ("bool", None, "data: the champion has a matchup at or below -300 gold @15 vs expectation (>= 3 weighted games)",
                                                      "{side} {lane} laner's champion has hard-losing matchups"),
    r"(our|their)\." + _L + r"\.star": ("bool", None, "data: player's lane gold @15 rating in the top 10% of players in that role (>= 5 weighted games)",
                                          "{side} {lane} laner is a star laner"),
    # ---- per team
    r"(our|their)\.timing": ("enum", ["Early", "Mid / tempo", "Late"], "tag sheet: comp timing (average of 5)", "{side} comp timing"),
    r"(our|their)\.early_score": ("number", None, "tag: Early score, summed over 5", "{side} comp's early score"),
    r"(our|their)\.strong_lanes": ("number", None, "lane priority", "{side} lanes with Strong priority (count)"),
    r"(our|their)\.weak_lanes": ("number", None, "lane priority", "{side} lanes with Weak priority (count)"),
    r"(our|their)\.lvl1_edge": ("number", None, "tag: Lvl 1 (Strong=+1, Weak=-1, summed, ours minus theirs)",
                                "{side} level-1 edge (Strong minus Weak champions, vs the enemy)"),
    r"(our|their)\.laners_lvl1_strong": ("number", None, "tag: Lvl 1", "{side} laners with Strong level 1 (count)"),
    r"(our|their)\.spike_edge": ("number", None, "tag: Early spike (steps earlier than the enemy's earliest)",
                                 "how much earlier {side} comp spikes than the enemy's"),
    r"(our|their)\.jng\.facilitator": ("bool", None, "tag: Primary style in Hard engage / Front-to-back / Protect the carry, or Engage",
                                       "{side} jungler is a facilitator (tank / engage)"),
    r"(our|their)\.jng\.carry": ("bool", None, "tag: Primary style in Skirmish / Dive / Pick / Split push and not a facilitator",
                                 "{side} jungler is a carry jungler"),
    r"(our|their)\.midjng": ("enum", LEVELS, "forecast: mid + jungle gold @15, top/bottom 10% of recent forecasts = Strong/Weak",
                             "{side} mid-jungle 2v2 strength"),
    r"(our|their)\.dive_threats": ("number", None, "tag: Dive threat", "{side} team's dive threats (count)"),
    r"(our|their)\.botlane_peel": ("number", None, "tag: Peel (ADC + support)", "{side} bot lane's peel (count)"),
    # ---- live only (never computed from a draft)
    r"live\.[a-z_]+": ("live", None, "live game information", "live: {name}"),
}
_COMPILED = [(re.compile(p + r"$"), spec) for p, spec in _SPECS.items()]


def fact_spec(name: str):
    """(type, allowed values, source, description) for a fact name, or None if the name is not a known fact."""
    for rx, (typ, values, source, desc) in _COMPILED:
        m = rx.match(name)
        if m:
            side = m.group(1) if m.groups() and m.group(1) in SIDES else ""
            role = m.group(2) if len(m.groups()) > 1 and m.group(2) in ROLE_WORD else ""
            words = desc.format(side={"our": "our", "their": "the enemy"}.get(side, side),
                                role=ROLE_WORD.get(role, role), lane=LANE_WORD.get(role, role), name=name.split(".", 1)[-1].replace("_", " "))
            return typ, values, source, words
    return None


def is_live(name: str) -> bool:
    return name.startswith("live.")
