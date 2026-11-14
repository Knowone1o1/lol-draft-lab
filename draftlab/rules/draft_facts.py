"""Compute the facts in draftlab/rules/facts.py for one team ("our") from a finished draft.

Every value is either known or None. None means "not labelled / not enough data", and rules that need it show as
"needs labels" on the card instead of guessing. Inputs are passed in (tags, early stats, forecast), so the same code
can run on any past game with stats built only from earlier games (Part 3: no leakage).
"""
from __future__ import annotations

from .facts import LANES, LEVELS, ROLES, SPIKES, TRAITS

# Lane priority cut: Strong / Weak = the lane forecast is in the top / bottom 10% of recent forecasts for that lane
# (|forecast| at or above its 80th percentile). Cuts come from data/rules/prio_cuts.json, written walk-forward by
# scripts/rule_tests.py; the fixed values below are only the fallback when no cuts are passed.
PRIO_PCT = 80
PRIO_GOLD = 700
MIDJNG_GOLD = 1000
DEFAULT_CUTS = {"top": PRIO_GOLD, "mid": PRIO_GOLD, "bot": PRIO_GOLD, "midjng": MIDJNG_GOLD}
FACILITATOR_STYLES = {"Hard engage", "Front-to-back", "Protect the carry"}
CARRY_STYLES = {"Skirmish", "Dive", "Pick", "Split push"}
MATCHUP_LOSS = -300     # a matchup lift at or below this (shrunk, gold @15) with >= MATCHUP_N weighted games
MATCHUP_N = 3.0         # counts as a hard-losing matchup
STAR_TOP = 0.10         # star laner: player's lane gold @15 rating in the top 10% of players in that role
STAR_MIN_N = 5.0        # ... among players with at least this many weighted games


def _level(x, cut):
    if x is None:
        return None
    return "Strong" if x >= cut else "Weak" if x <= -cut else "Even"


def _lvl_idx(v):
    return LEVELS.index(v) if v in LEVELS else None


def _sum_or_none(vals):
    return None if any(v is None for v in vals) else sum(vals)


class StatsLookups:
    """Thresholds from early stats that don't depend on the draft (computed once per stats object)."""

    def __init__(self, es):
        self.es = es
        self.star_cut = {}
        if es is None:
            return
        for r in ROLES:
            vals = sorted(v for (p, role), v in es.player.items() if role == r and es.player_n.get((p, role), 0) >= STAR_MIN_N)
            if len(vals) >= 20:
                self.star_cut[r] = vals[int(len(vals) * (1 - STAR_TOP))]
        self.worst = {}
        for (c, o, r), v in es.matchup.items():
            if es.matchup_n.get((c, o, r), 0) >= MATCHUP_N:
                key = (c, r)
                if v < self.worst.get(key, (0.0, None))[0]:
                    self.worst[key] = (v, o)

    def star(self, player, role):
        if self.es is None or not player or role not in self.star_cut:
            return None
        if self.es.player_n.get((player, role), 0) < STAR_MIN_N:
            return False
        return self.es.player_lane(player, role) >= self.star_cut[role]

    def matchup_dependent(self, champ, role):
        if self.es is None or not champ:
            return None
        v, _ = self.worst.get((champ, role), (0.0, None))
        return v <= MATCHUP_LOSS

    def worst_matchup(self, champ, role):
        return self.worst.get((champ, role))


def draft_facts(side: str, by_role: dict, tag_of, lane_gold: dict | None, pick_index: dict | None = None,
                rosters: dict | None = None, lookups: StatsLookups | None = None, comp_timing: dict | None = None,
                prio_cuts: dict | None = None):
    """Facts for `side` ("Blue" or "Red").

    by_role:     {"Blue": {role: champion}, "Red": {...}}
    tag_of:      champion -> tag dict from draftlab.comp (or None if untagged)
    lane_gold:   {role: Blue-view gold @15 forecast} for all 5 roles (None = no forecast)
    pick_index:  {champion: position in the pick order} (for blind picks)
    rosters:     {"Blue": {role: player key}, ...} (for star laners)
    comp_timing: {"Blue": "Early"|"Mid / tempo"|"Late", ...} from draftlab.comp.comp_profile
    Returns (facts, details) where details explains derived values in words for the card.
    """
    other = "Red" if side == "Blue" else "Blue"
    sides = {"our": side, "their": other}
    sign = {"our": 1 if side == "Blue" else -1}
    sign["their"] = -sign["our"]
    f, why = {}, {}

    def tag(s, r):
        c = by_role.get(sides[s], {}).get(r)
        t = tag_of(c) if c else None
        return (t or {}).get("early") or {}, (t or {})

    # ---- per champion tags
    for s in sides:
        for r in ROLES:
            e, t = tag(s, r)
            f[f"{s}.{r}.lvl1"] = e.get("lvl1") if e.get("lvl1") in LEVELS else None
            f[f"{s}.{r}.laning"] = e.get("laning") if e.get("laning") in LEVELS else None
            f[f"{s}.{r}.spike"] = e.get("spike") if e.get("spike") in SPIKES else None
            f[f"{s}.{r}.early_score"] = e.get("early_score")
            f[f"{s}.{r}.style"] = t.get("primary")
            f[f"{s}.{r}.timing"] = t.get("timing")
            for k in TRAITS:
                f[f"{s}.{r}.{k}"] = e.get(k)

    # ---- lane priority: gold @15 forecast, overridden by the Laning tags when both sides' tags are filled
    for s in sides:
        for lane in LANES:
            g = None if not lane_gold or lane_gold.get(lane) is None else sign[s] * lane_gold[lane]
            cut = (prio_cuts or DEFAULT_CUTS).get(lane, PRIO_GOLD)
            prio = _level(g, cut)
            how = f"forecast {g:+,.0f} gold at 15; Strong/Weak from ±{cut:,.0f}" if g is not None else None
            roles = ("bot", "sup") if lane == "bot" else (lane,)
            mine = [_lvl_idx(f[f"{s}.{r}.laning"]) for r in roles]
            opp = "their" if s == "our" else "our"
            theirs = [_lvl_idx(f[f"{opp}.{r}.laning"]) for r in roles]
            if None not in mine + theirs:
                d = sum(mine) - sum(theirs)
                prio = "Strong" if d > 0 else "Weak" if d < 0 else "Even"
                how = "Laning tags (" + ("ahead" if d > 0 else "behind" if d < 0 else "level") + ")" + \
                      (f"; forecast {g:+,.0f}" if g is not None else "")
            f[f"{s}.{lane}.prio"] = prio
            why[f"{s}.{lane}.prio"] = how
        f[f"{s}.strong_lanes"] = None if any(f[f"{s}.{l}.prio"] is None for l in LANES) else \
            sum(f[f"{s}.{l}.prio"] == "Strong" for l in LANES)
        f[f"{s}.weak_lanes"] = None if any(f[f"{s}.{l}.prio"] is None for l in LANES) else \
            sum(f[f"{s}.{l}.prio"] == "Weak" for l in LANES)
        if lane_gold and lane_gold.get("mid") is not None and lane_gold.get("jng") is not None:
            mj = sign[s] * (lane_gold["mid"] + lane_gold["jng"])
            cut = (prio_cuts or DEFAULT_CUTS).get("midjng", MIDJNG_GOLD)
            f[f"{s}.midjng"] = _level(mj, cut)
            why[f"{s}.midjng"] = f"mid + jungle forecast {mj:+,.0f} gold at 15; Strong/Weak from ±{cut:,.0f}"
        else:
            f[f"{s}.midjng"] = None

    # ---- pick order: blind = picked before the lane opponent
    for s in sides:
        opp = "their" if s == "our" else "our"
        for lane in LANES:
            a, b = by_role.get(sides[s], {}).get(lane), by_role.get(sides[opp], {}).get(lane)
            if pick_index and a in pick_index and b in pick_index:
                f[f"{s}.{lane}.blind"] = pick_index[a] < pick_index[b]
            else:
                f[f"{s}.{lane}.blind"] = None
            f[f"{s}.{lane}.matchup_dependent"] = lookups.matchup_dependent(a, lane) if lookups else None
            if lookups and a and f[f"{s}.{lane}.matchup_dependent"]:
                v, o = lookups.worst_matchup(a, lane)
                why[f"{s}.{lane}.matchup_dependent"] = f"{a} does {v:+,.0f} gold at 15 vs expectation into {o}"
            player = (rosters or {}).get(sides[s], {}).get(lane)
            f[f"{s}.{lane}.star"] = lookups.star(player, lane) if lookups else None

    # ---- team-level
    for s in sides:
        opp = "their" if s == "our" else "our"
        f[f"{s}.timing"] = (comp_timing or {}).get(sides[s])
        f[f"{s}.early_score"] = _sum_or_none([f[f"{s}.{r}.early_score"] for r in ROLES])
        mine = [f[f"{s}.{r}.lvl1"] for r in ROLES]
        theirs = [f[f"{opp}.{r}.lvl1"] for r in ROLES]
        pts = {"Strong": 1, "Even": 0, "Weak": -1}
        f[f"{s}.lvl1_edge"] = None if None in mine + theirs else \
            sum(pts[v] for v in mine) - sum(pts[v] for v in theirs)
        laners = [f[f"{s}.{r}.lvl1"] for r in ("top", "mid", "bot")]
        f[f"{s}.laners_lvl1_strong"] = None if None in laners else sum(v == "Strong" for v in laners)
        sp, osp = [f[f"{s}.{r}.spike"] for r in ROLES], [f[f"{opp}.{r}.spike"] for r in ROLES]
        f[f"{s}.spike_edge"] = None if None in sp + osp else \
            min(SPIKES.index(v) for v in osp) - min(SPIKES.index(v) for v in sp)
        style, engage = f[f"{s}.jng.style"], f[f"{s}.jng.engage"]
        if style is None:
            fac = None
        elif style in FACILITATOR_STYLES or engage:
            fac = True
        else:
            fac = False if engage is not None or style in CARRY_STYLES else None
        f[f"{s}.jng.facilitator"] = fac
        f[f"{s}.jng.carry"] = None if fac is None else (not fac and style in CARRY_STYLES)
        dives = [f[f"{s}.{r}.dive"] for r in ROLES]
        f[f"{s}.dive_threats"] = None if None in dives else sum(bool(v) for v in dives)
        peel = [f[f"{s}.{r}.peel"] for r in ("bot", "sup")]
        f[f"{s}.botlane_peel"] = None if None in peel else sum(bool(v) for v in peel)
    return f, why
