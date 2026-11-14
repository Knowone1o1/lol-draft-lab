"""Glue for the app: a finished draft -> one plan card per team (used by POST /api/plan_card)."""
from __future__ import annotations

from pathlib import Path

from ..comp import champ_tag, comp_profile
from ..config import ROOT
from .build import load_rules
from .card import build_card
from .draft_facts import StatsLookups, draft_facts

RULES_JSON = ROOT / "data" / "rules" / "rules.json"
EVIDENCE_JSON = ROOT / "data" / "rules" / "evidence.json"
CUTS_JSON = ROOT / "data" / "rules" / "prio_cuts.json"


class PlanEngine:
    def __init__(self, rules_path: Path | None = None, early_stats=None, evidence: dict | None = None):
        self.rules = load_rules(rules_path or RULES_JSON)
        self.lookups = StatsLookups(early_stats)
        if evidence is None and EVIDENCE_JSON.exists():
            import json
            evidence = json.loads(EVIDENCE_JSON.read_text(encoding="utf-8")).get("rules", {})
        self.evidence = evidence or {}
        # latest walk-forward lane-priority cuts (scripts/rule_tests.py); None -> fixed fallback in draft_facts
        self.prio_cuts = None
        if CUTS_JSON.exists():
            import json
            self.prio_cuts = json.loads(CUTS_JSON.read_text(encoding="utf-8")).get("current")

    def cards(self, teams: dict, by_role: dict, rosters: dict, pick_index: dict, forecast: dict | None,
              briefs: dict | None = None) -> dict:
        """teams/by_role/rosters keyed by "Blue"/"Red"; forecast = /api/early output (Blue view)."""
        lane_gold = {l["role"]: l["gold15"] for l in forecast["lanes"]} if forecast else None
        timing = {s: comp_profile([c for c in by_role.get(s, {}).values() if c])["timing"] for s in ("Blue", "Red")}
        out = {}
        for side, other in (("Blue", "Red"), ("Red", "Blue")):
            facts, why = draft_facts(side, by_role, champ_tag, lane_gold, pick_index, rosters, self.lookups, timing,
                                   self.prio_cuts)
            out[side] = build_card(self.rules, side, teams[side], teams[other], facts, why, by_role, forecast,
                                   (briefs or {}).get(side), self.evidence)
        return out
