"""Build data/rules/rules.json from the coaches' rules sheet + the conditions sheet.

Two inputs, both editable in a spreadsheet:
  early_game_rules.csv   the rules as coaches write them (ID, Phase, If, Then, Why, Inputs, Test, Sources)
  rule_conditions.csv    one row per rule: how its "If" is checked (Kind, Section, For each, When, Unless, Overrides,
                         Live check, Approximation, If hash)

"If hash" is a short fingerprint of the rule's "If" text at the time its condition was written. If someone edits the
"If" text in the rules sheet, the hashes no longer match and the build stops, so a condition never silently drifts
from the sentence a coach reads. After re-checking the condition, re-stamp with `--stamp R05 R12` (or `--stamp all`).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from .condition import ConditionError, parse, plain, validate
from .facts import fact_spec, is_live

KINDS = {"draft", "live", "scouting"}
SECTIONS = {  # order = order on the plan card
    "level1": "Level 1 plan",
    "jungle_path": "Jungle start and path",
    "lane_prio": "Lane priority",
    "first_objective": "First objective (Grubs vs Dragon)",
    "fight_windows": "Fight windows",
    "draft_warnings": "Draft warnings",
}
COND_COLS = ["ID", "Kind", "Section", "For each", "When", "Unless", "Overrides", "Live check", "Approximation", "If hash"]


class RulesBuildError(ValueError):
    def __init__(self, problems):
        self.problems = problems
        super().__init__("rules build failed:\n  - " + "\n  - ".join(problems))


def if_hash(text: str) -> str:
    norm = " ".join(text.split()).lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:8]


def _read(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [{k.strip(): (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(f)
                if any((v or "").strip() for v in row.values())]


def _for_each(text: str) -> dict:
    """'lane=top,bot' -> {'lane': ['top', 'bot']}"""
    if not text:
        return {}
    out = {}
    for part in text.split(";"):
        key, _, vals = part.partition("=")
        out[key.strip()] = [v.strip() for v in vals.split(",") if v.strip()]
    return out


def _oe_testable(test_text: str) -> bool:
    t = test_text.lower()
    return bool(t) and not t.startswith("not") and "needs" not in t


def build(rules_csv: Path, cond_csv: Path) -> dict:
    rules = _read(rules_csv)
    conds = {r["ID"]: r for r in _read(cond_csv)}
    problems, out = [], []
    ids = [r["ID"] for r in rules]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"{dup}: appears more than once in the rules sheet")
    for extra in sorted(set(conds) - set(ids)):
        problems.append(f"{extra}: has a condition but is not in the rules sheet")

    for r in rules:
        rid = r["ID"]
        c = conds.get(rid)
        if c is None:
            problems.append(f"{rid}: no row in rule_conditions.csv")
            continue
        kind, section = c.get("Kind", ""), c.get("Section", "")
        if kind not in KINDS:
            problems.append(f"{rid}: Kind {kind!r} must be one of {sorted(KINDS)}")
        if section not in SECTIONS:
            problems.append(f"{rid}: Section {section!r} must be one of {list(SECTIONS)}")
        h = if_hash(r["If (condition)"])
        if c.get("If hash") != h:
            problems.append(f"{rid}: the 'If' text changed since its condition was written "
                            f"(now: {r['If (condition)']!r}). Re-check the condition, then run with --stamp {rid}")
        placeholders = _for_each(c.get("For each", ""))
        entry = {
            "id": rid, "phase": r["Phase"], "kind": kind, "section": section,
            "if_text": r["If (condition)"], "then": r["Then (plan)"], "why": r["Why"],
            "inputs": r["Inputs used"], "oe_test": r["Test in Oracle's Elixir"],
            "oe_testable": _oe_testable(r["Test in Oracle's Elixir"]),
            "sources": [s.strip() for s in r["Sources"].replace(";", ",").split(",") if s.strip()],
            "for_each": placeholders, "when": None, "when_plain": None, "unless": None, "unless_plain": None,
            "facts": [], "overrides": [o.strip() for o in c.get("Overrides", "").split(",") if o.strip()],
            "live_check": c.get("Live check") or None, "approximation": c.get("Approximation") or None,
            "if_hash": h,
        }
        if kind == "draft":
            for field in ("When", "Unless"):
                text = c.get(field, "")
                if field == "When" and not text:
                    problems.append(f"{rid}: draft rule needs a When condition")
                    continue
                if not text:
                    continue
                try:
                    tree = parse(text)
                except ConditionError as e:
                    problems.append(f"{rid} {field}: {e}")
                    continue
                problems += [f"{rid} {field}: {p}" for p in validate(tree, placeholders)]
                key = field.lower()
                entry[key], entry[key + "_text"], entry[key + "_plain"] = tree, text, plain(tree)
                entry["facts"] += [f for f in _facts(tree) if f not in entry["facts"]]
            live = [f for f in entry["facts"] if is_live(f)]
            if live:
                problems.append(f"{rid}: draft rule uses live facts {live}; make it Kind=live or use Live check")
        elif c.get("When") or c.get("Unless"):
            problems.append(f"{rid}: {kind} rules are shown as text and must not have a When/Unless condition")
        out.append(entry)

    known = {e["id"] for e in out}
    for e in out:
        for o in e["overrides"]:
            if o not in known:
                problems.append(f"{e['id']}: overrides unknown rule {o}")
            elif o == e["id"]:
                problems.append(f"{e['id']}: cannot override itself")
    if problems:
        raise RulesBuildError(problems)

    conflicts = [{"winner": e["id"], "loser": o, "section": e["section"],
                  "note": f"When {e['id']} applies it replaces {o} on the card; {o} is shown struck through with the reason."}
                 for e in out for o in e["overrides"]]
    return {
        "version": 1,
        "built_from": {"rules": rules_csv.name, "conditions": cond_csv.name},
        "sections": SECTIONS,
        "counts": {k: sum(e["kind"] == k for e in out) for k in sorted(KINDS)},
        "conflicts": conflicts,
        "rules": out,
    }


def _facts(tree):
    from .condition import facts_used
    return list(dict.fromkeys(facts_used(tree)))


def stamp(rules_csv: Path, cond_csv: Path, which: list[str]) -> list[str]:
    """Write the current If-text fingerprint into rule_conditions.csv for the given IDs ('all' = every rule)."""
    hashes = {r["ID"]: if_hash(r["If (condition)"]) for r in _read(rules_csv)}
    rows = _read(cond_csv)
    done = []
    for row in rows:
        if row["ID"] in hashes and ("all" in which or row["ID"] in which) and row.get("If hash") != hashes[row["ID"]]:
            row["If hash"] = hashes[row["ID"]]
            done.append(row["ID"])
    with open(cond_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COND_COLS)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in COND_COLS})
    return done


def load_rules(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = ["build", "stamp", "load_rules", "if_hash", "RulesBuildError", "SECTIONS", "KINDS", "fact_spec"]
