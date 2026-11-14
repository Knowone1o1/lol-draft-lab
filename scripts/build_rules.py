"""Regenerate data/rules/rules.json from the rules sheet and the conditions sheet.

    .venv/Scripts/python scripts/build_rules.py                  # build (stops if a rule's 'If' text changed)
    .venv/Scripts/python scripts/build_rules.py --stamp R05 R12  # after re-checking those conditions
    .venv/Scripts/python scripts/build_rules.py --stamp all      # first time / accept all current texts

Workflow: edit the rules in the spreadsheet, export the rules tab to data/rules/early_game_rules.csv, run this.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.rules import RulesBuildError, build, stamp  # noqa: E402

RULES_DIR = ROOT / "data" / "rules"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", default=RULES_DIR / "early_game_rules.csv", type=Path)
    ap.add_argument("--conditions", default=RULES_DIR / "rule_conditions.csv", type=Path)
    ap.add_argument("--out", default=RULES_DIR / "rules.json", type=Path)
    ap.add_argument("--stamp", nargs="+", metavar="ID", help="accept the current 'If' text for these rules (or 'all')")
    a = ap.parse_args()
    if a.stamp:
        done = stamp(a.rules, a.conditions, a.stamp)
        print(f"stamped: {', '.join(done) or 'nothing to change'}")
    try:
        data = build(a.rules, a.conditions)
    except RulesBuildError as e:
        print(e)
        sys.exit(1)
    a.out.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    c = data["counts"]
    print(f"wrote {a.out.relative_to(ROOT)}: {len(data['rules'])} rules "
          f"({c['draft']} from the draft, {c['live']} watch-for, {c['scouting']} scouting), "
          f"{len(data['conflicts'])} override(s)")


if __name__ == "__main__":
    main()
