"""Early-game rules, Part 1: condition language, fact catalogue, and the CSV -> rules.json build."""
import csv
import shutil
from pathlib import Path

import pytest

from draftlab.rules import (ConditionError, RulesBuildError, build, evaluate, fact_spec, parse, plain, stamp,
                            substitute, validate)

RULES_DIR = Path(__file__).resolve().parent.parent / "data" / "rules"
RULES_CSV, COND_CSV = RULES_DIR / "early_game_rules.csv", RULES_DIR / "rule_conditions.csv"


# ---------------- condition language
def test_parse_and_evaluate_basic():
    t = parse("our.jng.lvl1 == Strong AND their.jng.lvl1 != Strong")
    assert evaluate(t, {"our.jng.lvl1": "Strong", "their.jng.lvl1": "Even"}) == (True, [])
    assert evaluate(t, {"our.jng.lvl1": "Weak", "their.jng.lvl1": "Even"}) == (False, [])


def test_precedence_and_parentheses():
    t = parse("our.timing == Early AND (our.bot.prio == Strong OR our.mid.prio == Strong)")
    f = {"our.timing": "Early", "our.bot.prio": "Weak", "our.mid.prio": "Strong"}
    assert evaluate(t, f)[0] is True
    assert evaluate(t, {**f, "our.timing": "Late"})[0] is False


def test_unknown_is_not_false():
    """A missing tag gives UNKNOWN (None), unless the known part already decides the answer."""
    t = parse("our.jng.lvl1 == Strong AND our.timing == Early")
    assert evaluate(t, {"our.timing": "Early"}) == (None, ["our.jng.lvl1"])
    assert evaluate(t, {"our.timing": "Late"})[0] is False           # False AND unknown = False
    t2 = parse("our.bot.prio == Strong OR our.mid.prio == Strong")
    assert evaluate(t2, {"our.bot.prio": "Strong"})[0] is True       # True OR unknown = True
    assert evaluate(parse("NOT our.jng.carry"), {})[0] is None


def test_ordinal_in_and_quoted_values():
    assert evaluate(parse("our.top.prio >= Even"), {"our.top.prio": "Strong"})[0] is True
    assert evaluate(parse("our.top.prio > Even"), {"our.top.prio": "Weak"})[0] is False
    assert evaluate(parse("our.bot.spike < \"2 items\""), {"our.bot.spike": "Level 6"})[0] is True
    assert evaluate(parse("our.sup.style IN [Pick, \"Hard engage\"]"), {"our.sup.style": "Hard engage"})[0] is True
    assert evaluate(parse("their.sup.style == \"Protect the carry\""), {"their.sup.style": "Pick"})[0] is False


def test_fact_to_fact_comparison():
    t = parse("our.laners_lvl1_strong > their.laners_lvl1_strong")
    assert evaluate(t, {"our.laners_lvl1_strong": 2, "their.laners_lvl1_strong": 1})[0] is True
    assert evaluate(t, {"our.laners_lvl1_strong": 1, "their.laners_lvl1_strong": 1})[0] is False
    assert evaluate(t, {"our.laners_lvl1_strong": 2}) == (None, ["their.laners_lvl1_strong"])
    assert "the enemy laners with Strong level 1" in plain(t)


def test_placeholders():
    t = parse("our.{lane}.blind AND our.{lane}.matchup_dependent")
    assert validate(t, {"lane": ["top", "mid", "bot"]}) == []
    top = substitute(t, {"lane": "top"})
    assert evaluate(top, {"our.top.blind": True, "our.top.matchup_dependent": True})[0] is True
    assert evaluate(top, {"our.mid.blind": True, "our.mid.matchup_dependent": True})[0] is None
    assert "our {lane} laner" in plain(t) and "our top laner" in plain(top)


def test_validate_catches_mistakes():
    assert validate(parse("our.jng.lvl1 == Stronk")) == ["our.jng.lvl1: ['Stronk'] not in ['Weak', 'Even', 'Strong']"]
    assert validate(parse("our.jng.dueling == Strong"))[0].startswith("unknown fact")
    assert validate(parse("our.{lane}.prio == Weak"), {"lane": ["top", "jng"]}) == ["unknown fact 'our.jng.prio'"]
    assert validate(parse("our.timing == their.strong_lanes"))  # enum vs number


def test_parse_errors():
    for bad in ("our.jng.lvl1 ==", "(our.jng.carry", "our.jng.carry AND", "== Strong", "our.jng.carry )"):
        with pytest.raises(ConditionError):
            parse(bad)


def test_fact_catalogue_words():
    assert fact_spec("their.bot.prio")[3] == "the enemy bot lane priority"
    assert fact_spec("our.sup.peel")[0] == "bool"
    assert fact_spec("live.enemy_jungler_seen")[0] == "live"
    assert fact_spec("our.jng.prio") is None          # junglers have no lane priority


# ---------------- build from the real sheets
@pytest.fixture(scope="module")
def built():
    return build(RULES_CSV, COND_CSV)


def test_real_sheet_builds(built):
    rules = {r["id"]: r for r in built["rules"]}
    assert len(rules) == 36 and set(rules) == {f"R{i:02d}" for i in range(5, 41)}
    assert built["counts"] == {"draft": 22, "live": 12, "scouting": 2}
    assert built["conflicts"] == [{"winner": "R20", "loser": "R08", "section": "jungle_path",
                                   "note": built["conflicts"][0]["note"]}]
    for r in rules.values():
        assert r["sources"] and all(s.startswith("S") for s in r["sources"])
        assert (r["when"] is not None) == (r["kind"] == "draft")
        assert r["when_plain"] is None or "{" not in r["when_plain"].replace("{lane}", "")
    assert rules["R19"]["unless_plain"].startswith("our top lane priority is Strong")
    assert rules["R19"]["oe_testable"] and not rules["R10"]["oe_testable"]


def test_every_draft_rule_runs_on_an_empty_draft(built):
    """With no tags filled in, nothing crashes and nothing fires except ALWAYS rules: the rest are UNKNOWN."""
    for r in built["rules"]:
        if r["kind"] != "draft":
            continue
        lanes = r["for_each"].get("lane", [None])
        for lane in lanes:
            tree = substitute(r["when"], {"lane": lane}) if lane else r["when"]
            res, missing = evaluate(tree, {})
            assert res is (True if r["when"].get("always") else None), r["id"]
            assert missing or r["when"].get("always")


def _copy(tmp_path):
    shutil.copy(RULES_CSV, tmp_path / "rules.csv")
    shutil.copy(COND_CSV, tmp_path / "cond.csv")
    return tmp_path / "rules.csv", tmp_path / "cond.csv"


def _edit(path, rid, col, value):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    for r in rows:
        if r["ID"] == rid:
            r[col] = value
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def test_edited_if_text_stops_the_build_until_restamped(tmp_path):
    rules, cond = _copy(tmp_path)
    _edit(rules, "R17", "If (condition)", "Our draft has at most one Strong lane")
    with pytest.raises(RulesBuildError) as e:
        build(rules, cond)
    assert len(e.value.problems) == 1 and e.value.problems[0].startswith("R17: the 'If' text changed")
    assert stamp(rules, cond, ["R17"]) == ["R17"]
    assert len(build(rules, cond)["rules"]) == 36


def test_build_reports_sheet_problems(tmp_path):
    rules, cond = _copy(tmp_path)
    _edit(cond, "R07", "When", "our.jng.lvl1 == Weakish")
    _edit(cond, "R12", "Section", "midgame")
    _edit(cond, "R10", "When", "our.jng.carry")
    _edit(cond, "R08", "Overrides", "R99")
    with pytest.raises(RulesBuildError) as e:
        build(rules, cond)
    text = "\n".join(e.value.problems)
    for rid in ("R07 When", "R12: Section", "R10: live rules", "R08: overrides unknown rule R99"):
        assert rid in text, text
