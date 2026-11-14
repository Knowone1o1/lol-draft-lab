"""Early-game rules, Part 2: draft facts and the plan card (synthetic inputs, no trained models needed)."""
from pathlib import Path

import pytest

from draftlab.comp import champ_tag
from draftlab.rules import build
from draftlab.rules.card import build_card, evaluate_rules, jungle_plan, objective_plan
from draftlab.rules.draft_facts import draft_facts

RULES_DIR = Path(__file__).resolve().parent.parent / "data" / "rules"
RULES = build(RULES_DIR / "early_game_rules.csv", RULES_DIR / "rule_conditions.csv")
R = ["top", "jng", "mid", "bot", "sup"]
BY_ROLE = {"Blue": dict(zip(R, ["B_top", "B_jng", "B_mid", "B_bot", "B_sup"])),
           "Red": dict(zip(R, ["R_top", "R_jng", "R_mid", "R_bot", "R_sup"]))}


def tags(overrides=None):
    """Fake tag sheet: every champion Even / Mid-tempo / Skirmish, early traits unlabelled unless overridden."""
    overrides = overrides or {}

    def tag_of(c):
        base = {"primary": "Skirmish", "timing": "Mid / tempo",
                "early": {"lvl1": None, "laning": None, "spike": None, "early_score": None,
                          "engage": None, "peel": None, "dive": None, "poke": None, "catch": None,
                          "waveclear": None, "roam": None, "invade": None}}
        o = overrides.get(c, {})
        return {**base, **{k: v for k, v in o.items() if k != "early"}, "early": {**base["early"], **o.get("early", {})}}
    return tag_of


def facts_for(side="Blue", gold=None, tag_over=None, pick_index=None, timing=None):
    gold = gold if gold is not None else {"top": 0, "jng": 0, "mid": 0, "bot": 0, "sup": 0}
    return draft_facts(side, BY_ROLE, tags(tag_over), gold, pick_index, None, None,
                       timing or {"Blue": "Mid / tempo", "Red": "Mid / tempo"})


def status(lines, rule, lane=None):
    return next(l["status"] for l in lines if l["rule"] == rule and l["lane"] == lane)


# ---------------- facts
def test_lane_priority_from_forecast_flips_for_red():
    f, why = facts_for("Blue", {"top": -900, "jng": 0, "mid": 200, "bot": 800, "sup": 0})
    assert (f["our.top.prio"], f["our.mid.prio"], f["our.bot.prio"]) == ("Weak", "Even", "Strong")
    assert f["their.top.prio"] == "Strong" and f["our.strong_lanes"] == 1 and f["our.weak_lanes"] == 1
    assert "-900" in why["our.top.prio"]
    f, _ = facts_for("Red", {"top": -900, "jng": 0, "mid": 200, "bot": 800, "sup": 0})
    assert (f["our.top.prio"], f["our.bot.prio"]) == ("Strong", "Weak")


def test_per_lane_percentile_cuts():
    cuts = {"top": 500, "mid": 350, "bot": 410, "midjng": 480}
    gold = {"top": 520, "jng": 200, "mid": -360, "bot": 400, "sup": 0}
    f, why = draft_facts("Blue", BY_ROLE, tags(), gold, None, None, None, {"Blue": "Early", "Red": "Late"}, cuts)
    assert (f["our.top.prio"], f["our.mid.prio"], f["our.bot.prio"]) == ("Strong", "Weak", "Even")
    assert f["our.midjng"] == "Even" and "±500" in why["our.top.prio"]
    f, _ = draft_facts("Blue", BY_ROLE, tags(), gold, None, None, None, {"Blue": "Early", "Red": "Late"})
    assert (f["our.top.prio"], f["our.mid.prio"]) == ("Even", "Even")       # fallback +/-700 without cuts


def test_laning_tags_override_forecast_when_both_filled():
    over = {"B_top": {"early": {"laning": "Strong"}}, "R_top": {"early": {"laning": "Even"}}}
    f, why = facts_for("Blue", {"top": -900, "jng": 0, "mid": 0, "bot": 0, "sup": 0}, over)
    assert f["our.top.prio"] == "Strong" and why["our.top.prio"].startswith("Laning tags")
    f, _ = facts_for("Blue", {"top": -900, "jng": 0, "mid": 0, "bot": 0, "sup": 0}, {"B_top": over["B_top"]})
    assert f["our.top.prio"] == "Weak"        # only one side tagged: forecast still decides


def test_unlabelled_tags_stay_unknown():
    f, _ = facts_for()
    for k in ("our.jng.lvl1", "our.lvl1_edge", "our.laners_lvl1_strong", "our.spike_edge", "our.dive_threats",
              "our.botlane_peel", "our.early_score", "our.sup.roam"):
        assert f[k] is None, k
    assert f["our.timing"] == "Mid / tempo" and f["our.jng.style"] == "Skirmish"


def test_jungler_roles_and_level1_edge():
    over = {"B_jng": {"primary": "Hard engage", "early": {"lvl1": "Weak"}}, "R_jng": {"early": {"lvl1": "Strong"}}}
    for c in ("B_top", "B_mid", "B_bot", "B_sup", "R_top", "R_mid", "R_bot", "R_sup"):
        over.setdefault(c, {"early": {"lvl1": "Even"}})
    f, _ = facts_for(tag_over=over)
    assert f["our.jng.facilitator"] is True and f["our.jng.carry"] is False
    assert f["their.jng.facilitator"] is False and f["their.jng.carry"] is True     # Skirmish jungler
    assert f["our.lvl1_edge"] == -2 and f["their.lvl1_edge"] == 2


def test_blind_from_pick_order():
    order = ["B_jng", "R_jng", "R_mid", "B_top", "B_mid", "R_top", "R_bot", "B_bot", "B_sup", "R_sup"]
    f, _ = facts_for(pick_index={c: i for i, c in enumerate(order)})
    assert f["our.top.blind"] is True and f["our.mid.blind"] is False and f["our.bot.blind"] is False
    assert f["their.mid.blind"] is True


# ---------------- rules on the card
def test_weak_lane_overrides_r08():
    f, why = facts_for(gold={"top": -900, "jng": 0, "mid": 0, "bot": 0, "sup": 0})
    lines = evaluate_rules(RULES, f, why)
    assert status(lines, "R20", "top") == "applies" and status(lines, "R20", "mid") == "off"
    r08 = next(l for l in lines if l["rule"] == "R08")
    assert r08["status"] == "overridden" and r08["overridden_by"] == "R20" and "Weak" in r08["override_reason"]
    assert jungle_plan(f, lines)["start"] == "top side"


def test_r08_path_toward_priority():
    f, why = facts_for(gold={"top": 900, "jng": 0, "mid": 0, "bot": 0, "sup": 0})
    lines = evaluate_rules(RULES, f, why)
    assert status(lines, "R08") == "applies"
    assert jungle_plan(f, lines) == {"rule": "R08", "start": "bot side", "finish": "top side",
                                     "path": "Start bot side and finish top side, where your lane has priority."}


def test_r19_unless_switches_to_grubs():
    f, why = facts_for(gold={"top": 900, "jng": 0, "mid": 0, "bot": 0, "sup": 0})
    lines = evaluate_rules(RULES, f, why)
    assert status(lines, "R19") == "unless"
    assert objective_plan(lines)["choice"] == "Grubs"
    f, why = facts_for(gold={"top": 900, "jng": 0, "mid": 0, "bot": 900, "sup": 0})
    assert objective_plan(evaluate_rules(RULES, f, why))["choice"] == "Dragon"


def test_late_vs_early_trades_objectives():
    f, why = facts_for(timing={"Blue": "Late", "Red": "Early"})
    lines = evaluate_rules(RULES, f, why)
    assert status(lines, "R12") == "applies" and objective_plan(lines)["choice"] == "Trade"


def test_needs_labels_lists_missing_tags_in_words():
    f, why = facts_for()
    r05 = next(l for l in evaluate_rules(RULES, f, why) if l["rule"] == "R05")
    assert r05["status"] == "needs_labels"
    assert "our jungler's level-1 strength" in r05["missing"]


def test_card_layout():
    f, why = facts_for(gold={"top": -900, "jng": 0, "mid": 0, "bot": 0, "sup": 0})
    fc = {"lanes": [{"role": r, "label": r, "gold15": g, "reasons": []} for r, g in zip(R, [-900, 0, 0, 0, 0])],
          "objectives": [{"key": "first_dragon", "label": "First dragon", "blue_prob": 0.3},
                         {"key": "grubs", "label": "Void grubs", "blue_prob": 0.5}]}
    card = build_card(RULES, "Blue", "A", "B", f, why, BY_ROLE, fc, {"break_even_gold": 500, "win_condition": {}})
    assert [s["key"] for s in card["sections"]] == list(RULES["sections"])
    jp = next(s for s in card["sections"] if s["key"] == "jungle_path")
    assert {l["rule"] for l in jp["lines"]} >= {"R08", "R20"}
    assert all(l["sources"] and l["evidence"] for s in card["sections"] for l in s["lines"])
    assert card["conflicts"][0]["active"] and card["gold15"]["break_even"] == 500
    assert card["first_objective"]["choice"] == "Dragon" and "30%" in card["first_objective"]["note"]
    assert len([w for w in card["watch_for"] if w["rule"] in ("R10", "R14")]) == 2
    assert card["level1"]["rule"] is None and "not labelled" in card["level1"]["text"]


def test_real_tag_sheet_marks_unlabelled_trait_columns_unknown():
    t = champ_tag("Aatrox")
    if t is None:
        pytest.skip("tag sheet not available")
    assert t["early"]["engage"] is None and t["early"]["lvl1"] is None and t["timing"]
