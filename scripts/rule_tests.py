"""Part 3: test the early-game rules walk-forward (2024-01 to the frozen holdout) and label them
Supported / Unclear / Contradicted. Writes reports/rule_tests.md and data/rules/evidence.json (shown on the plan card).

    .venv/Scripts/python scripts/rule_tests.py            # reuses data/rules/backtest_facts.pkl if present
    .venv/Scripts/python scripts/rule_tests.py --rebuild  # recompute the walk-forward fact table (slow)
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import HOLDOUT_START, REPORT_DIR  # noqa: E402
from draftlab.rules import load_rules  # noqa: E402
from draftlab.rules.backtest import (build_fact_table, label, logit_fit, ols_fit,  # noqa: E402
                                     series_bootstrap)

RULES_DIR = ROOT / "data" / "rules"
FACTS = RULES_DIR / "backtest_facts.pkl"
SEED, B = 0, 1000


def sig(x):
    return 1 / (1 + np.exp(-x))


def _b(v):
    return v.map(lambda x: np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(bool(x)))


# ----------------------------------------------------------------------------- tests
def t_win(df, xcols, k):
    """logit P(win) = strength (offset, coef 1) + b0 + b.x ; returns coefficient k."""
    X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy(float) for c in xcols])
    return logit_fit(X, df["win"].to_numpy(float), df["strength"].to_numpy(float))[k]


def t_gold(df, y, xcols, k):
    """y = a + c.strength + b.x (gold); returns coefficient k of xcols."""
    X = np.column_stack([np.ones(len(df)), df["strength"].to_numpy(float)] + [df[c].to_numpy(float) for c in xcols])
    return ols_fit(X, df[y].to_numpy(float))[2 + k]


def lane_rows(T, cols):
    parts = []
    for lane in ("top", "mid", "bot"):
        d = T[["series_id", "gameid", "strength", f"gd15_{lane}"] + [c.format(lane=lane) for c in cols]].copy()
        d.columns = ["series_id", "gameid", "strength", "y"] + [c.replace(".{lane}", "") for c in cols]
        d["lane"] = lane
        parts.append(d)
    return pd.concat(parts, ignore_index=True)


def define_tests(T):
    """Each test: (rule, what is tested, data, fit, direction, unit)."""
    T = T.copy()
    for c in [c for c in T.columns if c.startswith("fires.")]:
        T[c] = _b(T[c])
    out = []

    # R19: Dragon vs Grubs, games where one team took first Dragon and the other the Grubs majority
    d = T[(T["first_dragon"] == 1) & (T["grubs"] == 0) & T["strength"].notna()]
    out.append(("R19", "Games where one team took first Dragon and the other took the Grubs majority: does the Dragon "
                "team win more than team strength predicts?", d, lambda x: t_win(x, [], 0), +1, "win", None))

    # R11 / R12: value of early objectives for the comps the rule names
    base = T[T["first_dragon"].notna() & T["grubs"].notna() & T["strength"].notna()].copy()
    base["obj"] = (base["first_dragon"] + base["grubs"]) / 2
    for rid, sign, words in (("R11", +1, "Do early comps with bot/mid priority gain MORE from first Dragon + Grubs "
                                         "majority than other comps?"),
                             ("R12", -1, "Do late comps facing early comps lose LESS from conceding first Dragon + "
                                         "Grubs majority than other comps?")):
        d = base[base[f"fires.{rid}"].notna()].copy()
        d["f"], d["fx"] = d[f"fires.{rid}"], d[f"fires.{rid}"] * d["obj"]
        out.append((rid, words + " (interaction: rule fires x share of the two objectives taken)",
                    d, lambda x: t_win(x, ["f", "obj", "fx"], 3), sign, "win", "f"))

    # R15: facilitator jungler into a carry jungler loses the farm race (premise)
    d = T[T["fires.R15"].notna() & T["gd15_jng"].notna() & T["strength"].notna()]
    out.append(("R15", "Premise: a facilitator jungler facing a carry jungler is behind in jungle gold at 15 "
                "(the plan itself, ganking instead of farming, is not visible in the data).",
                d, lambda x: t_gold(x, "gd15_jng", ["fires.R15"], 0), -1, "gold", "fires.R15"))

    # R17: no Strong lane anywhere
    d = T[T["fires.R17"].notna() & T["strength"].notna()]
    out.append(("R17", "Drafts with no Strong lane and at least one Weak lane (forecast): do they win less than team strength predicts?",
                d, lambda x: t_win(x, ["fires.R17"], 1), -1, "win", "fires.R17"))

    # R24: losing top matchup
    d = T[T["fires.R24"].notna() & T["gd15_top"].notna() & T["strength"].notna()]
    out.append(("R24", "Top lanes forecast Weak: are they behind in top gold at 15 (beyond team strength)?",
                d, lambda x: t_gold(x, "gd15_top", ["fires.R24"], 0), -1, "gold", "fires.R24"))

    # R25: early comp with mid + bot priority converts into plates
    d = T[T["fires.R25"].notna() & T["plates_diff"].notna() & T["fc_team"].notna() & T["strength"].notna()].copy()
    d["fc_k"] = d["fc_team"] / 1000
    out.append(("R25", "Early comps with mid and bot priority: more tower plates than strength and the team gold "
                "forecast predict?", d, lambda x: t_gold(x, "plates_diff", ["fc_k", "fires.R25"], 1), +1, "plates", "fires.R25"))

    # R29: blind matchup-dependent picks
    L = lane_rows(T, ["our.{lane}.blind", "our.{lane}.matchup_dependent"])
    L = L[L["y"].notna() & L["strength"].notna()]
    L["blind"], L["dep"] = _b(L["our.blind"]), _b(L["our.matchup_dependent"])
    L = L[L["blind"].notna() & L["dep"].notna()]
    L["bd"] = L["blind"] * L["dep"]
    out.append(("R29", "Lane gold at 15: is blind-picking a matchup-dependent champion worse than blind-picking "
                "others (interaction blind x matchup-dependent)?", L, lambda x: t_gold(x, "y", ["blind", "dep", "bd"], 2),
                -1, "gold", "bd"))

    # R30: weak mid-jungle 2v2
    d = T[T["our.midjng"].notna() & T["gd15_jng"].notna() & T["gd15_mid"].notna() & T["strength"].notna()].copy()
    d["weak"] = (d["our.midjng"] == "Weak").astype(float)
    d["mj"] = d["gd15_jng"] + d["gd15_mid"]
    out.append(("R30", "Mid-jungle duos forecast Weak: behind in mid + jungle gold at 15 (beyond team strength)?",
                d, lambda x: t_gold(x, "mj", ["weak"], 0), -1, "gold", "weak"))

    # R40: countering a star laner
    L = lane_rows(T, ["their.{lane}.star", "their.{lane}.blind"])
    L = L[(L["their.star"] == True) & L["their.blind"].notna() & L["y"].notna() & L["strength"].notna()].copy()  # noqa: E712
    L["countered"] = _b(L["their.blind"])
    out.append(("R40", "Against an enemy star laner: does our laner do better at 15 when we counterpicked the star "
                "(star picked first) than when the star counterpicked us? (Jungle pressure is not visible.)",
                L, lambda x: t_gold(x, "y", ["countered"], 0), +1, "gold", "countered"))
    return out


NOT_TESTED = {
    "R05": "Needs Lvl 1 tags (empty in the sheet).", "R06": "Needs Lvl 1 tags and level-1 vision data.",
    "R07": "Needs Lvl 1 tags (empty in the sheet).", "R08": "Needs the jungler's start side (not in Oracle's Elixir).",
    "R13": "Needs Early spike tags and objective fight data.", "R18": "Needs vision at 15 (not loaded).",
    "R27": "Describes a habit after the lead exists (comeback rate is context, not a rule effect).",
    "R31": "Needs Laning / Roam / Peel / Dive tags (empty in the sheet).",
    "R32": "Needs Laning tags for bot and support (empty in the sheet).",
    "R38": "Needs the order of dragons (not in Oracle's Elixir).", "R39": "Scouting checklist, not a rule effect.",
}


def fmt(v, unit):
    if unit == "win":
        return f"{(sig(v) - 0.5) * 100:+.1f} pp"
    if unit == "plates":
        return f"{v:+.2f} plates"
    return f"{v:+,.0f} gold"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--start", default="2024-01-01")
    a = ap.parse_args()
    rules = load_rules(RULES_DIR / "rules.json")
    if a.rebuild or not FACTS.exists():
        T = build_fact_table(a.start, rules)
        T.to_pickle(FACTS)
        by_month = T.attrs.get("prio_cuts", {})
        last = max(by_month)
        (RULES_DIR / "prio_cuts.json").write_text(json.dumps(
            {"note": "Lane priority Strong/Weak cuts (gold @15 forecast): 80th percentile of |forecast| over the 12 "
                     "months before each month. 'current' = the latest month before the holdout, used by the app.",
             "current": by_month[last], "current_month": last, "by_month": by_month}, indent=1), encoding="utf-8")
    T = pd.read_pickle(FACTS)
    assert (T["date"] < pd.Timestamp(HOLDOUT_START)).all(), "holdout games in the fact table"
    by_id = {r["id"]: r for r in rules["rules"]}

    results = []
    for rid, words, d, fit, direction, unit, ind in define_tests(T):
        n_games = d["gameid"].nunique()
        # games where the rule's condition held (and where it didn't): the smaller group decides if we can tell
        n_fired = n_games if ind is None else int(d.loc[d[ind] == 1, "gameid"].nunique())
        n_other = n_games if ind is None else int(d.loc[d[ind] == 0, "gameid"].nunique())
        if min(n_fired, n_other) < 150:
            results.append({"rule": rid, "test": words, "unit": unit, "direction": "+" if direction > 0 else "-",
                            "effect": np.nan, "lo": np.nan, "hi": np.nan, "n_rows": len(d), "n_games": int(n_games),
                            "n_fired": n_fired, "label": "Unclear", "effect_text": "–",
                            "ci_text": f"too few cases (rule held in {n_fired} games)"})
            print(f"{rid}: too few cases ({n_fired} fired) -> Unclear")
            continue
        eff, lo, hi = series_bootstrap(fit, d, B=B, seed=SEED)
        lab = label(lo, hi, direction, min(n_fired, n_other))
        results.append({"rule": rid, "test": words, "unit": unit, "direction": "+" if direction > 0 else "-",
                        "effect": eff, "lo": lo, "hi": hi, "n_rows": len(d), "n_games": int(n_games),
                        "n_fired": n_fired, "label": lab, "effect_text": fmt(eff, unit),
                        "ci_text": f"[{fmt(lo, unit)}, {fmt(hi, unit)}]"})
        print(f"{rid}: {fmt(eff, unit)} {results[-1]['ci_text']}  n={n_games:,} games -> {lab}")

    fire = {c.split(".")[1]: float(pd.to_numeric(T[c].map(lambda v: np.nan if v is None else float(bool(v))),
                                                  errors="coerce").mean())
            for c in T.columns if c.startswith("fires.") and c.count(".") == 1}

    evidence = {r["rule"]: {"label": r["label"], "effect": r["effect_text"], "ci": r["ci_text"],
                            "n_games": r["n_games"], "n_fired": r["n_fired"], "test": r["test"]} for r in results}
    for rid, why in NOT_TESTED.items():
        evidence.setdefault(rid, {"label": "Not tested", "why": why})
    meta = {"window": f"{a.start} to {HOLDOUT_START} (holdout excluded)", "generated": f"{datetime.now():%Y-%m-%d %H:%M}",
            "method": "walk-forward facts; strength offset; series bootstrap 95% CI", "bootstrap": B}
    (RULES_DIR / "evidence.json").write_text(json.dumps({"meta": meta, "rules": evidence}, indent=1, ensure_ascii=False),
                                             encoding="utf-8")

    L = ["# Early-game rules: tests on past games (Part 3)", "",
         f"Generated {meta['generated']}. Window: {a.start} to {HOLDOUT_START} ({T['gameid'].nunique():,} games); the "
         "frozen holdout is not used, not even as history.", "",
         "**Method.** For each month, early-game stats are rebuilt from games played before it and the lane gold-at-15 "
         "forecast is refit on the 12 months before it; then every game's facts are computed with the same code as the "
         "plan card, and each rule's condition is evaluated. Win outcomes use the walk-forward team-strength logit "
         "(strength_g15_tier_cal) as a fixed offset with coefficient 1 (the A4 approach); gold and plate outcomes "
         "control for it. 95% intervals resample whole series (B = 1,000). ", "",
         "**Labels.** Supported = the interval excludes zero in the direction the rule claims. Contradicted = it "
         "excludes zero in the other direction. Unclear = it includes zero, or the rule held (or didn't hold) in fewer than 150 games.", "",
         "**What is tested.** Oracle's Elixir shows outcomes, not what teams tried. Most tests check the rule's "
         "premise (e.g. \"this lane is behind at 15\") or whether the objective it prioritises is worth more for the "
         "comps it names; none can show that following the plan wins more.", "",
         "| Rule | Test | Effect | 95% CI | Games tested | Rule held in | Label |", "|---|---|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['rule']} | {r['test']} | {r['effect_text']} | {r['ci_text']} | {r['n_games']:,} | {r['n_fired']:,} | **{r['label']}** |")
    L += ["", "Win effects are shown in percentage points at a 50% base chance; for R11 and R12 that is the change "
          "in the value of taking both objectives (vs neither).", "", "## Not tested", "",
          "| Rule | Why |", "|---|---|"]
    for rid, why in NOT_TESTED.items():
        L.append(f"| {rid} | {why} |")
    for r in rules["rules"]:
        if r["id"] not in evidence:
            L.append(f"| {r['id']} | {'Live / timeline information' if r['kind'] == 'live' else 'Not testable from match data'} |")
    L += ["", "## How often each draft rule fires (share of team-games where it is known)", "",
          "| Rule | Fires | Known in |", "|---|---|---|"]
    for c in sorted(c for c in T.columns if c.startswith("fires.")):
        v = T[c].map(lambda x: np.nan if x is None else float(bool(x)))
        L.append(f"| {c[6:]} | {v.mean():.0%} | {v.notna().mean():.0%} |" if v.notna().any() else f"| {c[6:]} | – | 0% (needs labels) |")
    L += ["", "## Caveats", "",
          "- Champion tags (comp timing, primary style) are today's sheet, not labels as of each game: a small hindsight risk.",
          "- Grubs = the team with more void grubs (Oracle's Elixir has counts, not who took the first camp).",
          "- Ten tests at 95%: about one in twenty labels could be wrong by chance alone.",
          "- R24 and R30 partly check the forecast that defines Weak lanes; they confirm the card's lane calls, not the advice.",
          "- Lane priority: Strong / Weak = the lane's forecast is in the top / bottom 10% of forecasts over the 12 months "
          "before each month (per lane; same for the mid-jungle 2v2). Cuts per month: data/rules/prio_cuts.json."]
    (REPORT_DIR / "rule_tests.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote reports/rule_tests.md and data/rules/evidence.json; fire rates: {json.dumps({k: round(v, 3) for k, v in fire.items()})}")


if __name__ == "__main__":
    main()
