"""Fit the 15-minute game-plan model and the win-condition tables.

1. How gold at 15 turns into wins depending on both comps' TIMING and STYLE. Four variants are scored
   on the 120 days before the frozen holdout (config.HOLDOUT_START); a richer one is used only if its paired
   bootstrap interval of improvement over timing-only excludes zero. The holdout is scored once afterwards.
2. A style-vs-style check of the team's Comp Types sheet against the data.
3. Win conditions: which champions need a lead, game length and objective path of similar comps.

Writes models/brief.pkl and reports/comp_timing.md.
Usage: .venv\\Scripts\\python scripts\\build_brief.py
"""
from __future__ import annotations

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.brief import (OBJ_COLS, SCENARIOS, TIMING_POINTS, VARIANTS, break_even, features,  # noqa: E402
                            obj_features)
from draftlab.comp import style_advantage, style_matrix  # noqa: E402
from draftlab.config import HOLDOUT_START, MODEL_DIR, REPORT_DIR, ROLE_LABELS, TRAIN_LEAGUES  # noqa: E402
from draftlab.validation import fmt_ci, logloss_terms, paired_diff  # noqa: E402
from draftlab.data import load_games  # noqa: E402
from draftlab.early import attach_early, load_early_raw  # noqa: E402
from draftlab.wincon import (OBJ_PATH, STYLES, SimilarComps, attach_extra, carry_table, load_extra,  # noqa: E402
                             team_rows)



def fit(X, y):
    return LogisticRegression(C=10.0, max_iter=3000).fit(X, y)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    games, _, _ = load_games()
    t, p = load_early_raw()
    g = attach_early(games, t, p)
    te, pe = load_extra()
    g = attach_extra(g, te, pe)
    d = g[g["league"].isin(TRAIN_LEAGUES) & (g["date"] >= "2023-01-01")]
    rows = team_rows(d)
    for c, src in (("first_tower", "first_tower"), ("grubs", "grubs"), ("first_dragon", "first_dragon")):
        b = d.set_index("gameid")[f"blue_{src}"]
        rows[c] = np.where(rows["side"] == "Blue", rows["gameid"].map(b), 1 - rows["gameid"].map(b))
    rows["sadv"] = [style_advantage(a, b, STYLES) for a, b in zip(rows["sv"], rows["osv"])]
    tr_all = rows[rows["gd"].notna()].reset_index(drop=True)
    H = pd.Timestamp(HOLDOUT_START)
    cut = H - pd.Timedelta(days=120)
    train, test = tr_all[tr_all["date"] < cut], tr_all[(tr_all["date"] >= cut) & (tr_all["date"] < H)]
    # The frozen holdout is scored only on the user's explicit final evaluation (--final-evaluation).
    hold = tr_all[tr_all["date"] >= H] if "--final-evaluation" in sys.argv else tr_all.iloc[0:0]

    # ------------------------------------------------------------- 1. variants (validation, before the holdout)
    scores, terms = {}, {}
    for v in VARIANTS:
        m = fit(features(train, v), train["win"])
        pr = m.predict_proba(features(test, v))[:, 1]
        scores[v] = (log_loss(test["win"], pr), roc_auc_score(test["win"], pr))
        terms[v] = logloss_terms(test["win"], pr)
    gid = test["gameid"].to_numpy()
    diffs = {v: paired_diff(terms[v], terms["timing"], gid) for v in VARIANTS}
    clear = [v for v in VARIANTS if v != "timing" and diffs[v][2] < 0]
    chosen = min(clear, key=lambda v: scores[v][0]) if clear else "timing"
    # holdout, scored once for the chosen variant and the timing-only reference
    dev = tr_all[tr_all["date"] < H]
    hold_ll = {}
    for v in {chosen, "timing"}:
        m = fit(features(dev, v), dev["win"])
        hold_ll[v] = logloss_terms(hold["win"], m.predict_proba(features(hold, v))[:, 1]) if len(hold) else np.array([])
    final = fit(features(tr_all, chosen), tr_all["win"])
    ok = tr_all[OBJ_COLS].notna().all(axis=1)
    final_obj = fit(obj_features(tr_all[ok], chosen), tr_all[ok]["win"])

    # ------------------------------------------------------------- 2. style-vs-style check
    tm = fit(features(train, "timing"), train["win"])
    allp = tm.predict_proba(features(tr_all, "timing"))[:, 1]
    tr_all = tr_all.assign(resid=tr_all["win"] - allp,
                           main=[STYLES[int(np.argmax(v))] if v.sum() else None for v in tr_all["sv"]],
                           omain=[STYLES[int(np.argmax(v))] if v.sum() else None for v in tr_all["osv"]])
    styles, M = style_matrix()
    pair = tr_all.groupby(["main", "omain"]).agg(n=("resid", "size"), resid=("resid", "mean")).reset_index()
    pair["sheet"] = [("strong" if M[styles.index(a), styles.index(b)] > 0 else "weak" if M[styles.index(a), styles.index(b)] < 0 else "—")
                     if a in styles and b in styles else "—" for a, b in zip(pair["main"], pair["omain"])]
    pair["se"] = 0.5 / np.sqrt(pair["n"])
    checked = pair[(pair["sheet"] != "—") & (pair["n"] >= 150)].copy()
    checked["agrees"] = np.where(checked["sheet"] == "strong", checked["resid"] > 0, checked["resid"] < 0)
    checked["clear"] = checked["resid"].abs() > 2 * checked["se"]

    # ------------------------------------------------------------- 3. win conditions
    recent = g[g["league"].isin(TRAIN_LEAGUES) & (g["date"] >= g["date"].max() - pd.Timedelta(days=730))]
    carry = carry_table(recent)
    # Same season only: objectives change between seasons (Atakhan existed in 2025, not in 2026),
    # so drake/Baron patterns from other seasons are not comparable.
    season = rows["date"].max().year
    similar = SimilarComps(rows[rows["date"].dt.year == season])

    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "brief.pkl", "wb") as f:
        pickle.dump({"model": final, "obj_model": final_obj, "variant": chosen, "variant_scores": scores,
                     "carry_table": carry, "similar": similar, "built": datetime.now().isoformat(), "n": len(tr_all)}, f)

    # ------------------------------------------------------------- report
    L = ["# Comp timing, style and win conditions", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. {len(tr_all):,} team-games (training leagues, 2023 onward, with gold "
         "at 15), each game seen from both teams. Comp timing and style mix come from the team's champion tag sheet "
         "(Early = 0, Mid = 1, Late = 2; style points primary = 2, secondary = 1). The style-matchup score multiplies "
         "the two style mixes through the sheet's Strong vs / Weak vs columns.", "",
         f"## 1. Which model predicts best (validation: {cut:%Y-%m-%d} to {H:%Y-%m-%d}, fitted on earlier games)", "",
         f"The frozen holdout (from {H:%Y-%m-%d}) is not used for this choice. Δ = variant minus timing-only, "
         "paired bootstrap 95% interval over games; a richer variant is used only if its interval is entirely below zero.", "",
         "| Variant | What it adds | Log-loss | Δ vs timing [95% CI] | AUC |", "|---|---|---|---|---|"]
    for v, (ll, auc) in scores.items():
        L.append(f"| {'**' + v + '**' if v == chosen else v} | {VARIANTS[v]} | {ll:.4f} | "
                 f"{fmt_ci(*diffs[v], sign=True) if v != 'timing' else '—'} | {auc:.3f} |")
    L += ["", f"Used in the app: **{chosen}**" + (" (no style variant clearly beat timing only)." if chosen == "timing"
          else f" ({-diffs[chosen][0]:.4f} better than timing only on validation)."), ""]
    if len(hold):
        hl = "; ".join(f"{v} {hold_ll[v].mean():.4f}" for v in hold_ll)
        hd = (" Chosen minus timing-only: " + fmt_ci(*paired_diff(hold_ll[chosen], hold_ll["timing"],
              hold["gameid"].to_numpy()), sign=True) + ".") if chosen != "timing" else ""
        L += [f"Frozen holdout ({len(hold)} team-games, scored once): log-loss {hl}.{hd}", ""]
    names = ["gold@15", "own timing", "enemy timing", "gold × own timing", "gold × enemy timing", "Elo diff"]
    if chosen != "timing":
        names.append("style-matchup score")
    if chosen == "timing+style+gold":
        names.append("style score × gold")
    if chosen == "timing+learned":
        names += [f"style mix: {s}" for s in STYLES]
    L += ["| Term | Coefficient |", "|---|---|"] + [f"| {n} | {c:+.3f} |" for n, c in zip(names, final.coef_[0])]
    L += ["", "## 2. Does the data agree with the sheet's style matchups?", "",
          "For each pair of main styles the sheet calls strong or weak (150+ games), the average result above or below "
          "what gold at 15, timing and Elo predict. 'Clear' = more than two standard errors from zero.", "",
          "| Your main style | Enemy main style | Sheet says | Games | Result vs expected | Agrees | Clear |", "|---|---|---|---|---|---|---|"]
    for x in checked.sort_values("resid", ascending=False).itertuples():
        L.append(f"| {x.main} | {x.omain} | {x.sheet} | {x.n} | {x.resid:+.1%} | {'yes' if x.agrees else 'no'} | {'yes' if x.clear else 'no'} |")
    if len(checked):
        L += ["", f"The data agrees with the sheet on {int(checked['agrees'].sum())} of {len(checked)} pairs; "
              f"{int((checked['agrees'] & checked['clear']).sum())} agree clearly and "
              f"{int((~checked['agrees'] & checked['clear']).sum())} disagree clearly."]
    L += ["", "## 3. Break-even gold at 15 by timing (equal teams, neutral style matchup)", "",
          "| Your comp ↓ / enemy comp → | " + " | ".join(TIMING_POINTS) + " |", "|---|" + "---|" * len(TIMING_POINTS)]
    for a, av in TIMING_POINTS.items():
        L.append(f"| **{a}** | " + " | ".join(f"{break_even(final, av, bv, 0, chosen):+,.0f}" for bv in TIMING_POINTS.values()) + " |")
    if chosen != "timing":
        L += ["", "Effect of the style matchup at equal timing (Mid vs Mid): break-even gold with a style score of "
              + ", ".join(f"{s:+.2f}: {break_even(final, 1, 1, 0, chosen, sadv=s):+,.0f}" for s in (-0.3, 0, 0.3)) + "."]
    L += ["", "## 4. Which champions need a lead", "",
          "Team win rate when the champion's lane is 500+ gold ahead at 15 vs 500+ behind (last two years, shrunk "
          "towards the role average). 'vs role' is that gap minus the role's usual gap: how much MORE the game rides on "
          "this champion's lane than on a typical champion in the same role. Support and jungle gaps are large for "
          "every champion because their gold mostly mirrors the whole game, so the app names win conditions by "
          "damage share plus 'vs role', not by the raw gap.", ""]
    carry = carry.assign(vs_role=carry["lead_dependency"] - carry["role_dependency"])
    for r in ROLE_LABELS:
        sub = carry[(carry["role"] == r) & (carry["n"] >= 60)].sort_values("vs_role", ascending=False).head(6)
        if sub.empty:
            continue
        L += [f"### {ROLE_LABELS[r]} (role average gap: {sub['role_dependency'].iloc[0]:.0%})", "",
              "| Champion | Games | Win when ahead | Win when behind | Gap | vs role | Damage share | Gold share |",
              "|---|---|---|---|---|---|---|---|"]
        for x in sub.itertuples():
            L.append(f"| {x.champ} | {x.n} | {x.wr_ahead:.0%} | {x.wr_behind:.0%} | {x.lead_dependency:+.0%} | "
                     f"{x.vs_role:+.0%} | {x.dmg:.0%} | {x.gold:.0%} |")
        L.append("")
    sr = similar.rows
    act = similar.active
    labels = ["below -3k", "-3k to -1.5k", "-1.5k to -0.5k", "even", "+0.5k to +1.5k", "+1.5k to +3k", "above +3k"]
    L += ["## 5. Objective paths and game length: why everything is compared at the same gold at 15", "",
          f"Season {season} only ({len(sr):,} team-games), because objectives change between seasons (Atakhan was in "
          f"2025 and is gone in 2026). Objectives in this season's data: {', '.join(OBJ_PATH[o] for o in act)}.", "",
          "Early leads cause objectives and short games: a team ahead at 15 takes drakes, towers and Baron because it is "
          "ahead, and stomps end early. So raw 'objectives in wins' says little about a comp's needs. The table shows how "
          "strongly the objectives in wins depend on gold at 15:", "",
          "| Gold at 15 | Team-games | Win rate | " + " | ".join(f"{OBJ_PATH[o]} in wins" for o in act) + " | Median win length |",
          "|---|---|---|" + "---|" * len(act) + "---|"]
    for gb, lab in enumerate(labels):
        x = sr[sr["gbin"] == gb]
        w = x[x["win"] == 1]
        if len(x) < 50:
            continue
        L.append(f"| {lab} | {len(x):,} | {x['win'].mean():.0%} | " + " | ".join(f"{w[o].mean():.0%}" for o in act)
                 + f" | {w['minutes'].median():.1f} min |")
    L += ["", "In the app, a comp's game-length and objective tendencies are its similar comps' results MINUS what an "
          "average comp with the same gold at 15 did. Only differences larger than the noise (1.5 standard errors, and at "
          "least 4 points) are called out. This removes the early-lead effect, but it is still an association: what "
          "happens after minute 15 (a Baron throw, a soul fight) is not in the data.", "",
          "Caveat: comp timing and style are expert labels from the sheet, not learned from the data."]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "comp_timing.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
