"""What has been winning on the current patch, with team strength held equal. Descriptive only.

Every effect is "beyond team strength": the walk-forward strength_g15_tier_cal logit is a fixed offset (coefficient
1, as in A4), so a team that wins because it is stronger doesn't count for its champions / objectives / comp.
Two windows: the current patch (16.20 + 16.22) and the recent patches (16.17-16.22) as a larger check.

NOTE: these games are inside the frozen holdout (from 2026-05-12). This script only DESCRIBES them; nothing here is
fed back into any model, rule test or score, so the holdout stays clean for the final evaluation.

    .venv/Scripts/python scripts/patch_strategy.py
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.comp import champ_tag, comp_profile  # noqa: E402
from draftlab.config import RAW_DIR, REPORT_DIR, ROLE_LABELS, ROLES  # noqa: E402
from draftlab.eval.dataset import load_eval_games  # noqa: E402
from draftlab.rules.backtest import logit_fit  # noqa: E402
from draftlab.winmodels import get_model  # noqa: E402

CURRENT = ["16.20", "16.22"]
RECENT = ["16.17", "16.18", "16.20", "16.22"]
OBJ = {"firstblood": "First blood", "firstdragon": "First Dragon", "grubs": "Grubs majority",
       "firstherald": "Rift Herald", "firsttower": "First tower", "firstmidtower": "First mid tower",
       "atakhan": "Atakhan", "firstbaron": "First Baron"}


def sig(x):
    return 1 / (1 + np.exp(-x))


def objectives_table():
    want = ["gameid", "side", "position", "firstblood", "firstdragon", "firstherald", "firsttower", "firstmidtower",
            "firstbaron", "void_grubs", "opp_void_grubs", "atakhans", "opp_atakhans", "dragons", "opp_dragons"]
    parts = []
    for p in sorted(RAW_DIR.glob("2026*.csv")):
        cols = pd.read_csv(p, nrows=0).columns
        d = pd.read_csv(p, usecols=[c for c in want if c in cols], low_memory=False)
        parts.append(d[d["position"] == "team"])
    t = pd.concat(parts).drop_duplicates(["gameid", "side"])
    t["grubs"] = np.sign(t["void_grubs"] - t["opp_void_grubs"]).replace(0, np.nan).map({1.0: 1.0, -1.0: 0.0})
    t["atakhan"] = np.sign(t["atakhans"] - t["opp_atakhans"]).replace(0, np.nan).map({1.0: 1.0, -1.0: 0.0})
    return t.set_index(["gameid", "side"])


def team_rows(g, strength):
    """Two rows per game (one per team): win, strength offset, side, first pick, picks by role, bans, comp."""
    obj = objectives_table()
    rows = []
    for i, r in g.iterrows():
        for side, other, sg in (("Blue", "Red", 1), ("Red", "Blue", -1)):
            picks = {ro: r[f"{side}_{ro}_champ"] for ro in ROLES}
            epicks = {ro: r[f"{other}_{ro}_champ"] for ro in ROLES}
            me, them = comp_profile([c for c in picks.values() if isinstance(c, str)]), \
                comp_profile([c for c in epicks.values() if isinstance(c, str)])
            rec = {"gameid": r["gameid"], "series_id": r["series_id"], "patch": r["patch"], "league": r["league"],
                   "side": side, "win": float(r["blue_win"] if side == "Blue" else 1 - r["blue_win"]),
                   "off": sg * strength[i], "first_pick": float((r["blue_first_pick"] == 1) == (side == "Blue")),
                   "picks": picks, "bans": [r.get(f"{side}_ban{k}") for k in range(1, 6)],
                   "timing": me["timing"], "etiming": them["timing"], "style": me["main_style"], "estyle": them["main_style"]}
            key = (r["gameid"], side)
            for o in OBJ:
                rec[o] = obj.at[key, o] if key in obj.index and o in obj.columns else np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def above(df, mask):
    """Win rate above expectation for rows in mask, with a 95% interval; expectation from the strength offset."""
    x = df[mask]
    n = len(x)
    if n == 0:
        return n, np.nan, np.nan, np.nan, np.nan
    w, e = x["win"].mean(), sig(x["off"]).mean()
    b = logit_fit(np.ones((n, 1)), x["win"].to_numpy(float), x["off"].to_numpy(float))[0]
    se = 1 / np.sqrt((sig(x["off"] + b) * (1 - sig(x["off"] + b))).sum())
    return n, w, e, (sig(b) - .5) * 100, ((sig(b - 1.96 * se) - .5) * 100, (sig(b + 1.96 * se) - .5) * 100)


def fmt_ci(t):
    return f"[{t[0]:+.1f}, {t[1]:+.1f}]"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    games = load_eval_games().sort_values("date", kind="stable").reset_index(drop=True)
    sel = np.nonzero(games["patch"].astype(str).isin(RECENT).to_numpy())[0]
    prm = json.loads((REPORT_DIR / "runs" / "strength_g15_tier_cal_tuned.json").read_text())["params"]
    p = np.clip(get_model("strength_g15_tier_cal", **prm).walk_forward(games, sel), 1e-6, 1 - 1e-6)
    strength = dict(zip(sel, np.log(p / (1 - p))))
    T = team_rows(games.loc[sel], strength)
    W = {"Current patch (16.20 + 16.22)": T[T["patch"].astype(str).isin(CURRENT)],
         "Recent patches (16.17–16.22)": T}
    n_cur, n_rec = W["Current patch (16.20 + 16.22)"]["gameid"].nunique(), T["gameid"].nunique()
    L = ["# What wins on the current patch (team strength held equal)", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Current patch: {n_cur} games; recent patches: {n_rec} games. "
         "\"Above expectation\" = win rate beyond what team strength predicts (walk-forward strength model as a fixed "
         "offset), in percentage points at an even matchup, with a 95% interval. Descriptive only: these games are in "
         "the frozen holdout and nothing here feeds a model or a score.", ""]

    # sides
    L += ["## 1. Side and first pick", "", "| Window | Blue side | First pick |", "|---|---|---|"]
    for name, d in W.items():
        b, f = above(d, d["side"] == "Blue"), above(d, d["first_pick"] == 1)
        L.append(f"| {name} | {b[3]:+.1f} pp {fmt_ci(b[4])} | {f[3]:+.1f} pp {fmt_ci(f[4])} |")

    # objectives
    L += ["", "## 2. Objectives: win rate when you take it (above expectation)", "",
          "Taking an objective goes with being ahead, so these are \"how much it's worth\" upper bounds, not proof "
          "that forcing it wins. Read them as a ranking.", "",
          "| Objective | " + " | ".join(f"{k} (share / effect)" for k in W) + " |", "|---|---|---|"]
    for o, lab in OBJ.items():
        cells = []
        for d in W.values():
            m = d[o].notna()
            got = above(d[m], d[m][o] == 1)
            cells.append(f"{got[0] / max(1, m.sum()):.0%} / {got[3]:+.1f} pp {fmt_ci(got[4])}" if m.sum() >= 30 else "–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    # dragon vs grubs head to head
    L += ["", "Dragon vs Grubs head to head (games where one team took first Dragon and the other the Grubs majority; "
          "effect for the Dragon team):", ""]
    for name, d in W.items():
        m = (d["firstdragon"] == 1) & (d["grubs"] == 0)
        r = above(d, m)
        L.append(f"- {name}: {r[3]:+.1f} pp {fmt_ci(r[4])} over {r[0]} games")

    # comp timing
    L += ["", "## 3. Comp timing vs the enemy's (above expectation)", "",
          "| Your comp | vs Early | vs Mid / tempo | vs Late |", "|---|---|---|---|"]
    for name, d in W.items():
        L.append(f"| **{name}** | | | |")
        for t in ("Early", "Mid / tempo", "Late"):
            cells = []
            for et in ("Early", "Mid / tempo", "Late"):
                r = above(d, (d["timing"] == t) & (d["etiming"] == et))
                cells.append(f"{r[3]:+.1f} pp (n={r[0]})" if r[0] >= 20 else f"n={r[0]}")
            L.append(f"| {t} | " + " | ".join(cells) + " |")

    # styles
    L += ["", "## 4. Comp style (main style from the tag sheet), above expectation", "",
          "| Style | " + " | ".join(W) + " |", "|---|---|---|"]
    styles = T["style"].dropna().value_counts().index
    for s in styles:
        cells = []
        for d in W.values():
            r = above(d, d["style"] == s)
            cells.append(f"{r[3]:+.1f} pp {fmt_ci(r[4])} (n={r[0]})" if r[0] >= 20 else f"n={r[0]}")
        L.append(f"| {s} | " + " | ".join(cells) + " |")
    L += ["", "Best and worst style pairings (recent patches, 25+ team-games):", "",
          "| Your style | Enemy style | Team-games | Above expectation |", "|---|---|---|---|"]
    pairs = []
    for (s, es), x in T.groupby(["style", "estyle"]):
        if len(x) >= 25:
            r = above(T, (T["style"] == s) & (T["estyle"] == es))
            pairs.append((r[3], s, es, r))
    pairs.sort(reverse=True)
    for v, s, es, r in pairs[:4] + pairs[-4:]:
        L.append(f"| {s} | {es} | {r[0]} | {v:+.1f} pp {fmt_ci(r[4])} |")

    # champions
    L += ["", "## 5. Champions: presence and wins above expectation", "",
          "Presence = picked or banned, share of games. Effect = win rate above expectation when picked (recent "
          "patches; current-patch effect in brackets where it has 15+ picks). Champions picked 25+ times.", "",
          "| Champion | Main role | Presence (current) | Picks (recent) | Above expectation (recent) | Current patch |",
          "|---|---|---|---|---|---|"]
    cur = W["Current patch (16.20 + 16.22)"]
    pres = defaultdict(set)
    for r in cur.itertuples():
        for c in list(r.picks.values()) + [b for b in r.bans if isinstance(b, str)]:
            pres[c].add(r.gameid)
    champ_rows = []
    allc = set(c for pk in T["picks"] for c in pk.values() if isinstance(c, str))
    for c in allc:
        m = T["picks"].map(lambda pk: c in pk.values())
        if m.sum() < 25:
            continue
        r = above(T, m)
        mc = cur["picks"].map(lambda pk: c in pk.values())
        rc = above(cur, mc) if mc.sum() >= 15 else None
        role = max(ROLES, key=lambda ro: (T.loc[m, "picks"].map(lambda pk: pk.get(ro) == c)).sum())
        champ_rows.append((r[3], c, role, len(pres.get(c, ())) / n_cur, int(m.sum()), r, rc))
    champ_rows.sort(key=lambda t: -t[3])
    hdr = L[-2:]                       # move the table header below the sub-heading
    del L[-2:]
    L += ["**Most contested on the current patch:**", ""] + hdr
    for v, c, role, pr, k, r, rc in champ_rows[:10]:
        L.append(f"| {c} | {ROLE_LABELS[role]} | {pr:.0%} | {k} | {v:+.1f} pp {fmt_ci(r[4])} | "
                 + (f"{rc[3]:+.1f} pp (n={rc[0]})" if rc else "–") + " |")
    champ_rows.sort(key=lambda t: -t[0])
    L += ["", "**Over-performers (recent, interval above zero first):**", "",
          "| Champion | Main role | Presence (current) | Picks (recent) | Above expectation (recent) | Current patch |",
          "|---|---|---|---|---|---|"]
    ranked = sorted(champ_rows, key=lambda t: (-int(t[5][4][0] > 0), -t[0]))
    for v, c, role, pr, k, r, rc in ranked[:8]:
        L.append(f"| {c} | {ROLE_LABELS[role]} | {pr:.0%} | {k} | {v:+.1f} pp {fmt_ci(r[4])} | "
                 + (f"{rc[3]:+.1f} pp (n={rc[0]})" if rc else "–") + " |")
    L += ["", "**Under-performers (recent):**", "",
          "| Champion | Main role | Presence (current) | Picks (recent) | Above expectation (recent) | Current patch |",
          "|---|---|---|---|---|---|"]
    ranked = sorted(champ_rows, key=lambda t: (-int(t[5][4][1] < 0), t[0]))
    for v, c, role, pr, k, r, rc in ranked[:6]:
        L.append(f"| {c} | {ROLE_LABELS[role]} | {pr:.0%} | {k} | {v:+.1f} pp {fmt_ci(r[4])} | "
                 + (f"{rc[3]:+.1f} pp (n={rc[0]})" if rc else "–") + " |")
    n_sig = sum(1 for t in champ_rows if t[5][4][0] > 0 or t[5][4][1] < 0)
    L += ["", f"{n_sig} of {len(champ_rows)} champions have an interval that excludes zero; by chance alone about "
          f"{0.05 * len(champ_rows):.0f} would. Treat single champions as leads, not proof.", "",
          "## Caveats", "",
          "- Small samples: the current patch has a few hundred games, so most effects have wide intervals.",
          "- Objective effects are associations: teams that are ahead take objectives. They rank objectives, they "
          "don't prove that forcing one wins.",
          "- Comp timing and style come from today's tag sheet.",
          "- Holdout games, described only. Not used to train, tune or score anything."]
    (REPORT_DIR / "patch_strategy.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
