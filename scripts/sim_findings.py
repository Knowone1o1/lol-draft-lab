"""Findings from a simulated series file (scripts/simulate_series.py). Writes reports/sim_findings_<a>_vs_<b>.md.

    .venv/Scripts/python scripts/sim_findings.py reports/sim_GenG_vs_Bilibili_Gaming.pkl
"""
import pickle
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import REPORT_DIR, ROLE_LABELS  # noqa: E402


def ci(k, n):
    p = k / n
    h = 1.96 * np.sqrt(p * (1 - p) / n)
    return f"{p:.1%} [{p - h:.1%}, {p + h:.1%}]"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    d = pickle.load(open(sys.argv[1], "rb"))
    G, A, Bn = d["games"].copy(), d["a"], d["b"]
    short = {A: "GEN" if A == "Gen.G" else A, Bn: "BLG" if Bn == "Bilibili Gaming" else Bn}
    n = G["series"].nunique()
    S = G.groupby("series").agg(winner=("winner", "last"), score=("score_after", "last"), games=("game", "max"),
                                form=("form", "first"))
    a34 = d.get("a3a4")
    if a34:
        sl = a34["strength_logit_blue"]
        pa = {k: 1 / (1 + np.exp(-v)) for k, v in sl.items()}
        scorer = ("the A3 strength model (strength_g15_tier_cal: tier-weighted gold-at-15 ratings, side and 2026 "
                  "first-pick terms, temperature-scaled) plus the A4 draft layer (5 champion features, coefficient-1 "
                  f"offset, refitted through {a34['draft']['fitted_through']}) scores each finished draft")
        rating = (f"A3 pre-draft chance for {short[A]}: blue side {pa[f'{A}|1']:.1%} (blue picks first) / "
                  f"{pa[f'{A}|0']:.1%} (red picks first); red side {1 - pa[f'{Bn}|1']:.1%} / {1 - pa[f'{Bn}|0']:.1%}.")
    else:
        scorer = "the app's draft win model scores each finished draft"
        rating = f"Ratings used: {short[A]} {d['elo'][A]:.0f}, {short[Bn]} {d['elo'][Bn]:.0f}."
    L = [f"# {short[A]} vs {short[Bn]}: {n:,} simulated Bo5 fearless series" + (" (A3 + A4 win model)" if a34 else ""), "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} with the app's models (built {d['info']['built'][:10]}, data to "
         f"{d['info']['data_range'][1]}, patch {d['info']['patch']}). Each team drafts like itself (imitation model, "
         "sampling among its 8 likeliest choices at temperature 0.7); fearless locks every champion already picked "
         f"in the series; {scorer}; each series gets a random form swing "
         "(SD 0.4 on the win logit, about ±70 rating points); sides / pick order follow how 2026 series actually "
         f"went. {rating}", ""]

    # 1. series result
    wa = int((S["winner"] == A).sum())
    sc = S["score"].value_counts().sort_index()
    L += ["## 1. Who wins", "", f"- **{short[A]} wins {ci(wa, n)}** of series; {short[Bn]} {(n - wa) / n:.1%}.",
          f"- Average series length: {S['games'].mean():.2f} games; goes to game 5 in {(S['games'] == 5).mean():.0%}.",
          "", "| Final score (" + short[A] + "–" + short[Bn] + ") | Share |", "|---|---|"]
    for s_, k in sc.items():
        L.append(f"| {s_} | {k / n:.1%} |")
    gw = (G["winner"] == A).mean()
    L += ["", f"Per game, {short[A]} wins {gw:.1%} of all simulated games; the draft model alone gives {short[A]} "
          f"{G['p_a'].mean():.1%} on average (before form)."]

    # 2. game number (fearless depletion)
    L += ["", "## 2. Does fearless change the matchup as the series goes on?", "",
          f"Draft-model chance for {short[A]} by game number (before form), and how many champions were already locked.", "",
          f"| Game | Games played | {short[A]} chance (draft) | Range (5th–95th pct) | Locked before the game |", "|---|---|---|---|---|"]
    for gno, x in G.groupby("game"):
        L.append(f"| {gno} | {len(x):,} | {x['p_a'].mean():.1%} | {x['p_a'].quantile(.05):.0%}–{x['p_a'].quantile(.95):.0%} | "
                 f"{x['locked_before'].mean():.0f} |")

    # 3. sides and pick order
    G["a_blue"] = G["blue"] == A
    G["a_first"] = G["first_pick_team"] == A
    L += ["", "## 3. Side and first pick", "", f"| Setup for {short[A]} | Games | {short[A]} chance (draft) |", "|---|---|---|"]
    for (ab, af), x in G.groupby(["a_blue", "a_first"]):
        L.append(f"| {'Blue' if ab else 'Red'} side, {'first' if af else 'second'} pick | {len(x):,} | {x['p_a'].mean():.1%} |")

    # 4. comebacks
    L += ["", "## 4. Comebacks", "", "| Situation | Series | Came back to win the series |", "|---|---|---|"]
    for team, other in ((A, Bn), (Bn, A)):
        for down in ("0-1", "0-2", "1-2"):
            hits = 0
            tot = 0
            for sid, x in G.groupby("series"):
                sc_ = [tuple(map(int, s_.split("-"))) for s_ in x["score_after"]]
                me = [s_[0] if team == A else s_[1] for s_ in sc_]
                op = [s_[1] if team == A else s_[0] for s_ in sc_]
                dm, do = map(int, down.split("-"))
                if any(m == dm and o == do for m, o in zip(me, op)):
                    tot += 1
                    hits += x["winner"].iloc[-1] == team
            if tot >= 20:
                L.append(f"| {short[team]} down {down} | {tot:,} | {ci(hits, tot)} |")

    # 5. most present champions
    L += ["", "## 5. The champions the series revolves around", "",
          "Presence = share of series where the champion is picked or banned at least once.", "",
          "| Champion | Presence | Picked by " + short[A] + " (series) | Picked by " + short[Bn] + " (series) | Banned (games) |",
          "|---|---|---|---|---|"]
    pres, pa, pb, bans = Counter(), Counter(), Counter(), Counter()
    for sid, x in G.groupby("series"):
        seen, ca, cb = set(), set(), set()
        for r in x.itertuples():
            ca |= set(r.picks[A])
            cb |= set(r.picks[Bn])
            seen |= set(r.picks[A]) | set(r.picks[Bn]) | set(r.bans[A]) | set(r.bans[Bn])
            for c in r.bans[A] + r.bans[Bn]:
                bans[c] += 1
        pres.update(seen)
        pa.update(ca)
        pb.update(cb)
    for c, k in pres.most_common(12):
        L.append(f"| {c} | {k / n:.0%} | {pa[c] / n:.0%} | {pb[c] / n:.0%} | {bans[c]:,} |")

    # 6. what each team bans at the start of game 1
    L += ["", "## 6. Game 1 first bans and first picks", ""]
    g1 = G[G["game"] == 1]
    for team in (A, Bn):
        fb = Counter(r.bans[team][0] for r in g1.itertuples() if r.bans[team])
        fp = Counter(r.picks[team][0] for r in g1.itertuples() if r.picks[team])
        L.append(f"- **{short[team]}** first ban: " + ", ".join(f"{c} {k / len(g1):.0%}" for c, k in fb.most_common(4)) +
                 "; first pick: " + ", ".join(f"{c} {k / len(g1):.0%}" for c, k in fp.most_common(4)))

    # 7. players' pools
    L += ["", "## 7. Player pools under fearless", "",
          "Each player's most-played champion in these simulations, and how often it's still available in the last game "
          "of a series that goes to 5.", "", "| Team | Role | Player | Most played here | Share of their games | Still open in game 5 |",
          "|---|---|---|---|---|---|"]
    g5 = G[G["game"] == 5]
    from draftlab.engine import DraftEngine
    teams_info = {x["name"]: x for x in DraftEngine().teams()}
    pname = {t: {r: v["name"] for r, v in teams_info[t]["roster"].items()} for t in (A, Bn)}
    for team in (A, Bn):
        for role in ("top", "jng", "mid", "bot", "sup"):
            cnt = Counter()
            for r in G.itertuples():
                for c, rl in r.roles[team].items():
                    if rl == role:
                        cnt[c] += 1
            if not cnt:
                continue
            top, k = cnt.most_common(1)[0]
            used_before_5 = 0
            for sid in g5["series"]:
                x = G[(G["series"] == sid) & (G["game"] < 5)]
                used_before_5 += any(top in (r.picks[A] + r.picks[Bn]) for r in x.itertuples())
            open5 = 1 - used_before_5 / len(g5) if len(g5) else float("nan")
            L.append(f"| {short[team]} | {ROLE_LABELS[role]} | {pname[team].get(role, '')} | "
                     f"{top} | {k / len(G):.0%} | {open5:.0%} |")

    # 8. champions that move the draft
    L += ["", f"## 8. Picks that move the draft model most ({short[A]}'s chance, before form)", "",
          "Average chance when the team picks the champion minus when it doesn't (champions in 150+ games).", "",
          "| Team | Champion | Games picked | Chance with | Without | Difference |", "|---|---|---|---|---|---|"]
    for team in (A, Bn):
        sign = 1 if team == A else -1
        rows = []
        allc = Counter(c for r in G.itertuples() for c in r.picks[team])
        for c, k in allc.items():
            if k < 150:
                continue
            m = G["picks"].map(lambda p: c in p[team])
            pa_, pn_ = G.loc[m, "p_a"].mean(), G.loc[~m, "p_a"].mean()
            mine = (pa_ if team == A else 1 - pa_), (pn_ if team == A else 1 - pn_)
            rows.append((mine[0] - mine[1], c, k, mine))
        rows.sort(reverse=True)
        for diff, c, k, (w, wo) in rows[:3] + rows[-2:]:
            L.append(f"| {short[team]} | {c} | {k:,} | {w:.1%} | {wo:.1%} | {diff * 100:+.1f} pp |")
    L += ["", "(Chance shown from that team's side.)", "",
          "## Caveats", "",
          "- The draft adds little to the win model (A4: draft champion features improve log-loss by ~0.004); most of "
          "the per-game chance is the rating gap. Draft-driven differences above are small by design.",
          ("- Win model: A3 strength + A4 draft layer, fitted on all games up to the match (a forecast; nothing is "
           "scored). The app itself still uses its older win model until A6." if a34 else
           "- The win model here is the app's current one; the better-calibrated strength model from A3 "
           "(strength_g15_tier_cal) isn't in the app yet (A6)."),
          "- Teams draft like their past selves; real finals prep (pocket picks, targeted bans) is not modelled.",
          "- The frozen holdout (from 2026-05-12) was not evaluated; the models are only used to forecast."]
    out = REPORT_DIR / f"sim_findings_{Path(sys.argv[1]).stem[4:]}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
