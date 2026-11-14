"""Simulate a Bo5 fearless series many times with the app's draft engine.

Each game: both teams draft like themselves (imitation model, mode "imitate", temperature 0.7), champions picked in
earlier games of the series are locked (fearless), the app's draft win model gives a win probability for the
finished draft, shifted by the series' form swing (random per series, SD 0.4 on the logit ~ +/-70 rating points),
and the winner is drawn from it.

Sides and pick order (2026 First Selection): Oracle's Elixir doesn't say whether a team chose side or pick order, so
setups are sampled from what actually happened in 2026 fearless series: game 1 from all game-1 setups (coin toss for
who is "team 1"), later games from the setups the previous game's LOSER ended up with.

    .venv/Scripts/python scripts/simulate_series.py --a "Gen.G" --b "Bilibili Gaming" --n 1000
Writes reports/sim_<a>_vs_<b>.pkl (every game) and prints a summary; analysis in scripts/sim_findings.py.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import REPORT_DIR  # noqa: E402
from draftlab.data import load_games  # noqa: E402
from draftlab.engine import DraftEngine  # noqa: E402
from draftlab.features import DraftState  # noqa: E402


def setup_tables():
    """Empirical 2026 setups. Returns (game1: P(blue picks first)), later: list of (loser_blue, loser_first) weights."""
    g, _, _ = load_games()
    g = g[(g["date"].dt.year == 2026) & g["series_id"].notna()].sort_values("date")
    g1 = g[g["game"] == 1]
    p_blue_first_g1 = float(g1["blue_first_pick"].mean())
    prev = g.set_index(["series_id", "game"])
    rows = []
    for r in g[g["game"] >= 2].itertuples():
        key = (r.series_id, r.game - 1)
        if key not in prev.index:
            continue
        pg = prev.loc[key]
        if isinstance(pg, pd.DataFrame):
            pg = pg.iloc[0]
        loser = pg["Red_teamname"] if pg["blue_win"] == 1 else pg["Blue_teamname"]
        if loser not in (r.Blue_teamname, r.Red_teamname):
            continue
        loser_blue = loser == r.Blue_teamname
        loser_first = (r.blue_first_pick == 1) == loser_blue
        rows.append((loser_blue, loser_first))
    later = pd.Series(rows).value_counts(normalize=True)
    return p_blue_first_g1, later, len(g1), len(rows)


def draft(e, blue, red, ros, locked, blue_first, rng, temperature):
    st = DraftState(blue, red, ros[blue], ros[red], [], frozenset(locked), blue_first=blue_first)
    while st.next_action():
        c, _ = e.ai_move(st, temperature=temperature, rng=rng, mode="imitate")
        st.actions.append(c)
    return st


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="Gen.G")
    ap.add_argument("--b", default="Bilibili Gaming")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bo", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--form-sd", type=float, default=0.4,
                    help="series-level form swing on team A's win logit (0.4 ~ +/-70 rating points of uncertainty)")
    ap.add_argument("--win-model", choices=["app", "a3a4"], default="app",
                    help="app = the app's draft win model; a3a4 = A3 strength (side + first pick) + A4 draft layer, "
                         "from reports/a3a4_forecast_<a>_vs_<b>.json (scripts/a3a4_forecast.py)")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    e = DraftEngine()
    tag = f"{a.a.replace('.', '').replace(' ', '_')}_vs_{a.b.replace('.', '').replace(' ', '_')}"
    A34 = None
    if a.win_model == "a3a4":
        from draftlab.config import MODEL_DIR
        from draftlab.features import comp_features
        from draftlab.winmodels.runtime import A3A4WinModel
        rt = A3A4WinModel(MODEL_DIR / "a3a4_win.json")      # the app's A6 model (scripts/build_a3a4.py)
        A34 = {"strength_logit_blue": {f"{bl}|{bf}": rt.strength_logit(bl, rd, bool(bf))
                                       for bl, rd in ((a.a, a.b), (a.b, a.a)) for bf in (1, 0)},
               "draft": rt.art["draft"], "built": rt.art["built"]}

        def win_prob(st):
            """A3 strength (side + first pick + team scores) + A4 draft layer, from the app's artifact."""
            x = np.asarray(comp_features(st, e.snap, 0.0, 0.0)[1:], float)
            return rt.prob(st.blue_team, st.red_team, st.blue_first, x)
    else:
        win_prob = e.win_prob
    T = {x["name"]: x for x in e.teams()}
    ros = {n: {r: v["key"] for r, v in T[n]["roster"].items()} for n in (a.a, a.b)}
    p_bf1, later, n1, n2 = setup_tables()
    later_keys, later_p = list(later.index), later.to_numpy()
    print(f"setups from 2026: game 1 blue picks first {p_bf1:.0%} ({n1} games); later games, loser's setup "
          f"({n2} games): " + ", ".join(f"{'blue' if b else 'red'}+{'first' if f else 'second'} pick {p:.0%}"
                                         for (b, f), p in zip(later_keys, later_p)))
    need = a.bo // 2 + 1
    games, t0 = [], time.time()
    for s in range(a.n):
        wins = {a.a: 0, a.b: 0}
        locked, gno, last_loser = set(), 0, None
        form = float(rng.normal(0, a.form_sd))      # this series' form swing for team A (logit), same all series long
        while max(wins.values()) < need:
            gno += 1
            if gno == 1:
                blue = a.a if rng.random() < 0.5 else a.b
                blue_first = bool(rng.random() < p_bf1)
            else:
                lb, lf = later_keys[rng.choice(len(later_keys), p=later_p)]
                other = a.b if last_loser == a.a else a.a
                blue = last_loser if lb else other
                blue_first = lf if lb else (not lf)
            red = a.b if blue == a.a else a.a
            st = draft(e, blue, red, ros, locked, blue_first, rng, a.temperature)
            p_blue = win_prob(st)
            p_a = p_blue if blue == a.a else 1 - p_blue
            p_a_form = 1 / (1 + np.exp(-(np.log(p_a / (1 - p_a)) + form)))
            blue_won = rng.random() < (p_a_form if blue == a.a else 1 - p_a_form)
            winner = blue if blue_won else red
            last_loser = red if blue_won else blue
            wins[winner] += 1
            roles = e.roles(st)
            games.append({"series": s, "game": gno, "blue": blue, "red": red, "blue_first": blue_first,
                          "first_pick_team": blue if blue_first else red, "p_blue": p_blue,
                          "p_a": p_a, "form": form, "p_a_form": p_a_form, "winner": winner,
                          "score_after": f"{wins[a.a]}-{wins[a.b]}",
                          "picks": {blue: st.picks("Blue"), red: st.picks("Red")},
                          "bans": {blue: st.bans("Blue"), red: st.bans("Red")},
                          "roles": {blue: roles.get("Blue", {}), red: roles.get("Red", {})},
                          "order": [(sd, k, c) for (sd, k, _), c in zip(st.order, st.actions)],
                          "locked_before": len(locked)})
            locked |= set(st.picks("Blue")) | set(st.picks("Red"))
        if (s + 1) % 100 == 0:
            print(f"  {s + 1} series, {time.time() - t0:.0f}s", flush=True)
    G = pd.DataFrame(games)
    out = REPORT_DIR / f"sim_{tag}{'_a3a4' if A34 else ''}.pkl"
    with open(out, "wb") as f:
        pickle.dump({"games": G, "a": a.a, "b": a.b, "args": vars(a), "info": e.info, "win_model": a.win_model,
                     "a3a4": A34,
                     "elo": {a.a: e.team_elo(a.a), a.b: e.team_elo(a.b)},
                     "setups": {"g1_blue_first": p_bf1, "later": later.to_dict()},
                     "names": {k: e.name(v) for t in ros.values() for k, v in t.items()}}, f)
    S = G.groupby("series").agg(winner=("winner", "last"), score=("score_after", "last"), games=("game", "max"))
    print(f"wrote {out.name}: {a.a} wins {(S['winner'] == a.a).mean():.1%} of {len(S)} series; scores:")
    print(S["score"].value_counts(normalize=True).sort_index().round(3).to_string())


if __name__ == "__main__":
    main()
