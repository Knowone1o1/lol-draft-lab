"""Score a locked pre-match forecast (scripts/lock_forecast.py) against what actually happened.

Uses ONLY the frozen model copies in the lock folder, after checking the forecast and model fingerprints, so a model
rebuilt with the final's own games can't leak into its score.

Result input, either:
  --result reports/forecast_lock_<date>/result.json   typed in by hand (see result_template.json in that folder)
  --from-data                                          read the games from Oracle's Elixir once it has them
                                                       (refresh the raw data first)

    .venv/Scripts/python scripts/score_final.py --lock reports/forecast_lock_2026-11-14 --from-data
    .venv/Scripts/python scripts/score_final.py --lock reports/forecast_lock_2026-11-14 --result reports/forecast_lock_2026-11-14/result.json

Scores (lower is better for Brier and log-loss):
  series   P(team A wins the series): model vs the old app model vs a coin flip; probability given to the exact score
  games    each game's chance from the frozen model WITH the actual draft and side, vs the strength-only part
           (no draft), the old app model and a coin flip
  draft    where each real pick / ban ranked in the imitation model's list at that moment (top-1 / top-5 rates),
           and the game-1 first-ban prediction
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ROLES = ["top", "jng", "mid", "bot", "sup"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def brier(p, y):
    return (p - y) ** 2


def logloss(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p if y else 1 - p)


def verify(lock: Path):
    text = (lock / "forecast.json").read_text(encoding="utf-8")
    want = (lock / "forecast.json.sha256").read_text(encoding="utf-8").split()[0]
    got = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if got != want:
        sys.exit(f"forecast.json was changed after locking (sha256 {got} != {want})")
    fc = json.loads(text)
    for f, h in fc["frozen_model_sha256"].items():
        if sha(lock / "models" / f) != h:
            sys.exit(f"frozen model {f} was changed after locking")
    return fc, got


def games_from_data(fc, date_window=3):
    import pandas as pd
    from draftlab.data import game_actions, load_games
    g, _, _ = load_games()
    d0 = pd.Timestamp(fc["match_date"])
    teams = {fc["team_a"], fc["team_b"]}
    sel = g[(g["date"] >= d0 - pd.Timedelta(days=1)) & (g["date"] < d0 + pd.Timedelta(days=date_window))
            & g.apply(lambda r: {r["Blue_teamname"], r["Red_teamname"]} == teams, axis=1)].sort_values("date")
    out = []
    for r in sel.to_dict("records"):
        out.append({"blue": r["Blue_teamname"], "red": r["Red_teamname"], "blue_first": bool(r["blue_first_pick"] == 1),
                    "actions": [c for (_, _, c) in game_actions(r)],
                    "winner": r["Blue_teamname"] if r["blue_win"] == 1 else r["Red_teamname"],
                    "rosters": {s: {ro: r[f"{s}_{ro}_player"] for ro in ROLES} for s in ("Blue", "Red")}})
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", required=True, type=Path)
    ap.add_argument("--result", type=Path)
    ap.add_argument("--from-data", action="store_true")
    a = ap.parse_args()
    lock = a.lock.resolve()
    fc, fc_hash = verify(lock)
    os.environ["DRAFTLAB_MODEL_DIR"] = str(lock / "models")       # frozen models only
    os.environ["DRAFTLAB_WIN_MODEL"] = "a3a4"
    from draftlab.engine import DraftEngine
    from draftlab.features import DraftState, comp_features

    A, B = fc["team_a"], fc["team_b"]
    games = games_from_data(fc) if a.from_data else json.loads(a.result.read_text(encoding="utf-8"))["games"]
    if not games:
        sys.exit("no games found for this match yet (refresh the data, or use --result)")
    e = DraftEngine(lock / "models" / "draftlab.pkl")
    rt = e.win_model
    T = {x["name"]: x for x in e.teams()}
    default_ros = {t: {r: v["key"] for r, v in T[t]["roster"].items()} for t in (A, B) if t in T}

    rows, ranks, locked = [], {"pick": [], "ban": []}, set()
    wins = {A: 0, B: 0}
    for k, gm in enumerate(games, 1):
        blue, red = gm["blue"], gm["red"]
        ros = gm.get("rosters") or {"Blue": default_ros.get(blue, {}), "Red": default_ros.get(red, {})}
        acts = gm["actions"]
        # draft predictions: rank of every real action in the imitation model's list at that moment
        for i, champ in enumerate(acts):
            st = DraftState(blue, red, ros["Blue"], ros["Red"], acts[:i], frozenset(locked), blue_first=gm["blue_first"])
            nxt = st.next_action()
            if nxt is None or not champ:
                continue
            side, kind, _ = nxt
            cands, p, _, _ = e.action_probs(st, side, kind, temperature=1.0)
            order = [cands[j] for j in sorted(range(len(cands)), key=lambda j: -p[j])]
            ranks[kind].append(order.index(champ) + 1 if champ in order else None)
        st = DraftState(blue, red, ros["Blue"], ros["Red"], acts, frozenset(locked), blue_first=gm["blue_first"])
        x = comp_features(st, e.snap, 0.0, 0.0)[1:]
        p_model = rt.prob(blue, red, gm["blue_first"], x)
        p_strength = 1 / (1 + math.exp(-rt.strength_logit(blue, red, gm["blue_first"])))
        legacy = e.models["win"].proba(comp_features(st, e.snap, e.team_elo(blue), e.team_elo(red)))[0]
        to_a = (lambda q: q) if blue == A else (lambda q: 1 - q)
        y = 1 if gm["winner"] == A else 0
        wins[gm["winner"]] += 1
        rows.append({"game": k, "blue": blue, "blue_first": gm["blue_first"], "winner": gm["winner"], "y": y,
                     "model": to_a(p_model), "strength_only": to_a(p_strength), "legacy": to_a(float(legacy)), "coin": 0.5})
        locked |= set(st.picks("Blue")) | set(st.picks("Red"))

    y_series = 1 if wins[A] > wins[B] else 0
    score = f"{wins[A]}-{wins[B]}"
    f, lg = fc["forecast"], fc["baselines"]["legacy_app_model"]
    L = [f"# Scorecard: {A} vs {B}, {fc['event']}", "",
         f"Forecast locked {fc['locked_at_utc']} (sha256 `{fc_hash}`), scored with the frozen models from that lock. "
         f"Result: **{A if y_series else B} wins {score if y_series else score[::-1]}** "
         f"({A}–{B} {score}). Lower Brier and log-loss are better.", "",
         "## Series", "", "| Forecast | P(" + A + " wins) | Brier | Log-loss | Chance given to the exact score |", "|---|---|---|---|---|"]
    for name, p, dist in (("Model (A3 + A4)", f["p_a_series"], f["score_dist"]), ("Old app model", lg["p_a_series"], lg["score_dist"]),
                          ("Coin flip", 0.5, None)):
        exact = f"{dist.get(score, 0):.1%}" if dist else "–"
        L.append(f"| {name} | {p:.1%} | {brier(p, y_series):.3f} | {logloss(p, y_series):.3f} | {exact} |")
    L += ["", "## Game by game (each with the actual draft and side)", "",
          "| Game | Blue | First pick | Winner | Model | Strength only | Old app model |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['game']} | {r['blue']} | {'blue' if r['blue_first'] else 'red'} | {r['winner']} | "
                 f"{r['model']:.1%} | {r['strength_only']:.1%} | {r['legacy']:.1%} |")
    L += ["", f"Chances are for {A}.", "", "| Forecast | Mean Brier | Mean log-loss | Favourite won |", "|---|---|---|---|"]
    for name in ("model", "strength_only", "legacy", "coin"):
        bs = sum(brier(r[name], r["y"]) for r in rows) / len(rows)
        ll = sum(logloss(r[name], r["y"]) for r in rows) / len(rows)
        hit = sum((r[name] > 0.5) == bool(r["y"]) for r in rows if r[name] != 0.5)
        n = sum(r[name] != 0.5 for r in rows)
        label = {"model": "Model (A3 + A4)", "strength_only": "Strength only (no draft)", "legacy": "Old app model", "coin": "Coin flip"}[name]
        L.append(f"| {label} | {bs:.3f} | {ll:.3f} | {f'{hit}/{n}' if n else '–'} |")
    L += ["", "## Draft predictions", ""]
    for kind in ("pick", "ban"):
        rk = ranks[kind]
        if rk:
            t1 = sum(1 for x in rk if x == 1) / len(rk)
            t5 = sum(1 for x in rk if x and x <= 5) / len(rk)
            L.append(f"- {kind.capitalize()}s: {len(rk)} actions, top-1 {t1:.0%}, top-5 {t5:.0%}"
                     + (" (backtest reference: about 50% top-5)." if kind == "pick" else "."))
    g1 = games[0]
    st1 = DraftState(g1["blue"], g1["red"], {}, {}, g1["actions"], frozenset(), blue_first=g1["blue_first"])
    for team in (A, B):
        side = "Blue" if g1["blue"] == team else "Red"
        bans = st1.bans(side)
        pred = fc["draft_predictions"][team]["game1_first_ban"]
        if bans:
            share = next((x["share"] for x in pred if x["champion"] == bans[0]), None)
            L.append(f"- {team} game-1 first ban: **{bans[0]}** (forecast gave it "
                     + (f"{share:.0%}" if share is not None else "under its top 5") + f"; top forecast {pred[0]['champion']} {pred[0]['share']:.0%}).")
    L += ["", "## How to read this", "",
          "- One series is a handful of games: it can't confirm or refute the model. The honest test is many games (the "
          "walk-forward backtest, and the frozen-holdout final evaluation, A5).",
          "- A 61% forecast is supposed to miss about 4 times in 10. Judge it by its scores against the baselines over "
          "many matches, not by whether the favourite won."]
    out = lock / "scorecard.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
