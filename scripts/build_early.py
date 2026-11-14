"""Build and backtest the early-game module; write models/early.pkl and reports/early_game.md.

Same no-look-ahead method as the draft models: for every month, early-game statistics are rebuilt
from games played before that month; models are fit on earlier months and scored on the last ones.

Usage:  .venv\\Scripts\\python scripts\\build_early.py [--test-months 4]
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, mean_absolute_error, r2_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.config import MODEL_DIR, REPORT_DIR, ROLE_LABELS, ROLES, TRAIN_LEAGUES  # noqa: E402
from draftlab.data import load_games  # noqa: E402
from draftlab.early import (LANE_FEATURES, OBJ_FEATURES, OBJECTIVES, TEAM_FEATURES, EarlyStats,  # noqa: E402
                            attach_early, game_lane_matrix, lane_long, load_early_raw, objective_features,
                            team_features, team_long)
from draftlab.training import month_starts  # noqa: E402

OBJ_LABEL = {"first_dragon": "First dragon", "grubs": "Void grubs majority", "first_blood": "First blood",
             "first_tower": "First tower"}


def game_inputs(row):
    picks = {s: {r: row[f"{s}_{r}_champ"] for r in ROLES} for s in ("Blue", "Red")}
    players = {s: {r: row[f"{s}_{r}_player"] for r in ROLES} for s in ("Blue", "Red")}
    return picks, players, (row["Blue_teamname"], row["Red_teamname"]), row["blue_elo"] - row["red_elo"]


def collect(g, lanes, teams, months, mask, test_months, log=print):
    ex = {s: {"lane": {r: ([], []) for r in ROLES}, "team": ([], []), "obj": {o: ([], []) for o in OBJECTIVES},
              "meta": []} for s in ("train", "test")}
    for m in months:
        nxt = m + pd.offsets.MonthBegin(1)
        sel = g[(g["date"] >= m) & (g["date"] < nxt) & mask & g["blue_gd15"].notna()]
        if sel.empty:
            continue
        split = "test" if m in test_months else "train"
        es = EarlyStats(lanes, teams, m)
        for row in sel.to_dict("records"):
            picks, players, tms, elo = game_inputs(row)
            LX = game_lane_matrix(es, picks, players, tms, elo)
            for i, r in enumerate(ROLES):
                y = row.get(f"blue_{r}_gd15")
                if y is not None and np.isfinite(y):
                    ex[split]["lane"][r][0].append(LX[i])
                    ex[split]["lane"][r][1].append(y)
            ex[split]["team"][0].append(team_features(LX))
            ex[split]["team"][1].append(row["blue_gd15"])
            for o in OBJECTIVES:
                y = row.get(f"blue_{o}")
                if y is not None and np.isfinite(y):
                    ex[split]["obj"][o][0].append(objective_features(es, o, picks, tms, LX))
                    ex[split]["obj"][o][1].append(int(y))
            ex[split]["meta"].append((row["gameid"], row.get("league"), row["blue_win"]))
        log(f"  {m:%Y-%m} [{split}] {len(sel)} games")
    return ex


def reg_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    big = np.abs(y) >= 300  # sign accuracy on lanes that actually had a lead
    return {"mae": mean_absolute_error(y, p), "r2": r2_score(y, p),
            "sign": float((np.sign(p[big]) == np.sign(y[big])).mean()) if big.any() else float("nan"), "n": len(y)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()

    games, elo, _ = load_games()
    team_raw, pl_raw = load_early_raw()
    g = attach_early(games, team_raw, pl_raw)
    print(f"{len(g)} games; {g['blue_gd15'].notna().mean():.0%} with gold at 15")
    lanes, teams = lane_long(g), team_long(g)
    mask = g["league"].isin(TRAIN_LEAGUES)
    months = month_starts(g["date"])[1:][-(args.train_months + args.test_months):]
    test_months = set(months[-args.test_months:])
    ex = collect(g, lanes, teams, months, mask, test_months)

    L = ["# Early game: forecasting the state at 15 minutes", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Every month, early-game statistics are rebuilt from games "
         "played *before* that month; models are fit on earlier months and scored on the most recent "
         f"{args.test_months} months ({min(test_months):%Y-%m} to {max(test_months):%Y-%m}). Gold difference is "
         "from the blue side's point of view; each lane is that player against the opposite laner.", ""]

    # ------------------------------------------------------------- lanes
    final = {"lane": {}, "obj": {}}
    L += ["## 1. Gold difference at 15, per lane", "",
          "MAE = average miss in gold. R² = share of the variation explained. Direction = how often the "
          "predicted leader was right, on lanes where someone led by 300+ gold.", "",
          "| Lane | Test lanes | Model MAE | Elo-only MAE | Model R² | Direction (model) | Direction (Elo only) |",
          "|---|---|---|---|---|---|---|"]
    lane_coefs = []
    for r in ROLES:
        Xtr, ytr = map(np.array, ex["train"]["lane"][r])
        Xte, yte = map(np.array, ex["test"]["lane"][r])
        m = Ridge(alpha=1.0).fit(Xtr, ytr)
        e = Ridge(alpha=1.0).fit(Xtr[:, [4]], ytr)
        a, b = reg_metrics(yte, m.predict(Xte)), reg_metrics(yte, e.predict(Xte[:, [4]]))
        L.append(f"| {ROLE_LABELS[r]} | {a['n']} | {a['mae']:.0f} | {b['mae']:.0f} | {a['r2']:.3f} | {a['sign']:.1%} | {b['sign']:.1%} |")
        final["lane"][r] = Ridge(alpha=1.0).fit(np.vstack([Xtr, Xte]), np.r_[ytr, yte])
        lane_coefs.append((r, final["lane"][r].coef_))
    L += ["", "Gold per unit of each feature (features are in thousands of gold; Elo difference in units of 400):", "", "",
          "| Lane | " + " | ".join(LANE_FEATURES) + " |", "|---|" + "---|" * len(LANE_FEATURES)]
    for r, c in lane_coefs:
        L.append(f"| {ROLE_LABELS[r]} | " + " | ".join(f"{v:+.0f}" for v in c) + " |")

    # ------------------------------------------------------------- team
    Xtr, ytr = map(np.array, ex["train"]["team"])
    Xte, yte = map(np.array, ex["test"]["team"])
    tm = Ridge(alpha=1.0).fit(Xtr, ytr)
    te = Ridge(alpha=1.0).fit(Xtr[:, [4]], ytr)
    a, b = reg_metrics(yte, tm.predict(Xte)), reg_metrics(yte, te.predict(Xte[:, [4]]))
    final["team"] = Ridge(alpha=1.0).fit(np.vstack([Xtr, Xte]), np.r_[ytr, yte])
    L += ["", "## 2. Team gold difference at 15", "",
          "| Model | MAE | R² | Direction |", "|---|---|---|---|",
          f"| Draft + players + teams | {a['mae']:.0f} | {a['r2']:.3f} | {a['sign']:.1%} |",
          f"| Elo only | {b['mae']:.0f} | {b['r2']:.3f} | {b['sign']:.1%} |",
          f"| Predict 0 | {mean_absolute_error(yte, np.zeros(len(yte))):.0f} | 0 | — |", "",
          "Weights: " + ", ".join(f"{n} {v:+.0f}" for n, v in zip(TEAM_FEATURES, final['team'].coef_)), ""]

    # ------------------------------------------------------------- objectives
    L += ["## 3. Early objectives", "", "| Objective | Test games | Base rate (blue) | Model log-loss | Base-rate log-loss | Elo-only log-loss | Model AUC | Accuracy |",
          "|---|---|---|---|---|---|---|---|"]
    for o in OBJECTIVES:
        Xtr, ytr = map(np.array, ex["train"]["obj"][o])
        Xte, yte = map(np.array, ex["test"]["obj"][o])
        if len(Xtr) < 100 or len(Xte) < 50:
            continue
        mo = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        eo = LogisticRegression(max_iter=1000).fit(Xtr[:, [4]], ytr)
        p, pe = mo.predict_proba(Xte)[:, 1], eo.predict_proba(Xte[:, [4]])[:, 1]
        base = np.full(len(yte), ytr.mean())
        L.append(f"| {OBJ_LABEL[o]} | {len(yte)} | {ytr.mean():.1%} | {log_loss(yte, p):.4f} | {log_loss(yte, base):.4f} | "
                 f"{log_loss(yte, pe):.4f} | {roc_auc_score(yte, p):.3f} | {((p > .5) == yte).mean():.1%} |")
        final["obj"][o] = LogisticRegression(max_iter=1000).fit(np.vstack([Xtr, Xte]), np.r_[ytr, yte])

    # ------------------------------------------------------------- team profiles
    cur = g[(g["date"].dt.year == g["date"].max().year) & mask & g["blue_gd15"].notna()]
    rows = []
    for side, sign in (("Blue", 1), ("Red", -1)):
        rows.append(pd.DataFrame({"team": cur[f"{side}_teamname"], "league": cur["league"],
                                  "gd15": sign * cur["blue_gd15"], "win": cur["blue_win"] if side == "Blue" else 1 - cur["blue_win"]}))
    tl = pd.concat(rows)
    prof = tl.groupby("team").agg(games=("win", "size"), league=("league", lambda s: s.mode().iloc[0]),
                                  win_rate=("win", "mean"), avg_gd15=("gd15", "mean"))
    ahead = tl[tl["gd15"] >= 1500].groupby("team")["win"].agg(["size", "mean"]).rename(columns={"size": "ahead_games", "mean": "win_when_ahead"})
    behind = tl[tl["gd15"] <= -1500].groupby("team")["win"].agg(["size", "mean"]).rename(columns={"size": "behind_games", "mean": "win_when_behind"})
    prof = prof.join(ahead).join(behind).fillna({"ahead_games": 0, "behind_games": 0})
    prof = prof[prof["games"] >= 20].sort_values("avg_gd15", ascending=False)
    REPORT_DIR.mkdir(exist_ok=True)
    prof.to_csv(REPORT_DIR / "team_early_profiles.csv")
    lg_ahead = (tl["gd15"] >= 1500)
    league_conv = tl[lg_ahead]["win"].mean()
    league_comeback = tl[tl["gd15"] <= -1500]["win"].mean()

    def top(df, col, n=10, asc=False, min_col=None, min_n=8):
        d = df if min_col is None else df[df[min_col] >= min_n]
        return d.sort_values(col, ascending=asc).head(n)

    L += ["", f"## 4. Team profiles ({g['date'].max().year}, training leagues, 20+ games)", "",
          f"Across all these games, a team 1,500+ gold ahead at 15 won **{league_conv:.0%}**; a team 1,500+ behind won "
          f"**{league_comeback:.0%}**. Full table: `reports/team_early_profiles.csv`.", ""]
    for title, df, extra in (
            ("Strongest early games (average gold lead at 15)", top(prof, "avg_gd15"), None),
            ("Most reliable closers (win rate when 1,500+ ahead, 8+ such games)", top(prof, "win_when_ahead", min_col="ahead_games"), "ahead"),
            ("Weakest closers: throw risk (win rate when 1,500+ ahead, 8+ such games)", top(prof, "win_when_ahead", asc=True, min_col="ahead_games"), "ahead"),
            ("Best comeback teams (win rate when 1,500+ behind, 8+ such games)", top(prof, "win_when_behind", min_col="behind_games"), "behind")):
        L += [f"### {title}", "", "| Team | League | Games | Win rate | Avg gold at 15 | Won when ahead | Won when behind |",
              "|---|---|---|---|---|---|---|"]
        for t, x in df.iterrows():
            wa = f"{x.win_when_ahead:.0%} ({int(x.ahead_games)})" if x.ahead_games else "—"
            wb = f"{x.win_when_behind:.0%} ({int(x.behind_games)})" if x.behind_games else "—"
            L.append(f"| {t} | {x.league} | {int(x.games)} | {x.win_rate:.0%} | {x.avg_gd15:+,.0f} | {wa} | {wb} |")
        L.append("")
    L += ["## Caveats", "",
          "- Gold at 15 is the outcome of draft *and* play; a draft that is 'ahead on paper' still depends on execution.",
          "- Player early ratings mix player skill with the champions they were given; the champion and matchup terms absorb part of that.",
          "- Some LPL games lack 15-minute data in Oracle's Elixir; they are left out of the early-game targets.", ""]

    stats_final = EarlyStats(lanes, teams, g["date"].max() + pd.Timedelta(days=1))
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "early.pkl", "wb") as f:
        pickle.dump({"stats": stats_final, "lane": final["lane"], "team": final["team"], "obj": final["obj"],
                     "elo": elo, "league_conversion": float(league_conv), "league_comeback": float(league_comeback),
                     "built": datetime.now().isoformat()}, f)
    (REPORT_DIR / "early_game.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nSaved models/early.pkl and reports/early_game.md in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
