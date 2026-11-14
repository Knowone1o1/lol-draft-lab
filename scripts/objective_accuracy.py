"""Backtest the plan card's first Dragon and Grubs forecasts walk-forward, 2024-01 to the frozen holdout.

Same model as the app (draftlab/early_engine.py, scripts/build_early.py): logistic regression on objective features
(team and jungler objective habits, side-lane strength, team early rating, Elo). For every month M, early stats are
rebuilt from games before M and the model is refit on the 12 months before M (training leagues), then scores M.
Grubs only exist from 2024, so the Grubs model starts once it has 500 training games.

    .venv/Scripts/python scripts/objective_accuracy.py
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import HOLDOUT_START, REPORT_DIR, ROLES, TRAIN_LEAGUES  # noqa: E402
from draftlab.early import (EarlyStats, attach_early, game_lane_matrix, lane_long, load_early_raw,  # noqa: E402
                            objective_features, team_long)
from draftlab.eval.bootstrap import mean_ci, stat_ci  # noqa: E402
from draftlab.eval.dataset import load_eval_games  # noqa: E402
from draftlab.training import month_starts  # noqa: E402

OBJ = {"first_dragon": "First Dragon", "grubs": "Grubs (majority of void grubs)"}
START, HISTORY, MIN_TRAIN = pd.Timestamp("2024-01-01"), 12, 500


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def calib_slope(y, p):
    from draftlab.rules.backtest import logit_fit
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))
    return logit_fit(np.column_stack([np.ones(len(z)), z]), y, np.zeros(len(z)))[1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    hold = pd.Timestamp(HOLDOUT_START)
    games = load_eval_games()
    team_raw, pl_raw = load_early_raw()
    g = attach_early(games, team_raw, pl_raw)
    g = g[g["date"] < hold].reset_index(drop=True)            # holdout never used, not even as history
    lanes, teams = lane_long(g), team_long(g)
    fit_mask = g["league"].isin(TRAIN_LEAGUES).to_numpy()
    months = [m for m in month_starts(g["date"]) if m >= START - pd.DateOffset(months=HISTORY)]

    X = {o: {} for o in OBJ}
    for m in months:
        nxt = m + pd.offsets.MonthBegin(1)
        idx = np.nonzero(((g["date"] >= m) & (g["date"] < nxt)).to_numpy())[0]
        if not len(idx):
            continue
        es = EarlyStats(lanes, teams, m)
        for i in idx:
            row = g.iloc[i]
            picks = {s: {r: row[f"{s}_{r}_champ"] for r in ROLES} for s in ("Blue", "Red")}
            players = {s: {r: row[f"{s}_{r}_player"] for r in ROLES} for s in ("Blue", "Red")}
            tms = (row["Blue_teamname"], row["Red_teamname"])
            LX = game_lane_matrix(es, picks, players, tms, row["blue_elo"] - row["red_elo"])
            for o in OBJ:
                X[o][i] = objective_features(es, o, picks, tms, LX)
        print(f"  {m:%Y-%m}: {len(idx)} games", flush=True)

    L = ["# Plan card: first Dragon and Grubs forecasts, backtested", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Walk-forward from 2024-01 to {HOLDOUT_START} (holdout not used): "
         "each month, early stats are rebuilt from earlier games and the app's objective model (logistic regression) is "
         f"refit on the {HISTORY} months before, then scores that month. Probabilities are from Blue's side; a team's "
         "chance on the card is the same number from its side. 95% intervals resample whole series.", ""]
    summary, calib, warn = [], [], []
    for o, label in OBJ.items():
        y_col = f"blue_{o}"
        preds = []
        for m in [m for m in months if m >= START]:
            nxt = m + pd.offsets.MonthBegin(1)
            lo = m - pd.DateOffset(months=HISTORY)
            tr = [i for i in X[o] if lo <= g.at[i, "date"] < m and fit_mask[i] and np.isfinite(g.at[i, y_col])]
            te = [i for i in X[o] if m <= g.at[i, "date"] < nxt and np.isfinite(g.at[i, y_col])]
            if len(tr) < MIN_TRAIN or not te:
                continue
            Xtr, ytr = np.array([X[o][i] for i in tr]), np.array([g.at[i, y_col] for i in tr])
            mo = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            eo = LogisticRegression(max_iter=1000).fit(Xtr[:, [4]], ytr)
            Xte = np.array([X[o][i] for i in te])
            for i, p, pe in zip(te, mo.predict_proba(Xte)[:, 1], eo.predict_proba(Xte[:, [4]])[:, 1]):
                preds.append({"i": i, "y": g.at[i, y_col], "p": p, "p_elo": pe, "p_base": ytr.mean(),
                              "series": g.at[i, "series_id"], "tier": g.at[i, "tier"], "month": m})
        P = pd.DataFrame(preds)
        y, p, ser = P["y"].to_numpy(), P["p"].to_numpy(), P["series"].to_numpy()
        acc = mean_ci(((p > 0.5) == y).astype(float), ser, B=1000)
        d_base = mean_ci(ll(y, p) - ll(y, P["p_base"].to_numpy()), ser, B=1000)
        d_elo = mean_ci(ll(y, p) - ll(y, P["p_elo"].to_numpy()), ser, B=1000)
        auc = stat_ci(roc_auc_score, y, p, ser, B=300)
        slope = stat_ci(calib_slope, y, p, ser, B=300)
        summary.append(f"| {label} | {P['month'].min():%Y-%m} | {len(P):,} | {y.mean():.0%} | "
                       f"{acc[0]:.1%} [{acc[1]:.1%}, {acc[2]:.1%}] | {auc[0]:.3f} [{auc[1]:.3f}, {auc[2]:.3f}] | "
                       f"{ll(y, p).mean():.4f} | {d_base[0]:+.4f} [{d_base[1]:+.4f}, {d_base[2]:+.4f}] | "
                       f"{d_elo[0]:+.4f} [{d_elo[1]:+.4f}, {d_elo[2]:+.4f}] | {slope[0]:.2f} [{slope[1]:.2f}, {slope[2]:.2f}] |")
        # calibration on the team's own chance (both sides), as the card shows it
        team_p = np.r_[p, 1 - p]
        team_y = np.r_[y, 1 - y]
        team_s = np.r_[ser, ser]
        for a, b in ((0, .3), (.3, .4), (.4, .5), (.5, .6), (.6, .7), (.7, 1.01)):
            sel = (team_p >= a) & (team_p < b)
            if sel.sum() >= 50:
                m_, lo_, hi_ = mean_ci(team_y[sel], team_s[sel], B=1000)
                calib.append(f"| {label} | {a:.0%}–{min(b, 1):.0%} | {sel.sum():,} | {team_p[sel].mean():.0%} | "
                             f"{m_:.0%} [{lo_:.0%}, {hi_:.0%}] |")
        sel = team_p < 0.40
        m_, lo_, hi_ = mean_ci(team_y[sel], team_s[sel], B=1000)
        warn.append(f"| {label} | {sel.sum():,} ({sel.mean():.0%} of team-games) | {team_p[sel].mean():.0%} | "
                    f"{m_:.0%} [{lo_:.0%}, {hi_:.0%}] |")
        print(summary[-1])

    L += ["## 1. Overall", "",
          "Accuracy = the side given over 50% got it. Log-loss: lower is better; Δ = model minus the baseline "
          "(negative = the model is better). Calibration slope 1 = probabilities are the right size; below 1 = too "
          "confident.", "",
          "| Objective | From | Games | Blue got it | Accuracy | AUC | Log-loss | Δ vs base rate | Δ vs Elo only | Calibration slope |",
          "|---|---|---|---|---|---|---|---|---|---|"] + summary
    L += ["", "## 2. Calibration: when the card says X%, how often did the team get it?", "",
          "| Objective | Card's chance | Team-games | Average card chance | Actually got it |", "|---|---|---|---|---|"] + calib
    L += ["", "## 3. The card's warning line", "",
          "The card warns when the forecast gives the team under 40% for the objective it plans to take.", "",
          "| Objective | Warned | Average card chance | Actually got it |", "|---|---|---|---|"] + warn
    L += ["", "## Caveats", "",
          "- Grubs = the team with more void grubs; Oracle's Elixir has counts, not who took the first camp.",
          "- These are the app's numbers before any draft-specific rule; the card's Dragon / Grubs choice comes from "
          "the rules (R11, R12, R19) and is tested in reports/rule_tests.md."]
    (REPORT_DIR / "objective_accuracy.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote reports/objective_accuracy.md")


if __name__ == "__main__":
    main()
