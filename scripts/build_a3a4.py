"""A6: build the app's win-model artifact models/a3a4_win.json (A3 strength + A4 draft layer).

A3 (strength_g15_tier_cal, tuned params) is fitted once on every game before today and stored as numbers:
  logit P(blue wins) = scale * (side + first_pick * (+1 blue / -1 red picks first) + score[blue] - score[red])
  score[team] = team effect + home-league effect + margin weight * current gold-at-15 rating
A4 (draft_champ, offset = that logit with coefficient 1) adds, refitted through the latest month:
  + sum_j beta_j * (x_j - mu_j) / sd_j     over the app's 5 champion features (blue minus red)
The app switches between this and the original model with config.WIN_MODEL ("a3a4" / "legacy").

This is a production fit for forecasting: it uses all games up to now, holdout period included. Nothing is scored.

    .venv/Scripts/python scripts/build_a3a4.py [--check reports/a3a4_forecast_GenG_vs_Bilibili_Gaming.json]
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
from draftlab.config import MODEL_DIR, REPORT_DIR  # noqa: E402
from draftlab.eval.dataset import load_eval_games  # noqa: E402
from draftlab.league import HOME_WINDOW_DAYS, home_leagues  # noqa: E402
from draftlab.winmodels import get_model  # noqa: E402
from draftlab.winmodels.base import day_of  # noqa: E402
from draftlab.winmodels.strength import tiered_gold15  # noqa: E402


def final_ratings(games, margins, k):
    """Each team's gold-at-15 margin rating AFTER all games (same update as strength.margin_rating)."""
    bt, rt = games["Blue_teamname"].to_numpy(), games["Red_teamname"].to_numpy()
    R, h = {}, 0.0
    for b, r, m in zip(bt, rt, margins):
        rb, rr = R.get(b, 0.0), R.get(r, 0.0)
        if not np.isnan(m):
            e = m - (rb - rr + h)
            R[b], R[r], h = rb + k * e, rr - k * e, h + 0.01 * e
    return R


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", type=Path, help="a3a4_forecast json to compare the 4 setups against")
    a = ap.parse_args()
    games = load_eval_games().sort_values("date", kind="stable").reset_index(drop=True)
    prm = json.loads((REPORT_DIR / "runs" / "strength_g15_tier_cal_tuned.json").read_text())["params"]
    m = get_model("strength_g15_tier_cal", **prm)
    dates = games["date"]
    days = day_of(dates)
    fit_day = (dates.max().normalize() + pd.Timedelta(days=1)).to_datetime64().astype("datetime64[D]")
    past = (dates >= pd.Timestamp(fit_day) - pd.Timedelta(days=HOME_WINDOW_DAYS)).to_numpy()
    home = home_leagues(games[past])
    D = m._dense(games)
    fit = m._fit_day(fit_day, dates, days, D, games["blue_win"].to_numpy(float),
                     games["Blue_teamname"].to_numpy(), games["Red_teamname"].to_numpy(), home, {})
    side, first_pick, w_margin = (float(v) for v in fit["dense"][:3])
    R = final_ratings(games, tiered_gold15(games, prm["w_major"]), prm["margin_k"])
    teams = sorted(set(games["Blue_teamname"]) | set(games["Red_teamname"]))
    score = {t: fit["team"].get(t, 0.0) + fit["league"].get(home.get(t, "?"), 0.0) + w_margin * R.get(t, 0.0)
             for t in teams}
    last = {t: str(d.date()) for t, d in pd.concat([
        games[["Blue_teamname", "date"]].rename(columns={"Blue_teamname": "t"}),
        games[["Red_teamname", "date"]].rename(columns={"Red_teamname": "t"})]).groupby("t")["date"].max().items()}
    print(f"A3 fitted for {fit_day}: side {side:+.3f}, first pick {first_pick:+.3f}, margin weight {w_margin:.3f}, "
          f"{len(teams)} teams")

    dm = get_model("draft_champ", base="strength_g15_tier_cal")
    rows = np.nonzero((dates >= dates.max().to_period("M").to_timestamp()).to_numpy())[0]
    dm.walk_forward(games, rows)
    mu, sd = dm._scale
    art = {
        "name": "a3a4", "built": datetime.now().isoformat(timespec="seconds"), "data_to": str(dates.max().date()),
        "strength": {"model": "strength_g15_tier_cal", "params": prm, "fit_day": str(fit_day), "scale": prm["logit_scale"],
                     "side": side, "first_pick": first_pick, "margin_weight": w_margin,
                     "score": score, "home": {t: home.get(t) for t in teams}, "last_game": last,
                     "league_score": {l: float(v) for l, v in fit["league"].items()}},
        "draft": {"model": "draft_champ", "features": ["champ_strength", "player_comfort", "synergy", "lane_counter",
                                                       "team_matchups"],
                  "beta_per_sd": [float(x) for x in dm._beta], "mu": [float(x) for x in mu], "sd": [float(x) for x in sd],
                  "coef": dm.coef_, "fitted_through": str(dates.max().to_period("M"))},
    }
    out = MODEL_DIR / "a3a4_win.json"
    out.write_text(json.dumps(art, indent=1), encoding="utf-8")
    print(f"wrote {out}; draft weights per SD: {dm.coef_}")

    if a.check and a.check.exists():
        ref = json.loads(a.check.read_text(encoding="utf-8"))
        from draftlab.winmodels.runtime import A3A4WinModel
        rt = A3A4WinModel(out)
        for k, v in ref["strength_logit_blue"].items():
            blue, bf = k.split("|")
            red = ref["b"] if blue == ref["a"] else ref["a"]
            mine = rt.strength_logit(blue, red, bf == "1")
            print(f"check {blue} blue, {'blue' if bf == '1' else 'red'} first: artifact {1 / (1 + np.exp(-mine)):.4f} "
                  f"vs walk-forward {1 / (1 + np.exp(-v)):.4f}")


if __name__ == "__main__":
    main()
