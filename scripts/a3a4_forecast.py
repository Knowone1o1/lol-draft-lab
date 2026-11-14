"""Forecast inputs for one matchup with the A3 strength model + A4 draft layer (for scripts/simulate_series.py).

  - A3: strength_g15_tier_cal (tuned params from reports/runs/strength_g15_tier_cal_tuned.json), fitted on every game
    before the match date, for the 4 setups: each team on blue side x blue / red picks first. Gives the pre-draft
    logit (blue's view), including the side term and the 2026 first-pick term.
  - A4: draft_champ (offset = that strength logit, coefficient 1) refitted on the latest month: draft weights per
    1 SD of the app's 5 champion features (champion strength, player comfort, synergy, lane counter, team matchups).

This is a FORECAST: models are fitted on all games up to now, holdout period included, and nothing is scored.

    .venv/Scripts/python scripts/a3a4_forecast.py --a "Gen.G" --b "Bilibili Gaming" --date 2026-11-14
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import REPORT_DIR  # noqa: E402
from draftlab.eval.dataset import load_eval_games  # noqa: E402
from draftlab.winmodels import get_model  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="Gen.G")
    ap.add_argument("--b", default="Bilibili Gaming")
    ap.add_argument("--date", default="2026-11-14")
    a = ap.parse_args()
    games = load_eval_games().sort_values("date", kind="stable").reset_index(drop=True)
    prm = json.loads((REPORT_DIR / "runs" / "strength_g15_tier_cal_tuned.json").read_text())["params"]

    # ---- A3: synthetic rows for the match, one per setup (copied from the latest international game)
    tmpl = games[games["international"]].iloc[-1]
    synth = []
    for blue, red in ((a.a, a.b), (a.b, a.a)):
        for bf in (1.0, 0.0):
            r = tmpl.copy()
            r["gameid"] = f"forecast:{blue}:{bf}"
            r["date"] = pd.Timestamp(a.date) + pd.Timedelta(hours=12)
            r["Blue_teamname"], r["Red_teamname"], r["blue_first_pick"] = blue, red, bf
            r["blue_win"] = np.nan
            for c in ("m_gold15", "m_gold25", "m_length"):
                if c in r:
                    r[c] = np.nan
            r["series_id"] = f"forecast:{blue}:{bf}"
            synth.append(r)
    ext = pd.concat([games, pd.DataFrame(synth)], ignore_index=True)
    idx = np.arange(len(games), len(ext))
    p = np.clip(get_model("strength_g15_tier_cal", **prm).walk_forward(ext, idx), 1e-6, 1 - 1e-6)
    strength = {f"{s['Blue_teamname']}|{int(s['blue_first_pick'])}": float(np.log(q / (1 - q)))
                for s, q in zip(synth, p)}
    for k, v in strength.items():
        blue, bf = k.split("|")
        print(f"A3 {blue} on blue, {'blue' if bf == '1' else 'red'} picks first: P(blue wins) = {1 / (1 + np.exp(-v)):.1%}")

    # ---- A4: draft weights on top of the A3 offset, refitted on everything before the latest month
    dm = get_model("draft_champ", base="strength_g15_tier_cal")
    last = np.nonzero((games["date"] >= games["date"].max().to_period("M").to_timestamp()).to_numpy())[0]
    dm.walk_forward(games, last)
    mu, sd = dm._scale
    out = {"a": a.a, "b": a.b, "date": a.date, "strength_logit_blue": strength,
           "draft": {"features": ["champ_strength", "player_comfort", "synergy", "lane_counter", "team_matchups"],
                     "beta_per_sd": [float(x) for x in dm._beta], "mu": [float(x) for x in mu],
                     "sd": [float(x) for x in sd], "fitted_through": f"{games['date'].max():%Y-%m}",
                     "coef": dm.coef_},
           "strength_params": prm}
    f = REPORT_DIR / f"a3a4_forecast_{a.a.replace('.', '').replace(' ', '_')}_vs_{a.b.replace('.', '').replace(' ', '_')}.json"
    f.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("A4 draft weights per SD:", dm.coef_)
    print(f"wrote {f.name}")


if __name__ == "__main__":
    main()
