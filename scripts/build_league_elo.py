"""League strength offsets for Elo (models/league_elo.json), used by the app.

Plain Elo has one pool per league: every league starts at 1500, so an ERL team that dominates its league
(e.g. Galions) ends up rated above Gen.G. Each team gets a home league, and each league an offset in Elo
points, fitted on games between teams from different home leagues (internationals, EMEA Masters, KeSPA
Cup...), with a prior: majors 0, an academy/ERL league = -200. Effective team strength = team Elo + offset.
The fitting code lives in draftlab/league.py; training uses the same code refitted month by month on
earlier games only (config.ELO_LEAGUE_ADJUST), and this script fits it once on the last two years for the app.
Usage: .venv\\Scripts\\python scripts\\build_league_elo.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import MODEL_DIR  # noqa: E402
from draftlab.data import load_games  # noqa: E402
from draftlab.league import HOME_WINDOW_DAYS, fit_offsets, home_leagues  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    g, elo, _ = load_games()
    recent = g[g["date"] >= g["date"].max() - np.timedelta64(HOME_WINDOW_DAYS, "D")]
    home = home_leagues(recent)
    # offsets are fitted on the plain (unadjusted) ratings, the same scale as the team ratings the app adds them to
    off, n = fit_offsets(recent, home, recent["blue_elo_raw"].to_numpy(), recent["red_elo_raw"].to_numpy())
    if not n:
        sys.exit("No cross-league games found.")
    off = {k: round(v, 1) for k, v in off.items()}
    out = {"offsets": off, "home": home, "n_cross_games": n,
           "note": "effective strength = team Elo + offsets[home league]"}
    MODEL_DIR.mkdir(exist_ok=True)
    (MODEL_DIR / "league_elo.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"{n} cross-league games. League offsets (Elo points):")
    for lg, v in sorted(off.items(), key=lambda t: -t[1])[:25]:
        print(f"  {lg:8s} {v:+7.1f}")
    for t in ("Gen.G", "T1", "T1 Esports Academy", "Bilibili Gaming", "G2 Esports", "Galions", "Karmine Corp"):
        if t in elo:
            print(f"  {t:20s} Elo {elo[t]:.0f} -> effective {elo[t] + off.get(home.get(t), 0):.0f} ({home.get(t)})")


if __name__ == "__main__":
    main()
