"""Recent-patch pair stats for the app's reasons ("this patch, pros paired them N times and won X%").

Takes the latest patches (default: the most played patch of the last 14 days and the one before it)
and counts, for every pair of champions on the same team: games, wins, and how many times more often
than chance they were picked together. Writes models/recent.pkl.
Usage: .venv\\Scripts\\python scripts\\build_recent.py [--patches 2]
"""
import argparse
import itertools
import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import MODEL_DIR, ROLES  # noqa: E402
from draftlab.data import load_games, patch_label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=int, default=2)
    args = ap.parse_args()
    g, _, _ = load_games()
    recent = g[g["date"] >= g["date"].max() - pd.Timedelta(days=14)]
    cur = int(recent["patch_idx"].dropna().mode().iloc[0])
    played = sorted(p for p in g["patch_idx"].dropna().unique() if p <= cur)
    use = [int(p) for p in played[-args.patches:]]
    d = g[g["patch_idx"].isin(use)]
    team_games = 2 * len(d)
    champ_n, champ_w, pair_n, pair_w = Counter(), Counter(), Counter(), Counter()
    for r in d.to_dict("records"):
        for s in ("Blue", "Red"):
            won = r["blue_win"] if s == "Blue" else 1 - r["blue_win"]
            cs = sorted(r[f"{s}_{x}_champ"] for x in ROLES)
            for c in cs:
                champ_n[c] += 1
                champ_w[c] += won
            for a, b in itertools.combinations(cs, 2):
                pair_n[(a, b)] += 1
                pair_w[(a, b)] += won
    pairs = {}
    for (a, b), n in pair_n.items():
        exp = team_games * (champ_n[a] / team_games) * (champ_n[b] / team_games)
        pairs[(a, b)] = {"games": n, "wins": pair_w[(a, b)], "lift": n / exp if exp else 0.0}
    label = " to ".join(sorted({patch_label(p) for p in (use[0], use[-1])}))
    out = {"patches": [patch_label(p) for p in use], "label": label, "games": len(d), "pairs": pairs,
           "champ": {c: {"games": n, "wins": champ_w[c]} for c, n in champ_n.items()}}
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "recent.pkl", "wb") as f:
        pickle.dump(out, f)
    k = tuple(sorted(("Ezreal", "Karma")))
    print(f"patches {out['patches']}: {len(d)} games, {len(pairs)} pairs. Ezreal+Karma: {pairs.get(k)}")


if __name__ == "__main__":
    main()
