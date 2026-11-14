"""Generate a synthetic Oracle's-Elixir-format CSV for testing the pipeline without real data.

Usage: python tests/make_synthetic.py OUT_DIR
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ["top", "jng", "mid", "bot", "sup"]
ORDER = [
    ("Blue", "ban"), ("Red", "ban"), ("Blue", "ban"), ("Red", "ban"), ("Blue", "ban"), ("Red", "ban"),
    ("Blue", "pick"), ("Red", "pick"), ("Red", "pick"), ("Blue", "pick"), ("Blue", "pick"), ("Red", "pick"),
    ("Red", "ban"), ("Blue", "ban"), ("Red", "ban"), ("Blue", "ban"),
    ("Red", "pick"), ("Blue", "pick"), ("Blue", "pick"), ("Red", "pick"),
]


def main(out_dir: Path, seed: int = 7):
    rng = np.random.default_rng(seed)
    champs = [f"Champ{i:02d}" for i in range(70)]
    champ_role = {c: [ROLES[i % 5]] + ([ROLES[(i + 2) % 5]] if i % 7 == 0 else []) for i, c in enumerate(champs)}
    strength = {c: rng.normal(0, 0.3) for c in champs}
    leagues = {"LCK": 8, "LEC": 8, "LFL": 6}
    teams, players = [], {}
    for lg, n in leagues.items():
        for t in range(n):
            name = f"{lg} Team {chr(65 + t)}"
            teams.append((lg, name, rng.normal(0, 0.6)))
            for r in ROLES:
                pool = [c for c in champs if r in champ_role[c]]
                players[(name, r)] = (f"{name.split()[0]}{chr(65 + t)}_{r}",
                                      list(rng.choice(pool, size=min(6, len(pool)), replace=False)))
    rows, gid = [], 0
    dates = pd.date_range("2025-01-10", "2026-10-30", freq="2D")
    for d in dates:
        for lg in leagues:
            lt = [t for t in teams if t[0] == lg]
            a, b = rng.choice(len(lt), size=2, replace=False)
            A, B = lt[a], lt[b]
            fearless = lg == "LCK" and d.year == 2026
            used = set()
            for gnum in (1, 2):
                gid += 1
                blue, red = (A, B) if gnum == 1 else (B, A)
                acts, picks = [], {"Blue": {}, "Red": {}}
                bans = {"Blue": [], "Red": []}
                order = {"Blue": [], "Red": []}
                for side, kind in ORDER:
                    team = blue if side == "Blue" else red
                    enemy = red if side == "Blue" else blue
                    taken = set(acts) | (used if fearless else set())
                    if kind == "ban":
                        pool = [c for r in ROLES for c in players[(enemy[1], r)][1] if c not in taken]
                        c = max(pool, key=lambda c: strength[c] + rng.normal(0, 0.4)) if pool else None
                        bans[side].append(c)
                    else:
                        open_r = [r for r in ROLES if r not in picks[side]]
                        r = rng.choice(open_r)
                        pool = [c for c in players[(team[1], r)][1] if c not in taken] or \
                               [c for c in champs if r in champ_role[c] and c not in taken]
                        c = max(pool, key=lambda c: strength[c] + rng.normal(0, 0.3))
                        picks[side][r] = c
                        order[side].append(c)
                    acts.append(c)
                used |= set(picks["Blue"].values()) | set(picks["Red"].values())
                score = (blue[2] - red[2]) + sum(strength[c] for c in picks["Blue"].values()) \
                    - sum(strength[c] for c in picks["Red"].values()) + 0.1
                bw = int(rng.random() < 1 / (1 + np.exp(-score)))
                base = dict(gameid=f"SYN{gid}", datacompleteness="complete", league=lg, year=d.year,
                            split="Spring", playoffs=0, date=f"{d:%Y-%m-%d} 10:{gnum:02d}:00", game=gnum,
                            patch=f"{d.year % 100 - 10}.{d.month:02d}")
                pid = 0
                for side, team in (("Blue", blue), ("Red", red)):
                    res = bw if side == "Blue" else 1 - bw
                    for r in ROLES:
                        pid += 1
                        pname = players[(team[1], r)][0]
                        rows.append({**base, "participantid": pid, "side": side, "position": r,
                                     "playername": pname, "playerid": f"oe:player:{pname}",
                                     "teamname": team[1], "teamid": f"oe:team:{team[1]}",
                                     "champion": picks[side][r], "result": res})
                    trow = {**base, "participantid": 100 if side == "Blue" else 200, "side": side,
                            "position": "team", "playername": None, "playerid": None, "teamname": team[1],
                            "teamid": f"oe:team:{team[1]}", "champion": None, "result": res,
                            "golddiffat15": rng.normal(0, 1500)}
                    for i in range(5):
                        trow[f"ban{i + 1}"] = bans[side][i]
                        trow[f"pick{i + 1}"] = order[side][i]
                    rows.append(trow)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "synthetic.csv", index=False)
    print(f"wrote {len(df) // 12} games to {out_dir / 'synthetic.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
