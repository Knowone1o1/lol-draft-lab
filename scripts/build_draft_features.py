"""Leakage-free draft features for the A4 ablation, one row per game, cached to data/draft_features.pkl.

Every game in month m uses statistics from games BEFORE month m only (monthly refresh, patch-targeted snapshot,
same settings as the pick/ban model's training). Blue-minus-red throughout.

    .venv\\Scripts\\python scripts\\build_draft_features.py [--start 2023-01-01]

Columns
  champ_*       5 champion features of the full draft (champ_strength, player_comfort, synergy, lane_counter,
                team_matchups), exactly the app's comp_features minus the Elo term
  b1_*, p1_*    the same after blue's first pick (7 actions) and after pick phase 1 (12 actions); NaN for games
                without a recorded order and for 2026 red-first-pick games (pick order is wrong there until C0)
  timing_diff   tag-sheet timing score, blue - red (0 = early ... 2 = late): negative = blue is the earlier comp
  timing_gap    |timing_diff|, the size of the timing mismatch
  style_adv     tag-sheet style matchup, blue's style mix x strong/weak matrix x red's style mix
  hv_diff       share of HIGH-VARIANCE style points (Dive, Pick, Skirmish, Hard engage) minus share of SAFE ones
                (Front-to-back, Protect the carry, Disengage, Poke / siege), blue - red. Split push counts as neutral.
                This split is an analyst assumption, not from the tag sheet.
  early_diff    data early score: sum over the 5 lanes of each champion's past lane gold lead at 15 in that
                role (k gold, shrunk to 0 with 20 pseudo-games, 180-day half-life), blue - red
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.comp import comp_profile, style_advantage, style_matrix  # noqa: E402
from draftlab.config import ROLES, ROOT  # noqa: E402
from draftlab.early import attach_early, lane_long, load_early_raw  # noqa: E402
from draftlab.eval import load_eval_games  # noqa: E402
from draftlab.features import actual_roles, comp_features, state_from_game  # noqa: E402
from draftlab.training import month_starts  # noqa: E402

OUT = ROOT / "data" / "draft_features.pkl"
CHAMP = ["champ_strength", "player_comfort", "synergy", "lane_counter", "team_matchups"]
STAGE_N = {"b1": 7, "p1": 12}
EARLY_K, EARLY_HL = 20.0, 180.0
HIGH_VAR = {"Dive", "Pick", "Skirmish", "Hard engage"}
SAFE = {"Front-to-back", "Protect the carry", "Disengage", "Poke / siege"}


def variance_score(prof) -> float:
    pts = prof["style_points"]
    tot = sum(pts.values()) or 1
    return (sum(v for k, v in pts.items() if k in HIGH_VAR) - sum(v for k, v in pts.items() if k in SAFE)) / tot


def early_table(lanes: pd.DataFrame, before: pd.Timestamp) -> dict:
    past = lanes[lanes["date"] < before]
    if past.empty:
        return {}
    w = np.power(0.5, (before - past["date"]).dt.days.to_numpy() / EARLY_HL)
    d = past.assign(w=w, wg=w * past["gd15"].to_numpy() / 1000)
    agg = d.groupby(["champ", "role"])[["wg", "w"]].sum()
    return (agg["wg"] / (agg["w"] + EARLY_K)).to_dict()


def style_vec(prof, styles):
    pts = prof["style_points"]
    tot = sum(pts.values()) or 1
    return [pts.get(s, 0) / tot for s in styles]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    a = ap.parse_args()
    from draftlab.fast_stats import StatsIndex
    from draftlab.params import load_params, snapshot_kwargs
    t0 = time.time()
    g = load_eval_games()
    team, pl = load_early_raw()
    lanes = lane_long(attach_early(g, team, pl))
    kw = dict(snapshot_kwargs(load_params()))
    kw["adjusted_strength"] = False
    idx = StatsIndex(g)
    styles, _ = style_matrix()
    rows = []
    for m in month_starts(g["date"]):
        if m < pd.Timestamp(a.start):
            continue
        nxt = m + pd.offsets.MonthBegin(1)
        mg = g[(g["date"] >= m) & (g["date"] < nxt)]
        if mg.empty:
            continue
        early = early_table(lanes, m)
        patches = mg["patch_idx"].fillna(-1)
        for p, grp in mg.groupby(patches):
            snap = idx.snapshot(m, target_patch=None if p < 0 else p, with_rosters=False, **kw)
            if snap.n_games == 0:
                continue
            for row in grp.to_dict("records"):
                r = {"gameid": row["gameid"]}
                st = state_from_game(row, 20)
                r.update(zip(CHAMP, comp_features(st, snap, 0.0, 0.0, role_hint=actual_roles(row))[1:]))
                ok = bool(row.get("has_order")) and not (row["date"].year >= 2026 and row["blue_first_pick"] == 0)
                for tag, n in STAGE_N.items():
                    v = comp_features(state_from_game(row, n), snap, 0.0, 0.0)[1:] if ok else [np.nan] * 5
                    r.update({f"{tag}_{c}": x for c, x in zip(CHAMP, v)})
                bp = comp_profile([row[f"Blue_{x}_champ"] for x in ROLES])
                rp = comp_profile([row[f"Red_{x}_champ"] for x in ROLES])
                r["timing_diff"] = bp["timing_score"] - rp["timing_score"]
                r["timing_gap"] = abs(r["timing_diff"])
                r["style_adv"] = style_advantage(style_vec(bp, styles), style_vec(rp, styles), styles)
                r["hv_diff"] = variance_score(bp) - variance_score(rp)
                r["early_diff"] = sum(early.get((row[f"Blue_{x}_champ"], x), 0.0) - early.get((row[f"Red_{x}_champ"], x), 0.0)
                                      for x in ROLES)
                rows.append(r)
        print(f"  {m:%Y-%m}: {len(mg)} games ({time.time() - t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows).set_index("gameid")
    df.to_pickle(OUT)
    print(f"{len(df):,} games -> {OUT} in {time.time() - t0:.0f}s")
    print(df.describe().T[["count", "mean", "std"]].round(3))


if __name__ == "__main__":
    main()
