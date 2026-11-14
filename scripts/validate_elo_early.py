"""Does league-adjusted / split-regressed Elo help the early-game and game-plan models? (judge M1)

Same protocol as build.py: rolling-origin validation on the 3 two-month folds just before the frozen holdout
(config.HOLDOUT_START), each fold predicted by models fitted on earlier months; paired bootstrap 95% intervals
over games. The holdout is not touched. Elo enters these models only as one feature (Elo difference / 400),
so each variant just swaps that column.

Writes reports/elo_validation.md.
Usage: .venv\\Scripts\\python scripts\\validate_elo_early.py
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.brief import features as brief_features  # noqa: E402
from draftlab.config import HOLDOUT_START, REPORT_DIR, ROLES, TRAIN_LEAGUES  # noqa: E402
from draftlab.data import load_games  # noqa: E402
from draftlab.early import (OBJECTIVES, EarlyStats, attach_early, game_lane_matrix, lane_long,  # noqa: E402
                            load_early_raw, objective_features, team_features, team_long)
from draftlab.training import month_starts  # noqa: E402
from draftlab.validation import boot_mean, fmt_ci, logloss_terms, paired_diff  # noqa: E402
from draftlab.wincon import team_rows  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build import ELO_VARIANTS, FOLD_MONTHS, N_FOLDS, elo_variant  # noqa: E402

TRAIN_MONTHS = 12


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    t0 = time.time()
    games, _, _ = load_games()
    t, p = load_early_raw()
    g = attach_early(games, t, p)
    H = pd.Timestamp(HOLDOUT_START)
    hm = H.to_period("M").to_timestamp()
    months = month_starts(g["date"])[1:]
    h_i = months.index(hm)
    val = months[h_i - N_FOLDS * FOLD_MONTHS + 1: h_i + 1]
    folds = [val[i:i + FOLD_MONTHS] for i in range(0, len(val), FOLD_MONTHS)]
    window = months[months.index(val[0]) - TRAIN_MONTHS: h_i + 1]
    variants = {}
    for name, reg, lg in ELO_VARIANTS:
        b, r, _ = elo_variant(games, 24.0, reg, lg)
        variants[name] = dict(zip(games["gameid"], (b - r)))
    print(f"Elo variants ready ({time.time() - t0:.0f}s)")

    # ------------------------------------------------------------- early-game examples (dev only)
    lanes, teams = lane_long(g), team_long(g)
    mask = g["league"].isin(TRAIN_LEAGUES) & (g["date"] < H) & g["blue_gd15"].notna()
    E = []
    for m in window:
        sel = g[(g["date"] >= m) & (g["date"] < m + pd.offsets.MonthBegin(1)) & mask]
        if sel.empty:
            continue
        es = EarlyStats(lanes, teams, m)
        for row in sel.to_dict("records"):
            picks = {s: {r: row[f"{s}_{r}_champ"] for r in ROLES} for s in ("Blue", "Red")}
            players = {s: {r: row[f"{s}_{r}_player"] for r in ROLES} for s in ("Blue", "Red")}
            tms = (row["Blue_teamname"], row["Red_teamname"])
            LX = game_lane_matrix(es, picks, players, tms, 0.0)
            E.append({"gid": row["gameid"], "month": m, "LX": LX,
                      "lane_y": np.array([row.get(f"blue_{r}_gd15", np.nan) for r in ROLES], float),
                      "team_y": row["blue_gd15"],
                      "obj": {o: (objective_features(es, o, picks, tms, LX), row.get(f"blue_{o}")) for o in OBJECTIVES}})
        print(f"  {m:%Y-%m} {len(sel)} games ({time.time() - t0:.0f}s)")
    gids = np.array([e["gid"] for e in E], dtype=object)
    emon = np.array([e["month"] for e in E], dtype="datetime64[ns]")

    def early_scores(vmap):
        ed = np.array([vmap[x] / 400 for x in gids])
        res = {"team_ae": [], "team_gid": [], "lane_ae": [], "lane_gid": [], "obj_ll": {o: [] for o in OBJECTIVES},
               "obj_gid": {o: [] for o in OBJECTIVES}}
        LXall = np.stack([e["LX"] for e in E]).copy()
        LXall[:, :, 4] = ed[:, None]
        TX = np.stack([team_features(lx) for lx in LXall])
        TY = np.array([e["team_y"] for e in E], float)
        LY = np.stack([e["lane_y"] for e in E])
        for f in folds:
            tr = emon < np.datetime64(f[0], "ns")
            te = np.isin(emon, [np.datetime64(x, "ns") for x in f])
            m = Ridge(alpha=1.0).fit(TX[tr], TY[tr])
            res["team_ae"].append(np.abs(m.predict(TX[te]) - TY[te]))
            res["team_gid"].append(gids[te])
            for ri in range(5):
                ok_tr, ok_te = tr & np.isfinite(LY[:, ri]), te & np.isfinite(LY[:, ri])
                lm = Ridge(alpha=1.0).fit(LXall[ok_tr, ri], LY[ok_tr, ri])
                res["lane_ae"].append(np.abs(lm.predict(LXall[ok_te, ri]) - LY[ok_te, ri]))
                res["lane_gid"].append(gids[ok_te])
            for o in OBJECTIVES:
                X = np.stack([e["obj"][o][0] for e in E]).copy()
                X[:, 4] = ed
                y = np.array([np.nan if e["obj"][o][1] is None else float(e["obj"][o][1]) for e in E])
                ok_tr, ok_te = tr & np.isfinite(y), te & np.isfinite(y)
                if ok_tr.sum() < 100 or ok_te.sum() < 50:
                    continue
                lm = LogisticRegression(max_iter=1000).fit(X[ok_tr], y[ok_tr])
                res["obj_ll"][o].append(logloss_terms(y[ok_te], lm.predict_proba(X[ok_te])[:, 1]))
                res["obj_gid"][o].append(gids[ok_te])
        out = {"team_ae": np.concatenate(res["team_ae"]), "team_gid": np.concatenate(res["team_gid"]),
               "lane_ae": np.concatenate(res["lane_ae"]), "lane_gid": np.concatenate(res["lane_gid"])}
        for o in OBJECTIVES:
            if res["obj_ll"][o]:
                out[f"obj_{o}"] = np.concatenate(res["obj_ll"][o])
                out[f"objgid_{o}"] = np.concatenate(res["obj_gid"][o])
        return out

    # ------------------------------------------------------------- game-plan (brief) rows (dev only)
    d = g[g["league"].isin(TRAIN_LEAGUES) & (g["date"] >= window[0] - pd.DateOffset(months=12)) & (g["date"] < H)]
    rows = team_rows(d)
    rows = rows[rows["gd"].notna()].reset_index(drop=True)
    rows["month"] = rows["date"].dt.to_period("M").dt.to_timestamp()
    sign = np.where(rows["side"] == "Blue", 1.0, -1.0)

    def brief_scores(vmap):
        r = rows.assign(elo=sign * rows["gameid"].map(vmap).to_numpy())
        ll, gg = [], []
        for f in folds:
            tr, te = r[r["month"] < f[0]], r[r["month"].isin(f)]
            m = LogisticRegression(C=10.0, max_iter=3000).fit(brief_features(tr, "timing"), tr["win"])
            ll.append(logloss_terms(te["win"], m.predict_proba(brief_features(te, "timing"))[:, 1]))
            gg.append(te["gameid"].to_numpy())
        return np.concatenate(ll), np.concatenate(gg)

    print("Scoring variants ...")
    res = {name: (early_scores(v), brief_scores(v)) for name, v in variants.items()}
    base_name = ELO_VARIANTS[0][0]
    be, bb = res[base_name]
    L = ["# Team-strength variants in the early-game and game-plan models (judge finding M1)", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Rolling-origin validation on {len(folds)} folds of {FOLD_MONTHS} "
         f"months before the frozen holdout ({'; '.join(f'{f[0]:%Y-%m}..{f[-1]:%Y-%m}' for f in folds)}, stopping at "
         f"{H:%Y-%m-%d}); each fold is predicted by models fitted on the earlier months from {window[0]:%Y-%m}. "
         "Training leagues, games with 15-minute data only. Δ = variant minus current (raw Elo); negative = better. "
         "Paired bootstrap 95% intervals over games. Elo enters these models as one feature (Elo difference / 400).", "",
         f"Validation size: {len(be['team_ae'])} games for team gold, {len(be['lane_ae'])} lanes, "
         f"{len(bb)} team-games for the game plan.", "",
         "| Variant | Team gold@15 MAE | Δ | Lane gold@15 MAE | Δ | First dragon log-loss | Δ | First tower log-loss | Δ | Game plan (timing) log-loss | Δ |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for name, (e, b) in res.items():
        cells = [name]
        dt = paired_diff(e["team_ae"], be["team_ae"], e["team_gid"])
        dl = paired_diff(e["lane_ae"], be["lane_ae"], e["lane_gid"])
        cells += [f"{e['team_ae'].mean():.1f}", fmt_ci(*dt, digits=1, sign=True),
                  f"{e['lane_ae'].mean():.1f}", fmt_ci(*dl, digits=1, sign=True)]
        for o in ("first_dragon", "first_tower"):
            k = f"obj_{o}"
            do = paired_diff(e[k], be[k], e[f"objgid_{o}"])
            cells += [f"{e[k].mean():.4f}", fmt_ci(*do, sign=True)]
        db = paired_diff(b[0], bb[0], b[1])
        cells += [f"{b[0].mean():.4f}", fmt_ci(*db, sign=True)]
        summary[name] = {"team": dt, "lane": dl, "brief": db}
        L.append("| " + " | ".join(cells) + " |")
    L += ["", "Reading: a variant clearly helps a model when its interval lies entirely below zero.", ""]
    for name, s in summary.items():
        if name == base_name:
            continue
        better = [k for k, v in s.items() if v[2] < 0]
        worse = [k for k, v in s.items() if v[1] > 0]
        L.append(f"- {name}: clearly better for {', '.join(better) or 'none'}; clearly worse for {', '.join(worse) or 'none'}.")
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "elo_validation.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"Done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
