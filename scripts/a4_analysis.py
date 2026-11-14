"""A4 analysis beyond the ablation table: interaction coefficients with CIs, level-strength draft win probability
and a residual check. Appends to reports/runs/<run_id>.md.

    .venv/Scripts/python scripts/a4_analysis.py <run_id> --base strength_g15_tier_cal [--boot 300]

1. Coefficients: draft_interact refitted ONCE on all games from 2023-01-01 up to the frozen holdout (the base
   strength logit is still a walk-forward, pre-game offset). Series-level bootstrap 95% CIs, natural units.
   This overlaps the 2025 report window: it is a description of the effect, not a model-selection step.
2. Level strength: P(blue wins) from the draft terms alone with the strength gap at 0 (interactions vanish, no
   side term), for every evaluated game.
3. Residual check (out-of-sample, from the run's predictions): does the draft's edge over the strength-only
   prediction show up in results? Bins of draft edge, observed - predicted with CIs, and the slope of
   (result - strength prediction) on (draft prediction - strength prediction): 1 = edge fully real, 0 = noise.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import HOLDOUT_START, ROOT  # noqa: E402
from draftlab.eval import load_eval_games  # noqa: E402
from draftlab.eval.bootstrap import mean_ci  # noqa: E402
from draftlab.winmodels import get_model  # noqa: E402
from draftlab.winmodels.draft import CHAMP, SIGNED_INTER, TRAIN_START, add_interactions  # noqa: E402

HYP = {"timing_diff x gap": "> 0 (favourite gains from the later comp)",
       "hv_diff x gap": "< 0 (underdog gains from high variance)"}
LABEL = {"timing_diff": "comp timing (blue later by 1 tag-sheet step)", "timing_gap": "timing mismatch size",
         "style_adv": "style matchup (tag sheet)", "hv_diff": "high-variance share (blue - red)",
         "early_diff": "data early score (k gold @15)", "champ_strength": "champion strength",
         "player_comfort": "player comfort", "synergy": "synergy", "lane_counter": "lane counter",
         "team_matchups": "whole-team matchups"}


def fit(Z, off, y, lam=1.0, x0=None):
    def f(b):
        z = off + Z @ b
        return (np.sum(np.logaddexp(0, z) - y * z) + 0.5 * lam * b @ b, Z.T @ (1 / (1 + np.exp(-z)) - y) + lam * b)
    return minimize(f, np.zeros(Z.shape[1]) if x0 is None else x0, jac=True, method="L-BFGS-B").x


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--base", required=True)
    ap.add_argument("--boot", type=int, default=300)
    a = ap.parse_args()
    runs = ROOT / "reports" / "runs"
    g = load_eval_games()
    rows = np.nonzero(((g["date"] >= TRAIN_START) & (g["date"] < HOLDOUT_START)).to_numpy())[0]
    m = get_model("draft_interact", base=a.base)
    off = np.full(len(g), np.nan)
    off[rows] = m._base_logit(g, rows)
    D = m._design(g)
    names = list(D.columns)
    X, names = add_interactions(D.to_numpy(float), names, off, rows, g["date"], rows)
    y = g["blue_win"].to_numpy(float)
    keep = rows[~np.isnan(X[rows]).any(axis=1)]
    sd = X[keep].std(0) + 1e-12
    Z, o, yy, ser = X[keep] / sd, off[keep], y[keep], g["series_id"].to_numpy()[keep]
    b = fit(Z, o, yy)
    rng = np.random.default_rng(0)
    us = np.unique(ser)
    pos = {s: np.nonzero(ser == s)[0] for s in us}
    boots = []
    for _ in range(a.boot):
        take = np.concatenate([pos[s] for s in rng.choice(us, len(us))])
        boots.append(fit(Z[take], o[take], yy[take], x0=b))
    B = np.array(boots)
    lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
    L = ["", "## A4 analysis", "",
         f"### 1. Draft coefficients (draft_interact, refitted once on {len(keep):,} games {TRAIN_START} to the holdout; "
         f"offset = `{a.base}`; {a.boot} series bootstrap draws)", "",
         "Natural units: logit change per 1 unit of the feature (per 1 unit x 1 logit of strength gap for "
         "interactions; gap > 0 = blue favoured). 1 logit near 50% is about 23 percentage points.", "",
         "| Term | Coefficient | 95% CI | Per 1 SD (logit) | Hypothesis |", "|---|---|---|---|---|"]
    for j, n in enumerate(names):
        base_n = n.split(" x ")[0]
        lab = LABEL.get(base_n.replace("b1_", "").replace("p1_", ""), base_n) + (" x " + n.split(" x ")[1] if " x " in n else "")
        sig = "" if lo[j] <= 0 <= hi[j] else " *"
        L.append(f"| {lab} | {b[j] / sd[j]:+.3f}{sig} | [{lo[j] / sd[j]:+.3f}, {hi[j] / sd[j]:+.3f}] | {b[j]:+.3f} | {HYP.get(n, '')} |")
    L.append("\n\\* = 95% CI excludes 0.")

    # 2. level strength
    d = pd.read_csv(runs / f"{a.run_id}_predictions.csv.gz", parse_dates=["date"])
    k = len(D.columns)
    Xe = D.reindex(g.index).to_numpy(float)
    gi = pd.Series(np.arange(len(g)), index=g["gameid"])
    ev = gi.reindex(d["gameid"]).to_numpy()
    xe = Xe[ev]
    ok = ~np.isnan(xe).any(axis=1)
    p_level = 1 / (1 + np.exp(-(xe[ok] / sd[:k]) @ b[:k]))
    dev = np.abs(p_level - 0.5)
    L += ["", "### 2. Level-strength draft win probability (strength gap = 0, no side term)", "",
          f"{ok.sum():,} evaluated games. How far the draft alone moves an even matchup:", "",
          "| Statistic | Value |", "|---|---|",
          f"| median distance from 50% | {np.median(dev) * 100:.1f} pp |",
          f"| 90th percentile | {np.percentile(dev, 90) * 100:.1f} pp |",
          f"| largest | {dev.max() * 100:.1f} pp |",
          f"| share outside 45-55% | {(dev > 0.05).mean():.1%} |",
          f"| share outside 40-60% | {(dev > 0.10).mean():.1%} |",
          f"| observed blue win rate when level-draft favours blue by > 5 pp | "
          f"{d['blue_win'].to_numpy()[ok][p_level > 0.55].mean():.1%} (n={(p_level > 0.55).sum():,}) |",
          f"| ... when it favours red by > 5 pp | {d['blue_win'].to_numpy()[ok][p_level < 0.45].mean():.1%} "
          f"(n={(p_level < 0.45).sum():,}) |",
          "", "App use: `draftlab.winmodels.draft.level_strength_prob(model, features)` after fitting; the "
          "coefficients above are the ones it applies."]

    # 3. residual check, out of sample
    yv, sv = d["blue_win"].to_numpy(float), d["series_id"].to_numpy()
    pb = d[a.base].to_numpy()
    L += ["", "### 3. Residual check (out-of-sample 2025 predictions)", "",
          "Edge = draft model probability minus strength-only probability (for the blue side). If the draft "
          "edge is real, games where the draft favours blue should beat the strength-only prediction.", ""]
    for mdl in [c for c in ("draft_champ_comp", "draft_interact", "draft_champ") if c in d]:
        e = d[mdl].to_numpy() - pb
        r = yv - pb
        q = pd.qcut(e, 5, labels=False, duplicates="drop")
        L += [f"**{mdl}**", "", "| Draft-edge quintile | n | mean edge | strength-only predicted | observed | "
              "observed - predicted (95% CI) |", "|---|---|---|---|---|---|"]
        for k_ in range(int(q.max()) + 1):
            mk = q == k_
            mci = mean_ci(r[mk], sv[mk])
            L.append(f"| Q{k_ + 1} | {mk.sum():,} | {e[mk].mean() * 100:+.1f} pp | {pb[mk].mean():.1%} | "
                     f"{yv[mk].mean():.1%} | {mci[0] * 100:+.1f} pp [{mci[1] * 100:+.1f}, {mci[2] * 100:+.1f}] |")
        # slope of residual on edge, series bootstrap
        us2 = np.unique(sv)
        pos2 = {s: np.nonzero(sv == s)[0] for s in us2}
        sl = []
        for _ in range(500):
            t = np.concatenate([pos2[s] for s in rng.choice(us2, len(us2))])
            sl.append(np.polyfit(e[t], r[t], 1)[0])
        s0 = np.polyfit(e, r, 1)[0]
        L += ["", f"Slope of (result - strength prediction) on draft edge: **{s0:.2f}** "
              f"[{np.percentile(sl, 2.5):.2f}, {np.percentile(sl, 97.5):.2f}] (1 = edge fully real, 0 = noise).", ""]
    (runs / f"{a.run_id}.md").open("a", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
