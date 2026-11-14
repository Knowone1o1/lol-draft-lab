"""A4 split tables from a run's predictions: paired log-loss gain of each draft model over its strength base.

    .venv/Scripts/python scripts/a4_splits.py <run_id> --base strength_g15_tier
Appends the tables to reports/runs/<run_id>.md. Series-level paired bootstrap (negative = draft model better).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import ROOT  # noqa: E402
from draftlab.eval.bootstrap import paired  # noqa: E402
from draftlab.eval.metrics import logloss_terms  # noqa: E402
from draftlab.winmodels.draft import first_selection  # noqa: E402


def fmt(t):
    m, lo, hi = t
    return f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--base", required=True)
    a = ap.parse_args()
    runs = ROOT / "reports" / "runs"
    d = pd.read_csv(runs / f"{a.run_id}_predictions.csv.gz", parse_dates=["date"])
    y = d["blue_win"].to_numpy()
    draft = [c for c in d.columns if c.startswith("draft_")]
    ll = {m: logloss_terms(y, d[m].to_numpy()) for m in draft + [a.base]}
    feats = pd.read_pickle(ROOT / "data" / "draft_features.pkl")
    has_stage = d["gameid"].map(feats["b1_champ_strength"]).notna().to_numpy() if "gameid" in d else None
    gnum = d["game"].to_numpy()
    fear = d["fearless"].astype(bool).to_numpy() if "fearless" in d else np.zeros(len(d), bool)
    splits = {"All": np.ones(len(d), bool), "Game 1": gnum == 1, "Game 2": gnum == 2, "Games 3-5": gnum >= 3,
              "Fearless games 3-5": fear & (gnum >= 3),
              "Major": (d["tier"] == "major").to_numpy(), "ERL": (d["tier"] == "erl").to_numpy(),
              "Other": (d["tier"] == "other").to_numpy(), "2026": (d["date"].dt.year == 2026).to_numpy()}
    s = d["series_id"].to_numpy()
    L = ["", "## A4: draft gain over the strength offset (log-loss, draft model minus "
         f"`{a.base}`, series-level 95% bootstrap; negative = draft helps)", "",
         "| Split | n | " + " | ".join(draft) + " |", "|---|---|" + "---|" * len(draft)]
    for name, m in splits.items():
        if m.sum() < 200:
            continue
        L.append(f"| {name} | {m.sum():,} | " + " | ".join(fmt(paired(ll[x][m], ll[a.base][m], s[m])) for x in draft) + " |")
    if has_stage is not None:
        L += ["", "### By draft stage (same games: every game with a recorded pick order; 2026 red-first-pick games "
              "excluded until the C0 fix)", "", "| Stage | n | gain vs base |", "|---|---|---|"]
        m = has_stage
        for name, col in (("After B1 (7 actions)", "draft_b1"), ("After pick phase 1 (12 actions)", "draft_p1"),
                          ("Full draft", "draft_champ")):
            if col in ll:
                L.append(f"| {name} | {m.sum():,} | {fmt(paired(ll[col][m], ll[a.base][m], s[m]))} |")
    (runs / f"{a.run_id}.md").open("a", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
