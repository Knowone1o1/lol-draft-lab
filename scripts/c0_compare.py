"""C0 before/after: pick/ban top-5 accuracy on the build's validation folds (before the frozen holdout), with the
old blue-always-first draft order vs the First Selection-aware order. Writes reports/c0_pick_order.md.

    .venv/Scripts/python scripts/c0_compare.py
Needs reports/val_ranks_legacy_order.pkl and reports/val_ranks_current.pkl (written by scripts/build.py).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import REPORT_DIR  # noqa: E402
from draftlab.data import load_games  # noqa: E402


def per_game(r):
    d = pd.DataFrame({"gid": r["gid"], "hit": (r["rank"] <= 5).astype(float)})
    return d.groupby("gid")["hit"].agg(["sum", "count"])


def paired_ci(a, b, B=2000, seed=0):
    """Top-5 difference (after - before), resampling whole games."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        diffs.append(a["sum"].values[i].sum() / a["count"].values[i].sum() - b["sum"].values[i].sum() / b["count"].values[i].sum())
    d = a["sum"].sum() / a["count"].sum() - b["sum"].sum() / b["count"].sum()
    return d, *np.percentile(diffs, [2.5, 97.5])


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    old = pickle.load(open(REPORT_DIR / "val_ranks_legacy_order.pkl", "rb"))
    new = pickle.load(open(REPORT_DIR / "val_ranks_current.pkl", "rb"))
    g, _, _ = load_games()
    bf = dict(zip(g["gameid"], g["blue_first_pick"]))
    L = ["# C0: pick-order fix (2026 First Selection), before vs after", "",
         "Validation folds of `scripts/build.py` (the 6 months before the frozen holdout; the holdout is not scored). "
         "Before = every game read as blue-picks-first; after = the side that actually picked first (Oracle's "
         "Elixir `firstPick`) takes the first-pick sequence, bans included. Top-5 = the champion actually chosen "
         "is in the model's top 5. Differences are after minus before with a 95% interval resampling whole games.", "",
         "| Action | Variant | Games | Before top-5 | After top-5 | Δ [95% CI] |", "|---|---|---|---|---|---|"]
    for key in sorted(set(old) & set(new)):
        kind, strv, vname = key
        a, b = per_game(new[key]), per_game(old[key])
        common = a.index.intersection(b.index)
        for label, sel in (("all", common), ("red picked first", [x for x in common if bf.get(x, 1) == 0]),
                           ("blue picked first", [x for x in common if bf.get(x, 1) == 1])):
            if len(sel) < 30:
                continue
            aa, bb = a.loc[sel], b.loc[sel]
            d, lo, hi = paired_ci(aa, bb)
            L.append(f"| {kind} ({label}) | {strv}/{vname} | {len(sel):,} | {bb['sum'].sum() / bb['count'].sum():.1%} | "
                     f"{aa['sum'].sum() / aa['count'].sum():.1%} | {d * 100:+.1f} pp [{lo * 100:+.1f}, {hi * 100:+.1f}] |")
    (REPORT_DIR / "c0_pick_order.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
