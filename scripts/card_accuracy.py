"""How often the plan card's lane calls were right, walk-forward, 2024-01 to the frozen holdout.

Uses data/rules/backtest_facts.pkl (scripts/rule_tests.py): every game's lane priority was computed with stats and
forecast models built only from earlier games, with the same code as the card. Blue-side rows only, so each game
counts once (Red's call is the mirror image). 95% intervals resample whole series.

    .venv/Scripts/python scripts/card_accuracy.py
"""
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import HOLDOUT_START, REPORT_DIR  # noqa: E402
from draftlab.eval.bootstrap import mean_ci  # noqa: E402

CLEAR = 500   # "clearly ahead" = 500+ gold at 15


def pct(v, ser):
    m, lo, hi = mean_ci(v, ser, B=1000, seed=0)
    return f"{m:.0%} [{lo:.0%}, {hi:.0%}]"


def call_table(d, call_col, gold_col, label):
    """Rows: Strong / Even / Weak calls -> share ahead, share clearly ahead/behind, mean gold."""
    d = d[d[call_col].notna() & d[gold_col].notna()]
    out = []
    for call in ("Strong", "Even", "Weak"):
        x = d[d[call_col] == call]
        if len(x) < 30:
            continue
        g, ser = x[gold_col].to_numpy(float), x["series_id"].to_numpy()
        out.append(f"| {label} | {call} | {len(x):,} ({len(x) / len(d):.0%}) | {pct(g > 0, ser)} | "
                   f"{pct(g >= CLEAR, ser)} | {pct(g <= -CLEAR, ser)} | {np.mean(g):+,.0f} |")
    sw = d[d[call_col].isin(["Strong", "Weak"])]
    hit = np.where(sw[call_col] == "Strong", sw[gold_col] > 0, sw[gold_col] < 0)
    hit = hit[sw[gold_col].to_numpy() != 0]
    ser = sw.loc[sw[gold_col] != 0, "series_id"].to_numpy()
    summary = (label, len(sw), pct(hit, ser) if len(sw) >= 30 else "–")
    return out, summary


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    T = pd.read_pickle(ROOT / "data" / "rules" / "backtest_facts.pkl")
    assert (T["date"] < pd.Timestamp(HOLDOUT_START)).all()
    B = T[T["side"] == "Blue"].copy()
    B["gd15_mj"] = B["gd15_mid"] + B["gd15_jng"]
    B["fc_mj"] = pd.to_numeric(B["fc_mid"], errors="coerce") + pd.to_numeric(B["fc_jng"], errors="coerce")

    L = ["# Plan card accuracy: were the lane calls right?", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M}. Games from 2024-01-01 to {HOLDOUT_START} (holdout not used), "
         "each scored with stats and forecast models built only from earlier games, using the same code as the card. "
         "One row per game (Blue's view). \"Ahead\" = gold lead at 15 in that lane (mid-jungle: mid + jungle). "
         f"\"Clearly\" = {CLEAR}+ gold. 95% intervals resample whole series.", "",
         "## 1. Strong / Weak calls: how often the favoured side was ahead at 15", "",
         "| Call | Strong/Weak calls | Favoured side ahead |", "|---|---|---|"]
    rows, summ = [], []
    for lane, label in (("top", "Top"), ("mid", "Mid"), ("bot", "Bot")):
        r, s = call_table(B, f"our.{lane}.prio", f"gd15_{lane}", label)
        rows += r
        summ.append(s)
    r, s = call_table(B, "our.midjng", "gd15_mj", "Mid-jungle")
    rows += r
    summ.append(s)
    for label, n, p in summ:
        L.append(f"| {label} | {n:,} | {p} |")

    # forecast sign on all lanes, as a reference for the calls
    L += ["", "For reference, the forecast's sign on every lane (not just Strong / Weak calls):", "",
          "| Lane | Lanes | Forecast leader was ahead |", "|---|---|---|"]
    for lane, label in (("top", "Top"), ("mid", "Mid"), ("bot", "Bot"), ("jng", "Jungle"), ("mj", "Mid-jungle")):
        fc = pd.to_numeric(B[f"fc_{lane}"], errors="coerce")
        g = B[f"gd15_{lane}"]
        ok = fc.notna() & g.notna() & (g != 0) & (fc != 0)
        hit = (np.sign(fc[ok]) == np.sign(g[ok])).to_numpy()
        L.append(f"| {label} | {ok.sum():,} | {pct(hit, B.loc[ok, 'series_id'].to_numpy())} |")

    L += ["", "## 2. All three calls, with what actually happened", "",
          "| Lane | Call | Games (share) | Ahead at 15 | Clearly ahead | Clearly behind | Average gold at 15 |",
          "|---|---|---|---|---|---|---|"] + rows

    # by tier and by year for the side lanes combined
    L += ["", "## 3. Strong / Weak calls by tier and season (top, mid, bot together)", "",
          "| Group | Calls | Favoured side ahead |", "|---|---|---|"]
    parts = []
    for lane in ("top", "mid", "bot"):
        x = B[B[f"our.{lane}.prio"].isin(["Strong", "Weak"]) & B[f"gd15_{lane}"].notna() & (B[f"gd15_{lane}"] != 0)]
        parts.append(pd.DataFrame({"tier": x["tier"], "year": x["date"].dt.year, "series_id": x["series_id"],
                                   "hit": np.where(x[f"our.{lane}.prio"] == "Strong", x[f"gd15_{lane}"] > 0,
                                                   x[f"gd15_{lane}"] < 0)}))
    P = pd.concat(parts)
    for col, vals in (("tier", ["major", "erl", "other"]), ("year", sorted(P["year"].unique()))):
        for v in vals:
            x = P[P[col] == v]
            if len(x) >= 50:
                L.append(f"| {v if col == 'year' else {'major': 'Major', 'erl': 'ERL', 'other': 'Other'}[v]} | {len(x):,} | {pct(x['hit'].to_numpy(), x['series_id'].to_numpy())} |")

    # team gold forecast
    fc = pd.to_numeric(B["fc_team"], errors="coerce")
    ok = fc.notna() & B["gd15"].notna() & (B["gd15"] != 0)
    hit = (np.sign(fc[ok]) == np.sign(B.loc[ok, "gd15"])).to_numpy()
    mae = np.abs(fc[ok] - B.loc[ok, "gd15"]).mean()
    L += ["", "## 4. Team gold at 15 (the number behind the break-even line)", "",
          f"- Forecast leader was ahead at 15 in {pct(hit, B.loc[ok, 'series_id'].to_numpy())} of {ok.sum():,} games.",
          f"- Average miss: {mae:,.0f} gold.", "",
          "## How to read this", "",
          "- 50% would be a coin flip. A Strong or Weak call is only made for about the top / bottom 10% of forecasts "
          "for each lane, so these are the card's most confident lane calls.",
          "- Even calls are not a prediction of an even lane; they mean the forecast isn't confident either way.",
          "- Laning tags are empty, so every call here comes from the gold forecast; filled tags would override it.",
          "- The rule labels (Supported / Unclear) are in reports/rule_tests.md; this report only checks the lane calls "
          "the rules are built on."]
    (REPORT_DIR / "plan_card_accuracy.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
