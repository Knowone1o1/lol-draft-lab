"""Write one comparable result set per run: <run_id>.md, <run_id>.csv (results table), <run_id>_meta.json,
<run_id>_predictions.csv.gz, <run_id>_calibration.svg, and a summary line in reports/runs/index.csv."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_LABEL = {"all": "All leagues", "major": "Major leagues (+ Worlds/MSI/FST/EWC)", "erl": "ERLs (EMEA regional + EMEA Masters)",
               "other": "Other leagues"}
COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]


def _fmt(v, lo, hi, metric):
    if not np.isfinite(v):
        return "—"
    pct = metric == "accuracy"
    f = (lambda x: f"{x:.1%}") if pct else (lambda x: f"{x:+.4f}" if "_vs_" in metric else f"{x:.4f}")
    return f"{f(v)} [{f(lo)}, {f(hi)}]" if np.isfinite(lo) else f(v)


def calibration_svg(rel: dict) -> str:
    W, H, m = 420, 420, 46
    sx = lambda v: m + v * (W - 2 * m)
    sy = lambda v: H - m - v * (H - 2 * m)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui,sans-serif" font-size="11">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
             f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="#9ca3af" stroke-dasharray="4 4"/>']
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(f'<text x="{sx(t)}" y="{H - m + 16}" text-anchor="middle" fill="#374151">{t:.0%}</text>')
        parts.append(f'<text x="{m - 8}" y="{sy(t) + 4}" text-anchor="end" fill="#374151">{t:.0%}</text>')
    parts.append(f'<text x="{W / 2}" y="{H - 8}" text-anchor="middle" fill="#111827">Predicted P(blue wins)</text>')
    parts.append(f'<text x="14" y="{H / 2}" text-anchor="middle" fill="#111827" transform="rotate(-90 14 {H / 2})">Observed blue win rate</text>')
    for i, (name, rows) in enumerate(rel.items()):
        pts = [(r[3], r[4], r[2]) for r in rows if r[2] >= 20]
        c = COLORS[i % len(COLORS)]
        if pts:
            parts.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(
                c, " ".join(f"{sx(px):.1f},{sy(py):.1f}" for px, py, _ in pts)))
            for px, py, n in pts:
                parts.append(f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="{2 + min(5, n / 400):.1f}" fill="{c}"><title>{name}: predicted {px:.2f}, observed {py:.2f}, n={n}</title></circle>')
        parts.append(f'<rect x="{m + 8}" y="{m - 8 + i * 16}" width="10" height="10" fill="{c}"/><text x="{m + 24}" y="{m + 1 + i * 16}" fill="#111827">{name}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_run(res, out_dir: Path):
    out_dir = Path(out_dir)
    rid, meta, t = res.run_id, res.meta, res.table
    t.to_csv(out_dir / f"{rid}.csv", index=False)
    res.predictions.to_csv(out_dir / f"{rid}_predictions.csv.gz", index=False, compression="gzip")
    (out_dir / f"{rid}_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    if res.reliability:
        (out_dir / f"{rid}_calibration.svg").write_text(calibration_svg(res.reliability), encoding="utf-8")

    L = [f"# Win-model evaluation run `{rid}`", "",
         f"{'**FINAL EVALUATION on the frozen holdout**' if meta['final_evaluation'] else 'Walk-forward evaluation before the frozen holdout'}"
         f" · games {meta['eval_window'][0]} to {meta['eval_window'][1]} · {meta['n_games_eval']:,} games in "
         f"{meta['n_series_eval']:,} series · holdout starts {meta['holdout_start']} · data cutoff {meta['data_cutoff']}"
         f" · seed {meta['seed']} · code {meta['code_hash']}" + (f" · {meta['note']}" if meta.get("note") else ""), "",
         "Each day's games are predicted by models refitted on all earlier days. Intervals are 95% bootstrap "
         "intervals resampling whole series; \"vs baseline\" rows are paired differences (negative = better than "
         f"`{meta['baseline']}`).", ""]
    models = [m["name"] for m in meta["models"]]
    metrics = ["log_loss", "brier", "accuracy", "auc", "cal_slope", "ece"]
    for split in ("all", "major", "erl", "other"):
        sub = t[t["split"] == split]
        if sub.empty:
            continue
        n = int(sub["n_games"].iloc[0])
        L += [f"## {SPLIT_LABEL[split]} ({n:,} games)", "",
              "| Model | Log-loss | Brier | Accuracy | AUC | Calibration slope | ECE |", "|---|---|---|---|---|---|---|"]
        for mname in models:
            cells = []
            for metric in metrics:
                r = sub[(sub["model"] == mname) & (sub["metric"] == metric)]
                cells.append(_fmt(*r[["value", "ci_low", "ci_high"]].iloc[0], metric) if len(r) else "—")
            L.append(f"| {mname} | " + " | ".join(cells) + " |")
        diffs = sub[sub["metric"].str.contains("_vs_")]
        if len(diffs):
            L += ["", "| Model | Log-loss vs baseline | Brier vs baseline |", "|---|---|---|"]
            for mname in models:
                d = diffs[diffs["model"] == mname]
                if len(d):
                    ll = d[d["metric"].str.startswith("log_loss")].iloc[0]
                    br = d[d["metric"].str.startswith("brier")].iloc[0]
                    L.append(f"| {mname} | {_fmt(ll.value, ll.ci_low, ll.ci_high, ll.metric)} | {_fmt(br.value, br.ci_low, br.ci_high, br.metric)} |")
        L.append("")
    if res.reliability:
        L += ["## Calibration (all leagues)", "", f"![calibration]({rid}_calibration.svg)", "",
              "| Model | " + " | ".join(f"{r[0]:.1f}-{r[1]:.1f}" for r in next(iter(res.reliability.values()))) + " |",
              "|---|" + "---|" * 10]
        for name, rows in res.reliability.items():
            L.append(f"| {name} (observed / predicted, n) | " + " | ".join(
                f"{r[4]:.2f} / {r[3]:.2f} ({r[2]})" if r[2] else "—" for r in rows) + " |")
        L.append("")
    L += ["## Roster changes", "",
          f"{meta['roster_change_3plus_games']:,} of {meta['n_games_eval']:,} evaluated games "
          f"({meta['roster_change_3plus_share']:.1%}) involve a team that changed 3 or more players since its previous "
          "game. Strength is keyed by team name, so those games start from a rating earned by a different roster.", "",
          "## Models", ""]
    for m in meta["models"]:
        L.append(f"- **{m['name']}**: params {m['params']}; features: {'; '.join(m['features'])}")
        if m.get("fitted"):
            L.append(f"  - coefficients at the last daily refit (logit scale): "
                     + ", ".join(f"{k} {v:+.3f}" for k, v in m["fitted"].items()))
        for c in m.get("caveats", []):
            L.append(f"  - *Caveat:* {c}")
    (out_dir / f"{rid}.md").write_text("\n".join(L), encoding="utf-8")

    # one comparable summary line per model per run
    idx = out_dir / "index.csv"
    summ = t[(t["split"] == "all") & t["metric"].isin(["log_loss", "brier", "accuracy", "auc", "cal_slope", "ece"])]
    wide = summ.pivot_table(index="model", columns="metric", values="value").reset_index()
    wide.insert(0, "run_id", rid)
    wide["final"] = meta["final_evaluation"]
    wide["window"] = "..".join(meta["eval_window"])
    wide["n_games"] = meta["n_games_eval"]
    wide.to_csv(idx, mode="a", header=not idx.exists(), index=False)
