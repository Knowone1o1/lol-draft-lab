"""Walk-forward evaluation harness (Track A, step 1). Every win model is measured with this.

- Frozen holdout: games from config.HOLDOUT_START on. Normal runs only score games before it; the holdout is
  scored only with final=True (the CLI's --final-evaluation), and each use is logged in
  reports/runs/holdout_log.json so it's visible if it was looked at more than once.
- Walk-forward: models predict each day's games after refitting on all earlier days (base.daily_refit).
- Metrics: log-loss and Brier (headline), accuracy and AUC, calibration slope + intercept and ECE, plus a
  reliability table and plot.
- Intervals: 95% bootstrap resampling whole series; model-vs-baseline differences are paired.
- Splits: all leagues (headline), major leagues (incl. Worlds/MSI/First Stand/EWC), ERLs (EMEA regional
  leagues + EMEA Masters), other.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from . import bootstrap as bs
from . import metrics as M

SPLITS = ("all", "major", "erl", "other")
CODE_FILES = ["draftlab/eval/harness.py", "draftlab/eval/metrics.py", "draftlab/eval/bootstrap.py",
              "draftlab/eval/dataset.py", "draftlab/winmodels"]


@dataclass
class RunResult:
    run_id: str
    meta: dict
    predictions: pd.DataFrame          # one row per evaluated game, one column per model
    table: pd.DataFrame                # model x split x metric with 95% intervals
    reliability: dict = field(default_factory=dict)


def code_hash(root: Path = config.ROOT) -> str:
    h = hashlib.sha1()
    for f in CODE_FILES:
        p = root / f
        files = sorted(p.rglob("*.py")) if p.is_dir() else [p]
        for q in files:
            if q.exists():
                h.update(q.read_bytes())
    return h.hexdigest()[:10]


def eval_mask(games: pd.DataFrame, start: str, final: bool) -> np.ndarray:
    hold = pd.Timestamp(config.HOLDOUT_START)
    d = games["date"]
    if final:
        return (d >= hold).to_numpy()
    return ((d >= pd.Timestamp(start)) & (d < hold)).to_numpy()


def _log_holdout_use(run_id: str, out_dir: Path):
    f = out_dir / "holdout_log.json"
    log = json.loads(f.read_text()) if f.exists() else []
    log.append({"run_id": run_id, "at": datetime.now().isoformat(timespec="seconds")})
    f.write_text(json.dumps(log, indent=1))
    return len(log)


def evaluate(games: pd.DataFrame, models: list, start: str = "2024-01-01", final: bool = False,
             baseline: str = "elo", B: int = 2000, seed: int = 0, out_dir: Path | None = None,
             note: str = "") -> RunResult:
    """Score `models` walk-forward on games from `start` up to the holdout (or on the holdout if final)."""
    np.random.seed(seed)
    out_dir = Path(out_dir or config.REPORT_DIR / "runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    games = games.sort_values("date", kind="stable").reset_index(drop=True)
    mask = eval_mask(games, start, final)
    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        raise ValueError("no games to evaluate in this window")
    preds = games.loc[idx, ["gameid", "date", "league", "tier", "series_id", "game", "Blue_teamname", "Red_teamname",
                            "blue_win", "fearless", "roster_change_3plus"]].reset_index(drop=True)
    for m in models:
        p = m.walk_forward(games, idx)
        if np.isnan(p).any():
            raise ValueError(f"{m.name} returned NaN predictions")
        preds[m.name] = p
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{code_hash()}"
    holdout_uses = _log_holdout_use(run_id, out_dir) if final else 0

    rows, rel = [], {}
    names = [m.name for m in models]
    for split in SPLITS:
        sub = preds if split == "all" else preds[preds["tier"] == split]
        if len(sub) < 50:
            continue
        y, ser = sub["blue_win"].to_numpy(), sub["series_id"].to_numpy()
        for name in names:
            p = sub[name].to_numpy()
            s = M.summary(y, p)
            for metric, terms in (("log_loss", M.logloss_terms(y, p)), ("brier", M.brier_terms(y, p)),
                                  ("accuracy", M.accuracy_terms(y, p))):
                mean, lo, hi = bs.mean_ci(terms, ser, B, seed)
                rows.append((name, split, metric, mean, lo, hi, len(sub)))
            constant = float(np.std(p)) < 1e-3   # e.g. always_blue: AUC and calibration slope are undefined
            for metric, fn in (("auc", M.auc), ("cal_slope", lambda yy, pp: M.calibration_slope(yy, pp)[0]),
                               ("ece", M.ece)):
                if constant and metric != "ece":
                    rows.append((name, split, metric, np.nan, np.nan, np.nan, len(sub)))
                    continue
                mean, lo, hi = bs.stat_ci(fn, y, p, ser, B=min(B, 300), seed=seed)
                rows.append((name, split, metric, mean, lo, hi, len(sub)))
            rows.append((name, split, "cal_intercept", s["cal_intercept"], np.nan, np.nan, len(sub)))
            if baseline in names and name != baseline:
                b = sub[baseline].to_numpy()
                for metric, fn in (("log_loss", M.logloss_terms), ("brier", M.brier_terms)):
                    mean, lo, hi = bs.paired(fn(y, p), fn(y, b), ser, B, seed)
                    rows.append((name, split, f"{metric}_vs_{baseline}", mean, lo, hi, len(sub)))
            if split == "all":
                rel[name] = M.reliability(y, p)
    table = pd.DataFrame(rows, columns=["model", "split", "metric", "value", "ci_low", "ci_high", "n_games"])
    ev = games.loc[idx]
    meta = {
        "run_id": run_id, "created": datetime.now().isoformat(timespec="seconds"), "note": note,
        "final_evaluation": final, "holdout_start": config.HOLDOUT_START, "holdout_uses_logged": holdout_uses,
        "eval_window": [str(ev["date"].min().date()), str(ev["date"].max().date())],
        "data_cutoff": str(games["date"].max().date()), "n_games_eval": int(len(idx)),
        "n_series_eval": int(ev["series_id"].nunique()),
        "n_by_split": {s: int((ev["tier"] == s).sum()) for s in ("major", "erl", "other")},
        "models": [m.describe() for m in models], "baseline": baseline, "bootstrap": {"B": B, "unit": "series"},
        "seed": seed, "code_hash": code_hash(), "refit": "daily (walk-forward)",
        "roster_change_3plus_games": int(ev["roster_change_3plus"].sum()),
        "roster_change_3plus_share": float(ev["roster_change_3plus"].mean()),
    }
    res = RunResult(run_id, meta, preds, table, rel)
    from .report import write_run
    write_run(res, out_dir)
    return res
