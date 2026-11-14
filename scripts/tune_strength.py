"""Tune the strength model's decay and shrinkage on the 2024 walk-forward ONLY (Track A, step 2).

The report window (2025-01-01 up to the frozen holdout) is never used for these choices.
Coordinate search on walk-forward log-loss; writes reports/runs/strength_tuning.md and strength_tuned.json.
    .venv\\Scripts\\python scripts\\tune_strength.py [--model strength|strength_noform]
Already-scored settings are reused from reports/runs/<model>_tuning_log.txt (same 2024 window), so reruns resume.
"""
import argparse
import ast
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import REPORT_DIR  # noqa: E402
from draftlab.eval import load_eval_games  # noqa: E402
from draftlab.eval.metrics import calibration_slope, logloss_terms  # noqa: E402
from draftlab.winmodels import get_model  # noqa: E402

TUNE_START, TUNE_END = "2024-01-01", "2025-01-01"
GRID = {"form_half_life": [3, 5, 10, 20, 40], "weight_half_life_days": [90, 180, 365, 730],
        "lam_team": [0.03, 0.1, 0.3, 1, 3, 10, 30], "lam_league": [0.01, 0.1, 1]}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="strength", choices=["strength", "strength_noform", "strength_gold15", "strength_gold25", "strength_length",
                                                         "strength_g15_tier"])
    a = ap.parse_args()
    margin = a.model in ("strength_gold15", "strength_gold25", "strength_length")
    grid = {k: v for k, v in GRID.items() if a.model == "strength" or k != "form_half_life"}
    if margin:   # A3: keep the tuned strength_noform settings fixed, tune only the margin-rating step size
        grid = {"margin_k": [0.01, 0.02, 0.04, 0.08, 0.16]}
    tier = a.model == "strength_g15_tier"
    if tier:     # A3 revision: step size above 0.16 + a separate margin weight for major-league games
        grid = {"margin_k": [0.08, 0.16, 0.32, 0.64, 1.0], "w_major": [0.0, 0.25, 0.5, 1.0]}
    g = load_eval_games()
    idx = np.nonzero(((g["date"] >= TUNE_START) & (g["date"] < TUNE_END)).to_numpy())[0]
    y = g["blue_win"].to_numpy()[idx]
    cache = {}
    log = REPORT_DIR / "runs" / f"{a.model}_tuning_log.txt"
    if log.exists():          # resume: settings already scored on this same 2024 window
        for line in log.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*(\{.*\}) -> log-loss ([0-9.]+), slope ([0-9.]+)", line)
            if m:
                cache[json.dumps(ast.literal_eval(m.group(1)), sort_keys=True)] = (float(m.group(2)), float(m.group(3)))

    def score(prm):
        key = json.dumps(prm, sort_keys=True)
        if key not in cache:
            t0 = time.time()
            p = get_model(a.model, **prm).walk_forward(g, idx)
            cache[key] = (float(logloss_terms(y, p).mean()), calibration_slope(y, p)[0])
            print(f"  {prm} -> log-loss {cache[key][0]:.4f}, slope {cache[key][1]:.2f} ({time.time() - t0:.0f}s)", flush=True)
        return cache[key][0]

    best = {"weight_half_life_days": 365, "lam_team": 3, "lam_league": 0.1}
    if a.model == "strength":
        best = {"form_half_life": 10, **best}
    if tier:
        best = {**json.loads((REPORT_DIR / "runs" / "strength_noform_tuned.json").read_text())["params"],
                "margin_k": 0.16, "w_major": 0.0}
    elif margin:
        best = {**json.loads((REPORT_DIR / "runs" / "strength_noform_tuned.json").read_text())["params"], "margin_k": 0.04}
    best_s = score(best)
    for _ in range(3):
        improved = False
        for k, vals in grid.items():
            for v in vals:
                if v == best[k]:
                    continue
                trial = dict(best, **{k: v})
                s = score(trial)
                if s < best_s - 1e-5:
                    best, best_s, improved = trial, s, True
        if not improved:
            break
    elo = get_model("elo_league").walk_forward(g, idx)
    elo_ll = float(logloss_terms(y, elo).mean())
    out = REPORT_DIR / "runs"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}_tuned.json").write_text(json.dumps({"params": best, "tune_window": [TUNE_START, TUNE_END],
                                                         "log_loss": best_s, "elo_league_log_loss": elo_ll,
                                                         "created": datetime.now().isoformat(timespec="seconds")}, indent=1))
    L = [f"# {a.model} tuning (2024 walk-forward only)", "",
         f"{len(idx):,} games, {TUNE_START} to {TUNE_END}. The report window (2025 onward) was not used. "
         f"elo_league on the same games: log-loss {elo_ll:.4f}.", "",
         "| form half-life (games) | weight half-life (days) | lam_team | lam_league | margin k | log-loss | calibration slope |",
         "|---|---|---|---|---|---|---|"]
    if any(best[k] in (v[0], v[-1]) for k, v in grid.items()):
        L.insert(3, "**Edge warning:** the chosen value sits at the edge of the searched range for at least one parameter.\n")
    for key, (ll, sl) in sorted(cache.items(), key=lambda kv: kv[1][0]):
        prm = json.loads(key)
        mark = " **(chosen)**" if prm == best else ""
        L.append(f"| {prm.get('form_half_life', '—')} | {prm['weight_half_life_days']} | {prm['lam_team']} | {prm['lam_league']} | {prm.get('margin_k', '—')}{' / maj ' + str(prm['w_major']) if 'w_major' in prm else ''} | {ll:.4f}{mark} | {sl:.2f} |")
    if tier:     # temperature scaling decided and fitted on the 2024 tuning predictions ONLY
        from scipy.optimize import minimize_scalar
        p24 = np.clip(get_model(a.model, **best).walk_forward(g, idx), 1e-6, 1 - 1e-6)
        z = np.log(p24 / (1 - p24))
        slope24 = calibration_slope(y, p24)[0]
        res = minimize_scalar(lambda c: float(logloss_terms(y, 1 / (1 + np.exp(-c * z))).mean()),
                              bounds=(0.3, 2), method="bounded")
        apply = slope24 < 0.93
        scale = float(res.x) if apply else 1.0
        p_cal = 1 / (1 + np.exp(-scale * z))
        ll_cal, slope_cal = float(logloss_terms(y, p_cal).mean()), calibration_slope(y, p_cal)[0]
        (out / "strength_g15_tier_cal_tuned.json").write_text(json.dumps({
            "params": {**best, "logit_scale": round(scale, 4)}, "tune_window": [TUNE_START, TUNE_END],
            "rule": "apply temperature only if the 2024 calibration slope < 0.93", "slope_2024_raw": slope24,
            "applied": bool(apply), "log_loss_2024_raw": best_s, "log_loss_2024_cal": ll_cal,
            "slope_2024_cal": slope_cal, "created": datetime.now().isoformat(timespec="seconds")}, indent=1))
        L += ["", f"**Temperature scaling (2024 predictions only):** raw slope {slope24:.3f} -> "
              + (f"applied, logit x {scale:.3f}, slope after {slope_cal:.3f}, log-loss {best_s:.4f} -> {ll_cal:.4f}"
                 if apply else "not applied (slope >= 0.93)")]
        print("temperature", slope24, apply, scale, flush=True)
    (out / f"{a.model}_tuning.md").write_text("\n".join(L), encoding="utf-8")
    print("best", best, best_s, "| elo_league", elo_ll)


if __name__ == "__main__":
    main()
