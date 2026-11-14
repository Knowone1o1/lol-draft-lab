"""Tune the model's weighting by deleting games inside random patches and predicting their winners.

For each randomly chosen patch (from --years):
  1. draw several scenarios, each keeping a random share (0-80%) of the patch's games known and
     deleting the rest (0% known = every game deleted = day one of a new patch);
  2. rebuild every statistic WITHOUT the deleted games and without anything from later patches or
     later dates (no future information);
  3. predict the winner of each deleted game from its draft, its players and team strength (Elo).
Every game gets an out-of-fold prediction. A logistic win model is then scored leave-one-patch-out,
so the score (log-loss) never uses a patch's own results to fit the coefficients.

A coordinate search adjusts the weighting parameters (half-lives, patch boost, fearless decay,
shrinkage, Elo K) towards the lowest log-loss on the TUNE patches, then the result is confirmed on
separate VALIDATION patches. It also runs the binary shadow-variable bounds (draftlab/bounds.py)
on the validation predictions.

Usage:
    .venv\\Scripts\\python scripts\\tune.py                    # tune on 2026 patches, write report
    .venv\\Scripts\\python scripts\\tune.py --years 2025 2026 --apply
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.bounds import bin_quantiles, shadow_bounds_ci, worst_case_bounds  # noqa: E402
from draftlab.config import REPORT_DIR, TRAIN_LEAGUES  # noqa: E402
from draftlab.data import add_elo, load_games, patch_label  # noqa: E402
from draftlab.fast_stats import StatsIndex  # noqa: E402
from draftlab.features import WIN_FEATURES, actual_roles, comp_features, state_from_game  # noqa: E402
from draftlab.params import DEFAULTS, PARAMS_FILE, fearless_weights, load_params, snapshot_kwargs  # noqa: E402

GRID = {
    "half_life": [30, 45, 60, 90, 135, 200],
    "matchup_half_life": [180, 365, 540, 900, 1500],
    "patch_boost": [0, 1, 3, 6, 10],
    "fearless_decay": [0.5, 0.65, 0.77, 0.9, 1.0],
    "k_champ": [10, 20, 30, 50, 80],
    "k_pair": [5, 10, 20, 40, 80],
    "k_lane": [5, 10, 15, 30, 60],
    "elo_k": [12, 18, 24, 32, 40],
}


LIMITS = {"fearless_decay": (0.2, 1.0), "patch_boost": (0, 40), "elo_k": (4, 80)}


def extend_grid(name, values, best_v):
    """If the best value is the largest/smallest tried, propose the next step outward."""
    lo, hi = min(values), max(values)
    a, b = LIMITS.get(name, (1, 5000))
    if best_v == hi and hi < b:
        nxt = min(b, round(hi * 1.5, 3) if hi > 0 else 1)
    elif best_v == lo and lo > a:
        nxt = max(a, round(lo / 1.5, 3))
    else:
        return None
    return None if nxt in values else nxt


class Evaluator:
    """Deletion scenarios: for each patch, `scenarios` draws of a random KNOWN share in [lo, hi].

    known = 0%  -> every game of the patch is deleted: day one of a new patch, only older patches known.
    known = 80% -> most of the patch is known: late in the patch.
    random mode: the known games are a random subset of the patch (all leagues); stats are as of the
                 patch's end minus the deleted games.
    chrono mode: the known games are the EARLIEST ones; stats are as of the first deleted game, so the
                 deleted games are genuinely in the future.
    The deleted games from the training leagues are the ones predicted.
    """

    def __init__(self, games, idx, patches, scenarios, seed, lo=0.0, hi=0.80, chrono=False, train_mask=None):
        self.games, self.idx = games, idx
        self.records = games.to_dict("records")
        self.patch_arr = games["patch_idx"].to_numpy(dtype=float)
        self.dates = games["date"].to_numpy(dtype="datetime64[ns]")
        train_mask = np.ones(len(games), bool) if train_mask is None else np.asarray(train_mask)
        self.plan = []
        rng = np.random.default_rng(seed)
        for P, _ in patches:
            allp = np.nonzero(self.patch_arr == P)[0]               # every league on this patch
            order = allp[np.argsort(self.dates[allp], kind="stable")]
            end = self.dates[allp].max()
            for _ in range(scenarios):
                frac = float(rng.uniform(lo, hi))                    # share of the patch KNOWN
                k = max(1, len(allp) - int(round(frac * len(allp))))   # games deleted (all if frac = 0)
                if chrono:
                    deleted = order[-k:]
                    cutoff = self.dates[deleted].min()
                    ref = pd.Timestamp(cutoff)
                    base = ((self.patch_arr <= P) | np.isnan(self.patch_arr)) & (self.dates < cutoff)
                else:
                    deleted = rng.choice(allp, size=k, replace=False)
                    ref = pd.Timestamp(end) + pd.Timedelta(seconds=1)
                    base = ((self.patch_arr <= P) | np.isnan(self.patch_arr)) & (self.dates <= end)
                include = base.copy()
                include[deleted] = False
                predict = deleted[train_mask[deleted]]
                if len(predict):
                    self.plan.append((P, frac, include, predict, ref))
        self._elo = {}
        self.cache = {}

    def elo(self, k):
        if k not in self._elo:
            g2, _ = add_elo(self.games[["Blue_teamname", "Red_teamname", "blue_win"]], k=k)
            self._elo[k] = (g2["blue_elo"].to_numpy(), g2["red_elo"].to_numpy())
        return self._elo[k]

    def oof(self, params):
        """Out-of-fold features for every planned game under these params."""
        key = json.dumps(params, sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        kw = snapshot_kwargs(params)
        be, re_ = self.elo(params["elo_k"])
        X, y, grp, fr = [], [], [], []
        for P, frac, include, predict, ref in self.plan:
            snap = self.idx.snapshot(ref, target_patch=P, include=include, with_rosters=False,
                                     win_only=True, **kw)
            for i in predict:
                row = self.records[i]
                X.append(comp_features(state_from_game(row, 20), snap, be[i], re_[i], role_hint=actual_roles(row)))
                y.append(row["blue_win"])
                grp.append(P)
                fr.append(frac)
        out = (np.array(X), np.array(y), np.array(grp), np.array(fr))
        self.cache[key] = out
        return out

    def score(self, params, cols=None):
        X, y, grp, fr = self.oof(params)
        if cols is not None:
            X = X[:, cols]
        p = np.zeros(len(y))
        for P in np.unique(grp):  # leave-one-patch-out
            tr, te = grp != P, grp == P
            m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
            p[te] = m.predict_proba(X[te])[:, 1]
        bands = {}
        for a, b in BANDS:
            m = (fr >= a) & (fr < b)
            if m.sum() >= 30:
                bands[f"{a:.0%}-{b:.0%}"] = (log_loss(y[m], p[m], labels=[0, 1]), int(m.sum()))
        return {"log_loss": log_loss(y, p), "auc": roc_auc_score(y, p), "acc": float(((p > 0.5) == y).mean()),
                "n": len(y), "pred": p, "y": y, "frac": fr, "bands": bands}


BANDS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.81)]


def pick_patches(games, years, leagues, min_games, n_tune, n_val, seed):
    m = games["date"].dt.year.isin(years) & games["league"].isin(leagues) & games["patch_idx"].notna()
    cand = games[m].groupby("patch_idx").size()
    cand = np.array(cand[cand >= min_games].index, dtype=float)
    rng = np.random.default_rng(seed)
    rng.shuffle(cand)
    need = n_tune + n_val
    if len(cand) < need:
        n_val = max(1, len(cand) // 3)
        n_tune = len(cand) - n_val
    sel = lambda ps: [(int(P), np.nonzero((m & (games["patch_idx"] == P)).to_numpy())[0]) for P in ps]
    return sel(cand[:n_tune]), sel(cand[n_tune:n_tune + n_val])


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # Windows console: allow ⟂, → etc.
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="*", type=int, default=[2026])
    ap.add_argument("--tune-patches", type=int, default=8)
    ap.add_argument("--val-patches", type=int, default=6)
    ap.add_argument("--scenarios", type=int, default=6, help="deletion scenarios per patch")
    ap.add_argument("--min-known", type=float, default=0.0, help="smallest share of the patch known (0 = all deleted, day one)")
    ap.add_argument("--max-known", type=float, default=0.80, help="largest share of the patch known (late patch)")
    ap.add_argument("--random", dest="chrono", action="store_false",
                    help="known games = a random subset of the patch (older mode; lets later games of the patch leak in)")
    ap.set_defaults(chrono=True)  # default: known games = the earliest of the patch, stats as of the first deleted game
    ap.add_argument("--min-games", type=int, default=120, help="minimum games for a patch to be eligible")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--apply", action="store_true", help="write the tuned params to params.json")
    args = ap.parse_args()
    t0 = time.time()

    games, _, _ = load_games(elo_k=24.0)
    idx = StatsIndex(games)
    tune_p, val_p = pick_patches(games, args.years, TRAIN_LEAGUES, args.min_games,
                                 args.tune_patches, args.val_patches, args.seed)
    if not tune_p:
        sys.exit("No eligible patches; lower --min-games or change --years.")
    print(f"Tune patches: {[patch_label(P) for P, _ in tune_p]} ({sum(len(g) for _, g in tune_p)} games)")
    print(f"Validation patches: {[patch_label(P) for P, _ in val_p]} ({sum(len(g) for _, g in val_p)} games)")
    tm = games["league"].isin(TRAIN_LEAGUES).to_numpy()
    ev = dict(lo=args.min_known, hi=args.max_known, chrono=args.chrono, train_mask=tm)
    tune = Evaluator(games, idx, tune_p, args.scenarios, args.seed, **ev)
    val = Evaluator(games, idx, val_p, args.scenarios, args.seed + 1, **ev)
    print(f"{len(tune.plan)} tune scenarios, {len(val.plan)} validation scenarios; known share of patch "
          f"{args.min_known:.0%}-{args.max_known:.0%} ({'earliest games' if args.chrono else 'random games'})", flush=True)

    start = load_params()
    if start.get("fearless_decay") is None:
        start["fearless_decay"] = 0.77  # ~ the hand-set config weights (1, .8, .6, .45, .35)
    best = dict(start)
    sc0 = tune.score(best)
    best_s = sc0["log_loss"]
    history = [("start", dict(best), best_s)]
    v0 = val.score(best)
    # Every trial, for the charts: tune score of each tried change; validation score at each accepted one.
    trials = [{"trial": 0, "pass": 0, "param": "start", "value": None, "log_loss": best_s, "acc": sc0["acc"],
               "auc": sc0["auc"], "accepted": True, "best_log_loss": best_s, "best_acc": sc0["acc"],
               "val_log_loss": v0["log_loss"], "val_acc": v0["acc"]}]
    best_acc = sc0["acc"]
    print(f"start log-loss {best_s:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    for ps in range(args.passes):
        improved = False
        for name in GRID:
            values = list(GRID[name])
            i = 0
            while i < len(values):
                v = values[i]
                i += 1
                if v == best[name]:
                    continue
                trial = dict(best, **{name: v})
                sc = tune.score(trial)
                s = sc["log_loss"]
                acc_ok = s < best_s - 1e-5
                print(f"  pass {ps + 1} {name}={v}: {s:.4f}{'  *' if acc_ok else ''}  ({time.time() - t0:.0f}s)", flush=True)
                rec = {"trial": len(trials), "pass": ps + 1, "param": name, "value": v, "log_loss": s,
                       "acc": sc["acc"], "auc": sc["auc"], "accepted": bool(acc_ok)}
                if acc_ok:
                    best, best_s, improved, best_acc = trial, s, True, sc["acc"]
                    vs = val.score(best)
                    rec.update(val_log_loss=vs["log_loss"], val_acc=vs["acc"])
                rec.update(best_log_loss=best_s, best_acc=best_acc)
                trials.append(rec)
                # Best value on the edge of the grid: extend it outward (up to 3 extra steps).
                if i == len(values) and len(values) < len(GRID[name]) + 3:
                    ext = extend_grid(name, values, best[name])
                    if ext is not None:
                        values.append(ext)
            GRID[name] = sorted(set(values), key=lambda x: (x is None, x))
            history.append((name, dict(best), best_s))
        if not improved:
            break

    # ----------------------------------------------------------- validation
    res = {}
    for label, prm in (("Start (current)", start), ("Tuned", best)):
        res[label] = {"tune": tune.score(prm), "val": val.score(prm)}
    elo_only = {"tune": tune.score(best, cols=[0]), "val": val.score(best, cols=[0])}
    val_gain = res["Start (current)"]["val"]["log_loss"] - res["Tuned"]["val"]["log_loss"]

    # ----------------------------------------------------------- shadow-variable bounds
    vb = res["Tuned"]["val"]
    yv, pv = vb["y"], vb["pred"]
    z = bin_quantiles(pv, 5)
    rng = np.random.default_rng(args.seed)
    bound_rows = []
    for name, keep in (("Random 30% deleted (MAR)", np.full(len(yv), 0.7)),
                       ("Blue wins deleted 50%, losses 15% (MNAR)", np.where(yv == 1, 0.5, 0.85)),
                       ("Upsets deleted 60% (MNAR)", np.where((pv > 0.5) != (yv == 1), 0.4, 0.9))):
        r = rng.random(len(yv)) < keep
        b = shadow_bounds_ci(yv, r, z, n_boot=150, seed=args.seed)
        wl, wh = worst_case_bounds(yv, r)
        bound_rows.append((name, yv.mean(), yv[r].mean(), b, (wl, wh)))

    # ----------------------------------------------------------- report
    L = ["# Weight tuning: within-patch hold-out", "",
         f"Generated {datetime.now():%Y-%m-%d %H:%M} in {time.time() - t0:.0f}s.", "",
         "**Method.** Patches are drawn at random from " + ", ".join(map(str, args.years)) + f". For each patch, "
         f"{args.scenarios} scenarios each keep a random share between {args.min_known:.0%} and {args.max_known:.0%} "
         f"of the patch's games known ({'the earliest games, with stats as of the first deleted game' if args.chrono else 'chosen at random, all leagues'}) "
         "and delete the rest — from 'patch just released' (0% known: every game of the patch deleted) to "
         "'late in the patch' (80% known). Every statistic is rebuilt "
         "without the deleted games (and without later patches/dates), and the deleted games' winners are predicted "
         "from draft + players + Elo. A logistic model is scored leave-one-patch-out. Parameters are tuned on the "
         "*tune* patches and confirmed on separate *validation* patches.", "",
         f"- Tune patches: {', '.join(patch_label(P) for P, _ in tune_p)} ({res['Tuned']['tune']['n']} games)",
         f"- Validation patches: {', '.join(patch_label(P) for P, _ in val_p)} ({res['Tuned']['val']['n']} games)", "",
         "## Result", "", "| Params | Tune log-loss | Val log-loss | Val AUC | Val accuracy |", "|---|---|---|---|---|"]
    for label, r in res.items():
        L.append(f"| {label} | {r['tune']['log_loss']:.4f} | {r['val']['log_loss']:.4f} | {r['val']['auc']:.3f} | {r['val']['acc']:.1%} |")
    L.append(f"| Elo only (tuned Elo K) | {elo_only['tune']['log_loss']:.4f} | {elo_only['val']['log_loss']:.4f} | "
             f"{elo_only['val']['auc']:.3f} | {elo_only['val']['acc']:.1%} |")
    L += ["", "### By share of the patch known (validation log-loss, lower is better)", "",
          "| Patch known | Games predicted | Start | Tuned | Elo only |", "|---|---|---|---|---|"]
    for band in res["Tuned"]["val"]["bands"]:
        a = res["Start (current)"]["val"]["bands"].get(band, (float("nan"), 0))
        b = res["Tuned"]["val"]["bands"][band]
        c = elo_only["val"]["bands"].get(band, (float("nan"), 0))
        L.append(f"| {band} | {b[1]} | {a[0]:.4f} | {b[0]:.4f} | {c[0]:.4f} |")
    verdict = ("**Tuned params improve the validation patches** — safe to apply." if val_gain > 0.0005 else
               "**No reliable improvement on validation patches** — keep current params (gains on tune patches "
               "were likely noise).")
    L += ["", verdict, "", "## Parameters", "", "| Parameter | Start | Tuned | Grid |", "|---|---|---|---|"]
    for k in GRID:
        L.append(f"| {k} | {start[k]} | **{best[k]}** | {GRID[k]} |")
    L += ["", f"Fearless game weights (tuned): {fearless_weights(best)}", "",
          "## Search path (tune log-loss after each parameter)", ""]
    L += [f"- {n}: {s:.4f}" for n, _, s in history]
    L += ["", "## Missing results: binary shadow-variable bounds", "",
          "Using our model's prediction (5 bins) as the shadow variable Z, Y = blue won, R = result kept. "
          "Assumption: Z ⟂ R | Y. The 90% interval comes from subsampling; worst-case assumes nothing.", "",
          "| Deletion pattern | True blue WR | Naive (kept games) | Shadow 90% CI | Worst case | Width saved |",
          "|---|---|---|---|---|---|"]
    for name, truth, naive, b, (wl, wh) in bound_rows:
        w = b["ci_upper"] - b["ci_lower"]
        L.append(f"| {name} | {truth:.3f} | {naive:.3f} | [{b['ci_lower']:.3f}, {b['ci_upper']:.3f}] | "
                 f"[{wl:.3f}, {wh:.3f}] | {1 - w / max(wh - wl, 1e-9):.0%} |")
    L += ["", "Note: the 'upsets deleted' pattern depends on the model's own prediction, which breaks the "
          "Z ⟂ R | Y assumption on purpose — it shows what happens when the assumption fails."]
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "tuning.md").write_text("\n".join(L), encoding="utf-8")
    (REPORT_DIR / "tuned_params.json").write_text(json.dumps({"params": best, "val_gain": val_gain}, indent=2))
    # Chart data: accuracy / log-loss vs share of the patch known, for start / tuned / Elo-only.
    def curve(r, step=0.1):
        y, p, fr = r["y"], r["pred"], r["frac"]
        out = []
        for a in np.arange(0, args.max_known + 1e-9, step):
            m = (fr >= a) & (fr < a + step)
            if m.sum() >= 20:
                out.append({"known_from": round(float(a), 2), "known_to": round(float(a + step), 2), "n": int(m.sum()),
                            "acc": float(((p[m] > 0.5) == y[m]).mean()),
                            "log_loss": float(log_loss(y[m], p[m], labels=[0, 1]))})
        return out
    chart = {"generated": datetime.now().isoformat(), "years": args.years, "chrono": args.chrono,
             "tune_patches": [patch_label(P) for P, _ in tune_p], "val_patches": [patch_label(P) for P, _ in val_p],
             "trials": trials, "start": start, "tuned": best, "grid": GRID, "val_gain": val_gain,
             "summary": {lab: {k: {m: r[k][m] for m in ("log_loss", "auc", "acc", "n")} for k in ("tune", "val")}
                         for lab, r in list(res.items()) + [("Elo only", elo_only)]},
             "known_curve": {lab: {k: curve(r[k]) for k in ("tune", "val")}
                             for lab, r in list(res.items()) + [("Elo only", elo_only)]},
             "bounds": [{"pattern": n, "truth": t, "naive": nv, "ci": [b["ci_lower"], b["ci_upper"]], "worst": [wl, wh]}
                        for n, t, nv, b, (wl, wh) in bound_rows]}
    (REPORT_DIR / "tuning_data.json").write_text(json.dumps(chart, indent=1, default=float), encoding="utf-8")
    print("\n".join(L))
    if args.apply:
        if val_gain > 0.0005:
            PARAMS_FILE.write_text(json.dumps({"params": best, "tuned": datetime.now().isoformat(),
                                               "val_log_loss_gain": val_gain}, indent=2))
            print(f"\nApplied: wrote {PARAMS_FILE}. Rebuild with scripts/build.py.")
        else:
            print("\nNot applied: no validation improvement.")


if __name__ == "__main__":
    main()
