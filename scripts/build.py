"""Build everything: load data -> validate variants -> score the frozen holdout -> fit final models -> save.

Protocol (judge finding M4):
  * Frozen holdout = every game from config.HOLDOUT_START (Worlds 2026 + the last 6 weeks). No choice looks
    at it; it is scored once, for the configuration chosen on validation.
  * Rolling-origin validation before it: draft models on 3 folds of 2 months (each fit on all earlier
    months), the win model on each of the last 12 months before the holdout (each fit on earlier months).
  * Every "model vs baseline" and "variant vs current" claim gets a paired bootstrap 95% interval
    (resampling whole games).
  * Variants (M1 Elo, M8 champion strength, M9 ban features) are adopted only if they clearly help on
    validation (interval excludes zero) and nothing clearly gets worse.

Usage:
    .venv\\Scripts\\python scripts\\build.py
    .venv\\Scripts\\python scripts\\build.py --cache data\\cache\\examples.pkl   (reuse built examples)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.config import (ADJUSTED_STRENGTH, ELO_LEAGUE_ADJUST, ELO_REGRESS,  # noqa: E402
                             ELO_REGRESS_GAP_DAYS, HOLDOUT_START, MODEL_DIR, REPORT_DIR, ROLES,
                             TRAIN_LEAGUES)
from draftlab.data import add_elo, load_games, patch_index, patch_label  # noqa: E402
from draftlab.fast_stats import StatsIndex  # noqa: E402
from draftlab.features import BAN_FEATURES, PICK_FEATURES, WIN_FEATURES, state_from_game  # noqa: E402
from draftlab.league import INTERNATIONAL, monthly_adjusted  # noqa: E402
from draftlab.matchups import counters_report, matchup_table, team_matchup_table  # noqa: E402
from draftlab.models import ConditionalLogit, WinModel, heuristic_pick_scores  # noqa: E402
from draftlab.params import fearless_weights, load_params, snapshot_kwargs  # noqa: E402
from draftlab.training import STAGES, build_examples, month_starts  # noqa: E402
from draftlab.validation import boot_mean, fmt_ci, logloss_terms, paired_diff, ranks  # noqa: E402

N_FOLDS, FOLD_MONTHS = 3, 2      # draft-model validation folds just before the holdout
WIN_VAL_MONTHS = 12              # win-model rolling origins (one per month)
PICK_NAMES = list(PICK_FEATURES)
BAN_PRESENCE = [f for f in BAN_FEATURES if f != "pick_rate"]       # current ban model
BAN_SPLIT = [f for f in BAN_FEATURES if f != "meta_presence"]      # M9: pick rate + ban rate, no presence
STR_COL = {"pick": PICK_FEATURES.index("champ_strength"), "ban": BAN_FEATURES.index("champ_strength")}
ELO_VARIANTS = [  # (name, regress, league offsets); the first is the current method
    ("raw Elo (current)", 0.0, False),
    ("raw Elo + league offsets", 0.0, True),
    ("Elo, 20% split regression", 0.2, False),
    ("Elo, 20% split regression + league offsets", 0.2, True),
    ("Elo, 40% split regression", 0.4, False),
    ("Elo, 40% split regression + league offsets", 0.4, True),
]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--leagues", nargs="*", help="league codes to train on (default: config.TRAIN_LEAGUES)")
    ap.add_argument("--train-months", type=int, default=12, help="months of drafts before the first validation fold")
    ap.add_argument("--neg", type=int, default=60, help="negative candidates sampled per training action")
    ap.add_argument("--boot", type=int, default=2000, help="bootstrap resamples for the intervals")
    ap.add_argument("--final-evaluation", action="store_true",
                    help="score the frozen holdout (only when the user explicitly asks for the final evaluation)")
    ap.add_argument("--cache", help="pickle file to save/reuse the built examples (rebuilt if inputs changed)")
    ap.add_argument("--refresh-stats", action="store_true",
                    help="only rebuild stats/Elo/rosters from the latest data, keep trained models (fast)")
    ap.add_argument("--patch", help="patch the app should prepare for, e.g. 16.20 (default: most played in last 14 days)")
    ap.add_argument("--patch-boost", type=float, default=None,
                    help="override the patch boost from params.json/config (0 = recency only)")
    args = ap.parse_args()
    params = load_params()
    if args.patch_boost is not None:
        params["patch_boost"] = args.patch_boost
    kw = snapshot_kwargs(params)
    print("Params:", params)

    t0 = time.time()
    print("Loading data ...")
    games, elo, fearless = load_games(elo_k=params["elo_k"])
    idx = StatsIndex(games)
    target = patch_index(args.patch) if args.patch else current_patch(games)
    print(f"  target patch for the app: {patch_label(target)}")
    if args.refresh_stats:
        path = MODEL_DIR / "draftlab.pkl"
        with open(path, "rb") as f:
            art = pickle.load(f)
        art["snapshot"] = idx.snapshot(games["date"].max() + pd.Timedelta(days=1), target_patch=target, **kw)
        art["params"] = params
        art["patch"] = patch_label(target)
        art["matchups"] = write_matchups(games, art["snapshot"])
        art["elo"] = elo
        art["data_range"] = (str(games["date"].min().date()), str(games["date"].max().date()))
        with open(path, "wb") as f:
            pickle.dump(art, f)
        print(f"Refreshed stats in {path} ({time.time() - t0:.0f}s); models unchanged.")
        return
    print(f"  {len(games)} games, {games['date'].min():%Y-%m-%d} -> {games['date'].max():%Y-%m-%d}, "
          f"{games['has_order'].mean():.0%} with full pick order")

    leagues = set(args.leagues) if args.leagues else TRAIN_LEAGUES
    counts = games["league"].value_counts()
    used = [l for l in counts.index if l in leagues]
    if not used:
        sys.exit("None of the training leagues are in the data.")
    mask = games["league"].isin(leagues)

    # ------------------------------------------------------------- months, folds, holdout
    H = pd.Timestamp(HOLDOUT_START)
    hm = H.to_period("M").to_timestamp()
    all_months = month_starts(games["date"])[1:]
    if hm not in all_months:
        sys.exit(f"Holdout start {H:%Y-%m-%d} is outside the data.")
    h_i = all_months.index(hm)
    val_months = all_months[h_i - N_FOLDS * FOLD_MONTHS + 1: h_i + 1]
    folds = [val_months[i:i + FOLD_MONTHS] for i in range(0, len(val_months), FOLD_MONTHS)]
    months = all_months[max(0, all_months.index(val_months[0]) - args.train_months):]
    win_val = [m for m in months if m <= hm][-WIN_VAL_MONTHS:]
    print(f"Window {months[0]:%Y-%m}..{months[-1]:%Y-%m}; holdout from {H:%Y-%m-%d}; draft folds "
          + ", ".join("+".join(f"{m:%Y-%m}" for m in f) for f in folds))

    ex = load_or_build(args, games, mask, months, set(val_months), H, params, idx)
    n_hold_games = int(np.sum(ex["win"]["holdout"]))
    print(f"  examples ready ({time.time() - t0:.0f}s); holdout games: {n_hold_games}")

    W = ex["win"]
    Xw0 = np.array(W["X"], dtype=float)
    yw = np.array(W["y"], dtype=float)
    gid = np.array(W["gameid"], dtype=object)
    wmon = np.array(W["month"], dtype="datetime64[ns]")
    hold = np.array(W["holdout"], dtype=bool)
    wleague = np.array(W["league"], dtype=object)
    intl = np.isin(wleague, list(INTERNATIONAL))
    rows_by_id = {r["gameid"]: r for r in games[games["gameid"].isin(set(gid))].to_dict("records")}
    stage_ids = [g for g in gid if g in ex["stage"]]
    S0 = {s: np.array([ex["stage"][g][s] for g in stage_ids], dtype=float) for s in STAGES}
    s_pos = {g: i for i, g in enumerate(gid)}
    s_rows = np.array([s_pos[g] for g in stage_ids], dtype=int)

    R = Report()
    R.h1("LoL Draft Lab: backtest report")
    R.p(f"Generated {datetime.now():%Y-%m-%d %H:%M}. Data: {len(games)} games "
        f"({games['date'].min():%Y-%m-%d} to {games['date'].max():%Y-%m-%d}). Training leagues: {', '.join(used)}.")
    R.p("**Protocol.** Every month, statistics are rebuilt from games played *before* that month, so no game "
        f"ever sees its own result. A **frozen holdout** (every game from {H:%Y-%m-%d}: Worlds 2026 plus the last "
        f"6 weeks; {n_hold_games} games in the training leagues) is never used for any choice: all variant "
        "comparisons below use **rolling-origin validation** before it, and the holdout is scored once for the "
        "chosen configuration. Draft models are validated on "
        f"{len(folds)} folds of {FOLD_MONTHS} months ("
        + "; ".join(f"{f[0]:%Y-%m}..{f[-1]:%Y-%m}" for f in folds)
        + f", the last fold stopping at {H:%Y-%m-%d}), each fitted on all earlier months from {months[0]:%Y-%m}. "
        f"The win model is validated month by month over the {len(win_val)} months before the holdout, each "
        "month predicted by a model fitted on the months before it. "
        f"Intervals are paired bootstrap 95% intervals ({args.boot} resamples of whole games).")
    R.p(f"Weighting: meta stats use a {params['half_life']:g}-day half-life with patch boost "
        f"{params['patch_boost']:g}; counters and synergies a {params['matchup_half_life']:g}-day half-life; "
        f"fearless game weights {fearless_weights(params)}; shrinkage k_champ={params['k_champ']:g}, "
        f"k_pair={params['k_pair']:g}, k_lane={params['k_lane']:g}; Elo K={params['elo_k']:g}.")

    # ============================================================== M1: Elo variants (win model)
    print("Elo variants ...")
    elo_maps = {}
    elo_res = []
    for name, reg, lg in ELO_VARIANTS:
        b, r, _ = elo_variant(games, params["elo_k"], reg, lg)
        elo_maps[name] = (b, r)
        d = dict(zip(games["gameid"], (b - r) / 400.0))
        X = Xw0.copy()
        X[:, 0] = [d[g] for g in gid]
        res = rolling_win(X, yw, wmon, hold, win_val)
        elo_res.append((name, reg, lg, X, res))
        print(f"  {name}: Elo-only {res['ll_elo'].mean():.4f}, Elo+draft {res['ll_full'].mean():.4f}")
    base = elo_res[0][4]
    vmask = base["intl"] = intl[base["rows"]]
    R.h2("1. Variant selection on validation (before the holdout)")
    R.p("Adoption rule: a variant replaces the current method only if its paired 95% interval of improvement "
        "excludes zero and no other metric gets clearly worse. Every variant tried is listed.")
    R.h3("1a. Team strength (M1): Elo variants, win model, rolling-origin validation")
    R.p(f"{len(base['y'])} validation games ({int(vmask.sum())} at international events / EMEA Masters, where league "
        "strength matters most). Log-loss, lower is better; Δ = variant minus current (negative = better). "
        "League offsets are refitted every month on earlier games only; split regression pulls a team's rating "
        f"toward its league's mean after a break of more than {ELO_REGRESS_GAP_DAYS} days.")
    rows, elo_ok = [], []
    for name, reg, lg, X, res in elo_res:
        de = paired_diff(res["ll_elo"], base["ll_elo"], B=args.boot)
        dfull = paired_diff(res["ll_full"], base["ll_full"], B=args.boot)
        di = paired_diff(res["ll_elo"][vmask], base["ll_elo"][vmask], B=args.boot)
        rows.append((name, f"{res['ll_elo'].mean():.4f}", fmt_ci(*de, sign=True), f"{res['ll_full'].mean():.4f}",
                     fmt_ci(*dfull, sign=True), f"{res['ll_elo'][vmask].mean():.4f}", fmt_ci(*di, sign=True)))
        if name != ELO_VARIANTS[0][0] and de[2] < 0 and dfull[1] <= 0 and di[1] <= 0:
            # interval of the main metric excludes zero, and the others are not clearly worse
            elo_ok.append((res["ll_elo"].mean(), name))
    R.table(["Variant", "Elo-only log-loss", "Δ vs current", "Elo+draft log-loss", "Δ vs current",
             "International: Elo-only", "Δ vs current"], rows)
    chosen_elo = min(elo_ok)[1] if elo_ok else ELO_VARIANTS[0][0]
    ce = next(v for v in ELO_VARIANTS if v[0] == chosen_elo)
    R.p(f"**Chosen: {chosen_elo}.**" + (" No variant clearly beat the current Elo." if not elo_ok else
        f" Clearly better than the current Elo on validation ({len(elo_ok)} of {len(ELO_VARIANTS) - 1} variants passed; "
        "the best one is used)."))
    _, _, _, Xw, _ = next(e for e in elo_res if e[0] == chosen_elo)
    eb, er = elo_maps[chosen_elo]
    ediff = dict(zip(games["gameid"], (eb - er) / 400.0))
    S = {s: v.copy() for s, v in S0.items()}
    for s in S:
        S[s][:, 0] = [ediff[g] for g in stage_ids]

    # ============================================================== M8: champion strength
    print("Team-adjusted champion strength per month ...")
    adj = {m: idx.adjusted_strength(m, params["half_life"], fearless_weights(params), params["k_champ"],
                                    elo_diff=eb - er) for m in months}
    Xw_adj = Xw.copy()
    Xw_adj[:, 1] = [team_strength(rows_by_id[g], 20, adj[pd.Timestamp(m)]) for g, m in zip(gid, wmon)]
    S_adj = {s: v.copy() for s, v in S.items()}
    for s, n in STAGES.items():
        S_adj[s][:, 1] = [team_strength(rows_by_id[g], n, adj[pd.Timestamp(wmon[i])]) for g, i in zip(stage_ids, s_rows)]
    for kind in ex["draft"]:
        col = STR_COL["pick" if kind == "pick" else "ban"]
        for X, y, meta in ex["draft"][kind]:
            meta["str_raw"] = X[:, col].astype(np.float32)
            a = adj[pd.Timestamp(meta["month"])]
            meta["str_adj"] = np.array([a.get(c, 0.0) for c in meta["cands"]], np.float32)
    top_adj = sorted(adj[months[-1]].items(), key=lambda t: -t[1])[:8]

    res_raw = next(e for e in elo_res if e[0] == chosen_elo)[4]
    res_adj = rolling_win(Xw_adj, yw, wmon, hold, win_val)

    print("Draft models, rolling-origin folds ...")
    rng = np.random.default_rng(7)
    for kind in ex["draft"]:
        for X, y, meta in ex["draft"][kind]:
            if meta["full"] and len(X) - 1 > args.neg:
                meta["sub"] = np.r_[0, 1 + rng.choice(len(X) - 1, size=args.neg, replace=False)]
    val = {}
    for kind, names_list in (("pick", [("base", PICK_NAMES)]), ("ban1", [("base", BAN_PRESENCE), ("split", BAN_SPLIT)]),
                             ("ban2", [("base", BAN_PRESENCE), ("split", BAN_SPLIT)])):
        for strv in ("raw", "adj"):
            set_strength(ex["draft"][kind], STR_COL["pick" if kind == "pick" else "ban"], strv)
            for vname, names in names_list:
                if vname == "split" and strv == "adj":
                    continue
                val[(kind, strv, vname)] = rolling_draft(ex["draft"][kind], names, folds)
                r = val[(kind, strv, vname)]
                print(f"  {kind} {strv} {vname}: top5 {np.mean(r['rank'] <= 5):.3f} ({time.time() - t0:.0f}s)")
    # per-action validation ranks, for before/after comparisons of data fixes (e.g. C0 pick order)
    import os
    tag = "legacy_order" if os.environ.get("DRAFTLAB_LEGACY_ORDER") == "1" else "current"
    with open(REPORT_DIR / f"val_ranks_{tag}.pkl", "wb") as f:
        pickle.dump({k: {"rank": np.asarray(v["rank"]), "gid": np.asarray(v["gid"])} for k, v in val.items()}, f)

    R.h3("1b. Champion strength (M8): raw win rate vs team-adjusted")
    R.p("Team-adjusted strength: a recency-weighted logistic regression of each game's result on both teams' "
        "(chosen) ratings plus one effect per champion (+ for blue, − for red), with the same pull toward 50% as "
        "the raw win rate (L2 = k_champ/4). It is refitted every month on earlier games only. Δ = adjusted minus raw.")
    rows = []
    dll = paired_diff(res_adj["ll_full"], res_raw["ll_full"], B=args.boot)
    rows.append(("Win model (Elo + draft), log-loss", f"{res_raw['ll_full'].mean():.4f}", f"{res_adj['ll_full'].mean():.4f}",
                 fmt_ci(*dll, sign=True)))
    gains = [("win", -dll[0], dll[1], dll[2])]  # lower log-loss = gain; keep raw interval for the rule
    for kind in ("pick", "ban1", "ban2"):
        a, b = val[(kind, "adj", "base")], val[(kind, "raw", "base")]
        d = paired_diff(a["rank"] <= 5, b["rank"] <= 5, a["gid"], B=args.boot)
        rows.append((f"{kind}: top-5", f"{np.mean(b['rank'] <= 5):.1%}", f"{np.mean(a['rank'] <= 5):.1%}",
                     fmt_ci(*d, pct=True, sign=True)))
        gains.append((kind, d[0], d[1], d[2]))
    R.table(["Metric", "Raw win rate", "Team-adjusted", "Δ [95% CI]"], rows)
    win_better = dll[2] < 0
    win_worse = dll[1] > 0
    draft_better = any(lo > 0 for k, m, lo, hi in gains[1:])
    draft_worse = any(hi < 0 for k, m, lo, hi in gains[1:])
    use_adj = (win_better or draft_better) and not (win_worse or draft_worse)
    R.p(f"**Chosen: {'team-adjusted' if use_adj else 'raw win rate (unchanged)'}.** "
        + ("It clearly improves at least one metric and harms none." if use_adj else
           "The adjusted version did not clearly improve any metric (or clearly hurt one).")
        + " Strongest champions by the adjusted measure in the latest month: "
        + ", ".join(f"{c} {v * 100:+.1f}%" for c, v in top_adj) + ".")
    strv = "adj" if use_adj else "raw"
    Xw_final = Xw_adj if use_adj else Xw
    S_final = S_adj if use_adj else S
    res_val = res_adj if use_adj else res_raw

    R.h3("1c. Ban model features (M9): presence vs separate pick rate and ban rate")
    R.p("Presence = pick rate + ban rate, so the old ban model's `ban_rate` and `meta_presence` weights were "
        "near-collinear and unreadable. The split variant replaces presence with the pick rate. Adoption rule "
        "for this interpretability fix: used unless it clearly hurts top-5 (interval entirely below zero).")
    rows, use_split = [], {}
    for kind in ("ban1", "ban2"):
        a, b = val[(kind, "raw", "split")], val[(kind, "raw", "base")]
        d = paired_diff(a["rank"] <= 5, b["rank"] <= 5, a["gid"], B=args.boot)
        use_split[kind] = d[2] >= 0
        rows.append((kind, f"{np.mean(b['rank'] <= 5):.1%}", f"{np.mean(a['rank'] <= 5):.1%}", fmt_ci(*d, pct=True, sign=True),
                     "split" if use_split[kind] else "presence"))
    R.table(["Action", "Presence (current)", "Pick rate + ban rate", "Δ top-5 [95% CI]", "Chosen"], rows)
    feat_for = {"pick": PICK_NAMES,
                "ban1": BAN_SPLIT if use_split["ban1"] else BAN_PRESENCE,
                "ban2": BAN_SPLIT if use_split["ban2"] else BAN_PRESENCE}

    R.p("Variants tried in this build: 6 Elo variants (table 1a), 2 champion-strength measures (1b), 2 ban "
        "feature sets (1c); nothing else. Earlier sessions also compared, on 2026-08..2026-11 (months that overlap "
        "this holdout): one vs two memories, 2025-26 vs 2022-26 data, and patch boost 3 vs 10 "
        "(`reports/backtest_*.md`). Those three settings were kept as they were, so the holdout is untouched by "
        "every choice made here but not by those three earlier ones.")

    # ============================================================== 2. validation results (chosen config)
    R.h2("2. Predicting pro draft actions (validation, chosen configuration)")
    for kind in ex["draft"]:
        set_strength(ex["draft"][kind], STR_COL["pick" if kind == "pick" else "ban"], strv)
    vkey = {"pick": (strv, "base"), "ban1": (strv, "split" if use_split["ban1"] else "base"),
            "ban2": (strv, "split" if use_split["ban2"] else "base")}
    for kind in ("ban1", "ban2"):  # the split variant was only run with raw strength
        if (kind, *vkey[kind]) not in val:
            val[(kind, *vkey[kind])] = rolling_draft(ex["draft"][kind], feat_for[kind], folds)
    R.p("For each real pick/ban, every available champion is ranked. Top-k = how often the champion actually "
        "chosen was in our top k (ties count half). Baselines need no training. Role fit = probability the "
        "champion is played in the best role the team still has open; comfort = log(1 + the player's recent "
        "games on it). Δ = model minus the strongest baseline, with a paired 95% interval.")
    draft_table(R, {k: val[(k, *vkey[k])] for k in ("pick", "ban1", "ban2")}, args.boot)

    R.h2("3. Win probability: does the draft add information beyond team strength? (validation)")
    win_table(R, res_val, args.boot)

    # ============================================================== 4. M13 stages (validation)
    R.h2("4. Win model by draft stage (M13)")
    R.p("The win model is trained on complete drafts with true roles, but the app uses it from the first ban "
        "with roles inferred from the picks so far. Here the same rolling-origin models score each game's "
        "draft at fixed stages (roles inferred, as in the app). Δ = Elo + draft minus Elo only; negative = the "
        "draft terms help. 'bans only' has no picks, so the model reduces to Elo with its own coefficient.")
    stage_val = stage_scores(S_final, s_rows, res_val, wmon, hold, yw)
    stage_table(R, stage_val, args.boot, "Validation games (draft folds)")

    # ============================================================== 5. frozen holdout, scored once
    if not args.final_evaluation:
        R.h2("5. Frozen holdout")
        R.p(f"Not scored (games from {H:%Y-%m-%d} stay untouched until the final evaluation, `--final-evaluation`).")
    if args.final_evaluation:
        print("Scoring the frozen holdout ...")
        R.h2(f"5. Frozen holdout (games from {H:%Y-%m-%d}), scored once")
        R.p(f"Chosen configuration: Elo = {chosen_elo}; champion strength = {'team-adjusted' if use_adj else 'raw win rate'}; "
            f"ban features = ban1 {'pick+ban rate' if use_split['ban1'] else 'presence'}, ban2 "
            f"{'pick+ban rate' if use_split['ban2'] else 'presence'}. Models are fitted on every month before the "
            "holdout and never refitted on it for these numbers.")
        hres = {}
        for kind in ("pick", "ban1", "ban2"):
            exs = ex["draft"][kind]
            tr = [train_view(X, meta, feat_for[kind]) for X, y, meta in exs if not meta["holdout"]]
            te = [(X, y, meta) for X, y, meta in exs if meta["holdout"]]
            m = ConditionalLogit(feat_for[kind]).fit(tr)
            hres[kind] = score_draft(m, feat_for[kind], te)
        draft_table(R, hres, args.boot)
        trn = ~hold
        full_m = WinModel(WIN_FEATURES).fit(Xw_final[trn], yw[trn])
        elo_m = WinModel(WIN_FEATURES[:1]).fit(Xw_final[trn][:, :1], yw[trn])
        drf_m = WinModel(WIN_FEATURES[1:]).fit(Xw_final[trn][:, 1:], yw[trn])
        hr = {"y": yw[hold], "p_full": full_m.proba(Xw_final[hold]), "p_elo": elo_m.proba(Xw_final[hold][:, :1]),
              "p_draft": drf_m.proba(Xw_final[hold][:, 1:]), "p_base": np.full(hold.sum(), yw[trn].mean())}
        hr["ll_full"], hr["ll_elo"] = logloss_terms(hr["y"], hr["p_full"]), logloss_terms(hr["y"], hr["p_elo"])
        R.h3("Win probability on the holdout")
        win_table(R, hr, args.boot)
        hmods = {pd.Timestamp(m): (full_m, elo_m) for m in np.unique(wmon[hold])}
        stage_h = stage_scores(S_final, s_rows, {"models": hmods}, wmon, hold, yw, holdout=True)
        stage_table(R, stage_h, args.boot, "Holdout games")
        R.p(stage_verdict(stage_val, stage_h, args.boot))
    else:
        R.p(stage_verdict(stage_val, None, args.boot))

    # ============================================================== final fit (all months, for the app)
    print("Fitting final models on all months ...")
    final = {}
    for kind in ("pick", "ban1", "ban2"):
        allx = [train_view(X, meta, feat_for[kind]) for X, y, meta in ex["draft"][kind]]
        final[kind] = ConditionalLogit(feat_for[kind]).fit(allx)
    final["win"] = WinModel(WIN_FEATURES).fit(Xw_final, yw)
    R.h2("6. Final models (fitted on every month, including the holdout, for the app)")
    for kind, m in final.items():
        if kind == "win":
            continue
        R.h3(f"What drives {kind} decisions (standardised weights)")
        R.table(["Feature", "Weight"], [(n, f"{w:+.2f}") for n, w in m.coefficients()])
    R.h3("Win model weights (Elo + draft)")
    R.table(["Feature", "Coefficient"], [(n, f"{w:+.3f}") for n, w in final["win"].coefficients()])

    want = (ce[1], ce[2], use_adj)
    have = (float(ELO_REGRESS), bool(ELO_LEAGUE_ADJUST), bool(ADJUSTED_STRENGTH or params.get("adjusted_strength")))
    if want != have:
        msg = (f"config.py does not match the validation choice: want ELO_REGRESS={ce[1]}, ELO_LEAGUE_ADJUST={ce[2]}, "
               f"ADJUSTED_STRENGTH={use_adj}; config has {have}. The saved app model follows the validation choice; "
               "update config.py so the app's league offsets, early-game and game-plan models use the same ratings.")
        print("WARNING:", msg)
        R.p(f"**Warning:** {msg}")
    else:
        R.p("config.py matches the validation choice.")

    # app snapshot uses the chosen ratings and strength measure
    idx.elo_diff = eb - er
    _, elo_final = add_elo(games, k=params["elo_k"], regress=ce[1], gap_days=ELO_REGRESS_GAP_DAYS)
    ref = games["date"].max() + pd.Timedelta(days=1)
    kw_final = dict(kw, adjusted_strength=use_adj)
    snap = idx.snapshot(ref, target_patch=target, **kw_final)
    params_out = dict(params, adjusted_strength=use_adj)
    matchups = write_matchups(games, snap)
    MODEL_DIR.mkdir(exist_ok=True)
    with open(MODEL_DIR / "draftlab.pkl", "wb") as f:
        pickle.dump({"snapshot": snap, "models": final, "elo": elo_final, "matchups": matchups, "params": params_out,
                     "patch": patch_label(target), "built": datetime.now().isoformat(),
                     "elo_method": chosen_elo, "league_adjusted": ce[2],
                     "data_range": (str(games["date"].min().date()), str(games["date"].max().date())),
                     "holdout_start": str(H.date()), "train_leagues": used}, f)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "backtest.md").write_text(R.text(), encoding="utf-8")
    print(f"Saved models/draftlab.pkl and reports/backtest.md in {time.time() - t0:.0f}s")
    print()
    print(R.text())


# ------------------------------------------------------------------ helpers
def load_or_build(args, games, mask, months, full_months, H, params, idx):
    key = hashlib.sha1(json.dumps({"p": params, "m": [str(m) for m in months], "f": sorted(str(m) for m in full_months),
                                   "h": str(H), "n": len(games), "neg": args.neg, "last": str(games["date"].max()),
                                   "leagues": sorted(set(games.loc[mask, "league"]))},
                                  sort_keys=True, default=str).encode()).hexdigest()
    if args.cache and Path(args.cache).exists():
        with open(args.cache, "rb") as f:
            c = pickle.load(f)
        if c.get("key") == key:
            print("  using cached examples", args.cache)
            return c["ex"]
        print("  cache is stale, rebuilding")
    print(f"Building examples for {len(months)} months ...")
    ex = build_examples(games, mask, months, n_neg_train=args.neg, full_months=full_months, holdout_start=H,
                        params=params, index=idx)
    if args.cache:
        Path(args.cache).parent.mkdir(parents=True, exist_ok=True)
        with open(args.cache, "wb") as f:
            pickle.dump({"key": key, "ex": ex}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return ex


def elo_variant(games, k, regress, league):
    g2, final = add_elo(games, k=k, regress=regress, gap_days=ELO_REGRESS_GAP_DAYS)
    b, r = g2["blue_elo"].to_numpy(float), g2["red_elo"].to_numpy(float)
    if league:
        b, r = monthly_adjusted(g2, b, r)
    return b, r, final


def team_strength(row, n_actions, adj):
    st = state_from_game(row, n_actions)
    return (sum(adj.get(c, 0.0) for c in st.picks("Blue")) - sum(adj.get(c, 0.0) for c in st.picks("Red"))) / 5


def rolling_win(X, y, mon, hold, val_months):
    out = {k: [] for k in ("y", "p_full", "p_elo", "p_draft", "p_base", "rows")}
    models = {}
    for m in val_months:
        m64 = np.datetime64(m)
        tr = (mon < m64) & ~hold
        te = (mon == m64) & ~hold
        if tr.sum() < 200 or te.sum() == 0:
            continue
        full = WinModel(WIN_FEATURES).fit(X[tr], y[tr])
        eo = WinModel(WIN_FEATURES[:1]).fit(X[tr][:, :1], y[tr])
        dr = WinModel(WIN_FEATURES[1:]).fit(X[tr][:, 1:], y[tr])
        models[pd.Timestamp(m)] = (full, eo)
        out["y"].append(y[te])
        out["p_full"].append(full.proba(X[te]))
        out["p_elo"].append(eo.proba(X[te][:, :1]))
        out["p_draft"].append(dr.proba(X[te][:, 1:]))
        out["p_base"].append(np.full(te.sum(), y[tr].mean()))
        out["rows"].append(np.nonzero(te)[0])
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["ll_full"] = logloss_terms(out["y"], out["p_full"])
    out["ll_elo"] = logloss_terms(out["y"], out["p_elo"])
    out["models"] = models
    return out


def set_strength(exs, col, variant):
    key = "str_" + variant
    for X, y, meta in exs:
        X[:, col] = meta[key]


def train_view(X, meta, names):
    allnames = PICK_FEATURES if "role_fit" in names else BAN_FEATURES
    cols = [allnames.index(n) for n in names]
    sub = meta.get("sub")
    Xs = X[sub] if sub is not None else X
    return np.asarray(Xs[:, cols], dtype=np.float64), 0


def score_draft(model, names, exs):
    """Ranks of the chosen champion for the model and every baseline."""
    is_pick = "role_fit" in names
    out = {"rank": [], "gid": [], "base": {}}
    for X, y, meta in exs:
        X = np.asarray(X, dtype=np.float64)
        cols = [(PICK_FEATURES if is_pick else BAN_FEATURES).index(n) for n in names]
        out["rank"].append(ranks([(model.scores(X[:, cols]), y)])[0])
        out["gid"].append(meta["gameid"])
        for bname, s in baselines(X, meta, is_pick).items():
            out["base"].setdefault(bname, []).append(ranks([(s, y)])[0])
    out["rank"] = np.array(out["rank"])
    out["gid"] = np.array(out["gid"], dtype=object)
    out["base"] = {k: np.array(v) for k, v in out["base"].items()}
    return out


def baselines(X, meta, is_pick):
    lpr = np.log(meta["pr"].astype(float) + 1e-3)
    if is_pick:
        f = {n: X[:, PICK_FEATURES.index(n)] for n in ("meta_presence", "role_fit", "player_comfort")}
        return {
            "Presence (pick + ban rate)": f["meta_presence"],
            "Pick rate": lpr,
            "Presence × role fit": f["meta_presence"] + f["role_fit"],
            "Presence × role fit + 0.5 × comfort": f["meta_presence"] + f["role_fit"] + 0.5 * f["player_comfort"],
            "Pick rate × role fit + 0.5 × comfort": lpr + f["role_fit"] + 0.5 * f["player_comfort"],
            "Fixed-weight heuristic": heuristic_pick_scores(X, PICK_FEATURES),
        }
    f = {n: X[:, BAN_FEATURES.index(n)] for n in ("meta_presence", "ban_rate", "enemy_role_fit", "enemy_comfort")}
    return {
        "Presence (pick + ban rate)": f["meta_presence"],
        "Pick rate": lpr,
        "Ban rate": f["ban_rate"],
        "Ban rate × enemy role fit + 0.5 × enemy comfort": f["ban_rate"] + f["enemy_role_fit"] + 0.5 * f["enemy_comfort"],
    }


def rolling_draft(exs, names, folds):
    parts = []
    for f in folds:
        f0 = pd.Timestamp(min(f))
        fset = {pd.Timestamp(m) for m in f}
        tr = [train_view(X, meta, names) for X, y, meta in exs
              if pd.Timestamp(meta["month"]) < f0 and not meta["holdout"]]
        te = [(X, y, meta) for X, y, meta in exs if pd.Timestamp(meta["month"]) in fset and not meta["holdout"]]
        if not tr or not te:
            continue
        m = ConditionalLogit(names).fit(tr)
        parts.append(score_draft(m, names, te))
    out = {"rank": np.concatenate([p["rank"] for p in parts]), "gid": np.concatenate([p["gid"] for p in parts]),
           "base": {k: np.concatenate([p["base"][k] for p in parts]) for k in parts[0]["base"]}}
    return out


def draft_table(R, res: dict, B):
    rows = []
    for kind, r in res.items():
        rk, g = r["rank"], r["gid"]
        best = max(r["base"], key=lambda k: np.mean(r["base"][k] <= 5))
        for name, v in [("Model (conditional logit)", rk)] + list(r["base"].items()):
            t5 = boot_mean(v <= 5, g, B)
            d = paired_diff(rk <= 5, r["base"][best] <= 5, g, B) if name.startswith("Model") else None
            rows.append((kind, name + (" ← strongest baseline" if name == best else ""), len(v),
                         f"{np.mean(v <= 1):.1%}", fmt_ci(*t5, pct=True), f"{np.mean(v <= 10):.1%}",
                         f"{v.mean():.1f}", fmt_ci(*d, pct=True, sign=True) if d else ""))
    R.table(["Action", "Method", "n", "Top-1", "Top-5 [95% CI]", "Top-10", "Mean rank",
             "Δ top-5 vs strongest baseline"], rows)


def win_table(R, r, B):
    y = r["y"]
    rows = []
    for name, key in (("Always the training blue-win rate", "p_base"), ("Elo only", "p_elo"),
                      ("Draft only", "p_draft"), ("Elo + draft", "p_full")):
        p = r[key]
        ll = boot_mean(logloss_terms(y, p), None, B)
        auc = roc_auc_score(y, p) if key != "p_base" else 0.5
        rows.append((name, fmt_ci(*ll), f"{np.mean((p - y) ** 2):.4f}", f"{((p > 0.5) == y).mean():.1%}", f"{auc:.3f}"))
    R.p(f"{len(y)} games. Log-loss with 95% interval; lower log-loss/Brier is better.")
    R.table(["Model", "Log-loss [95% CI]", "Brier", "Accuracy", "AUC"], rows)
    d = paired_diff(logloss_terms(y, r["p_full"]), logloss_terms(y, r["p_elo"]), None, B)
    verdict = ("the draft clearly adds information" if d[2] < 0 else
               "the draft clearly makes it worse" if d[1] > 0 else "not distinguishable from zero")
    R.p(f"**Elo + draft minus Elo only: {fmt_ci(*d, sign=True)} log-loss** ({verdict}).")


def stage_scores(S, s_rows, res, mon, hold, yall, holdout=False):
    out = {}
    models = res["models"]
    for s, X in S.items():
        y, pf, pe, g = [], [], [], []
        for j, i in enumerate(s_rows):
            if hold[i] != holdout:
                continue
            m = pd.Timestamp(mon[i])
            if m not in models:
                continue
            full, eo = models[m]
            pf.append(full.proba(X[j])[0])
            pe.append(eo.proba(X[j][:1])[0])
            y.append(yall[i])
            g.append(i)
        out[s] = (np.array(y, float), np.array(pf), np.array(pe))
    return out


def stage_table(R, st, B, label):
    rows = []
    for s, (y, pf, pe) in st.items():
        if len(y) == 0:
            continue
        d = paired_diff(logloss_terms(y, pf), logloss_terms(y, pe), None, B)
        rows.append((s, len(y), f"{logloss_terms(y, pe).mean():.4f}", f"{logloss_terms(y, pf).mean():.4f}",
                     fmt_ci(*d, sign=True), f"{np.mean(np.abs(pf - pe)) * 100:.1f}"))
    R.p(f"**{label}.**")
    R.table(["Stage", "Games", "Elo only", "Elo + draft", "Δ [95% CI]", "Avg. draft shift (pts)"], rows)


def stage_verdict(sv, sh, B):
    good, bad = [], []
    for s in STAGES:
        for st, lab in ((sv, "validation"), (sh, "holdout")) if sh is not None else ((sv, "validation"),):
            y, pf, pe = st[s]
            if not len(y):
                continue
            d = paired_diff(logloss_terms(y, pf), logloss_terms(y, pe), None, B)
            if d[2] < 0:
                good.append(f"{s} ({lab})")
            if d[1] > 0:
                bad.append(f"{s} ({lab})")
    return ("**Which stages are reliable.** "
            + (f"The draft terms clearly improve on Elo at: {', '.join(good)}. " if good else
               "At no stage do the draft terms clearly improve on Elo alone. ")
            + (f"They clearly hurt at: {', '.join(bad)}. " if bad else "At no stage do they clearly hurt. ")
            + "Stages with neither are 'not validated': the win chance there is Elo's, and the draft shift "
              "should be shown as a slight/unclear tilt, not a number.")


def current_patch(games, days: int = 14):
    recent = games[games["date"] >= games["date"].max() - pd.Timedelta(days=days)]
    return int(recent["patch_idx"].dropna().mode().iloc[0])


def write_matchups(games, snap):
    """Cross-season counter table -> reports/matchups.csv + reports/counters.md."""
    t = pd.concat([matchup_table(games), team_matchup_table(games)], ignore_index=True)
    current = {c for c in snap.champions if snap.presence(c) >= 0.01}
    REPORT_DIR.mkdir(exist_ok=True)
    t.to_csv(REPORT_DIR / "matchups.csv", index=False)
    (REPORT_DIR / "counters.md").write_text(counters_report(t, current), encoding="utf-8")
    print(f"  matchups: {len(t)} lane matchups with 10+ games -> reports/counters.md")
    return t


class Report:
    def __init__(self):
        self.lines = []

    def h1(self, t): self.lines += [f"# {t}", ""]
    def h2(self, t): self.lines += [f"## {t}", ""]
    def h3(self, t): self.lines += [f"### {t}", ""]
    def p(self, t): self.lines += [t, ""]

    def table(self, head, rows):
        self.lines.append("| " + " | ".join(head) + " |")
        self.lines.append("|" + "---|" * len(head))
        for r in rows:
            self.lines.append("| " + " | ".join(str(x) for x in r) + " |")
        self.lines.append("")

    def text(self): return "\n".join(self.lines)


if __name__ == "__main__":
    main()
